import csv
import random
import argparse
import os
from math import sqrt, ceil
import subprocess
from matplotlib import rcParams
import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
import copy
from dataclasses import dataclass

import RandomParameters
from RandomParameters import DynamicParameter, DynamicParameterList, StaticParameter, StaticParameterList
import MergeRandomOutputs
from DynamicEquivalent import *


if __name__ == "__main__":
    random.seed(1)
    start_time = time.time()
    parser = argparse.ArgumentParser('Takes as input the same files as a systematic analysis (SA), and generates a randomised version of those files. '
    'Those can then be used to perform another SA. The ouput files are all written to working_dir/RandomisedInputs/')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--fic_MULTIPLE', type=str, required=True,
                        help='Input file containing the different scenarios to run')
    parser.add_argument('--reduced_fic_MULTIPLE', type=str, required=True,
                        help='Input file containing the different scenarios to run for the reduced model')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the network')
    parser.add_argument('--reduced_name', type=str, required=True,
                        help='Name of the reduced network')
    parser.add_argument('--nb_threads', type=str, required=True,
                        help="Number of threads (to use in the SA's)")
    # Random runs
    parser.add_argument('--csv_par', type=str, required=True,
                        help='Csv file containing the list of dynamic parameters to be randomised and associated distributions')
    parser.add_argument('--csv_iidm', type=str, required=True,
                        help='Csv file containing the list of static parameters to be randomised and associated distributions')
    parser.add_argument('--nb_runs_random', type=int, required=True,
                        help='Maximum number of random SA\'s to run')
    parser.add_argument('--curve_names', '-c', type=str, required=True, action='append',
                        help='List of the names of the curves that have to be merged')
    parser.add_argument('--time_precision', type=float, required=True,
                        help='Time precision of the output curves (linear interpolation from the input ones)')
    # Parameter optimisation
    parser.add_argument('--csv_par_bounds', type=str, required=True,
                        help='Csv file containing the list of dynamic parameters to be optimised and their bounds')
    parser.add_argument('--csv_iidm_bounds', type=str, required=True,
                        help='Csv file containing the list of static parameters to be optimised and their bounds')
    # Balancing (assume same names for full and reduced model)
    parser.add_argument('--target_Q', type=str, required=True,
                        help='Reactive power that should be produced by the infinite bus')
    parser.add_argument('--slack_load_id', type=str, required=True,
                        help='Id of the slack bus that balances the reactive power (should be connected to the slack bus). '
                        'In future versions of pypowsybl, it will be possible to add new elements (then remove them I suppose) '
                        '-> The slack load and generators won\'t be needed anymore.')
    parser.add_argument('--slack_gen_id', type=str, required=True,
                        help='Id of the generator that emulates the infinite bus in powsybl')
    args = parser.parse_args()

    working_dir = args.working_dir
    fic_MULTIPLE = os.path.join(working_dir, args.fic_MULTIPLE)
    reduced_fic_MULTIPLE = os.path.join(working_dir, args.reduced_fic_MULTIPLE)
    network_name = args.name
    reduced_network_name = args.reduced_name
    csv_par = os.path.join(working_dir, args.csv_par)
    csv_iidm = os.path.join(working_dir, args.csv_iidm)
    csv_par_bounds = os.path.join(working_dir, args.csv_par_bounds)
    csv_iidm_bounds = os.path.join(working_dir, args.csv_iidm_bounds)
    target_Q = float(args.target_Q)
    nb_runs_random = args.nb_runs_random
    slack_load_id = args.slack_load_id
    slack_gen_id = args.slack_gen_id
    slack_gen_type = "Line"
    nb_threads = args.nb_threads
    curve_names = args.curve_names
    time_precision = args.time_precision

    MIN_NB_RUNS = 10

    rerun_random = False
    rerun_protection = False
    rerun_average = False
    rerun_lasso = False
    rerun_de = False

    if rerun_random:  # If intermediary results changes, need to rerun following steps
        rerun_protection = True
    if rerun_protection:
        rerun_average = True
    if rerun_average:
        rerun_lasso = True
    if rerun_lasso:
        rerun_de = True

    ###
    # Part 1: random runs
    ###

    run_fic_MULTIPLE = []
    sigma_thr = 0.01

    for run_id in range(max(nb_runs_random, MIN_NB_RUNS)):
        output_dir_name = os.path.join('RandomRuns_update', 'It_%03d' % run_id)
        if rerun_random:
            runRandomSA(csv_iidm, csv_par, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q, slack_load_id, slack_gen_id)

        current_fic = os.path.join(working_dir, output_dir_name, 'fic_MULTIPLE.xml')
        run_fic_MULTIPLE.append(current_fic)

        new_curves = -100 * MergeRandomOutputs.mergeCurves([current_fic], curve_names, time_precision)  # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
        if run_id == 0:
            random_curves = new_curves
        else:
            random_curves = np.concatenate((random_curves, new_curves), axis=2)  # ndarray(curve_name, scenario, run, t_step)
        if np.any(np.abs(new_curves[:,:,0, 0] - new_curves[:,:,0, 50]) > new_curves[:,:,0, 0] * 0.1):
            print('Initialisation does not lead to steady-state')

        if run_id > MIN_NB_RUNS:
            mean = np.mean(random_curves, axis=2)  # average over runs
            sigma = np.std(random_curves, axis=2, ddof=1)  # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)

            for (x,y,z), value in np.ndenumerate(sigma):
                sigma[x,y,z] = max(value, 1e-3)  # Avoid division by 0

            std_error = sigma / abs(mean[:,:,[0 for i in range(sigma.shape[2])]]) / sqrt(run_id + 1)  # Normalise with respect to value at t = 0
            print('Run_id: %d, Std error: %f' % (run_id, std_error.max()*100), end='\n')
            if std_error.max() < sigma_thr:
                print("\nThreshold reached in %d iterations" % (run_id + 1))
                break

    if std_error.max() > sigma_thr:
        print("Warning: maximum number of random runs ({}) reached with sigma ({}%) > tol ({}%)".format(nb_runs_random, std_error.max()*100, sigma_thr*100))

    # Merge all curves and write to csv (only done once at the end for performance)
    output_dir_curves = os.path.join(working_dir, "MergedCurves")
    # MergeRandomOutputs.mergeCurves(run_fic_MULTIPLE, curve_names, time_precision, write_to_csv=True, output_dir=output_dir_curves)

    """ centroids = Clustering.getCentroids(random_curves)
    worst_tripping = np.argmax(centroids[0,3,:,-1])
    print(worst_tripping)
    centroid = centroids[:,:,worst_tripping,:]  # Worst centroid probably """
    centroids = None
    centroid = None # Not used atm anyway

    representative_index = 0
    representative_dist = 999999
    nb_runs = random_curves.shape[2]
    for i in range(nb_runs):
        dist = np.sum(np.mean(np.mean(((random_curves[:,:,i,:] - mean) / sigma)**2, axis=2), axis=1))
        if dist < representative_dist:
            representative_dist = dist
            representative_index = i
    print('\nRepresentative index:', representative_index)

    plotCurves(random_curves, centroids, 'Random_updated')
    randomising_time = time.time()
    print('Spent %.1fs on randomising the full model' % (randomising_time-start_time))

    ###
    # Part 2: optimisation
    ###

    np.random.seed(int(42))

    dyn_bounds = DynamicParameterListWithBounds(csv_par_bounds)
    dyn_bounds_list = dyn_bounds.bounds
    nb_dyn_params = len(dyn_bounds_list)

    static_bounds = StaticParameterListWithBounds(csv_iidm_bounds)
    static_bounds_list = static_bounds.bounds

    bounds = dyn_bounds_list + static_bounds_list
    # refineBounds_ = lambda trial_denorm : refineBounds(trial_denorm, dyn_bounds, static_bounds)
    parameter_names = getParameterNames(dyn_bounds, static_bounds)

    def fobj2(value_list, lasso_factor, output_dir, disturbance_ids = None, convergence_criteria = 1.0, rerun = True, tripping_only = False):
        global run_id 
        output_dir_name = os.path.join(output_dir, "It_%03d" % run_id)
        if rerun:
            runSAFromValueList(value_list, nb_dyn_params, dyn_bounds, static_bounds, working_dir, output_dir_name, reduced_fic_MULTIPLE, reduced_network_name, run_id, target_Q, slack_load_id, slack_gen_id, slack_gen_type, disturbance_ids=disturbance_ids)
        current_fic = os.path.join(working_dir, output_dir_name, 'fic_MULTIPLE.xml')
        curves = -100 * MergeRandomOutputs.mergeCurves([current_fic], curve_names, time_precision)  # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
        curves = curves[:,:,0,:]  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)
        if disturbance_ids is not None:
            random_curves_considered = random_curves[:,disturbance_ids,:]
        else:
            random_curves_considered = random_curves
        obj = objective(curves, random_curves_considered, centroid, value_list, bounds, lasso_factor, parameter_names, convergence_criteria, tripping_only)
        print('Run id: %d, objective: %f' %(run_id, obj.total_obj))
        run_id += 1
        return obj


    with open(os.path.join(working_dir, 'Best_parameters.csv'), newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            best_parameters = row
            for i in range(len(best_parameters)):
                best_parameters[i] = float(best_parameters[i])

    # 'Heuristics' to make best_parameters better
    parameter_names = getParameterNames(dyn_bounds, static_bounds)
    for i in range(len(best_parameters)):
        if parameter_names[i] == 'IBG-2_target_p':
            best_parameters[i] /= 2
        if 'load_ActiveMotorShare' in parameter_names[i]:
            best_parameters[i] *= 2

    run_id = 0
    result = fobj2(best_parameters, 0, 'Update_test')

    fitted_curves = -100 * MergeRandomOutputs.mergeCurves([os.path.join(working_dir, 'Update_test', "It_%03d" % 0, 'fic_MULTIPLE.xml')], curve_names, time_precision)
    fitted_curves = fitted_curves[:,:,0,:]  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)
    plotCurves(random_curves, centroids, 'Fit_update_no_DE', fitted_curves)
    if result.converged:
        raise Exception('Success')

    # DE parameters
    pop_size = 20
    nb_max_iterations_DE = 20

    # DE
    print('\nDE')
    def fobj(value_list):
        return fobj2(value_list, 0, 'Optimisation_update', rerun=rerun_de)
    results = list(de(fobj, bounds, popsize=pop_size, its=nb_max_iterations_DE, init=best_parameters))
    print('DE', ": objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))

    # Results
    printParameters(results[-1], dyn_bounds, static_bounds, working_dir, 'results.txt')
    best_parameters = results[-1].best_parameters
    with open(os.path.join(working_dir, 'Best_parameters_update.csv'), 'w') as file:
        file.write(','.join([str(param) for param in best_parameters]))

    print("Objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))
    convergence_evolution = [r.best_obj.total_obj for r in results]
    plt.plot(convergence_evolution)
    plt.savefig('Convergence_Update.png', bbox_inches='tight')
    plt.close()

    optimising_time = time.time()
    print('Spent %.1fs on randomising the full model' % (randomising_time-start_time))
    print('Spent %.1fs on optimising the reduced model' % (optimising_time-randomising_time))

    best_run_id = results[-1].best_index + pop_size * (results[-1].best_iteration + 1)
    fitted_curves = -100 * MergeRandomOutputs.mergeCurves([os.path.join(working_dir, 'Optimisation_update', "It_%03d" % best_run_id, 'fic_MULTIPLE.xml')], curve_names, time_precision)
    fitted_curves = fitted_curves[:,:,0,:]  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)

    plotCurves(random_curves, centroids, 'Fit_update', fitted_curves)

    # Find sample most representative of full model for comparison with equivalent
    mean = np.mean(random_curves, axis=2)  # average over runs
    sigma = np.std(random_curves, axis=2, ddof=1)
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(value, 1e-3)  # Avoid division by 0
    percentile_5, percentile_95 = np.percentile(random_curves, axis=2, q=[5, 95])

    print('\nDE best run id:', best_run_id)
    print('\nRepresentative index:', representative_index)
