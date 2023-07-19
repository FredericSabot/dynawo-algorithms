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
from dataclasses import dataclass

import RandomParameters
from RandomParameters import DynamicParameter, DynamicParameterList, StaticParameter, StaticParameterList
import MergeRandomOutputs

def normaliseParameters(parameters, bounds):
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    norm_parameters = []
    for i in range(len(parameters)):
        if diff[i] == 0:
            norm_parameters.append(0.5)
        else:
            norm_parameters.append((parameters[i] - min_b[i]) / diff[i])
    return norm_parameters

def denormaliseParameters(normalised_parameters, bounds):
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    return min_b + normalised_parameters * diff

def de(fobj, bounds, mut=0.8, crossp=0.95, popsize=20, its=1000, init=None):
    """
    Differential Evolution (DE) algorithm, adapted from https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
    """
    global run_id
    run_id = 0  # Reset run_id
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    pop_denorm = denormaliseParameters(pop, bounds)

    if init is not None:
        pop_denorm[0] = init
        pop_denorm[-1] = init
    objs = [fobj(ind) for ind in pop_denorm]
    fitness = np.asarray([obj.total_obj for obj in objs])
    best_idx = np.argmin(fitness)
    best_obj = objs[best_idx]
    best = pop_denorm[best_idx]
    best_it = 0
    best_converged = False
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = denormaliseParameters(trial, bounds)
            obj = fobj(trial_denorm)
            f = obj.total_obj
            converged = obj.converged
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best_obj = obj
                    best = trial_denorm
                    best_it = i
                    best_converged = converged
        yield DEResults(best_idx, best_it, best, best_obj, best_converged)
        if best_converged:
            return

@dataclass
class Objective:
    total_obj: float
    obj_name_disturb: np.ndarray
    parameter_dist: list
    converged: bool

@dataclass
class DEResults:
    best_index: int
    best_iteration: int
    best_parameters: list
    best_obj: Objective
    converged: bool

    def __repr__(self) -> str:
        return str(self.best_index) + ',' + str(self.best_iteration) + ',' + str(self.best_parameters) + ',' + str(self.best_obj) + ',' + str(self.converged)

def refineStaticParameters(static_bounds):
    pass
    return static_bounds

class DynamicParameterListWithBounds:
    def __init__(self, bounds_csv):
        self.param_list = DynamicParameterList()
        self.bounds = []
 
        with open(bounds_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            row = spamreader.__next__()
            if row != ['ParamSet_id', 'Param_id', 'L_bound', 'U_bound']:
                raise Exception("Incorrect format of %s" % bounds_csv)

            for row in spamreader:
                paramSetId = row[0]
                paramId = row[1]
                bounds = [float(row[2]), float(row[3])]

                self.param_list.append(DynamicParameter(paramSetId, paramId, None))
                self.bounds.append(bounds)

    def __append__(self, parameter, bounds):
        """
        @param parameter DynamicParameter object to append
        @param bounds tuple (min_bound, max_bound)
        """
        self.param_list.append(parameter)
        self.bounds.append(bounds)
    
    def valueListToParameterList(self, value_list):
        if len(value_list) != len(self.bounds):
            raise ValueError('value_list and self_bounds should have the same length')
        
        parameter_list = DynamicParameterList()
        for i in range(len(value_list)):
            parameter_list.append(self.param_list[i])
            parameter_list[i].value = value_list[i]
        return parameter_list


class StaticParameterListWithBounds:
    def __init__(self, bounds_csv):
        self.param_list = StaticParameterList()
        self.bounds = []

        with open(bounds_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            row = spamreader.__next__()
            if row != ['Component_type', 'Component_name', 'Param_id', 'L_bound', 'U_bound']:
                print(row)
                raise Exception("Incorrect format of %s" % bounds_csv)

            for row in spamreader:
                componentType = row[0]
                componentName = row[1]
                paramId = row[2]
                bounds = [float(row[3]), float(row[4])]

                self.param_list.append(StaticParameter(componentType, componentName, paramId, None))
                self.bounds.append(bounds)

    def __append__(self, parameter, bounds):
        """
        @param parameter DynamicParameter object to append
        @param bounds tuple (min_bound, max_bound)
        """
        self.param_list.append(parameter)
        self.bounds.append(bounds)

    def valueListToParameterList(self, value_list):
        if len(value_list) != len(self.bounds):
            raise ValueError('value_list and self_bounds should have the same length')
        
        parameter_list = StaticParameterList()
        for i in range(len(value_list)):
            parameter_list.append(self.param_list[i])
            parameter_list[-1].value = value_list[i]
        return parameter_list


def runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q=None, slack_load_id=None, slack_gen_id=None, slack_gen_type=None, disturbance_ids=None, nb_threads='6'):
    RandomParameters.writeParametricSAInputs(working_dir, fic_MULTIPLE, network_name, output_dir_name, static_parameters, dyn_parameters,
            run_id, target_Q, slack_load_id, slack_gen_id, slack_gen_type, disturbance_ids)

    output_dir = os.path.join(working_dir, output_dir_name)
    cmd = ['./myEnvDynawoAlgorithms.sh', 'SA', '--directory', output_dir, '--input', 'fic_MULTIPLE.xml',
            '--output' , 'aggregatedResults.xml', '--nbThreads', nb_threads]
    output = subprocess.run(cmd, capture_output=True, text=True)
    # if output.stderr != '':
    #     print(output.stderr, end='')


def runRandomSA(static_csv, dynamic_csv, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q=None, slack_load_id=None, slack_gen_id=None, slack_gen_type=None):
    full_network_name = os.path.join(working_dir, network_name)
    static_parameters = RandomParameters.randomiseStaticParams(full_network_name + '.iidm', static_csv)
    dyn_parameters = RandomParameters.randomiseDynamicParams(full_network_name + '.par', dynamic_csv)

    runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q, slack_load_id, slack_gen_id, slack_gen_type)


def runSAFromValueList(value_list, nb_dyn_params, dyn_bounds, static_bounds, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q=None, slack_load_id=None, slack_gen_id=None, slack_gen_type=None, total_load=None, total_gen=None, disturbance_ids=None):
    dyn_value_list, static_value_list = value_list[:nb_dyn_params], value_list[nb_dyn_params:]
    dyn_parameters = dyn_bounds.valueListToParameterList(dyn_value_list)
    static_parameters = static_bounds.valueListToParameterList(static_value_list)

    if total_load is not None and total_gen is not None:
        static_parameters = refineStaticParameters(static_parameters, total_load, total_gen)

    runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q, slack_load_id, slack_gen_id, slack_gen_type, disturbance_ids)


def objective(fitted_curves, random_curves, value_list, bounds, lasso_factor, parameter_names, convergence_criteria = 1, tripping_only = False):
    median = np.median(random_curves, axis=2)  # average over runs
    sigma = np.std(random_curves, axis=2, ddof=1)
    sigma_fixed = sigma.copy()
    for (x,y,z), value in np.ndenumerate(sigma_fixed):
        sigma_fixed[x,y,z] = max(value, 1e-3)  # Avoid division by 0
    percentile_5, percentile_95 = np.percentile(random_curves, axis=2, q=[5, 95])
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(max(value, 1e-2), 1e-2 * abs(median[x,y,z]))  # Avoid division by 0 + allow some tolerance
        sigma[x,y,z] = min(sigma[x,y,z], abs(0.1 * max(median[x,y,0], median[x,y,z])))  # Limit tolerance when very high dispersion # Issue if median close to 0 at some point

    obj = ((fitted_curves - percentile_95) / sigma)
    for (x,y,z), value in np.ndenumerate(obj):
        if value > 0:
            obj[x, y, z] = value / 2
    obj = obj**2
    obj = np.clip(obj, 0, 100)  # Allow for large errors if they only occur during a very short period.
                                # Could also use dynamic time warping or another similar metric

    obj_name_disturb = np.mean(obj, axis=2)  # 0.5 * np.percentile(obj, axis=2, q=95) + 0.5 * np.mean(obj, axis=2)  # average over time

    if tripping_only:  # Only consider active power steady-state error
        for (x,y), value in np.ndenumerate(obj_name_disturb):
            obj_name_disturb[x,y] = 0 + abs(percentile_95[x,y,-1] - fitted_curves[x,y,-1]) / sigma[x,y,-1]  # Only look at steady-state deviation error
            if x == 1:
                obj_name_disturb[x,y] = 0  # Neglect reactive power

    if np.max(obj_name_disturb) <= convergence_criteria:
        converged = True
    else:
        converged = False

    obj = np.sum(obj_name_disturb, axis=0)  # sum over curve_names (typically P and Q at point of common coupling)

    if tripping_only:
        obj = np.mean(obj) + np.max(obj)
    else:
        obj = np.mean(obj)  # average over disturbances

    if obj < 6:
        pass
    normalised_parameters = normaliseParameters(value_list, bounds)
    parameter_dist = [abs(i - 0.5) for i in normalised_parameters]
    total_obj = obj + lasso_factor * sum([0 if isTrippingParameter(parameter_names[i]) else parameter_dist[i] for i in range(len(parameter_dist))])

    return Objective(total_obj, obj_name_disturb, parameter_dist, converged)

def plotCurves(curves, plot_name, fitted_curves = None, time_precision=1e-2):  # ndarray(curve_name, scenario, run, t_step)
    nb_disturb = curves.shape[1]
    nb_runs = curves.shape[2]
    nb_time_steps = curves.shape[3]
    t_axis = np.array([i * time_precision for i in range(nb_time_steps)])
    sqrt_d = int(ceil(sqrt(nb_disturb)))
    rcParams['figure.figsize'] = 12, 7.2
    curve_names = ['P (MW)', 'Q (MVar)']

    median = np.median(curves, axis=2)  # average over runs
    sigma = np.std(curves, axis=2, ddof=1)  # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)
    percentile_5, percentile_95 = np.percentile(curves, axis=2, q=[5, 95])
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(max(value, 1e-2), 1e-2 * abs(median[x,y,z]))  # Avoid division by 0 + allow some tolerance
        sigma[x,y,z] = min(sigma[x,y,z], abs(0.1 * median[x,y,0]))  # Limit tolerance when very high dispersion # Issue if median close to 0 at some point

    if fitted_curves is not None:
        obj = ((fitted_curves - percentile_95) / sigma)
        for (x,y,z), value in np.ndenumerate(obj):
            if value < 0:
                obj[x, y, z] = value / 2
        obj = obj**2
        obj = np.clip(obj, 0, 100)  # Allow for large errors if they only occur during a very short period
    else:
        obj = None

    representative_index = 0
    representative_dist = 999999
    nb_runs = curves.shape[2]
    for i in range(nb_runs):
        dist = np.sum(np.mean(np.mean(((curves[:,:,i,:] - percentile_95) / sigma)**2, axis=2), axis=1))
        if dist < representative_dist:
            representative_dist = dist
            representative_index = i

    for c in range(curves.shape[0]):
        fig, axs = plt.subplots(sqrt_d, sqrt_d)
        for d in range(curves.shape[1]):
            axs2 = axs[d//sqrt_d, d%sqrt_d].twinx()
            axs[d//sqrt_d, d%sqrt_d].set_title('Disturbance %d' % (d + 1))
            axs[d//sqrt_d, d%sqrt_d].set_xlabel('Time (s)')
            axs[d//sqrt_d, d%sqrt_d].set_ylabel(curve_names[c])
            axs2.set_ylabel('Error')
            axs2.set_ylim([0, 10])
            for r in range(nb_runs):
                axs[d//sqrt_d, d%sqrt_d].plot(t_axis, curves[c,d,r,:], ':', linewidth=1, alpha=0.3)
            axs[d//sqrt_d, d%sqrt_d].plot(t_axis, percentile_5[c,d,:], label='5th percentile', zorder=1000, alpha=0.7)
            axs[d//sqrt_d, d%sqrt_d].plot(t_axis, percentile_95[c,d,:], label='95th percentile', zorder=1000, alpha=0.7)
            # axs[d//sqrt_d, d%sqrt_d].plot(t_axis, median[c,d,:], 'green', label='Median', zorder=1000, alpha=0.7)

            if obj is not None:
                axs2.plot(t_axis, obj[c,d,:], label='Error', zorder=1000, alpha=0.3)
            # axs[d//sqrt_d, d%sqrt_d].plot(t_axis, curves[c,d,representative_index,:], label='Representative', zorder=2000)

            if fitted_curves is not None:
                axs[d//sqrt_d, d%sqrt_d].plot(t_axis, fitted_curves[c,d,:], 'red', label='Fit', zorder=3000)
        handles, labels = axs[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')
        plt.tight_layout()
        plt.savefig(plot_name + '_%d.pdf' % c, bbox_inches='tight')
        plt.close()

def getParameterNames(dyn_bounds, static_bounds):
    parameter_names = []
    for parameter in dyn_bounds.param_list:
        parameter_names.append(parameter.set_id + '_' + parameter.id)
    for parameter in static_bounds.param_list:
        parameter_names.append(parameter.component_name + '_' + parameter.id)
    return parameter_names

def printParameters(de_results : DEResults, dyn_bounds, static_bounds, working_dir = None, output_file = None, output_file_human_readable = None):
    parameter_names = getParameterNames(dyn_bounds, static_bounds)

    parameter_values = de_results.best_parameters
    for i in range(len(parameter_values)):
        print(parameter_names[i], parameter_values[i])

    if working_dir is not None and output_file_human_readable is not None:
        with open(os.path.join(working_dir, output_file_human_readable), 'w') as file:
            parameter_values = de_results.best_parameters
            for i in range(len(parameter_values)):
                file.write(parameter_names[i] + ' ' + str(parameter_values[i]))
                file.write('\n')

    if working_dir is not None and output_file is not None:
        with open(os.path.join(working_dir, output_file), 'w') as file:
            file.write(repr(de_results))
            file.write('\n')

def isTrippingParameter(parameter_name):
    if 'LVRT' in parameter_name:
        return True
    else:
        return False


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
    rerun_de = False

    if rerun_random:  # If intermediary results changes, need to rerun following steps
        rerun_de = True

    ###
    # Part 1: random runs
    ###

    run_fic_MULTIPLE = []
    sigma_thr = 0.01

    for run_id in range(max(nb_runs_random, MIN_NB_RUNS)):
        output_dir_name = os.path.join('RandomRuns', 'It_%03d' % run_id)
        if rerun_random:
            runRandomSA(csv_iidm, csv_par, working_dir, output_dir_name, fic_MULTIPLE, network_name, run_id, target_Q, slack_load_id, slack_gen_id, slack_gen_type)

        current_fic = os.path.join(working_dir, output_dir_name, 'fic_MULTIPLE.xml')
        run_fic_MULTIPLE.append(current_fic)

        new_curves = -100 * MergeRandomOutputs.mergeCurvesFromFics([current_fic], curve_names, time_precision)  # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
        if run_id == 0:
            random_curves = new_curves
        else:
            random_curves = np.concatenate((random_curves, new_curves), axis=2)  # ndarray(curve_name, scenario, run, t_step)
        if np.any(np.abs(new_curves[:,:,0, 0] - new_curves[:,:,0, 50]) > 0.1):
            print(run_id)
            print('Initialisation does not lead to steady-state')

        if run_id > MIN_NB_RUNS:
            median = np.median(random_curves, axis=2)  # average over runs
            sigma = np.std(random_curves, axis=2, ddof=1)  # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)

            for (x,y,z), value in np.ndenumerate(sigma):
                sigma[x,y,z] = max(value, 1e-3)  # Avoid division by 0

            std_error = sigma / abs(median[:,:,[0 for i in range(sigma.shape[2])]]) / sqrt(run_id + 1)  # Normalise with respect to value at t = 0
            print('Run_id: %d, Std error: %f' % (run_id, std_error.max()*100), end='\n')
            if std_error.max() < sigma_thr:
                print("\nThreshold reached in %d iterations" % (run_id + 1))
                break

    if std_error.max() > sigma_thr:
        print("Warning: maximum number of random runs ({}) reached with sigma ({}%) > tol ({}%)".format(nb_runs_random, std_error.max()*100, sigma_thr*100))

    # Merge all curves and write to csv (only done once at the end for performance)
    output_dir_curves = os.path.join(working_dir, "MergedCurves")
    # MergeRandomOutputs.mergeCurvesFromFics(run_fic_MULTIPLE, curve_names, time_precision, write_to_csv=True, output_dir=output_dir_curves)

    randomising_time = time.time()

    plotCurves(random_curves, 'Random')
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
    parameter_names = getParameterNames(dyn_bounds, static_bounds)

    def fobj2(value_list, lasso_factor, output_dir, disturbance_ids = None, convergence_criteria = 1.0, rerun = True, tripping_only = False):
        global run_id
        output_dir_name = os.path.join(output_dir, "It_%03d" % run_id)
        if rerun:
            runSAFromValueList(value_list, nb_dyn_params, dyn_bounds, static_bounds, working_dir, output_dir_name, reduced_fic_MULTIPLE, reduced_network_name, run_id, target_Q, slack_load_id, slack_gen_id, slack_gen_type, disturbance_ids=disturbance_ids)
        current_fic = os.path.join(working_dir, output_dir_name, 'fic_MULTIPLE.xml')
        curves = -100 * MergeRandomOutputs.mergeCurvesFromFics([current_fic], curve_names, time_precision)  # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
        curves = curves[:,:,0,:]  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)
        if disturbance_ids is not None:
            random_curves_considered = random_curves[:,disturbance_ids,:]
        else:
            random_curves_considered = random_curves
        obj = objective(curves, random_curves_considered, value_list, bounds, lasso_factor, parameter_names, convergence_criteria, tripping_only)
        print('Run id: %d, objective: %f' %(run_id, obj.total_obj))
        run_id += 1
        return obj

    # DE parameters
    pop_size = 20
    nb_max_iterations_DE = 20

    """ # Identify disturbances that do not lead to tripping
    percentile_5, percentile_95 = np.percentile(random_curves, axis=2, q=[5, 95])
    nb_disturb = random_curves.shape[1]
    nb_time_steps = random_curves.shape[2]
    no_trip_cases = []
    for d in range(nb_disturb):
        if abs(percentile_5[0, d, 0] - percentile_5[0, d, -1]) < 1e-2 and abs(percentile_95[0, d, 0] - percentile_95[0, d, -1]) < 1e-2:
            no_trip_cases.append(d)
    print(no_trip_cases) """

    # DE trip
    convergence_criteria = 1 # max(1, np.max(results[-1].best_obj.obj_name_disturb))
    nb_max_iterations_DE = 20
    print('\nDE trip')
    def fobj(value_list):
        return fobj2(value_list, 0, 'Optimisation_trip', rerun=rerun_de, convergence_criteria=convergence_criteria)
    results = list(de(fobj, bounds, popsize=pop_size, its=nb_max_iterations_DE))
    print('DE', ": objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))

    # Results
    printParameters(results[-1], dyn_bounds, static_bounds, working_dir, 'results.txt', 'Optimised parameters.txt')
    best_parameters = results[-1].best_parameters
    with open(os.path.join(working_dir, 'Best_parameters.csv'), 'w') as file:
        file.write(','.join([str(param) for param in best_parameters]))

    print("Objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))
    convergence_evolution = [r.best_obj.total_obj for r in results]
    plt.plot(convergence_evolution)
    plt.savefig('Convergence.png', bbox_inches='tight')
    plt.close()

    optimising_time = time.time()
    print('Spent %.1fs on randomising the full model' % (randomising_time-start_time))
    print('Spent %.1fs on optimising the reduced model' % (optimising_time-randomising_time))

    best_run_id = results[-1].best_index + pop_size * (results[-1].best_iteration + 1)
    fitted_curves = -100 * MergeRandomOutputs.mergeCurvesFromFics([os.path.join(working_dir, 'Optimisation_trip', "It_%03d" % best_run_id, 'fic_MULTIPLE.xml')], curve_names, time_precision)
    # fitted_curves = -100 * MergeRandomOutputs.mergeCurvesFromFics([os.path.join(working_dir, 'OptimisationCheck' + '_' + '_'.join(str(id) for id in worst_disturbances), "It_%03d" % 0, 'fic_MULTIPLE.xml')], curve_names, time_precision)
    fitted_curves = fitted_curves[:,:,0,:]  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)

    # random_curves = random_curves[:,no_trip_cases,:,:]
    plotCurves(random_curves, 'Fit', fitted_curves)

    # Find sample most representative of full model for comparison with equivalent
    median = np.median(random_curves, axis=2)  # average over runs
    sigma = np.std(random_curves, axis=2, ddof=1)
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(value, 1e-3)  # Avoid division by 0
    percentile_5, percentile_95 = np.percentile(random_curves, axis=2, q=[5, 95])

    representative_index = 0
    representative_dist = 999999
    nb_runs = random_curves.shape[2]
    for i in range(nb_runs):
        dist = np.sum(np.mean(np.mean(((random_curves[:,:,i,:] - percentile_95) / sigma)**2, axis=2), axis=1))
        if dist < representative_dist:
            representative_dist = dist
            representative_index = i
    print('\nRepresentative index:', representative_index)
    print('Distance:', representative_dist)

    print('\nDE best run id:', best_run_id)
    print("Objective: %f, converged: %d" % (results[-1].best_obj.total_obj, results[-1].converged))
