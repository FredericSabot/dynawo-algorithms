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

import RandomParameters
from RandomParameters import DynamicParameter, DynamicParameterList, StaticParameter, StaticParameterList
import MergeRandomOutputs

def de(fobj, bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    """
    Differential Evolution (DE) algorithm from https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
    """
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best_idx, best, fitness[best_idx]

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
                bounds = (float(row[2]), float(row[3]))

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
            raise ValueError('value_list and self_bounds list should have the same length')
        
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
                bounds = (float(row[3]), float(row[4]))

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
            raise ValueError('value_list and self_bounds list should have the same length')
        
        parameter_list = StaticParameterList()
        for i in range(len(value_list)):
            parameter_list.append(self.param_list[i])
            parameter_list[-1].value = value_list[i]
        return parameter_list


def runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, target_Q=None, slack_load_id=None, slack_gen_id=None):
    RandomParameters.writeParametricSAInputs(working_dir, fic_MULTIPLE, network_name, output_dir_name, static_parameters, dyn_parameters,
            run_id, target_Q, slack_load_id, slack_gen_id)

    output_dir = os.path.join(working_dir, output_dir_name)
    cmd = ['./myEnvDynawoAlgorithms.sh', 'SA', '--directory', output_dir, '--input', 'fic_MULTIPLE.xml',
            '--output' , 'aggregatedResults.xml', '--nbThreads', nb_threads]
    subprocess.run(cmd)


def runRandomSA(static_csv, dynamic_csv, working_dir, output_dir_name, fic_MULTIPLE, network_name, target_Q=None, slack_load_id=None, slack_gen_id=None):
    full_network_name = os.path.join(working_dir, network_name)
    static_parameters = RandomParameters.randomiseStaticParams(full_network_name + '.iidm', static_csv)
    dyn_parameters = RandomParameters.randomiseDynamicParams(full_network_name + '.par', dynamic_csv)

    runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, target_Q, slack_load_id, slack_gen_id)


def runSAFromValueList(value_list, nb_dyn_params, dyn_bounds, static_bounds, working_dir, output_dir_name, fic_MULTIPLE, network_name, target_Q=None, slack_load_id=None, slack_gen_id=None):
    dyn_value_list, static_value_list = value_list[:nb_dyn_params], value_list[nb_dyn_params:]
    dyn_parameters = dyn_bounds.valueListToParameterList(dyn_value_list)
    static_parameters = static_bounds.valueListToParameterList(static_value_list)

    runSA(static_parameters, dyn_parameters, working_dir, output_dir_name, fic_MULTIPLE, network_name, target_Q, slack_load_id, slack_gen_id)


def objective(fitted_curves, random_curves):
    mean = np.mean(random_curves, axis=2)  # average over runs
    sigma = np.std(random_curves, axis=2, ddof=1)
    for (x,y,z), value in np.ndenumerate(sigma):
        sigma[x,y,z] = max(value, 1e-3)  # Avoid division by 0 + allow some tolerance

    obj = ((fitted_curves - mean) / sigma)**2

    obj = np.mean(obj, axis=2)  # average over time
    obj = np.mean(obj, axis=1)  # average over disturbances
    obj = np.sum(obj)  # sum over curve_names (typically P and Q at point of common coupling)

    return obj


def plotCurves(curves, plot_name, fitted_curves = None):  # ndarray(curve_name, scenario, run, t_step)
    nb_disturb = curves.shape[1]
    nb_runs = curves.shape[2]
    nb_time_steps = curves.shape[3]
    t_axis = np.array([i * time_precision for i in range(nb_time_steps)])
    sqrt_d = int(ceil(sqrt(nb_disturb)))
    rcParams['figure.figsize'] = 12, 7.2

    mean = np.mean(curves, axis=2)  # average over runs
    # sigma = np.std(curves, axis=2, ddof=1)  # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)
    percentile_5, percentile_95 = np.percentile(curves, axis=2, q=[5, 95])

    for c in range(curves.shape[0]):
        fig, axs = plt.subplots(sqrt_d, sqrt_d)
        for d in range(curves.shape[1]):
            axs[d//sqrt_d, d%sqrt_d].set_title('Disturbance %d' % d)
            for r in range(nb_runs):
                axs[d//sqrt_d, d%sqrt_d].plot(t_axis, curves[c,d,r,:], ':', linewidth=1, alpha=0.3)
            # axs[d//sqrt_d, d%sqrt_d].plot(t_axis, mean[c,d,:], label='Mean', zorder=1000)
            # axs[d//sqrt_d, d%sqrt_d].plot(t_axis, mean[c,d,:] + 3*sigma[c,d,:], label='Mean + 3 sigma', zorder=1000)
            # axs[d//sqrt_d, d%sqrt_d].plot(t_axis, mean[c,d,:] - 3*sigma[c,d,:], label='Mean - 3 sigma', zorder=1000)
            axs[d//sqrt_d, d%sqrt_d].plot(t_axis, percentile_5[c,d,:], label='5th percentile', zorder=1000)
            axs[d//sqrt_d, d%sqrt_d].plot(t_axis, percentile_95[c,d,:], label='95th percentile', zorder=1000)

            if fitted_curves is not None:
                axs[d//sqrt_d, d%sqrt_d].plot(t_axis, fitted_curves[c,d,:], 'red', label='Fit')
            axs[d//sqrt_d, d%sqrt_d].legend()
        plt.savefig(plot_name + '_%d.png' % c, bbox_inches='tight')
        plt.close()


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
    nb_threads = args.nb_threads
    curve_names = args.curve_names
    time_precision = args.time_precision

    MIN_NB_RUNS = 10

    ###
    # Part 1: random runs
    ###

    run_fic_MULTIPLE = []
    sigma_thr = 0.01

    for run_id in range(max(nb_runs_random, MIN_NB_RUNS)):
        output_dir_name = os.path.join('RandomRuns', 'It_%03d' % run_id)
        runRandomSA(csv_iidm, csv_par, working_dir, output_dir_name, fic_MULTIPLE, network_name, target_Q, slack_load_id, slack_gen_id)

        current_fic = os.path.join(working_dir, output_dir_name, 'fic_MULTIPLE.xml')
        run_fic_MULTIPLE.append(current_fic)

        new_curves = -100 * MergeRandomOutputs.mergeCurves([current_fic], curve_names, time_precision)  # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
        if run_id == 0:
            random_curves = new_curves
        else:
            random_curves = np.concatenate((random_curves, new_curves), axis=2)  # ndarray(curve_name, scenario, run, t_step)

        if run_id >= MIN_NB_RUNS - 1:
            mean = np.mean(random_curves, axis=2)  # average over runs
            sigma = np.std(random_curves, axis=2, ddof=1)  # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)

            for (x,y,z), value in np.ndenumerate(sigma):
                sigma[x,y,z] = max(value, 1e-3)  # Avoid division by 0

            std_error = sigma / abs(mean[:,:,[0 for i in range(sigma.shape[2])]]) / sqrt(run_id + 1)  # Normalise with respect to value at t = 0
            print('Run_id: %d, Std error: %f' % (run_id, std_error.max()*100))
            if std_error.max() < sigma_thr:
                print("Threshold reached in %d iterations" % (run_id + 1))
                break

    if std_error.max() > sigma_thr:
        print("Warning: maximum number of random runs ({}) reached with sigma ({}%) > tol ({}%)".format(nb_runs_random, std_error.max()*100, sigma_thr*100))

    # Merge all curves and write to csv (only done once at the end for performance)
    output_dir_curves = os.path.join(working_dir, "MergedCurves")
    MergeRandomOutputs.mergeCurves(run_fic_MULTIPLE, curve_names, time_precision, write_to_csv=True, output_dir=output_dir_curves)

    randomising_time = time.time()

    plotCurves(random_curves, 'Random')

    ###
    # Part 2: optimisation
    ###

    run_id = 0  # Reset run_id
    np.random.seed(int(42))  # Different seed that the one used for the random runs, although the random generator is a priori different anyway

    dyn_bounds = DynamicParameterListWithBounds(csv_par_bounds)
    dyn_bounds_list = dyn_bounds.bounds
    nb_dyn_params = len(dyn_bounds_list)

    static_bounds = StaticParameterListWithBounds(csv_iidm_bounds)
    static_bounds_list = static_bounds.bounds

    bounds = dyn_bounds_list + static_bounds_list

    def fobj(value_list):
        global run_id
        output_dir_name = os.path.join('Optimisation', "It_%03d" % run_id)
        runSAFromValueList(value_list, nb_dyn_params, dyn_bounds, static_bounds, working_dir, output_dir_name, reduced_fic_MULTIPLE, reduced_network_name, target_Q, slack_load_id, slack_gen_id)
        current_fic = os.path.join(working_dir, output_dir_name, 'fic_MULTIPLE.xml')
        curves = -100 * MergeRandomOutputs.mergeCurves([current_fic], curve_names, time_precision)  # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
        curves = np.mean(curves, axis=2)  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)
        obj = objective(curves, random_curves)
        print('Run id: %d, objective: %f' %(run_id, obj))
        run_id += 1
        return obj

    pop_size = 10
    nb_iterations = 20
    results = list(de(fobj, bounds, popsize=pop_size, its=nb_iterations))
    
    with open(os.path.join(working_dir, 'results.txt'), 'w') as file:
        for r in results:
            print(r)
            file.write(', '.join([str(r_) for r_ in r]))
            file.write('\n')

    best_run_id = results[-1][0] + pop_size * nb_iterations
    obj = results[-1][2]
    print("Best run id: %d, objective: %f" % (best_run_id, obj))
    _, x, f = zip(*results)
    plt.plot(f)
    plt.savefig('Convergence.png', bbox_inches='tight')
    plt.close()

    optimising_time = time.time()
    print('Spent %.1fs on randomising the full model' % (randomising_time-start_time))
    print('Spent %.1fs on optimising the reduced model' % (optimising_time-randomising_time))

    fitted_curves = -100 * MergeRandomOutputs.mergeCurves([os.path.join(working_dir, 'Optimisation', "It_%03d" % best_run_id, 'fic_MULTIPLE.xml')], curve_names, time_precision)
    fitted_curves = np.mean(fitted_curves, axis=2)  # Only a single run, so replace (curve_name, scenario, run, t_step) -> (curve_name, scenario, t_step)

    plotCurves(random_curves, 'Fit', fitted_curves)
