from lxml import etree
import csv
import random
import argparse
import os
from math import sqrt, ceil
import subprocess
from matplotlib import rcParams
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import time

import RandomParameters
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


class DynamicParametersBounds:
    def __init__(self, bounds_csv, fic_MULTIPLE):
        """
        Creates an orderedDict of structure {parFile: [paramSetId, paramID, bounds]}. It is used to
        stored the bounds of some dynamic parameters to a format that can easily be sent to dynawo.

        parFile is the file where the parameter is stored
        paramSetId and paramID locate the parameter in the file
        bounds are bounds on the possible value that-the parameter can take 
        """
        self.d = OrderedDict()
        input_pars = RandomParameters.ficGetPars(fic_MULTIPLE)

        pars_root = []
        pars_namespace = []
        pars_prefix_root = []
        pars_prefix_root_string = []

        for input_par in input_pars:
            pars_root.append(etree.parse(input_par).getroot())
            pars_namespace.append(pars_root[-1].nsmap)
            pars_prefix_root.append(pars_root[-1].prefix)
            if pars_prefix_root[-1] is None:
                pars_prefix_root_string.append('')
            else:
                pars_prefix_root_string.append(pars_prefix_root[-1] + ':')

        with open(bounds_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            row = spamreader.__next__()
            if row != ['parFile', 'ParamSet_id', 'Param_id', 'L_bound', 'U_bound']:
                raise Exception("Incorrect format of %s" % bounds_csv)

            for row in spamreader:
                working_dir = os.path.dirname(fic_MULTIPLE)
                parFile = os.path.join(working_dir, row[0])
                paramSetId = row[1]
                paramId = row[2]
                bounds = (float(row[3]), float(row[4]))

                try:
                    index = input_pars.index(parFile)
                except ValueError:
                    raise Exception("Input par %s given in %s is not used in the SA" % (parFile, bounds_csv))
                parameterSet = RandomParameters.findParameterSet(pars_root[index], paramSetId)
                parameter = RandomParameters.findParameter(parameterSet, paramId) # Check that the parameters given in bounds_csv do exists
                # the value of the parameter is not used

                value = self.d.setdefault(parFile, [[],[],[]])
                value[0].append(paramSetId)
                value[1].append(paramId)
                value[2].append(bounds)
                

    def toBoundsList(self):
        """
        Create a list of the bounds contained in self.d (that has the structure {parFile: [paramSetId, paramID, bounds]})
        by iterating on self.d. The bounds are always returned in the same order as self.d is an orderedDict
        """
        out = []
        for value in self.d.values():
            for v in value[2]:
                out.append(v)
        return out
    
    def valueListToDict(self, v_lst):
        """
        Creates a dictionary that has the same structure as self.d (i.e. {parFile: [paramSetId, paramID, bounds]}),
        but with the element 'bounds' (a tuple) replaced by the values in v_lst (that are floats)

        Also checks that the values are within the bounds.
        The order of the values should be the same as the ordre used in self.toBoundsList().
        """

        if len(v_lst) != len(self.toBoundsList()):
            raise ValueError("Length of v_lst (%d) should be %d" % (len(v_lst), len(self.toBoundsList())))
        
        new_d = dict()
        v_index = 0

        for parFile, value in self.d.items():
            for i in range(len(value[0])):
                new_v = new_d.setdefault(parFile, [[],[],[]])
                new_v[0].append(value[0][i])
                new_v[1].append(value[1][i])
                
                bounds = value[2][i]
                if v_lst[v_index] >= bounds[0] and v_lst[v_index] <= bounds[1]:
                    new_v[2].append(v_lst[v_index])
                else:
                    raise ValueError("Value no %d of v_lst (%f) not within (%f, %f)" % (v_index, v_lst[v_index], bounds[0], bounds[1]))
                v_index += 1
        return new_d


class StaticParametersBounds:
    def __init__(self, bounds_csv, fic_MULTIPLE):
        """
        Creates an orderedDict of structure {iidmFile: [Component_type, Component_name, Param_id, bounds]}. It is used to
        stored the bounds of some static parameters to a format that can easily be sent to dynawo.

        iidmFile is the file where the parameter is stored
        Component_type, Component_name and Param_id locate the parameter in the file
        bounds are bounds on the possible value that the parameter can take
        """
        self.d = OrderedDict()
        with open(bounds_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            row = spamreader.__next__()
            if row != ['Component_type', 'Component_name', 'Param_id', 'L_bound', 'U_bound']:
                print(row)
                raise Exception("Incorrect format of %s" % bounds_csv)
            
            component_types = []
            component_names = []
            param_ids = []
            param_bounds = []

            for row in spamreader:
                componentType = row[0]
                componentName = row[1]
                paramId = row[2]
                bounds = (float(row[3]), float(row[4]))

                if componentType not in ['Load', 'Line', 'Bus', 'Generator']:
                    raise Exception("Component type '%s' not considered" % componentType)

                component_types.append(componentType)
                component_names.append(componentName)
                param_ids.append(paramId)
                param_bounds.append(bounds)

        for input_iidm in RandomParameters.ficGetIidms(fic_MULTIPLE): # Assume all iidm's have the same parameters/bounds
            self.d[input_iidm] = [component_types, component_names, param_ids, param_bounds]

    def toBoundsList(self):
        """
        Create a list of the bounds contained in self.d (that has the structure {iidmFile: [Component_type, Component_name, Param_id, bounds]})
        by iterating on self.d. The bounds are always returned in the same order as self.d is an orderedDict
        """
        out = []
        for value in self.d.values():
            for v in value[3]:
                out.append(v)
        return out
    
    def valueListToDict(self, v_lst):
        """
        Creates a dictionary that has the same structure as self.d (i.e. {iidmFile: [Component_type, Component_name, Param_id, bounds]}),
        but with the element 'bounds' (a tuple) replaced by the values in v_lst (that are floats)

        Also checks that the values are within the bounds.
        The order of the values should be the same as the ordre used in self.toBoundsList().
        """

        if len(v_lst) != len(self.toBoundsList()):
            raise ValueError("Length of v_lst (%d) should be %d" % (len(v_lst), len(self.toBoundsList())))
        
        new_d = dict()
        v_index = 0

        for parFile, value in self.d.items():
            for i in range(len(value[0])):
                new_v = new_d.setdefault(parFile, [[],[],[],[]])
                new_v[0].append(value[0][i])
                new_v[1].append(value[1][i])
                new_v[2].append(value[2][i])
                
                bounds = value[3][i]
                if v_lst[v_index] >= bounds[0] and v_lst[v_index] <= bounds[1]:
                    new_v[3].append(v_lst[v_index])
                else:
                    raise ValueError("Value no %d of v_lst (%f) not within (%f, %f)" % (v_index, v_lst[v_index], bounds[0], bounds[1]))
                v_index += 1
        return new_d


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

    args.fic_MULTIPLE = os.path.join(args.working_dir, args.fic_MULTIPLE)
    args.reduced_fic_MULTIPLE = os.path.join(args.working_dir, args.reduced_fic_MULTIPLE)
    args.csv_par = os.path.join(args.working_dir, args.csv_par)
    args.csv_iidm = os.path.join(args.working_dir, args.csv_iidm)
    args.csv_par_bounds = os.path.join(args.working_dir, args.csv_par_bounds)
    args.csv_iidm_bounds = os.path.join(args.working_dir, args.csv_iidm_bounds)

    args.target_Q = float(args.target_Q)

    # Part 1: random runs
    run_fic_MULTIPLE = []
    sigma_thr = 0.01
    for run_id in range(args.nb_runs_random):
        output_dir_name = "RandomisedInputs_%03d" % run_id
        output_dir = os.path.join(args.working_dir, output_dir_name)

        output_dir_curves = os.path.join(args.working_dir, "MergedCurves")

        static_data_dic = RandomParameters.randomiseStaticParams(args.fic_MULTIPLE, args.csv_iidm)
        dyn_data_dic = RandomParameters.randomiseDynamicParams(args.fic_MULTIPLE, args.csv_par)

        RandomParameters.writeParametricSAInputs(args.working_dir, args.fic_MULTIPLE, output_dir_name, static_data_dic, dyn_data_dic,
                run_id, args.target_Q, args.slack_load_id, args.slack_gen_id)
        cmd = ['./myEnvDynawoAlgorithms.sh', 'SA', '--directory', output_dir, '--input', 'fic_MULTIPLE.xml',
                '--output' , 'aggregatedResults.xml', '--nbThreads', args.nb_threads]
        subprocess.run(cmd)
        current_fic = os.path.join(output_dir, 'fic_MULTIPLE.xml')
        run_fic_MULTIPLE.append(current_fic)

        if run_id == 0:
            # Minus because infinite bus has receptor convention (minus sign only affects the curves), 100 is from pu to MW
            curves = -100 * MergeRandomOutputs.mergeCurves([current_fic], args.curve_names, args.time_precision)
        else:
            new_curves = -100 * MergeRandomOutputs.mergeCurves([current_fic], args.curve_names, args.time_precision)
            curves = np.concatenate((curves, new_curves), axis=2)
        # Order of indices:
            # curves = ndarray(nb_curve_names, nb_scenarios_per_fic, nb_runs, nb_t_steps)
            # mean = ndarray(nb_curve_names, nb_scenarios_per_fic, nb_t_steps)
        mean = np.mean(curves, axis=2)
        sigma = np.std(curves, axis=2, ddof=1) # ddof = 1 means divide by sqrt(N-1) instead of sqrt(N)

        if run_id >= 10: # Do at least 5 runs
            std_error = sigma / abs(mean[:,:,[0 for i in range(sigma.shape[2])]]) / sqrt(run_id + 1) # Normalise with respect to value at t = 0 (avoid div by 0)
            print('Run_id: %d, Std error: %f%%' % (run_id, std_error.max()*100))
            if std_error.max() < sigma_thr: 
                print("Threshold reached in %d iterations" % (run_id + 1))
                break
    if std_error.max() > sigma_thr:
        print("Warning: maximum number of random runs (%d) reached with sigma (%f) > tol (%f)" % (args.nb_runs_random, std_error.max(), sigma_thr))

    # Merge all curves and write to csv (only done at the end for performance)
    MergeRandomOutputs.mergeCurves(run_fic_MULTIPLE, output_dir_curves, args.curve_names, args.time_precision, write_to_csv=True)
    disturb = curves.shape[1]
    sqrt_d = int(ceil(sqrt(disturb)))
    rcParams['figure.figsize'] = 12, 7.2
    for c in range(curves.shape[0]):
        fig, axs = plt.subplots(sqrt_d, sqrt_d)
        for d in range(curves.shape[1]):
            axs[d//sqrt_d, d%sqrt_d].set_title('Disturbance %d' % d)
            axs[d//sqrt_d, d%sqrt_d].plot(mean[c,d,:], label='Mean', zorder=1000)
            axs[d//sqrt_d, d%sqrt_d].plot(mean[c,d,:] + 3*sigma[c,d,:], label='Mean + 3 sigma', zorder=1000)
            axs[d//sqrt_d, d%sqrt_d].plot(mean[c,d,:] - 3*sigma[c,d,:], label='Mean - 3 sigma', zorder=1000)

            for i in range(run_id):
                axs[d//sqrt_d, d%sqrt_d].plot(curves[c,d,i,:], ':', linewidth=1)
            axs[d//sqrt_d, d%sqrt_d].legend()
        plt.savefig('Random%d.png' % c, bbox_inches='tight')
        plt.close()
    randomising_time = time.time()

    # Part 2: optimisation
    dyn_bounds_dic = DynamicParametersBounds(args.csv_par_bounds, args.reduced_fic_MULTIPLE)
    dyn_bounds = dyn_bounds_dic.toBoundsList()
    nb_dyn_params = len(dyn_bounds)

    iidm_bounds_dic = StaticParametersBounds(args.csv_iidm_bounds, args.reduced_fic_MULTIPLE)
    iidm_bounds = iidm_bounds_dic.toBoundsList()

    bounds = dyn_bounds + iidm_bounds

    run_id = 0
    np.random.seed(int(2e9)) # Different seed that the one used for the random runs, although the random generator is a priori different anyway
    # 2e9 is around half the max 32-bit unsigned int
    def fobj2(v_lst, verbose=False):
        global run_id
        dyn_data_dic = dyn_bounds_dic.valueListToDict(v_lst[:nb_dyn_params])
        static_data_dic = iidm_bounds_dic.valueListToDict(v_lst[nb_dyn_params:])

        output_dir_name = os.path.join('Optimisation', "It_%03d" % run_id)
        output_dir = os.path.join(args.working_dir, output_dir_name)
        RandomParameters.writeParametricSAInputs(args.working_dir, args.reduced_fic_MULTIPLE, output_dir_name, static_data_dic, dyn_data_dic,
                run_id, args.target_Q, args.slack_load_id, args.slack_gen_id)
        cmd = ['./myEnvDynawoAlgorithms.sh', 'SA', '--directory', output_dir, '--input', 'fic_MULTIPLE.xml',
                '--output' , 'aggregatedResults.xml', '--nbThreads', args.nb_threads]
        subprocess.run(cmd)

        output_dir_curves = os.path.join(args.working_dir, 'Optimisation', "MergedCurves_it_%03d" % run_id)
        curves = -100 * MergeRandomOutputs.mergeCurves([os.path.join(output_dir, 'fic_MULTIPLE.xml')], args.curve_names, args.time_precision, write_to_csv=output_dir_curves, output_dir=output_dir_curves)

        obj = ((curves - mean) / sigma)**2

        obj = np.mean(obj, axis=2) # average over time
        obj = np.mean(obj, axis=1) # average over disturbances
        if verbose:
            print('Objective:')
            print(obj)
        obj = np.sum(obj) # sum over curve_names (typically P and Q at point of common coupling)

        print("Run id: %d" % run_id)
        run_id += 1

        return obj, curves
    
    def fobj(v_lst):
        return fobj2(v_lst)[0]

    pop_size = 10
    nb_iterations = 30
    result = list(de(fobj, bounds, popsize=pop_size, its=nb_iterations))
    
    with open(os.path.join(args.working_dir, 'results.txt'), 'w') as file:
        for r in result:
            print(r)
            file.write(', '.join([str(r_) for r_ in r]))
            file.write('\n')

    best_run_id = result[-1][0] + pop_size * nb_iterations
    print("Best run id: %d" % best_run_id)
    _, x, f = zip(*result)
    plt.plot(f)
    plt.savefig('Convergence.png', bbox_inches='tight')
    plt.close()

    optimising_time = time.time()
    print('Spent %.1fs on randomising the full model' % (randomising_time-start_time))
    print('Spent %.1fs on optimising the reduced model' % (optimising_time-randomising_time))

    _, curves = fobj2(x[-1], verbose=True)

    for c in range(curves.shape[0]):
        fig, axs = plt.subplots(sqrt_d, sqrt_d)
        for d in range(curves.shape[1]):
            axs[d//sqrt_d, d%sqrt_d].set_title('Disturbance %d' % d)
            axs[d//sqrt_d, d%sqrt_d].plot(mean[c,d,:], label='Mean')
            axs[d//sqrt_d, d%sqrt_d].plot(mean[c,d,:] + 3*sigma[c,d,:], label='Mean + 3 sigma')
            axs[d//sqrt_d, d%sqrt_d].plot(mean[c,d,:] - 3*sigma[c,d,:], label='Mean - 3 sigma')

            axs[d//sqrt_d, d%sqrt_d].plot(curves[c,d,:], label='Fit')
            axs[d//sqrt_d, d%sqrt_d].legend()
        plt.savefig('Fit%d.png' % c, bbox_inches='tight')
