from lxml import etree
import csv
import argparse
import numpy as np
import os

def mergeCurvesFromFics(fic_MULTIPLEs, curve_names, time_precision, write_to_csv=False, output_dir=None):
    """
    Merge the output curves of a parametric systematic analysis

    @param fic_MULTIPLE fic_MULTIPLE List of fic_MULTIPLE files for which the ouputs curves have to be merged
    @param output_dir Full path of the directory where the merged curves should be written
    @param curve_names List of the names of the curves that have to be merged
    @param time_precision Time precision of the output curves (linear interpolation from the input ones)
    """
    output_curves_all_fic = dict()
    scenario_ids_all_fic = dict()

    for fic_MULTIPLE in fic_MULTIPLEs:
        working_dir = os.path.dirname(fic_MULTIPLE)
        fic = etree.parse(fic_MULTIPLE)
        fic_root = fic.getroot()
        fic_namespace = fic_root.nsmap
        fic_prefix_root = fic_root.prefix
        fic_namespace_uri = fic_namespace[fic_prefix_root]

        if len(fic_root) == 0:
            raise Exception('fic_MULTIPLE file is empty')
        scenarios = fic_root[0]
        if scenarios.tag != '{' + fic_namespace_uri + '}' + 'scenarios':
            raise Exception('Invalid fic_MULTIPLE file')

        scenario_ids = []
        for scenario in scenarios:
            scenario_id = scenario.get('id')
            if scenario_id == None:
                raise Exception('fic_MULTIPLE.xml has scenario without id')
            scenario_ids.append(scenario_id)

        jobsFile = scenarios.get('jobsFile')
        jobs = etree.parse(os.path.join(working_dir, os.path.dirname(fic_MULTIPLE), jobsFile))
        jobs_root = jobs.getroot()
        jobs_namespace = jobs_root.nsmap
        jobs_prefix_root = jobs_root.prefix
        jobs_namespace_uri = jobs_namespace[jobs_prefix_root]

        simulation = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'simulation[@startTime]')
        if len(simulation) != 1:
            raise Exception("Jobs file should contain exactly one 'startTime' entry, %s found" % (len(simulation)))
        startTime = float(simulation[0].get('startTime'))

        simulation = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'simulation[@stopTime]')
        if len(simulation) != 1:
            raise Exception("Jobs file should contain exactly one 'stopTime' entry, %s found" % (len(simulation)))
        stopTime = float(simulation[0].get('stopTime'))

        output_time_axis = [t for t in np.arange(startTime, stopTime, time_precision)]
        output_time_axis += [stopTime] # np.arange creates a half-open interval

        curves = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'curves[@exportMode]')
        if len(curves) != 1:
            raise Exception("Jobs file should contain exactly one 'exportMode' entry, %s found" % (len(curves)))
        if curves[0].get('exportMode') != 'CSV':
            raise NotImplementedError('Can only merge curves in .csv format')

        outputs = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'outputs[@directory]')
        if len(outputs) != 1:
            raise Exception("Jobs file should contain exactly one 'directory' entry, %s found" % (len(outputs)))
        sim_outputs_dir = outputs[0].get('directory')

        output_curves_one_fic = []
        for scenario_id in scenario_ids:
            input_csv = os.path.join(working_dir, os.path.dirname(fic_MULTIPLE), scenario_id, sim_outputs_dir, 'curves', 'curves.csv')

            with open(input_csv) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=';')
                row = spamreader.__next__()
                input_time_axis = []
                input_curves = [[] for i in range(len(curve_names))]
                indexes = []
                for value in curve_names:
                    indexes.append(row.index(value))
                for row in spamreader:
                    input_time_axis.append(float(row[0]))
                    for i in range(len(curve_names)):
                        input_curves[i].append(float(row[indexes[i]]))

            output_curves = [np.interp(output_time_axis, input_time_axis, input_curves[i]) for i in range(len(curve_names))]
            output_curves_one_fic.append(output_curves)
        scenario_ids_all_fic[fic_MULTIPLE] = scenario_ids
        output_curves_all_fic[fic_MULTIPLE] = output_curves_one_fic
    nb_scenarios_per_fic = len(list(output_curves_all_fic.values())[0])  # Assume that all fic_MULTIPLE have the same number of scenarios

    if write_to_csv:
        if output_dir is None:
            raise ValueError("output_dir should be given if write_to_csv is True")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for curve_index in range(len(curve_names)):
            output_csv = os.path.join(output_dir, curve_names[curve_index] + '.csv')
            with open(output_csv, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=';')

                header = ['time']
                for scenario_nb in range(nb_scenarios_per_fic):
                    for scenario_ids in scenario_ids_all_fic.values():
                        header.append(scenario_ids[scenario_nb])
                spamwriter.writerow(header)

                for t in range(len(output_time_axis)):
                    row = [output_time_axis[t]]
                    for scenario_nb in range(nb_scenarios_per_fic):
                        for curves_one_fic in output_curves_all_fic.values():
                            row.append(curves_one_fic[scenario_nb][curve_index][t])
                    spamwriter.writerow(row)
    
    # output_curves_all_fic -> np.narray
    nb_curve_names = len(curve_names)
    nb_runs = len(fic_MULTIPLEs)
    nb_t_steps = len(list(output_curves_all_fic.values())[0][0][0])
    curves = np.zeros((nb_curve_names, nb_scenarios_per_fic, nb_runs, nb_t_steps))

    for curve_nb in range(nb_curve_names):
        for scenario_nb in range(nb_scenarios_per_fic):
            for run_nb in range(len(output_curves_all_fic.values())):
                curves[curve_nb, scenario_nb, run_nb, :] = list(output_curves_all_fic.values())[run_nb][scenario_nb][curve_nb]   

    return curves

def mergeCurvesFromCurves(full_curve_paths, start_time, stop_time, curve_names, time_precision, write_to_csv=False, output_dir=None):
    """
    Merge the output curves from a list

    @param full_curve_paths Full paths of the curves to be merged
    @param output_dir Full path of the directory where the merged curves should be written
    @param curve_names List of the names of the curves that have to be merged
    @param time_precision Time precision of the output curves (linear interpolation from the input ones)
    """

    output_time_axis = [t for t in np.arange(start_time, stop_time, time_precision)]
    output_time_axis += [stop_time] # np.arange creates a half-open interval
    output_curves_all = []

    for input_csv in full_curve_paths:
        with open(input_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';')
            row = spamreader.__next__()
            input_time_axis = []
            input_curves = [[] for i in range(len(curve_names))]
            indexes = []
            for value in curve_names:
                indexes.append(row.index(value))
            for row in spamreader:
                input_time_axis.append(float(row[0]))
                for i in range(len(curve_names)):
                    input_curves[i].append(float(row[indexes[i]]))

            output_curves = [np.interp(output_time_axis, input_time_axis, input_curves[i]) for i in range(len(curve_names))]
            output_curves_all.append(output_curves)

    if write_to_csv:
        if output_dir is None:
            raise ValueError("output_dir should be given if write_to_csv is True")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for curve_index in range(len(curve_names)):
            output_csv = os.path.join(output_dir, curve_names[curve_index] + '.csv')
            with open(output_csv, 'w') as csvfile:
                spamwriter = csv.writer(csvfile, delimiter=';')

                header = ['time']
                for i in range(len(full_curve_paths)):
                    header.append(i)
                spamwriter.writerow(header)

                for t in range(len(output_time_axis)):
                    row = [output_time_axis[t]]
                    for it in range(len(full_curve_paths)):
                        row.append(output_curves_all[it][curve_index][t])
                    spamwriter.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Merge the output curves of a parametric systematic analysis')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--fic_MULTIPLEs', '-f', type=str, required=True, action='append',
                        help='List of files for which the ouputs have to be merged')
    parser.add_argument('--curve_names', '-c', type=str, required=True, action='append',
                        help='List of the names of the curves that have to be merged')
    parser.add_argument('--time_precision', type=float, required=True,
                        help='Time precision of the output curves (linear interpolation from the input ones)')
    args = parser.parse_args()

    fic_MULTIPLEs = []
    for fic_MULTIPLE in args.fic_MULTIPLEs:
        fic_MULTIPLEs.append(os.path.join(args.working_dir, fic_MULTIPLE))
    output_dir = os.path.join(args.working_dir, 'MergedCurves')

    mergeCurvesFromFics(fic_MULTIPLEs, args.curve_names, args.time_precision, write_to_csv=True, output_dir=output_dir)