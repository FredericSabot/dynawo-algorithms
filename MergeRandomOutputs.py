from lxml import etree
import csv
import argparse
import numpy as np
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Merge the output curves of a "random SA"')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--fic_MULTIPLE', type=str, required=True,
                        help='fic_MULTIPLE file for which the ouputs have to be merged')
    parser.add_argument('--curve_names', '-c', type=str, required=True, action='append',
                        help='List of the names of the curves that have to be merged')
    parser.add_argument('--time_precision', type=float, required=True,
                        help='Time precision of the output curves (linear interpolation from the input ones)')
    parser.add_argument('--nb_runs', type=int, required=True,
                        help='Number of randomised copies of the base scenarios to merge (has to be lower or equal to the number actually simulated)')
    args = parser.parse_args()

    args.working_dir += '/'
    args.fic_MULTIPLE = args.working_dir + args.fic_MULTIPLE
    output_dir = args.working_dir + 'MergedCurves/'

    fic = etree.parse(args.fic_MULTIPLE)
    fic_root = fic.getroot()
    fic_namespace = fic_root.nsmap
    fic_prefix_root = fic_root.prefix
    fic_namespace_uri = fic_namespace[fic_prefix_root]

    if len(fic_root) == 0:
        raise Exception('fic_MULTIPLE file is empty')
    scenarios = fic_root[0]
    if scenarios.tag != '{' + fic_namespace_uri+ '}' + 'scenarios':
        raise Exception('Invalid fic_MULTIPLE file')

    scenario_ids = []
    for scenario in scenarios:
        scenario_id = scenario.get('id')
        if scenario_id == None:
            raise Exception('fic_MULTIPLE.xml has scenario without id')
        scenario_ids.append(scenario_id)

    jobsFile = scenarios.get('jobsFile')
    jobs = etree.parse(args.working_dir + jobsFile)
    jobs_root = jobs.getroot()
    jobs_namespace = jobs_root.nsmap
    jobs_prefix_root = jobs_root.prefix
    jobs_namespace_uri = jobs_namespace[jobs_prefix_root]
    # paths (scenario id's + output name), start/stop time

    simulation = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'simulation[@startTime]')
    if len(simulation) != 1:
        raise Exception("Jobs file should contain exactly one 'startTime' entry, %s found" % (len(simulation)))
    startTime = float(simulation[0].get('startTime'))

    simulation = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'simulation[@stopTime]')
    if len(simulation) != 1:
        raise Exception("Jobs file should contain exactly one 'stopTime' entry, %s found" % (len(simulation)))
    stopTime = float(simulation[0].get('stopTime'))

    output_time_axis = [t for t in np.arange(startTime, stopTime, args.time_precision)]
    output_time_axis += [stopTime] # arange creates a half-open interval

    curves = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'curves[@exportMode]')
    if len(curves) != 1:
        raise Exception("Jobs file should contain exactly one 'exportMode' entry, %s found" % (len(curves)))
    if curves[0].get('exportMode') != 'CSV':
        raise NotImplementedError('Can only merge curves in .csv format')

    outputs = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'outputs[@directory]')
    if len(outputs) != 1:
        raise Exception("Jobs file should contain exactly one 'directory' entry, %s found" % (len(outputs)))
    outputs_dir = outputs[0].get('directory')

    output_curves_all_scenarios = []
    for scenario_id in scenario_ids:
        for nb_run in range(args.nb_runs):
            suffix = "_%03d" % (nb_run)
            input_csv = args.working_dir + 'RandomisedInputs' + '/' + scenario_id + suffix  + '/' + outputs_dir + '/' + 'curves' + '/' + 'curves.csv'

            with open(input_csv) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=';')
                row = spamreader.__next__()
                input_time_axis = []
                input_curves = [[] for i in range(len(args.curve_names))]
                indexes = []
                for value in args.curve_names:
                    indexes.append(row.index(value))
# time, name, scenario
                for row in spamreader:
                    input_time_axis.append(float(row[0]))
                    for i in range(len(args.curve_names)):
                        input_curves[i].append(float(row[indexes[i]]))

            output_curves = [np.interp(output_time_axis, input_time_axis, input_curves[i]) for i in range(len(args.curve_names))]
            output_curves_all_scenarios.append(output_curves)


    output_dir = args.working_dir + 'MergedCurves/'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for curve_index in range(len(args.curve_names)):
        output_csv = output_dir + args.curve_names[curve_index] + '.csv'
        with open(output_csv, 'w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=';')

            header = ['time']
            for scenario_id in scenario_ids:
                for nb_run in range(args.nb_runs):
                    suffix = "_%03d" % (nb_run)
                    header.append(scenario_id + suffix)
            spamwriter.writerow(header)

            for t in range(len(output_time_axis)):
                row = [output_time_axis[t]]
                scenario_index = 0
                for scenario_nb in range(len(scenario_ids)):
                    for nb_run in range(args.nb_runs):
                        row.append(output_curves_all_scenarios[scenario_index][curve_index][t])
                        scenario_index += 1
                spamwriter.writerow(row)
