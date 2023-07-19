import argparse
import csv
import os
import statistics
from natsort import natsorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Merge timeline_analysis files contained in a given folder'
    ''
    ''
    'Hypotheses:'
    '   - Naming convention TimelineAnalysisX.csv where X is the runID'
    '   - CSV delimiter = ,')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')

    args = parser.parse_args()
    working_dir = args.working_dir

    filenames = [filename for filename in os.listdir(working_dir) if os.path.isfile(os.path.join(working_dir, filename)) and
            filename[:16] == 'TimelineAnalysis' and filename[-4:] == '.csv' and
            filename != 'TimelineAnalysis.csv']  # Skip the non-randomised test (that is not run at the same time and could thus have different parameters)
    filenames = natsorted(filenames, key=lambda y: y.lower())

    with open(os.path.join(working_dir, 'MergedTimelineAnalysis.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for filename in filenames:
            runID = filename[16:].split('.')[0]

            with open(os.path.join(working_dir, filename), newline='') as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                header = next(csv_reader)

                if filename == filenames[0]:  # Only print header once
                    header.insert(1, 'Run ID')
                    writer.writerow(header)
                for row in csv_reader:
                    row.insert(1, runID)
                    writer.writerow(row)


    results_per_scenario = {}
    for filename in filenames:
        with open(os.path.join(working_dir, filename), newline='') as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            header = next(csv_reader)

            for row in csv_reader:
                scenario, load_shedding, nb_trips, distance_trips, UVA_trip, speed_trips = row[0:6]
                results_per_scenario.setdefault(scenario,[]).append([load_shedding, nb_trips, distance_trips, UVA_trip, speed_trips])
    
    with open(os.path.join(working_dir, 'MergedTimelineAnalysisStdDev.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['Init event', 'Average shedding', 'Std dev', 'Load sheddings'])

        for (scenario, results) in results_per_scenario.items():
            load_sheddings = [float(result[0]) for result in results]
            average_load_shedding = statistics.mean(load_sheddings)
            std_dev_load_shedding = statistics.stdev(load_sheddings)  # (Sample variance)

            writer.writerow([scenario, average_load_shedding, std_dev_load_shedding] + load_sheddings)
