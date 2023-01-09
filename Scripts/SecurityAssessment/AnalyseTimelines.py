from lxml import etree
import argparse
import pypowsybl as pp
import pandas as pd
import os, shutil, glob
import csv

""" def isSameModel(model1, model2):
    ""
    Return true if model1 and model2 disconnect the same element
    ""
    if len(model1) >= 23 and len(model2) >= 23:  # e.g. 5.41972 _BUS___26-BUS___29-1_AC_side2_Distance
                                                 #      5.41972 _BUS___28-BUS___29-1_AC_side1_Distance
        if model1[:23] == model2[:23]:
            return True
    elif len(model1) >= 8 and len(model2) >= 8:  # e.g. 5.5 GEN___38_SM_UVA Under-voltage generator trip
                                                 #      5.59184 GEN___38_SM_Speed Speed protection trip
        if model1[:8] == model2[:8]:
            return True
    return False """


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Analyse all timeline files in a directory'
    ''
    ''
    'Hypotheses:'
    '   - Timelines are exported in TXT mode'
    '   - Timeline files are in a "timeLine/" subfolder of working_dir (as created in as SA)'
    '   - Timelines have been filtered by Dynawo (filter="true" in the .jobs file)'
    '   - Load shedding:'
    '       - Loads can be disconnected completely (switchoff) or partially via a "centralised" UFLS relay'
    '       - The amount of load shed by each step of the UFLS in hard coded in this program'
    '   - Distance timings hardcoded: 300 and 600ms for tripping, (80ms for CB)')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--name', type=str, required=True,
                        help='Full path of the iidm file describing the network')

    args = parser.parse_args()
    working_dir = args.working_dir
    name = args.name

    timeline_dir = os.path.join(working_dir, 'timeLine/')
    filenames = [filename for filename in os.listdir(timeline_dir) if os.path.isfile(os.path.join(timeline_dir, filename))]

    full_name = os.path.join(working_dir, name)


    # XML boilerplate
    XMLparser = etree.XMLParser(remove_blank_text=True)
    namespace = 'http://www.rte-france.com/dynawo'
    jobs_prefix = 'dyn'
    jobs_namespace_map = {jobs_prefix: namespace}
    ficMultiple_prefix = None
    fic_namespace_map = {ficMultiple_prefix: namespace}
    fic_rootName = etree.QName(namespace, 'multipleJobs')
    # dyd_prefix = 'dyn'
    # dyd_namespace = 'http://www.rte-france.com/dynawo'
    # dyd_rootName = etree.QName(dyd_namespace, 'dynamicModelsArchitecture')
    # dyd_namespace_map = {dyd_prefix: dyd_namespace}
    par_prefix = 'par'
    par_namespace_map = {par_prefix: namespace}



    outputs = {}
    close_events_dic = {}
    count_occurences = {}
    Z_armings_dic = {}
    Z_disarmings_dic = {}
    gen_disc_dic = {}
    for filename in filenames:
        ###
        # Filter events
        ###
        output = {}
        output['Init event'] = filename[9:-4]  # Remove 'timeline_', and '.txt'
        with open(os.path.join(timeline_dir, filename), 'r') as doc:
            timeline_ = doc.readlines()

            timeline = []
            trip_timeline = []  # Timeline that only contains trip events
            Z_armings = []
            Z_disarmings = []
            gen_disconnections = []
            for event in timeline_:
                (time, model, event) = event.strip().split(' | ')
                
                timeline.append([time, model, event])
                if 'trip' in event:  # Does not include UFLS (might need update if other protections are added)
                    trip_timeline.append((time, model, event))
                    count_occurences[model] = count_occurences.get(model, 0) + 1

                if 'Distance protection zone' in event:
                    if 'disarming' in event:
                        Z_disarmings.append([time, model, event])
                    elif 'arming' in event:  # elif -> does not include disarmings
                        Z_armings.append([time, model, event])
                
                if 'GENERATOR : disconnecting' in event:
                    gen_disconnections.append([time, model, event])
        
        Z_armings_dic[filename] = Z_armings
        Z_disarmings_dic[filename] = Z_disarmings

        ###
        # Compute load shedding
        ###
        n = pp.network.load(os.path.join(working_dir, name + '.iidm'))
        loads = n.get_loads()
        for load in loads.index:
            if 'Dummy' in load:  # Remove dummy loads
                loads = loads.drop(load)
        
        total_load = sum([loads.at[load, 'p0'] for load in loads.index])

        UFLS_ratio = 1
        disconnected_load = 0
        for (time, model, event) in timeline:
            if event == 'UFLS step 1 activated':
                UFLS_ratio += -0.1
            elif event == 'UFLS step 2 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 3 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 4 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 5 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 6 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 7 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 8 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 9 activated':
                UFLS_ratio += -0.05
            elif event == 'UFLS step 10 activated':
                UFLS_ratio += -0.05
            
            elif event == 'LOAD : disconnecting':
                if 'Dummy' in model:
                    continue
                disconnected_load += loads.at[model, 'p0']
            else:
                pass

        remaining_load = (total_load - disconnected_load) * UFLS_ratio

        load_shedding = (total_load - remaining_load) / total_load * 100

        # Check for execution problem/divergence
        namespace_map = '{http://www.rte-france.com/dynawo}'
        root = etree.parse(os.path.join(working_dir, "aggregatedResults.xml"), XMLparser).getroot()
        execution_status = root.find('.//{}scenarioResults[@id="{}"]'.format(namespace_map, filename[9:-4])).get('status')

        if execution_status == "DIVERGENCE":
            load_shedding = 100.1  # Mark it as 100.1% load shedding to not affect averages, but still see there is a numerical issue
        
        if len(gen_disconnections) == 10:  # All machines are disconnected (not really a convergence issue)
            load_shedding = 100

        output['Load shedding (%)'] = load_shedding

        # TODO: check for low voltages using the new final values API (load_terminal_V_re, and _im -> compute abs)

        ###
        # Event counts
        ###
        output['Trips'] = len(trip_timeline)
        output['Distance trips'] = sum(1 for (_, __, event) in trip_timeline if 'Distance' in event)
        output['UVA trips'] = sum(1 for (_, __, event) in trip_timeline if 'Under-voltage' in event)
        output['Speed trips'] = sum(1 for (_, __, event) in trip_timeline if 'Speed' in event)

        ###
        # Check for events that occurs in close succession
        ###
        old_time = 0
        close_events_list = []
        for i in range(len(trip_timeline)):
            (time, model, event) = trip_timeline[i]
            close_events_sublist = [(time, model, event)]

            if time == old_time:
                continue
            old_time = time

            if float(time) == 5.1 or float(time) == 5.2:  # Events related to the initiating event
                continue
                
            if 'Speed' in event or 'Under-voltage' in event:  # Only consider distance protections
                continue

            for j in range(i + 1, len(trip_timeline)):
                (next_time, next_model, next_event) = trip_timeline[j]
                if trip_timeline[i]==trip_timeline[j]:
                    print('Error')

                if float(next_time) > float(time) + 0.1:
                    break
                
                if 'Speed' in next_event or 'Under-voltage' in next_event:
                    continue

                #if isSameModel(model, next_model):
                #    continue

                close_events_sublist.append((next_time, next_model, next_event))

            if len(close_events_sublist) > 1:
                close_events_list.append(close_events_sublist) 
        if close_events_list != []:
            close_events_dic[filename] = close_events_list

        output['Trip events'] = trip_timeline
        outputs[filename] = output

    """
    ###
    # Create small variations of simulations files for cases with close events
    ###
    output_dir = os.path.join(working_dir, "Close_events_rerun")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir_timeline = os.path.join(output_dir, 'timeLine')
    if not os.path.exists(output_dir_timeline):
        os.makedirs(output_dir_timeline)

    # Copy input files
    shutil.copy(full_name + '.iidm', output_dir)

    # shutil.copy(full_name + '.dyd', output_dir)
    dyd_files = glob.iglob(os.path.join(working_dir, "*.dyd"))  # Copy all .dyd files for a potential recursion
    for file in dyd_files:
        if os.path.isfile(file):
            shutil.copy2(file, output_dir)

    shutil.copy(full_name + '.par', output_dir)  # Only kept for the "NETWORK" and "solver" parameter sets
    # shutil.copy(full_name + '.jobs', output_dir)
    if os.path.isfile(full_name + '.crv'):
        shutil.copy(full_name + '.crv', output_dir)
    if os.path.isfile(full_name + '.crt'):
        shutil.copy(full_name + '.crt', output_dir)
    
    # Jobs file: remove reference to old dyd
    jobs_root = etree.parse(full_name + '.jobs', XMLparser).getroot()
    for ref in jobs_root.findall('.//dyn:dynModels', {jobs_prefix: namespace}):
        ref.getparent().remove(ref)

    with open(os.path.join(output_dir, name + '.jobs'), 'wb') as doc:
        doc.write(etree.tostring(jobs_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Fic_multiple file
    fic_root = etree.Element(fic_rootName, nsmap=fic_namespace_map)
    scenarios_attrib = {'jobsFile': name + '.jobs'}
    scenarios = etree.SubElement(fic_root, etree.QName(namespace, 'scenarios'), scenarios_attrib)


    for (filename, close_event_list) in close_events_dic.items():
        print(filename)
        for close_events_sublist in close_event_list:
            (time, model, event) = close_events_sublist[0]
            print(time, model, event)

            if 'zone 1' in event:
                evnt = 'Z1'
            elif 'zone 2' in event:
                evnt = 'Z2'
            elif 'zone 3' in event:
                evnt = 'Z3'
            elif 'zone 4' in event:
                evnt = 'Z4'
            else:
                raise NotImplementedError('Only consider distance protections')

            
            init_event = filename[9:-4]  # Strips leading "timeline_" and ending ".xml"
            scenarioID = init_event + '-' + model + '-' + evnt

            # Add fic scenarios
            scenario_attrib = {'id': scenarioID, 'dydFile': scenarioID + '.dyd'}
            scenario =  etree.SubElement(scenarios, etree.QName(namespace, 'scenario'), scenario_attrib)

            # Dyd file: whole dyd file, but refer to the new par + include the initiating event

            dyd_root = etree.parse(full_name + '.dyd', XMLparser).getroot()
            init_event_root = etree.parse(os.path.join(working_dir, init_event + '.dyd'), XMLparser).getroot()
            dyd_root.append(etree.Comment('Init event'))
            for dyd_model in init_event_root:
                dyd_root.append(dyd_model)
            
            for dyd_model in dyd_root:
                if dyd_model.get('parFile') is not None:
                    dyd_model.set('parFile', scenarioID + '.par')
            
            with open(os.path.join(output_dir, scenarioID + '.dyd'), 'wb') as doc:
                doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

            # Par file
            par_root = etree.parse(full_name + '.par', XMLparser).getroot()

            # Increase delay of the relay that tripped first in a series of close events
            set = par_root.find('.//par:set[@id="' + model + '"]', par_namespace_map)
            if evnt == 'Z1':
                par = set.find('.//par:par[@name="distance_T1"]', par_namespace_map)
            elif evnt == 'Z2':
                par = set.find('.//par:par[@name="distance_T2"]', par_namespace_map)
            elif evnt == 'Z3':
                par = set.find('.//par:par[@name="distance_T3"]', par_namespace_map)
            elif evnt == 'Z4':
                par = set.find('.//par:par[@name="distance_T4"]', par_namespace_map)
            else:
                raise NotImplementedError()
            par.set('value', str(float(par.get('value')) + 0.1))

            with open(os.path.join(output_dir, scenarioID + '.par'), 'wb') as doc:
                doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

            for (time, model, event) in close_events_sublist[1:]:
                print('\t', time, model, event)
        print()
    
        # Copy results of originals to not rerun them (only what is in the timeline folder)
        shutil.copy(os.path.join(working_dir, 'timeLine', filename), output_dir_timeline)

    with open(os.path.join(output_dir, 'fic_MULTIPLE.xml'), 'wb') as doc:
        doc.write(etree.tostring(fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    """

    ###
    # Statistics
    ###
    print("Nb cases with close events: ", len(close_events_dic)); print()

    for (model, count) in sorted(count_occurences.items(), key=lambda item: item[1]):
        print(model, count)
    
    with open(os.path.join(working_dir, 'TimelineAnalysis.csv'), 'w', newline='')  as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(next(iter(outputs.values())).keys())
        for output in outputs.values():
            out = list(output.values())[:-1]
            for out_i in list(output.values())[-1]:  # Unrol the events
                for out_j in out_i:
                    out.append(out_j)
            writer.writerow(out)


    outputs_lst = list(outputs.values())
    outputs_lst = sorted(outputs_lst, key=lambda d: d['Init event'], reverse=True)  # Sort by filename (alphabetically)
    for i in range(len(outputs_lst)):
        filename, load_shedding, _, __, ___, ____, events = outputs_lst[i].values()

        for j in range(i+1, len(outputs_lst)):
            next_filename, next_load_shedding, _, __, ___, ____, next_events = outputs_lst[j].values()

            short_filename = filename[:-4]  # Remove .xml extension
            if short_filename in next_filename:  # Means next is the same simulation but with a slowed down protection
                print(short_filename, next_load_shedding - load_shedding, load_shedding, next_load_shedding)
            else:
                break

        """ # Not very interesting
        for i in range(len(outputs_lst)):
            filename, load_shedding, _, __, ___, ____, _____, events = outputs_lst[i].values()
            first_event_time = float(events[0][0])  # Out of bound error

            for j in range(i+1, len(outputs_lst)):
                next_filename, next_load_shedding, _, __, ___, ____, _____, next_events = outputs_lst[j].values()
                next_first_event_time = float(next_events[0][0])

                short_filename = filename[:-4]  # Remove .xml extension
                if short_filename in next_filename:  # Means next is the same simulation but with a slowed down protection
                    print(short_filename, next_first_event_time - first_event_time, first_event_time, next_first_event_time)
                else:
                    break """

    for (filename, close_event_list) in close_events_dic.items():
        print(filename, end=' ')
        for close_events_sublist in close_event_list:
            (time, model, event) = close_events_sublist[0]
            # print(time, model, event, newline=' ')
            for (next_time, next_model, next_event) in close_events_sublist[1:]:
                print(float(next_time) - float(time), end=' ')
        print()

    print('###')
    print('#')
    print('###')


    for filename in filenames:
        Z_armings = Z_armings_dic[filename]
        Z_disarmings = Z_disarmings_dic[filename]

        if len(Z_disarmings) == 0:
            continue
        print(filename, end=' ')

        min_remaining_time = 999999
        remaining_times = []
        for (time, model, event) in Z_armings:
            time = float(time)
            found = False
            for (next_time, next_model, next_event) in Z_disarmings:
                next_time = float(next_time)
                if next_time > time and model == next_model and event[:26] == next_event[:26]:  # Last condition checks it is the same zone
                    disarming_time = next_time - time
                    found = True
                    break
            
            if not found:  # Disarming not found (i.e. tripped)
                continue

            if 'zone 1' in event:
                continue # remaining_time = 0.02 - disarming_time
            elif 'zone 2' in event:
                remaining_time = 0.3 - disarming_time
            elif 'zone 3' in event:
                remaining_time = 0.6 - disarming_time
            elif 'zone 4' in event:
                continue
            else:
                print(event)
                raise NotImplementedError('')

            remaining_times.append(remaining_time)

            # if remaining_time < 0:
                # continue
                # raise ValueError('Either caused by incorrect settings or numerical issue')
            
            if remaining_time < min_remaining_time:
                min_remaining_time = remaining_time
                min_time1 = time
                min_time2 = next_time
                min_model = next_model
                min_event = next_event
        print(min_remaining_time, min_time1, min_time2, min_model, min_event , end=' ')
        #print(remaining_times)
        print()

    ###
    # Special
    ###

    special = True
    if special:
        print()
        print('###')
        print('# Special')
        print('###')
        print()
        for filename in filenames:
            print(filename, end=' ')

            trip_timeline = outputs[filename]['Trip events']

            actual_trip_timeline = [(time, model, event) for (time, model, event) in trip_timeline if 'trip zone 2' in event or 'trip zone 3' in event]
            fake_trip_timeline = [(time, model, event) for (time, model, event) in trip_timeline if 'trip zone 1' in event or 'trip zone 4' in event]

            translated_fake_trip_timeline = []
            for (time, model, event) in fake_trip_timeline:
                if event == 'Distance protection trip zone 1':
                    translated_fake_trip_timeline.append((time, model, 'Distance protection trip zone 2'))
                elif event == 'Distance protection trip zone 4':
                    translated_fake_trip_timeline.append((time, model, 'Distance protection trip zone 3'))
                else:
                    raise ValueError('')

            """
            same_order = True
            for i in range(len(actual_trip_timeline)):
                (_, model, event) = actual_trip_timeline[i]
                try:
                    (_, fake_model, fake_event) = translated_fake_trip_timeline[i]
                except:
                    print()
                    print(actual_trip_timeline)
                    print(translated_fake_trip_timeline)
                    print()
                    raise

                if model != fake_model or event != fake_event:
                    same_order = False
                    break
            """

            same_order = True
            for (time, model, event) in trip_timeline:
                time = float(time)
                found = False
                for (fake_time, fake_model, fake_event) in translated_fake_trip_timeline:
                    fake_time = float(fake_time)
                    if fake_model == model and fake_event == event:
                        found = True
                        continue
                    elif not found:  # Look for events that occured after (time, model, event) in the original timeline,
                        continue
                    else:
                        if fake_time + 0.07 < time:  # check if they can occur before the considered event
                            same_order = False
                            break



            missing_events = False
            for (_, fake_model, fake_event) in translated_fake_trip_timeline:
                found = False
                for (_, model, event) in actual_trip_timeline:
                    if fake_model == model and fake_event == event:
                        found = True
                        break
                if not found:
                    missing_events = True
                    break
            
            if not same_order:
                print('1', end=' ')
            else:
                print('0', end=' ')

            if missing_events:
                print('1', end=' ')
            else:
                print('0', end=' ')

            if not same_order or missing_events:
                print('1', end=' ')
            else:
                print('0', end=' ')
            print()
