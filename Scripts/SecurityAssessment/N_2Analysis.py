from locale import T_FMT
from lxml import etree
import argparse
import pypowsybl as pp
import pandas as pd
import os
import shutil
import Protections


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generates the fic_MULTIPLE and dyd files necessary to perform a N-2 analysis on the input iidm. '
    'The considered contingencies are short circuits at each end of each line (cleared by opening the line), and short circuits on each generators '
    '(cleared by disconnecting the generator). The parameters of these contingencies have to be added manually in the par file (assumed to have the same '
    'name as the iidm file), in the parameter sets "Fault", "LineDisc", and "GeneratorDisc"'
    ''
    ''
    'Hypotheses:'
    '   - Bus-breaker model'
    '   - Generators id\'s are the same in the iidm and dyd files')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--output', type=str,
                        help='Output directory (relative path)', default="N-2_Analysis")
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the network')

    args = parser.parse_args()

    working_dir = args.working_dir
    output_dir = os.path.join(working_dir, args.output)
    full_name = os.path.join(working_dir, args.name)
    name = args.name

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # XML boilerplate
    XMLparser = etree.XMLParser(remove_blank_text=True) # Necessary for pretty_print to work
    namespace = 'http://www.rte-france.com/dynawo'
    ficMultiple_prefix = None
    fic_namespace_map = {ficMultiple_prefix: namespace}
    fic_rootName = etree.QName(namespace, 'multipleJobs')
    dyd_prefix = 'dyn'
    dyd_rootName = etree.QName(namespace, 'dynamicModelsArchitecture')
    dyd_namespace_map = {dyd_prefix: namespace}
    par_prefix = 'par'
    par_namespace_map = {par_prefix: namespace}
    

    # Create fic_multiple file
    fic_root = etree.Element(fic_rootName, nsmap=fic_namespace_map)
    scenarios_attrib = {'jobsFile': name + '.jobs'}
    scenarios = etree.SubElement(fic_root, etree.QName(namespace, 'scenarios'), scenarios_attrib)

    # Read inputs
    dyd_root = etree.parse(full_name + '.dyd', XMLparser).getroot()
    par_root = etree.parse(full_name + '.par', XMLparser).getroot()

    # Read network
    n = pp.network.load(full_name + '.iidm')
    lines = n.get_lines()
    gens = n.get_generators()

    # Base case (no faults)
    scenario_attrib = {'id': 'Base'}
    scenario =  etree.SubElement(scenarios, etree.QName(namespace, 'scenario'), scenario_attrib)

    # Only consider lines fault/disconnections for now (esp. since will use ~node-breaker later)
    # Fault on each end of the line + disconnection of the line + CB failure

    # For now, CB failure assumed to disconnect one adjacent element (don't know which -> do them all)

    # Line faults are actually on buses, not lines ==>  To model fault of line_end1 with stuck CB on end2:
    #   - open the line and clear fault after 100ms (zone 1 time)
    #   - create a new fault at _end2 with resistance = line resistance + line impedance (incl. capa if possible) after 100ms
    #   - clear the new fault + adjacent line after 200ms (CB failure protection time)

    # Fault on line_end1 with stuck CB on same end is more straighforward:
    #   - open line at 100ms
    #   - disconnect the other element at 200ms
    #   - remove the fault at 200ms

    t_init = 5
    t_clearing = '{}'.format(t_init + 0.1)
    t_backup = '{}'.format(t_init + 0.2)
    t_init = '{}'.format(t_init)
    r_fault = 0.0001
    x_fault = 0.0001

    # Initiating fault parameter set
    fault_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'InitFault'})
    par_attribs = [
        {'type':'DOUBLE', 'name':'fault_RPu', 'value':'{}'.format(r_fault)},
        {'type':'DOUBLE', 'name':'fault_XPu', 'value':'{}'.format(x_fault)},
        {'type':'DOUBLE', 'name':'fault_tBegin', 'value': t_init},
        {'type':'DOUBLE', 'name':'fault_tEnd', 'value': t_clearing}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(fault_par_set, etree.QName(namespace, 'par'), par_attrib)
    
    # Initiating fault cleared in backup time parameter set
    fault_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'InitFaultBackup'})
    par_attribs[-1]['value'] = t_backup
    for par_attrib in par_attribs:
        etree.SubElement(fault_par_set, etree.QName(namespace, 'par'), par_attrib)

    # Replacement fault parameter set
    for lineID in lines.index:
        replacement_fault_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'Replacement fault_' + lineID})
        
        voltage_level = lines.at[lineID, 'voltage_level1_id']
        Vb = float(n.get_voltage_levels().at[voltage_level, 'nominal_v']) * 1000
        Sb = 100e6  # 100MW is default base in Dynawo
        Zb = Vb**2/Sb
        r_line = lines.at[lineID, 'r'] / Zb
        x_line = lines.at[lineID, 'x'] / Zb

        r = r_fault + r_line
        x = r_fault + x_line

        par_attribs = [
            {'type':'DOUBLE', 'name':'fault_RPu', 'value':'{}'.format(r)},
            {'type':'DOUBLE', 'name':'fault_XPu', 'value':'{}'.format(x)},
            {'type':'DOUBLE', 'name':'fault_tBegin', 'value': t_clearing}, # Starts when the initial fault is cleared
            {'type':'DOUBLE', 'name':'fault_tEnd', 'value': t_backup}
        ]
        for par_attrib in par_attribs:
            etree.SubElement(replacement_fault_par_set, etree.QName(namespace, 'par'), par_attrib)

    # Line disconnection parameter set
    line_disc_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'LineDisc'})
    par_attribs = [
        {'type':'DOUBLE', 'name':'event_tEvent', 'value': t_clearing},
        {'type':'BOOL', 'name':'event_disconnectOrigin', 'value':'true'},
        {'type':'BOOL', 'name':'event_disconnectExtremity', 'value':'true'},
    ]
    for par_attrib in par_attribs:
        etree.SubElement(line_disc_par_set, etree.QName(namespace, 'par'), par_attrib)

    # Adjacent line disconnection parameter set
    adj_line_disc_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'AdjLineDisc'})
    par_attribs = [
        {'type':'DOUBLE', 'name':'event_tEvent', 'value': t_backup},
        {'type':'BOOL', 'name':'event_disconnectOrigin', 'value':'true'},
        {'type':'BOOL', 'name':'event_disconnectExtremity', 'value':'true'}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(adj_line_disc_par_set, etree.QName(namespace, 'par'), par_attrib)

    # Generator disconnection parameter set
    gen_disc_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'GenDisc'})
    par_attribs = [
        {'type':'DOUBLE', 'name':'event_tEvent', 'value': t_clearing},
        {'type':'BOOL', 'name':'event_stateEvent1', 'value':'true'}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(gen_disc_par_set, etree.QName(namespace, 'par'), par_attrib)

    # Adjacent gen disconnection parameter set
    adj_gen_disc_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'AdjGenDisc'})
    par_attribs = [
        {'type':'DOUBLE', 'name':'event_tEvent', 'value': t_backup},
        {'type':'BOOL', 'name':'event_stateEvent1', 'value':'true'}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(adj_gen_disc_par_set, etree.QName(namespace, 'par'), par_attrib)


    bus2lines = Protections.get_buses_to_lines(n)

    for lineID in lines.index:
        for fault_side in [1, 2]:
            for CB_fail_side in [1, 2]:
                for adj_lineID in Protections.get_adjacent_lines(bus2lines, n, lineID, CB_fail_side):

                    scenarioID = lineID + '_end{}-CB_end{}-'.format(fault_side, CB_fail_side) + adj_lineID
                    # Add scenarios to the fic_MULTIPLE.xml
                    scenario_attrib = {'id': scenarioID, 'dydFile': scenarioID + '.dyd'}
                    scenario =  etree.SubElement(scenarios, etree.QName(namespace, 'scenario'), scenario_attrib)

                    # Create dyd
                    dyd_root = etree.Element(dyd_rootName, nsmap=dyd_namespace_map)
                    # Initial fault
                    busID = '@' + lineID + '@@NODE{}@'.format(fault_side)
                    if fault_side == CB_fail_side:
                        par_set_name = 'InitFaultBackup' # Fault cleared in backup up time
                    else:
                        par_set_name = 'InitFault' # Fault "cleared" in normal time, but replaced by another one
                    blackbox_attrib = {'id': 'FAULT_' + lineID + '_end{}'.format(fault_side), 'lib': 'NodeFault', 'parFile': name + '.par', 'parId': par_set_name}
                    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
                    connect_attrib = {'id1': 'FAULT_' + lineID + '_end{}'.format(fault_side), 'var1': 'fault_terminal', 'id2': 'NETWORK', 'var2': busID + '_ACPIN'}
                    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

                    # Line disconnection
                    blackbox_attrib = {'id': 'DISC_' + lineID, 'lib': 'EventQuadripoleDisconnection', 'parFile': name + '.par', 'parId': 'LineDisc'}
                    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
                    connect_attrib = {'id1': 'DISC_' + lineID, 'var1': 'event_state1_value', 'id2': 'NETWORK', 'var2': lineID + '_state_value'}
                    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

                    # Adjacent line disconnection
                    blackbox_attrib = {'id': 'DISC_' + adj_lineID, 'lib': 'EventQuadripoleDisconnection', 'parFile': name + '.par', 'parId': 'AdjLineDisc'}
                    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
                    connect_attrib = {'id1': 'DISC_' + adj_lineID, 'var1': 'event_state1_value', 'id2': 'NETWORK', 'var2': adj_lineID + '_state_value'}
                    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

                    if fault_side != CB_fail_side:
                        # "Replacement" fault
                        par_set_name = 'Replacement fault_' + lineID
                        location = '@' + lineID + '@@NODE{}@_ACPIN'.format(CB_fail_side)

                        blackbox_attrib = {'id': 'Replacement fault', 'lib': 'NodeFault', 'parFile': name + '.par', 'parId': par_set_name}
                        etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
                        connect_attrib = {'id1': 'Replacement fault', 'var1': 'fault_terminal', 'id2': 'NETWORK', 'var2': location}
                        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)


                    with open(os.path.join(output_dir, scenarioID + '.dyd'), 'wb') as doc:
                        doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))




    """
    for genID in gens.index:
        scenario_attrib = {'id': genID, 'dydFile': genID + '.dyd'}
        scenario =  etree.SubElement(scenarios, etree.QName(namespace, 'scenario'), scenario_attrib)

        busID = '@' + genID + '@@NODE@'

        # Create dyd
        dyd_root = etree.Element(dyd_rootName, nsmap=dyd_namespace_map)
        blackbox_attrib = {'id': 'FAULT_' + busID, 'lib': 'NodeFault', 'parFile': name + '.par', 'parId': 'Fault'}
        etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
        # Connect fault
        connect_attrib = {'id1': 'FAULT_' + busID, 'var1': 'fault_terminal', 'id2': 'NETWORK', 'var2': busID + '_ACPIN'}
        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)
        # Generator disconnection
        blackbox_attrib = {'id': 'DISC_' + genID, 'lib': 'EventSetPointBoolean', 'parFile': name + '.par', 'parId': 'GenDisc'}
        etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
        connect_attrib = {'id1': 'DISC_' + genID, 'var1': 'event_state1', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

        with open(os.path.join(output_dir, genID + '.dyd' ), 'wb') as doc:
            doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    """

    with open(os.path.join(output_dir, 'fic_MULTIPLE.xml'), 'wb') as doc:
        doc.write(etree.tostring(fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    with open(os.path.join(output_dir, name + '.par'), 'wb') as doc:
        doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    shutil.copy(full_name + '.iidm', output_dir)
    shutil.copy(full_name + '.dyd', output_dir)
    # shutil.copy(full_name + '.par', output_dir)
    shutil.copy(full_name + '.jobs', output_dir)

    if os.path.isfile(full_name + '.crv'):
        shutil.copy(full_name + '.crv', output_dir)
    if os.path.isfile(full_name + '.crt'):
        shutil.copy(full_name + '.crt', output_dir)

    # ./myEnvDynawoAlgorithms.sh SA --directory examples/RBTS --input fic_MULTIPLE.xml --output aggregatedResults.xml --nbThreads 6
