from lxml import etree
import argparse
import pypowsybl as pp
import pandas as pd
import os
import shutil

def add_scenario_to_fic_scenarios(fic_scenarios, namespace, scenarioID, with_dyd = True):
    if with_dyd:
        scenario_attrib = {'id': scenarioID, 'dydFile': scenarioID + '.dyd'}
    else:
        scenario_attrib = {'id': scenarioID}
    etree.SubElement(fic_scenarios, etree.QName(namespace, 'scenario'), scenario_attrib)
    # return fic_scenarios  # Not needed


def add_bus_fault_to_etree(dyd_root, namespace, network_name, faultID, busID, parID):
    blackbox_attrib = {'id': faultID, 'lib': 'NodeFault', 'parFile': network_name + '.par', 'parId': parID}
    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
    connect_attrib = {'id1': faultID, 'var1': 'fault_terminal', 'id2': 'NETWORK', 'var2': busID + '_ACPIN'}
    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)


def add_line_disc_to_etree(dyd_root, namespace, network_name, lineID, parID = 'LineDisc'):
    blackbox_attrib = {'id': 'DISC_' + lineID, 'lib': 'EventQuadripoleDisconnection', 'parFile': network_name + '.par', 'parId': parID}
    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
    connect_attrib = {'id1': 'DISC_' + lineID, 'var1': 'event_state1_value', 'id2': 'NETWORK', 'var2': lineID + '_state_value'}
    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)


def add_gen_disc_to_etree(dyd_root, namespace, network_name, genID, parID = 'GenDisc'):
    blackbox_attrib = {'id': 'DISC_' + genID, 'lib': 'EventSetPointBoolean', 'parFile': network_name + '.par', 'parId': parID}
    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), blackbox_attrib)
    connect_attrib = {'id1': 'DISC_' + genID, 'var1': 'event_state1', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)


def add_bus_fault_parameters_to_etree(par_root, namespace, parID, t_init, t_clearing, r_fault, x_fault):
    fault_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : parID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'fault_RPu', 'value':'{}'.format(r_fault)},
        {'type':'DOUBLE', 'name':'fault_XPu', 'value':'{}'.format(x_fault)},
        {'type':'DOUBLE', 'name':'fault_tBegin', 'value': t_init},
        {'type':'DOUBLE', 'name':'fault_tEnd', 'value': t_clearing}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(fault_par_set, etree.QName(namespace, 'par'), par_attrib)


def add_line_disc_parameters_to_etree(par_root, namespace, parID, t_disc):
    line_disc_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : parID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'event_tEvent', 'value': t_disc},
        {'type':'BOOL', 'name':'event_disconnectOrigin', 'value':'true'},
        {'type':'BOOL', 'name':'event_disconnectExtremity', 'value':'true'},
    ]
    for par_attrib in par_attribs:
        etree.SubElement(line_disc_par_set, etree.QName(namespace, 'par'), par_attrib)


def add_gen_disc_parameters_to_etree(par_root, namespace, parID, t_disc):
    gen_disc_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : parID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'event_tEvent', 'value': t_disc},
        {'type':'BOOL', 'name':'event_stateEvent1', 'value':'true'}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(gen_disc_par_set, etree.QName(namespace, 'par'), par_attrib)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generates the fic_MULTIPLE and dyd files necessary to perform a N-1 analysis on the input iidm. '
    'The considered contingencies are short circuits at each end of each line (cleared by opening the line), and short circuits on each generators '
    '(cleared by disconnecting the generator). The parameters of these contingencies have to be added manually in the par file (assumed to have the same '
    'name as the iidm file), in the parameter sets "Fault", "LineDisc", and "GeneratorDisc"'
    ''
    ''
    'Hypothesis:'
    '   - Bus-breaker model'
    '   - Generators id\'s are the same in the iidm and dyd files'
    '   - All files describing the network use the same name (e.g. IEEE14.iidm, IEEE14.dyd, etc.) and are in the "working_dir" folder')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--output', type=str,
                        help='Output directory (relative path)', default="N-1_Analysis")
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
    dyd_prefix = 'dyn'
    dyd_rootName = etree.QName(namespace, 'dynamicModelsArchitecture')
    dyd_namespace_map = {dyd_prefix: namespace}
    par_prefix = 'par'
    par_namespace_map = {par_prefix: namespace}
    fic_rootName = etree.QName(namespace, 'multipleJobs')

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
    add_scenario_to_fic_scenarios(scenarios, namespace, 'Base', with_dyd=False)

    # Fault on each end of the line + disconnection of the line
    for lineID in lines.index:
        busIDs = ['@' + lineID + '@@NODE1@', '@' + lineID + '@@NODE2@']
        ends = [1, 2]
        for busID, end in zip(busIDs, ends):
            scenarioID = lineID + '_end{}'.format(end)
            add_scenario_to_fic_scenarios(scenarios, namespace, scenarioID)

            # Create dyd
            dyd_root = etree.Element(dyd_rootName, nsmap=dyd_namespace_map)
            # Fault
            faultID = 'FAULT_' + lineID + '_end{}'.format(end)
            add_bus_fault_to_etree(dyd_root, namespace, name, faultID, busID, parID='InitFault')
            # Line disconnection
            add_line_disc_to_etree(dyd_root, namespace, name, lineID)

            with open(os.path.join(output_dir, scenarioID + '.dyd'), 'wb') as doc:
                doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    
    for genID in gens.index:
        scenarioID = genID
        add_scenario_to_fic_scenarios(scenarios, namespace, scenarioID)

        # Create dyd
        dyd_root = etree.Element(dyd_rootName, nsmap=dyd_namespace_map)
        # Fault
        busID = '@' + genID + '@@NODE@'
        faultID = 'FAULT_' + busID
        add_bus_fault_to_etree(dyd_root, namespace, name, faultID, busID, parID='InitFault')
        # Generator disconnection
        add_gen_disc_to_etree(dyd_root, namespace, name, genID)        

        with open(os.path.join(output_dir, genID + '.dyd' ), 'wb') as doc:
            doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Write fic
    with open(os.path.join(output_dir, 'fic_MULTIPLE.xml'), 'wb') as doc:
        doc.write(etree.tostring(fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Parameters
    t_init = 5
    t_clearing = str(t_init + 0.1)
    t_init = str(t_init)
    r_fault = str(0.0001)
    x_fault = str(0.0001)

    add_bus_fault_parameters_to_etree(par_root, namespace, 'InitFault', t_init, t_clearing, r_fault, x_fault)
    add_line_disc_parameters_to_etree(par_root, namespace, 'LineDisc', t_clearing)
    add_gen_disc_parameters_to_etree(par_root, namespace, 'GenDisc', t_clearing)

    with open(os.path.join(output_dir, name + '.par'), 'wb') as doc:
        doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Copy non-modified files
    shutil.copy(full_name + '.iidm', output_dir)
    shutil.copy(full_name + '.dyd', output_dir)
    shutil.copy(full_name + '.jobs', output_dir)

    if os.path.isfile(full_name + '.crv'):
        shutil.copy(full_name + '.crv', output_dir)
    if os.path.isfile(full_name + '.crt'):
        shutil.copy(full_name + '.crt', output_dir)
    if os.path.isfile(full_name + '.fsv'):
        shutil.copy(full_name + '.fsv', output_dir)

    # ./myEnvDynawoAlgorithms.sh SA --directory examples/RBTS --input fic_MULTIPLE.xml --output aggregatedResults.xml --nbThreads 6
