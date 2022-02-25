from lxml import etree
import argparse
import pypowsybl as pp
import pandas as pd
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generates the fic_MULTIPLE and dyd files necessary to perform a N-1 analysis on the input iidm. '
    'The considered contingencies are short circuits at each end of each line (cleared by opening the line), and short circuits on each generators '
    '(cleared by disconnecting the generator). The parameters of these contingencies have to be added manually in the par file (assumed to have the same '
    'name as the iidm file), in the parameter sets "Fault", "LineDisc", and "GeneratorDisc"')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--input_iidm', type=str, required=True,
                        help='Input iidm')

    args = parser.parse_args()


    [iidm_name, extension] = args.input_iidm.rsplit('.', 1)
    args.input_iidm = os.path.join(args.working_dir, args.input_iidm)

    # Fic_multiple file
    ficMultiple_prefix = None
    ficMultiple_namespace = 'http://www.rte-france.com/dynawo'
    fic_namespace_map = {ficMultiple_prefix: ficMultiple_namespace}

    rootName = etree.QName(ficMultiple_namespace, 'multipleJobs')
    fic_root = etree.Element(rootName, nsmap=fic_namespace_map)

    scenarios_attrib = {'jobsFile': iidm_name + '.jobs'}  # Assume the jobs file has the same name as the iidm
    scenarios = etree.SubElement(fic_root, etree.QName(ficMultiple_namespace, 'scenarios'), scenarios_attrib)

    # Dyd files
    dydMultiple_prefix = 'dyd'
    dydMultiple_namespace = 'http://www.rte-france.com/dynawo'
    dyd_namespace_map = {dydMultiple_prefix: dydMultiple_namespace}

    rootName = etree.QName(dydMultiple_namespace, 'dynamicModelsArchitecture')

    # Read network
    n = pp.network.load(args.input_iidm)
    lines = n.get_lines()
    gens = n.get_generators()

    # Fault on each end of the line + disconnection of the line
    for lineId in lines.index:
        # Fault on bus at end 1 of line
        # Add scenarios to fic
        scenario_attrib = {'id': lineId + '_end1', 'dydFile': lineId + '_end1' + '.dyd'}
        scenario =  etree.SubElement(scenarios, etree.QName(ficMultiple_namespace, 'scenario'), scenario_attrib)
        scenario_attrib = {'id': lineId + '_end2', 'dydFile': lineId + '_end2' + '.dyd'}
        scenario =  etree.SubElement(scenarios, etree.QName(ficMultiple_namespace, 'scenario'), scenario_attrib)

        busIds = [lines.at[lineId, 'bus1_id'], lines.at[lineId, 'bus2_id']]
        # TODO use n.get_bus_breaker_topology().buses, when version 0.13 of pypowsybl is available
        # Currently, buses ID are read from the "bus view", i.e. automatically generated from the voltage level IDs
        ends = ['1', '2']
        for i in range(len(busIds)):
            # Create dyd
            busId = busIds[i]
            end = ends[i]
            dyd_root = etree.Element(rootName, nsmap=dyd_namespace_map)
            blackbox_attrib = {'id': 'FAULT_' + busId, 'lib': 'NodeFault', 'parFile': iidm_name + '.par', 'parId': 'Fault'}
            blackbox =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'blackBoxModel'), blackbox_attrib)
            # Connect fault
            connect_attrib = {'id1': 'FAULT_' + busId, 'var1': 'fault_terminal', 'id2': 'NETWORK', 'var2': busId + '_ACPIN'}
            connect =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'connect'), connect_attrib)
            # Line disconnection
            blackbox_attrib = {'id': 'DISC_' + lineId, 'lib': 'EventQuadripoleDisconnection', 'parFile': iidm_name + '.par', 'parId': 'LineDisc'}
            blackbox =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'blackBoxModel'), blackbox_attrib)
            connect_attrib = {'id1': 'DISC_' + lineId, 'var1': 'event_state1_value', 'id2': 'NETWORK', 'var2': lineId + '_state_value'}
            connect =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'connect'), connect_attrib)

            with open(os.path.join(args.working_dir, lineId + '_end' + end + '.dyd'), 'wb') as doc:
                doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    
    for genId in gens.index:
        scenario_attrib = {'id': genId, 'dydFile': genId + '.dyd'}
        scenario =  etree.SubElement(scenarios, etree.QName(ficMultiple_namespace, 'scenario'), scenario_attrib)

        busId = gens.at[genId, 'bus_id']

        dyd_root = etree.Element(rootName, nsmap=dyd_namespace_map)
        blackbox_attrib = {'id': 'FAULT_' + busId, 'lib': 'NodeFault', 'parFile': iidm_name + '.par', 'parId': 'Fault'}
        blackbox =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'blackBoxModel'), blackbox_attrib)
        # Connect fault
        connect_attrib = {'id1': 'FAULT_' + busId, 'var1': 'fault_terminal', 'id2': 'NETWORK', 'var2': busId + '_ACPIN'}
        connect =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'connect'), connect_attrib)
        # Generator disconnection
        blackbox_attrib = {'id': 'DISC_' + genId, 'lib': 'EventSetPointBoolean', 'parFile': iidm_name + '.par', 'parId': 'GenDisc'}
        blackbox =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'blackBoxModel'), blackbox_attrib)
        connect_attrib = {'id1': 'DISC_' + genId, 'var1': 'event_state1', 'id2': genId, 'var2': 'generator_switchOffSignal2'}
        connect =  etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'connect'), connect_attrib)

        with open(os.path.join(args.working_dir, genId + '.dyd' ), 'wb') as doc:
            doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))


    with open(os.path.join(args.working_dir, 'fic_MULTIPLE.xml'), 'wb') as doc:
        doc.write(etree.tostring(fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # ./myEnvDynawoAlgorithms.sh SA --directory examples/RBTS --input fic_MULTIPLE.xml --output aggregatedResults.xml --nbThreads 6
