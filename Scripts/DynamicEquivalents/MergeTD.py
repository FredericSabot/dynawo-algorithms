from lxml import etree
import argparse
import pypowsybl as pp
import os
import shutil
import re
import glob
from random import randrange

def getCleanXMLTag(tag):
    """
    E.g. '{http://www.rte-france.com/dynawo}blackBoxModel' -> 'blackBoxModel'
    """
    if type(tag) != str:  # Occurs if the element is a comment
        return ''
    return re.sub("[\{].*?[\}]", "", tag)

def addPrefixToXMLElementAttributes(element, prefix : str, keys : list):
    """
    keys is the keys of the element attributes to which the prefix should be added
    """
    for key in keys:
        value = element.get(key)
        if value != 'NETWORK' and value != 'OMEGA_REF':  # Iidms are merged -> NETWORK contains all D networks
            element.set(key, prefix + element.get(key))

def mergeDyds(full_T_name, full_D_name, full_TD_name, nb_it=None, with_ufls=False):
    XMLparser = etree.XMLParser(remove_blank_text=True)
    T_root = etree.parse(full_T_name + '.dyd', XMLparser).getroot()

    T = pp.network.load(full_T_name + '.iidm')
    T_loads = T.get_loads()
    for load in T_loads.index:
        if T_loads.at[load, 'p0'] < 0.01 or load == 'LOAD___39_EC':
            continue

        D_root = etree.parse(full_D_name + '.dyd', XMLparser).getroot()
        # r = randrange(50)
        # D_path = os.path.join('/home/fsabot/Desktop/dynawo-algorithms/examples/CIGRE_MV_Wind_voltage/SA/RandomRuns', 'It_%03d' % r, 'CIGRE_MV_Wind.dyd')
        # D_root = etree.parse(D_path, XMLparser).getroot()
        for element in D_root:
            tag = getCleanXMLTag(element.tag)
            if nb_it:
                if tag == 'blackBoxModel':
                    if element.get('id') == 'OMEGA_REF':  # Neglect all elements after <blackBoxModel id="OMEGA_REF"
                        break

            if tag == 'blackBoxModel':
                addPrefixToXMLElementAttributes(element, load + '_', ['id', 'parId', 'staticId'])
                for subelement in element:
                    subtag = getCleanXMLTag(subelement.tag)
                    if subtag == 'macroStaticRef':
                        addPrefixToXMLElementAttributes(subelement, load + '_', ['id'])
            elif tag == 'connect':
                addPrefixToXMLElementAttributes(element, load + '_', ['id1', 'id2'])
                if element.get('var1') == 'infiniteBus_omegaPu':
                    element.set('id1', 'OMEGA_REF')
                    element.set('var1', 'omegaRef_grp_0')
                    if element.get('var2') == 'load_omegaRefPu_value':  # Connections are not exactly the same with OMEGA_REF and inf bus
                        element.set('var2', 'load_omegaRefPu')
                    elif element.get('var2') == 'ibg_omegaRefPu':
                        element.set('var1', 'omegaRef_grp_0_value')
                if element.get('var2') == 'infiniteBus_omegaPu':
                    element.set('id2', 'OMEGA_REF')
                    element.set('var2', 'omegaRef_grp_0')
                    if element.get('var1') == 'load_omegaRefPu_value':
                        element.set('var1', 'load_omegaRefPu')
                    elif element.get('var1') == 'ibg_omegaRefPu':
                        element.set('var2', 'omegaRef_grp_0_value')
            elif tag == 'macroConnect':
                addPrefixToXMLElementAttributes(element, load + '_', ['id1', 'id2', 'connector'])
            elif tag == 'macroStaticReference' or tag == 'macroConnector':
                addPrefixToXMLElementAttributes(element, load + '_', ['id'])
            else:
                print(tag)
                raise NotImplementedError('Did I forget something?')
            T_root.append(element)

        # Remove dynamic load models (and their connection) of the loads replaced by distribution systems
        for element in list(T_root):
            if T_loads.at[load, 'p0'] < 0.01:
                continue
            tag = getCleanXMLTag(element.tag)
            if tag == 'blackBoxModel':
                if element.get('id') == load:
                    T_root.remove(element)
            elif tag == 'connect' or tag == 'macroConnect':
                if element.get('id1') == load or element.get('id2') == load:
                    T_root.remove(element)

    for element in T_root:
        tag = getCleanXMLTag(element.tag)
        if tag == 'blackBoxModel':
            element.set('parFile', os.path.basename(full_TD_name) + '.par')

        if with_ufls:
            if tag == 'blackBoxModel' and '_EC_L' in element.get('id') and 'slack' not in element.get('id'):  # TODO: make something less hardcoded
                connect_attribs = [
                    {'id1': 'UFLS', 'var1': 'ufls_deltaPQfiltered', 'id2': element.get('id'), 'var2': 'load_deltaP'},
                    {'id1': 'UFLS', 'var1': 'ufls_deltaPQfiltered', 'id2': element.get('id'), 'var2': 'load_deltaQ'}
                ]
                for connect_attrib in connect_attribs:
                    etree.SubElement(T_root, etree.QName('http://www.rte-france.com/dynawo', 'connect'), connect_attrib)

    with open(full_TD_name + '.dyd', 'wb') as doc:
        doc.write(etree.tostring(T_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

def mergePars(full_T_name, full_D_name, full_TD_name):
    XMLparser = etree.XMLParser(remove_blank_text=True)
    T_root = etree.parse(full_T_name + '.par', XMLparser).getroot()

    T = pp.network.load(full_T_name + '.iidm')
    T_loads = T.get_loads()
    for load in T_loads.index:
        if T_loads.at[load, 'p0'] < 0.01 or load == 'LOAD___39_EC':
            continue
        D_root = etree.parse(full_D_name + '.par', XMLparser).getroot()
        # r = randrange(50)
        # D_path = os.path.join('/home/fsabot/Desktop/dynawo-algorithms/examples/CIGRE_MV_Wind_voltage/SA/RandomRuns', 'It_%03d' % r, 'CIGRE_MV_Wind.par')
        # D_root = etree.parse(D_path, XMLparser).getroot()
        for element in D_root:
            element.set('id', load + '_' + element.get('id'))
            T_root.append(element)

    with open(full_TD_name + '.par', 'wb') as doc:
        doc.write(etree.tostring(T_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

def mergeIidms(full_T_name, full_D_name, full_TD_name, nb_it = None):
    T = pp.network.load(full_T_name + '.iidm')
    D = pp.network.load(full_D_name + '.iidm')

    if nb_it:
        D.remove_elements('LINE-GEN')
        D.remove_elements('P1')
        D.create_generators(id='GEN-slack', max_p=1e9, min_p=-1e9, voltage_regulator_on=True, target_p=0.9853699928339129, target_v=113.3, voltage_level_id='VL-0', 
                            bus_id='B-0', connectable_bus_id='B-0')

    P_load = 0

    T_loads = T.get_loads()
    for load in T_loads.index:
        # r = randrange(50)
        # D_path = os.path.join('/home/fsabot/Desktop/dynawo-algorithms/examples/CIGRE_MV_Wind_voltage/SA/RandomRuns', 'It_%03d' % r, 'CIGRE_MV_Wind.iidm')
        # D = pp.network.load(D_path)
        # D.remove_elements('LINE-GEN')
        # D.remove_elements('P1')
        # D.create_generators(id='GEN-slack', max_p=1e9, min_p=-1e9, voltage_regulator_on=True, target_p=0.9853699928339129, target_v=113.3, voltage_level_id='VL-0', 
        #                     bus_id='B-0', connectable_bus_id='B-0')

        # Scaling
        if nb_it or True:
            # P_D = 0.9853699928339129
            # Q_D = 0.624464919206555

            # P_D = 2.5756359475314063
            # Q_D = 0.9681799419061042
            
            P_D = 5.0219189866505980  # Gross load of the system -> will cause a decrease of total net load in the transmission system that has to be compensated beforehand
            # Q_D = 1.5474945827433073

        else:
            P_D = D.get_generators().at['GEN-slack', 'target_p']
            Q_D = -D.get_generators().at['GEN-slack', 'q']
        P_T = T_loads.at[load, 'p0']
        Q_T = T_loads.at[load, 'q0']
        scaling_factor = P_T / P_D

        if P_T < 0.01 or load == 'LOAD___39_EC':
            continue

        if not nb_it and False:
            q_mismatch = Q_T - Q_D * scaling_factor
            bus = T_loads.at[load, 'bus_id']
            vl = T_loads.at[load, 'voltage_level_id']
            bus_breaker_view = T.get_bus_breaker_topology(vl).buses
            bus = bus_breaker_view.index[bus_breaker_view['bus_id'] == bus].tolist()[0]  # Damn, I hate pandas
            T.create_loads(id=load + '_mismatch', p0=0, q0=q_mismatch, voltage_level_id=vl, bus_id=bus)

        T.remove_elements(load)

        P_load += P_T

        # Copy all elements from D in place of the load in T
        subs = D.get_substations()
        sub_ids = [load + '_' + sub for sub in subs.index]
        T.create_substations(id=sub_ids)

        vls = D.get_voltage_levels()
        vls.index = [load + '_' + vl for vl in vls.index]
        for index in vls.index:
            vls.at[index, 'substation_id'] = load + '_' + vls.at[index, 'substation_id']
            if index == load + '_' + 'VL-0':
                # Match nominal voltage of point of common coupling
                vls.at[index, 'nominal_v'] = T.get_voltage_levels().at[T_loads.at[load, 'voltage_level_id'], 'nominal_v']
        topology_kind = ['BUS_BREAKER'] * len(vls.index)
        vls['topology_kind'] = topology_kind
        T.create_voltage_levels(vls)

        buses = D.get_buses()
        actual_bus_ids = []
        for bus in buses.index:
            vl = buses.at[bus, 'voltage_level_id']
            bus_view = D.get_bus_breaker_topology(vl).buses
            actual_bus_ids.append(bus_view.index[bus_view['bus_id'] == bus].tolist()[0])
            buses.at[bus, 'voltage_level_id'] = load + '_' + vl
        buses.index = actual_bus_ids
        buses.index = [load + '_' + bus for bus in buses.index]
        T.create_buses(id=buses.index, voltage_level_id=buses.voltage_level_id)

        D_loads = D.get_loads()
        D_loads.index = [load + '_' + D_load for D_load in D_loads.index]
        for D_load in D_loads.index:
            vl = D_loads.at[D_load, 'voltage_level_id']
            bus_view = D.get_bus_breaker_topology(vl).buses
            bus_id = bus_view.index[bus_view['bus_id'] == D_loads.at[D_load, 'bus_id']].tolist()[0]
            D_loads.at[D_load, 'bus_id'] = load + '_' + bus_id
            D_loads.at[D_load, 'voltage_level_id'] = load + '_' + D_loads.at[D_load, 'voltage_level_id']
        T.create_loads(id=D_loads.index, p0=D_loads.p0 * scaling_factor, q0=D_loads.q0 * scaling_factor, voltage_level_id=D_loads.voltage_level_id, bus_id=D_loads.bus_id)

        gens = D.get_generators()
        gens = gens.drop('GEN-slack', axis=0)  # Remove 'external grid' source (i.e. T system) from the D system
        gens.index = [load + '_' + gen for gen in gens.index]
        for gen in gens.index:
            vl = gens.at[gen, 'voltage_level_id']
            bus_view = D.get_bus_breaker_topology(vl).buses
            bus_id = bus_view.index[bus_view['bus_id'] == gens.at[gen, 'bus_id']].tolist()[0]
            gens.at[gen, 'bus_id'] = load + '_' + bus_id
            gens.at[gen, 'voltage_level_id'] = load + '_' + gens.at[gen, 'voltage_level_id']
        T.create_generators(id=gens.index, target_p=gens.target_p * scaling_factor, target_q=gens.target_q * scaling_factor, target_v=gens.target_v,
                            min_p=gens.min_p * scaling_factor, max_p=gens.max_p * scaling_factor, voltage_regulator_on=gens.voltage_regulator_on,
                            voltage_level_id=gens.voltage_level_id, bus_id=gens.bus_id)

        tfos = D.get_2_windings_transformers()
        tfos.index = [load + '_' + tfo for tfo in tfos.index]
        for tfo in tfos.index:
            bus1 = tfos.at[tfo, 'bus1_id']
            vl1 = tfos.at[tfo, 'voltage_level1_id']
            bus_breaker_view = D.get_bus_breaker_topology(vl1).buses
            tfos.at[tfo, 'bus1_id'] = load + '_' + bus_breaker_view.index[bus_breaker_view['bus_id'] == bus1].tolist()[0]
            tfos.at[tfo, 'voltage_level1_id'] = load + '_' + vl1
            bus2 = tfos.at[tfo, 'bus2_id']
            vl2 = tfos.at[tfo, 'voltage_level2_id']
            bus_breaker_view = D.get_bus_breaker_topology(vl2).buses
            tfos.at[tfo, 'bus2_id'] = load + '_' + bus_breaker_view.index[bus_breaker_view['bus_id'] == bus2].tolist()[0]
            tfos.at[tfo, 'voltage_level2_id'] = load + '_' + vl2
            if vl1 == 'VL-0':
                # Match side 1 tfo voltage with the T voltage
                tfos.at[tfo, 'rated_u1'] = T.get_voltage_levels().at[T_loads.at[load, 'voltage_level_id'], 'nominal_v']
        T.create_2_windings_transformers(id=tfos.index, r=tfos.r / scaling_factor, x=tfos.x / scaling_factor, g=tfos.g * scaling_factor, b=tfos.b * scaling_factor,
                                         rated_u1=tfos.rated_u1, rated_u2=tfos.rated_u2, rated_s=tfos.rated_s * scaling_factor,
                                         voltage_level1_id=tfos.voltage_level1_id, voltage_level2_id=tfos.voltage_level2_id,
                                         bus1_id=tfos.bus1_id, bus2_id=tfos.bus2_id)

        lines = D.get_lines()
        lines.index = [load + '_' + line for line in lines.index]
        for line in lines.index:
            bus1 = lines.at[line, 'bus1_id']
            vl1 = lines.at[line, 'voltage_level1_id']
            bus_breaker_view = D.get_bus_breaker_topology(vl1).buses
            lines.at[line, 'bus1_id'] = load + '_' + bus_breaker_view.index[bus_breaker_view['bus_id'] == bus1].tolist()[0]
            lines.at[line, 'voltage_level1_id'] = load + '_' + vl1
            bus2 = lines.at[line, 'bus2_id']
            vl2 = lines.at[line, 'voltage_level2_id']
            bus_breaker_view = D.get_bus_breaker_topology(vl2).buses
            lines.at[line, 'bus2_id'] = load + '_' + bus_breaker_view.index[bus_breaker_view['bus_id'] == bus2].tolist()[0]
            lines.at[line, 'voltage_level2_id'] = load + '_' + vl2
        T.create_lines(id=lines.index, r=lines.r / scaling_factor, x=lines.x / scaling_factor, g1=lines.g1 * scaling_factor, b1=lines.b1 * scaling_factor,
                       g2=lines.g2 * scaling_factor, b2=lines.b2 * scaling_factor, voltage_level1_id=lines.voltage_level1_id,
                       voltage_level2_id=lines.voltage_level2_id, bus1_id=lines.bus1_id, bus2_id=lines.bus2_id)

        # Create a line between the T and D systems
        bus1 = T_loads.at[load, 'bus_id']
        vl1 = T_loads.at[load, 'voltage_level_id']
        bus_breaker_view = T.get_bus_breaker_topology(vl1).buses
        bus1 = bus_breaker_view.index[bus_breaker_view['bus_id'] == bus1].tolist()[0]  # Damn, I hate pandas

        bus2 = D.get_generators().at['GEN-slack', 'bus_id']
        vl2 = D.get_generators().at['GEN-slack', 'voltage_level_id']
        bus_breaker_view = D.get_bus_breaker_topology(vl2).buses
        bus2 = load + '_' + bus_breaker_view.index[bus_breaker_view['bus_id'] == bus2].tolist()[0]
        vl2 = load + '_' + vl2

        T.create_lines(id='LD-{}'.format(load), voltage_level1_id=vl1, bus1_id=bus1, voltage_level2_id=vl2, bus2_id=bus2, r=0, x=0.001)


    lf_parameters = pp.loadflow.Parameters(distributed_slack=False)
    print(pp.loadflow.run_ac(T, lf_parameters))
    T.dump(full_TD_name, 'XIIDM', {'iidm.export.xml.version' : '1.4'})  # Latest version support by Dynawo
    os.rename(full_TD_name + '.xiidm', full_TD_name + '.iidm')


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Takes as input a transmission system and a distribution system, and replace all loads of the T system by the D system'
    'Hypothesis:'
    '   - The T files are in working_dir/T_dir/ and named T_name.*, similar for the D files',
    '   - In the initial D system, the T system is represented by a generator named "GEN-slack"'
    '   - The D system only contains buses (bus breaker view), loads, lines, generators and two windings transformers'
    '   - Reuse the same D system for all T loads:'
    '       - Scaling: the total P of the D system is matched to each ones of the T loads it replaces'
    '       - Similarly, Q is also scaled, but keeping constant the power factor of the D system'
    '       - The nominal voltages of the transformer(s) and buses at the point of common coupling are matched between the two systems'
    '           - If the TFO is modelled in the T system, it is assumed already matched'
    '           - If the TFO is modelled in the D system, its nominal voltage on side 1 is matched to the nominal voltage of the T load is replaces.'
    '               The nominal voltage of the voltage level at the point of common coupling (D side) is also matched. It is assumed that this VL is "VL-0".')
    '               The transformers to be matched have their side 1 connected to the VL called "VL-0.'

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--T_dir', type=str, required=True,
                        help='Name of the directory containing the transmission network')
    parser.add_argument('--T_name', type=str, required=True,
                        help='Name of the transmission network')
    parser.add_argument('--D_dir', type=str, required=True,
                        help='Name of the directory containing the distribution network')
    parser.add_argument('--D_name', type=str, required=True,
                        help='Name of the distribution network')
    parser.add_argument('--TD_dir', type=str, required=True,
                        help='Name of the directory where to write the merged network')
    parser.add_argument('--TD_name', type=str, required=True,
                        help='Name of the merged network')
    parser.add_argument('--nb_it', type=str, required=False,
                        help='Number of random copies of the distribution network to consider')
    parser.add_argument('--with_ufls', type=str, required=True,
                        help='True if there is an UFLS relay connected to the loads')

    args = parser.parse_args()
    working_dir = args.working_dir
    T_dir = args.T_dir
    T_name = args.T_name
    D_dir = args.D_dir
    D_name = args.D_name
    TD_dir = args.TD_dir
    TD_name = args.TD_name
    if args.nb_it:
        nb_it = int(args.nb_it)
    else:
        nb_it = None

    if args.with_ufls == "True":
        with_ufls = True
    elif args.with_ufls == "False":
        with_ufls = False
    else:
        raise

    full_T_name = os.path.join(working_dir, T_dir, T_name)

    if nb_it is None:
        full_D_name = os.path.join(working_dir, D_dir, D_name)
        full_TD_name = os.path.join(working_dir, TD_dir, TD_name)
        os.makedirs(os.path.dirname(full_TD_name), exist_ok=True)

        mergeIidms(full_T_name, full_D_name, full_TD_name)
        mergeDyds(full_T_name, full_D_name, full_TD_name, with_ufls=with_ufls)
        mergePars(full_T_name, full_D_name, full_TD_name)

        shutil.copy(full_T_name + '.jobs', full_TD_name + '.jobs')
        if os.path.isfile(full_T_name + '.crv'):
            shutil.copy(full_T_name + '.crv', full_TD_name + '.crv')
        if os.path.isfile(full_T_name + '.fsv'):
            shutil.copy(full_T_name + '.fsv', full_TD_name + '.fsv')
        if os.path.isfile(full_T_name + '.crt'):
            shutil.copy(full_T_name + '.crt', full_TD_name + '.crt')

        # Copy files for SA if any
        full_T_dir = os.path.join(working_dir, T_dir)
        full_TD_dir = os.path.join(working_dir, TD_dir)
        fic = os.path.join(full_T_dir, 'fic_MULTIPLE.xml')
        if os.path.isfile(fic):
            shutil.copy(fic, full_TD_dir)
        dyd_files = glob.glob(os.path.join(full_T_dir, '*.dyd'))
        dyd_files.remove(full_T_name + '.dyd')
        for dyd_file in dyd_files:
            shutil.copy(dyd_file, full_TD_dir)    
    else:
        for i in range(nb_it):
            full_D_name = os.path.join(working_dir, D_dir, 'It_%03d' % i, D_name)
            full_TD_name = os.path.join(working_dir, TD_dir + '_Random', 'It_%03d' % i, TD_name)
            os.makedirs(os.path.dirname(full_TD_name), exist_ok=True)

            mergeIidms(full_T_name, full_D_name, full_TD_name, nb_it)
            mergeDyds(full_T_name, full_D_name, full_TD_name, nb_it, with_ufls)
            mergePars(full_T_name, full_D_name, full_TD_name)

            shutil.copy(full_T_name + '.jobs', full_TD_name + '.jobs')
            if os.path.isfile(full_T_name + '.crv'):
                shutil.copy(full_T_name + '.crv', full_TD_name + '.crv')
            if os.path.isfile(full_T_name + '.fsv'):
                shutil.copy(full_T_name + '.fsv', full_TD_name + '.fsv')
            if os.path.isfile(full_T_name + '.crt'):
                shutil.copy(full_T_name + '.crt', full_TD_name + '.crt')
            
            # Copy files for SA if any
            full_T_dir = os.path.join(working_dir, T_dir)
            full_TD_dir = os.path.join(working_dir, TD_dir + '_Random', 'It_%03d' % i)
            fic = os.path.join(full_T_dir, 'fic_MULTIPLE.xml')
            if os.path.isfile(fic):
                shutil.copy(fic, full_TD_dir)
            dyd_files = glob.glob(os.path.join(full_T_dir, '*.dyd'))
            dyd_files.remove(full_T_name + '.dyd')
            for dyd_file in dyd_files:
                shutil.copy(dyd_file, full_TD_dir)
