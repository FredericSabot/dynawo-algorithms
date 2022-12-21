from lxml import etree
import argparse
import pypowsybl as pp
import os

def get_voltage_level_to_loads(network):
    """
    Compute a dictionary where the keys are all the voltage level ids in the network, and values are a list of all loads connected
    to said voltage levels.
    """
    loads = network.get_loads()
    vls = network.get_voltage_levels()
    out = {}

    for vlID in vls.index:
        value = []
        for loadID in loads.index:
            if loads.at[loadID, 'voltage_level_id'] == vlID:
                value.append(loadID)
        out[vlID] = value
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Add dummy loads to voltage levels that do not have a load to allow to do short-circuits'
    ''
    'Hypothesis:'
    '   - All files describing the network use the same name (e.g. IEEE14.iidm, IEEE14.dyd, etc.) and are in the "working_dir" folder')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the network')

    args = parser.parse_args()

    working_dir = args.working_dir
    full_name = os.path.join(working_dir, args.name)
    name = args.name

    # Read iidm
    n = pp.network.load(full_name + ".iidm")
    vls = n.get_voltage_levels()
    
    # Read dyd
    dydMultiple_prefix = 'dyn'
    dydMultiple_namespace = 'http://www.rte-france.com/dynawo'
    dyd_namespace_map = {dydMultiple_prefix: dydMultiple_namespace}
    XMLparser = etree.XMLParser(remove_blank_text=True) # Necessary for pretty_print to work
    dyd_root = etree.parse(full_name + '.dyd', XMLparser).getroot()

    # Read par
    parMultiple_prefix = 'par'
    parMultiple_namespace = 'http://www.rte-france.com/dynawo'
    par_namespace_map = {parMultiple_prefix: parMultiple_namespace}
    par_root = etree.parse(full_name + '.par', XMLparser).getroot()

    # Add dummy loads
    vl2loads = get_voltage_level_to_loads(n)

    for vlID in vls.index:
        if vl2loads[vlID] == []: # Only add dummy load if no load is already present
            loadID = vlID + '_DummyLoad'
            busID = vlID
            busID = busID.replace('VL', 'TN')

            n.create_loads(id= loadID, voltage_level_id = vlID, bus_id=busID, p0=0.001, q0=0.001)
    
            # Dyn model for load
            """ Example
            <dyn:blackBoxModel id="_BUS____1_VL_DummyLoad" lib="LoadAlphaBeta" parFile="IEEE39.par" parId="DummyLoad" staticId="_BUS____1_VL_DummyLoad">
                <dyn:staticRef var="load_PPu" staticVar="p"/>
                <dyn:staticRef var="load_QPu" staticVar="q"/>
                <dyn:staticRef var="load_state" staticVar="state"/>
            </dyn:blackBoxModel>
            <dyn:connect id1="_BUS____1_VL_DummyLoad" var1="load_terminal" id2="NETWORK" var2="_BUS____1_TN_ACPIN"/>
            <dyn:connect id1="_BUS____1_VL_DummyLoad" var1="load_switchOffSignal1" id2="NETWORK" var2="_BUS____1_TN_switchOff"/>
            """
            load_attrib = {'id': loadID, 'lib': 'LoadAlphaBeta', 'parFile': name + '.par', 'parId': 'DummyLoad', 'staticId': loadID}
            load = etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'blackBoxModel'), load_attrib)

            staticRef_attribs = [
                {'var': 'load_PPu', 'staticVar': 'p'},
                {'var': 'load_QPu', 'staticVar': 'q'},
                {'var': 'load_state', 'staticVar': 'state'},
            ]
            for staticRef_attrib in staticRef_attribs:
                etree.SubElement(load, etree.QName(dydMultiple_namespace, 'staticRef'), staticRef_attrib)
            
            connect_attribs = [
                {'id1': loadID, 'var1': 'load_terminal', 'id2': 'NETWORK', 'var2': busID + '_ACPIN'},
                {'id1': loadID, 'var1': 'load_switchOffSignal1', 'id2': 'NETWORK', 'var2': busID + '_switchOff'},
            ]
            for connect_attrib in connect_attribs:
                etree.SubElement(dyd_root, etree.QName(dydMultiple_namespace, 'connect'), connect_attrib)

    # Dummy load parameters
    load_par_set = etree.SubElement(par_root, etree.QName(parMultiple_namespace, 'set'), {'id' : 'DummyLoad'})
    par_attribs = [
        {'type':'DOUBLE', 'name':'load_alpha', 'value':'2'},
        {'type':'DOUBLE', 'name':'load_beta', 'value':'2'}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(load_par_set, etree.QName(parMultiple_namespace, 'par'), par_attrib)
    
    ref_attribs = [
        {'type':'DOUBLE', 'name':'load_P0Pu', 'origData':'IIDM', 'origName':'p_pu'},
        {'type':'DOUBLE', 'name':'load_Q0Pu', 'origData':'IIDM', 'origName':'q_pu'},
        {'type':'DOUBLE', 'name':'load_U0Pu', 'origData':'IIDM', 'origName':'v_pu'},
        {'type':'DOUBLE', 'name':'load_UPhase0', 'origData':'IIDM', 'origName':'angle_pu'},
    ]
    for ref_attrib in ref_attribs:
        etree.SubElement(load_par_set, etree.QName(parMultiple_namespace, 'reference'), ref_attrib)

    # Write iidm
    # pp.loadflow.run_ac(n)
    n.dump(full_name + '_dummy', 'XIIDM', {'iidm.export.xml.version' : '1.4'}) # Latest version supported by Dynawo
    os.rename(full_name + '_dummy.xiidm', full_name + '_dummy.iidm') # Set back original extension (powsybl always set it to XIIDM)

    # Write the modified dyd and par files
    with open(full_name + '_dummy.dyd', 'wb') as doc:
        doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    with open(full_name + '_dummy.par', 'wb') as doc:
        doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
