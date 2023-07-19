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


def add_dyn_load_model_to_etree(dyd_root, namespace, loadID, busID, parID):
    load_attrib = {'id': loadID, 'lib': 'LoadAlphaBeta', 'parFile': name + '.par', 'parId': parID, 'staticId': loadID}
    load = etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), load_attrib)

    staticRef_attribs = [
        {'var': 'load_PPu', 'staticVar': 'p'},
        {'var': 'load_QPu', 'staticVar': 'q'},
        {'var': 'load_state', 'staticVar': 'state'},
    ]
    for staticRef_attrib in staticRef_attribs:
        etree.SubElement(load, etree.QName(namespace, 'staticRef'), staticRef_attrib)
    
    connect_attribs = [
        {'id1': loadID, 'var1': 'load_terminal', 'id2': 'NETWORK', 'var2': busID + '_ACPIN'},
        {'id1': loadID, 'var1': 'load_switchOffSignal1', 'id2': 'NETWORK', 'var2': busID + '_switchOff'},
    ]
    for connect_attrib in connect_attribs:
        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)


def add_dyn_load_parameters_to_etree(par_root, namespace, parID):
    load_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : parID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'load_alpha', 'value':'2'},
        {'type':'DOUBLE', 'name':'load_beta', 'value':'2'}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(load_par_set, etree.QName(namespace, 'par'), par_attrib)
    
    ref_attribs = [
        {'type':'DOUBLE', 'name':'load_P0Pu', 'origData':'IIDM', 'origName':'p_pu'},
        {'type':'DOUBLE', 'name':'load_Q0Pu', 'origData':'IIDM', 'origName':'q_pu'},
        {'type':'DOUBLE', 'name':'load_U0Pu', 'origData':'IIDM', 'origName':'v_pu'},
        {'type':'DOUBLE', 'name':'load_UPhase0', 'origData':'IIDM', 'origName':'angle_pu'},
    ]
    for ref_attrib in ref_attribs:
        etree.SubElement(load_par_set, etree.QName(namespace, 'reference'), ref_attrib)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Add dummy loads (i.e. with negligeable consomption) to voltage levels that do not have a load'
    'to allow to simulate short-circuits on those voltage levels'
    ''
    'Hypothesis:'
    '   - All files describing the network use the same name (e.g. IEEE14.iidm, IEEE14.dyd, etc.) and are in the "working_dir" folder'
    '   - Bus IDs end with "TN" and voltage level IDs end with "VL"')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the network')

    args = parser.parse_args()

    working_dir = args.working_dir
    full_name = os.path.join(working_dir, args.name)
    name = args.name

    # XML boilerplate
    XMLparser = etree.XMLParser(remove_blank_text=True)  # Necessary for pretty_print to work
    namespace = 'http://www.rte-france.com/dynawo'
    dydMultiple_prefix = 'dyn'
    dyd_namespace_map = {dydMultiple_prefix: namespace}
    parMultiple_prefix = 'par'
    par_namespace_map = {parMultiple_prefix: namespace}

    # Read inputs
    n = pp.network.load(full_name + ".iidm")
    vls = n.get_voltage_levels()
    
    dyd_root = etree.parse(full_name + '.dyd', XMLparser).getroot()
    par_root = etree.parse(full_name + '.par', XMLparser).getroot()

    # Add dummy loads
    vl2loads = get_voltage_level_to_loads(n)
    for vlID in vls.index:
        if vl2loads[vlID] == []:  # Only add dummy load if no load is already present
            loadID = vlID + '_DummyLoad'
            busID = vlID
            busID = busID.replace('VL', 'TN')

            # Add static load model
            n.create_loads(id= loadID, voltage_level_id = vlID, bus_id=busID, p0=0.001, q0=0.001)
            # Add dynamic load model
            add_dyn_load_model_to_etree(dyd_root, namespace, loadID, busID, parID='DummyLoad')

    # Add parameters for the dynamic load models (same are used for all dummy loads)
    add_dyn_load_parameters_to_etree(par_root, namespace, parID='DymmyLoad')

    # Write iidm
    # pp.loadflow.run_ac(n)  # Running a load flow is not necessary (negligible changes) and can sometimes cause issues
    n.dump(full_name + '_dummy', 'XIIDM', {'iidm.export.xml.version' : '1.4'})  # Latest version supported by Dynawo
    os.rename(full_name + '_dummy.xiidm', full_name + '_dummy.iidm')  # Set back original extension (powsybl always set it to XIIDM)

    # Write the modified dyd and par files
    with open(full_name + '_dummy.dyd', 'wb') as doc:
        doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
    with open(full_name + '_dummy.par', 'wb') as doc:
        doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
