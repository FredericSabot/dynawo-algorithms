from lxml import etree
import argparse
import pypowsybl as pp
import os
from math import pi
import random
import shutil

def get_buses_to_lines(network):
    """
    Compute a dictionary where the keys are the bus ids of all buses in the network, and values are a list of all lines connected
    to said buses.
    """
    lines = network.get_lines()
    buses = network.get_buses()
    out = {}

    for busID in buses.index:
        out[busID] = []
        for lineID in lines.index:
            if lines.at[lineID, 'bus1_id'] == busID or lines.at[lineID, 'bus2_id'] == busID:
                out[busID].append(lineID)
    return out


def get_adjacent_lines(bus_to_lines, network, lineID, side):
    """
    Get the list of lines that are connected to the side 'side' (1 or 2) or the line with ID 'lineID'. 'bus_to_lines' is the dict that
    maps buses to the lines that are connected to them (computed with function get_buses_to_lines(network))
    """
    lines = network.get_lines()
    common_bus = lines.at[lineID, 'bus{}_id'.format(side)]

    adj_lines = bus_to_lines[common_bus].copy()
    adj_lines.remove(lineID) # Remove the line itself from adjacent elements
    return adj_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Add some protections to an existing grid'
    ''
    ''
    'Hypotheses:'
    '   - Generators id\'s are the same in the iidm and dyd files'
    '   - All files describing the network use the same name (e.g. IEEE14.iidm, IEEE14.dyd, etc.) and are in the "working_dir" folder')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory (relative path)')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the network')
    parser.add_argument("--randomise", action="store_true", dest="randomise",
                      help='Randomises the settings of protections', default=False)
    parser.add_argument('--nb_runs', type=int, required=False, default='50',
                        help='Number of random samples to draw')
    parser.add_argument("--special", action="store_true", dest="special",
                      help='Use special distance protection scheme', default=False)

    args = parser.parse_args()

    working_dir = args.working_dir
    full_name = os.path.join(working_dir, args.name)
    name = args.name
    randomise = args.randomise
    nb_runs = args.nb_runs

    # XML boilerplate
    XMLparser = etree.XMLParser(remove_blank_text=True)  # Necessary for pretty_print to work
    dydMultiple_prefix = 'dyd'
    namespace = 'http://www.rte-france.com/dynawo'
    dyd_namespace_map = {dydMultiple_prefix: namespace}
    par_prefix = 'par'
    par_namespace_map = {par_prefix: namespace}


    CB_time = 0.08
    CB_max_error = 0.01  # +/- 10ms

    if randomise:
        runIDs = range(nb_runs)
    else:
        runIDs = [0]

    for runID in runIDs:
        if randomise:
            output_dir = os.path.join(working_dir, args.output + str(runID))
        else:
            output_dir = os.path.join(working_dir, args.output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        random.seed(runID)

        dyd_root = etree.parse(full_name + '.dyd', XMLparser).getroot()
        par_root = etree.parse(full_name + '.par', XMLparser).getroot()

        # Read iidm
        n = pp.network.load(full_name + ".iidm")
        lines = n.get_lines()
        gens = n.get_generators()

        # Generator protections
        dyd_root.append(etree.Comment('Generator protections'))
        once = True # Only write the parameters once (same value for all generators)
        for genID in gens.index:
            # Over-/under-speed protection
            """ Example
            <dyn:blackBoxModel id="GEN___30_SM_Speed" lib="SpeedProtection" parFile="IEEE39.par" parId="Speed"/>
            <dyn:connect id1="GEN___30_SM_Speed" var1="speedProtection_omegaMonitoredPu" id2="GEN___30_SM" var2="generator_omegaPu_value"/>
            <dyn:connect id1="GEN___30_SM_Speed" var1="speedProtection_switchOffSignal" id2="GEN___30_SM" var2="generator_switchOffSignal2"/>
            """
            speed_attrib = {'id': genID + '_Speed', 'lib': 'SpeedProtection', 'parFile': name + '.par', 'parId': 'Speed_' + genID}
            etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), speed_attrib)
            
            connect_attribs = [
                {'id1': genID + '_Speed', 'var1': 'speedProtection_omegaMonitoredPu', 'id2': genID, 'var2': 'generator_omegaPu_value'},
                {'id1': genID + '_Speed', 'var1': 'speedProtection_switchOffSignal', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
            ]
            for connect_attrib in connect_attribs:
                etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

            if randomise:
                rand_omegaMax = random.uniform(-0.01, 0.01)
                rand_omegaMin = random.uniform(-0.01, 0.01)
                rand_CB = random.uniform(-CB_max_error, CB_max_error)
            else:
                rand_omegaMax = 0
                rand_omegaMin = 0
                rand_CB = 0

            speed_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'Speed_' + genID})
            par_attribs = [
                {'type':'DOUBLE', 'name':'speedProtection_omegaMaxPu', 'value':'{}'.format(1.05 + rand_omegaMax)},
                {'type':'DOUBLE', 'name':'speedProtection_omegaMinPu', 'value':'{}'.format(0.95 + rand_omegaMin)},
                {'type':'DOUBLE', 'name':'speedProtection_tLagAction', 'value':'{}'.format(0.02 + CB_time + rand_CB)}
            ]
            for par_attrib in par_attribs:
                etree.SubElement(speed_par_set, etree.QName(namespace, 'par'), par_attrib)
            
            # Under-voltage protection
            """ Example
            dyn:blackBoxModel id="GEN___30_SM_UVA" lib="UnderVoltageAutomaton" parFile="IEEE39.par" parId="UVA"/>
            <dyn:connect id1="GEN___30_SM_UVA" var1="underVoltageAutomaton_UMonitoredPu" id2="NETWORK" var2="@GEN___30_SM@@NODE@_Upu_value"/>
            <dyn:connect id1="GEN___30_SM_UVA" var1="underVoltageAutomaton_switchOffSignal" id2="GEN___30_SM" var2="generator_switchOffSignal2"/>
            """
            uva_attrib = {'id': genID + '_UVA', 'lib': 'UnderVoltageAutomaton', 'parFile': name + '.par', 'parId': 'UVA_' + genID}
            etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), uva_attrib)

            connect_attribs = [
                {'id1': genID + '_UVA', 'var1': 'underVoltageAutomaton_UMonitoredPu', 'id2': 'NETWORK', 'var2': '@' + genID + '@@NODE@_Upu_value'},
                {'id1': genID + '_UVA', 'var1': 'underVoltageAutomaton_switchOffSignal', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
            ]
            for connect_attrib in connect_attribs:
                etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)
            
            if randomise:
                rand_UMin = random.uniform(-0.05, 0)
                rand_CB = random.uniform(-CB_max_error, CB_max_error)
            else:
                rand_UMin = 0
                rand_CB = 0

            uva_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'UVA_' + genID})
            par_attribs = [
                {'type':'DOUBLE', 'name':'underVoltageAutomaton_UMinPu', 'value':'{}'.format(0.85 + rand_UMin)},
                {'type':'DOUBLE', 'name':'underVoltageAutomaton_tLagAction', 'value':'{}'.format(1.5 + rand_CB)}
            ]
            for par_attrib in par_attribs:
                etree.SubElement(uva_par_set, etree.QName(namespace, 'par'), par_attrib)
            
            # Out-of-step protection
            """ Example
            <dyn:blackBoxModel id="Gen39_OOS" lib="OutOfStep" parFile="IEEE39.par" parId="OOS"/>
            <dyn:connect id1="Gen39_OOS" var1="oos_thetaMonitoredPu" id2="GEN___39_SM" var2="generator_thetaInternal_value"/>
            <dyn:connect id1="Gen39_OOS" var1="oos_switchOffSignal" id2="GEN___39_SM" var2="generator_switchOffSignal2"/>
            """
            oos_attrib = {'id': genID + '_OOS', 'lib': 'OutOfStep', 'parFile': name + '.par', 'parId': 'OOS_' + genID}
            etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), oos_attrib)

            connect_attribs = [
                {'id1': genID + '_OOS', 'var1': 'oos_thetaMonitoredPu',  'id2': genID, 'var2': 'generator_thetaInternal_value'},
                {'id1': genID + '_OOS', 'var1': 'oos_switchOffSignal', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
            ]
            for connect_attrib in connect_attribs:
                etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)
            
            if randomise:
                rand_thetaMin = random.uniform(-10, 10) * pi/180
                rand_CB = random.uniform(-CB_max_error, CB_max_error)
            else:
                rand_thetaMin = 0
                rand_CB = 0

            oos_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'OOS_' + genID})
            par_attribs = [
                {'type':'DOUBLE', 'name':'oos_thetaMinPu', 'value':'{}'.format(3*pi/4 + rand_thetaMin)},
                {'type':'DOUBLE', 'name':'oos_tLagAction', 'value':'{}'.format(0.02 + CB_time + rand_CB)}
            ]
            for par_attrib in par_attribs:
                etree.SubElement(oos_par_set, etree.QName(namespace, 'par'), par_attrib)

            once = False
        
        # UFLS, for now, all loads are connected to the same relay
        dyd_root.append(etree.Comment('Under-frequency load shedding'))
        """ Example
        if centralised:
            <dyn:blackBoxModel id="UFLS" lib="UFLS10Steps" parFile="IEEE39.par" parId="UFLS"/>
            <dyn:connect id1="UFLS" var1="ufls_omegaMonitoredPu" id2="OMEGA_REF" var2="omegaRef_0_value"/>
            <dyn:connect id1="UFLS" var1="ufls_deltaPQfiltered" id2="LOAD____3_EC" var2="load_deltaP"/>
            <dyn:connect id1="UFLS" var1="ufls_deltaPQfiltered" id2="LOAD____3_EC" var2="load_deltaQ"/>
            [..., connect to all other loads]
        else:
            
        """
        centralised = True
        if centralised: # All UFLSs have the same parameters, and read the same frequency -> only one model to save variables
            ufls_attrib = {'id': 'UFLS', 'lib': 'UFLS10Steps', 'parFile': name + '.par', 'parId': 'UFLS'}
            etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), ufls_attrib)

            connect_attribs = [
                {'id1': 'UFLS', 'var1': 'ufls_omegaMonitoredPu',  'id2': 'OMEGA_REF', 'var2': 'omegaRef_0_value'}
            ]
            for loadID in n.get_loads().index:
                if 'Dummy' not in loadID: # Skip dummy loads
                    connect_attribs += [
                        {'id1': 'UFLS', 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaP'},
                        {'id1': 'UFLS', 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaQ'}
                    ]
            for connect_attrib in connect_attribs:
                etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)
            
            # UFLS parameters
            ufls_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : 'UFLS'})
            par_attribs = [
                {'type':'DOUBLE', 'name':'ufls_UFLSStep1', 'value':'0.1'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep2', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep3', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep4', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep5', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep6', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep7', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep8', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep9', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_UFLSStep10', 'value':'0.05'},
                {'type':'DOUBLE', 'name':'ufls_omega1Pu', 'value':'0.98'},
                {'type':'DOUBLE', 'name':'ufls_omega2Pu', 'value':'0.978'},
                {'type':'DOUBLE', 'name':'ufls_omega3Pu', 'value':'0.976'},
                {'type':'DOUBLE', 'name':'ufls_omega4Pu', 'value':'0.974'},
                {'type':'DOUBLE', 'name':'ufls_omega5Pu', 'value':'0.972'},
                {'type':'DOUBLE', 'name':'ufls_omega6Pu', 'value':'0.97'},
                {'type':'DOUBLE', 'name':'ufls_omega7Pu', 'value':'0.968'},
                {'type':'DOUBLE', 'name':'ufls_omega8Pu', 'value':'0.966'},
                {'type':'DOUBLE', 'name':'ufls_omega9Pu', 'value':'0.964'},
                {'type':'DOUBLE', 'name':'ufls_omega10Pu', 'value':'0.962'},
                {'type':'DOUBLE', 'name':'ufls_tLagAction', 'value':'0.1'}
            ]
            for par_attrib in par_attribs:
                etree.SubElement(ufls_par_set, etree.QName(namespace, 'par'), par_attrib)
        """ else: # Decentralised UFLS
            # random.seed(0)
            for loadID in n.get_loads().index:
                if 'Dummy' in loadID: # Skip dummy loads
                    continue
                ufls_attrib = {'id': loadID + '_UFLS', 'lib': 'UFLS10Steps', 'parFile': name + '.par', 'parId': loadID + '_UFLS'}
                etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), ufls_attrib)

                connect_attribs = [
                    {'id1': loadID + '_UFLS', 'var1': 'ufls_omegaMonitoredPu', 'id2': 'OMEGA_REF', 'var2': 'omegaRef_0_value'},
                    {'id1': loadID + '_UFLS', 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaP'},
                    {'id1': loadID + '_UFLS', 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaQ'}
                ]
                for connect_attrib in connect_attribs:
                    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)
                
                # UFLS parameters
                ufls_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : loadID + '_UFLS'})
                par_attribs = [
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep1', 'value':'0.1'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep2', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep3', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep4', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep5', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep6', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep7', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep8', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep9', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_UFLSStep10', 'value':'0.05'},
                    {'type':'DOUBLE', 'name':'ufls_omega1Pu', 'value': '{}'.format(0.99 + random.gauss(0, 0.0005))}, # TODO: put realistic values
                    {'type':'DOUBLE', 'name':'ufls_omega2Pu', 'value': '{}'.format(0.988 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega3Pu', 'value': '{}'.format(0.986 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega4Pu', 'value': '{}'.format(0.984 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega5Pu', 'value': '{}'.format(0.982 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega6Pu', 'value': '{}'.format(0.98 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega7Pu', 'value': '{}'.format(0.978 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega8Pu', 'value': '{}'.format(0.976 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega9Pu', 'value': '{}'.format(0.974 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_omega10Pu', 'value': '{}'.format(0.972 + random.gauss(0, 0.0005))},
                    {'type':'DOUBLE', 'name':'ufls_tLagAction', 'value':'0.1'}
                ]
                for par_attrib in par_attribs:
                    etree.SubElement(ufls_par_set, etree.QName(namespace, 'par'), par_attrib) """

        # Line/distance protection
        dyd_root.append(etree.Comment('Line protection'))
        bus2lines = get_buses_to_lines(n)
        line_limits = [600, 1000, 500, 500, 500, 500, 600, 500, 1200, 900, 900, 480, 900, 900,  # From IEEE PES Power Grid Library - Optimal Power Flow - v21.07
                    900, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 600, 900, 600,
                    600, 600, 600, 600, 600, 600]

        line_number = 0
        for lineID in lines.index:
            line_limit = line_limits[line_number]
            line_number += 1
            for side in [1,2]:
                opposite_side = 3-side # 2 if side == 1, 1 if side == 2
                id = lineID + '_side{}'.format(side)
                if not args.special:
                    dist_attrib = {'id': id + '_Distance', 'lib': 'DistanceProtectionLineSide{}Disarm'.format(side), 'parFile': name + '.par',
                    'parId': id + '_Distance'}
                else:
                    dist_attrib = {'id': id + '_Distance', 'lib': 'DistanceProtectionLineSide{}Special'.format(side), 'parFile': name + '.par',
                    'parId': id + '_Distance'}
                etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), dist_attrib)

                connect_attribs = [
                    {'id1': id + '_Distance', 'var1': 'distance_UMonitoredPu', 'id2': 'NETWORK', 'var2': lineID + '_U{}_value'.format(side)},
                    {'id1': id + '_Distance', 'var1': 'distance_PMonitoredPu', 'id2': 'NETWORK', 'var2': lineID + '_P{}_value'.format(side)},
                    {'id1': id + '_Distance', 'var1': 'distance_QMonitoredPu', 'id2': 'NETWORK', 'var2': lineID + '_Q{}_value'.format(side)},
                    {'id1': id + '_Distance', 'var1': 'distance_lineState', 'id2': 'NETWORK', 'var2': lineID + '_state'}
                ]
                for connect_attrib in connect_attribs:
                    etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

                # Parameters
                dist_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : id + '_Distance'})

                voltage_level = lines.at[lineID, 'voltage_level1_id']
                Ub = float(n.get_voltage_levels().at[voltage_level, 'nominal_v']) * 1000
                Sb = 100e6  # 100MW is the default base in Dynawo
                Zb = Ub**2/Sb

                X = lines.at[lineID, 'x'] / Zb
                R = lines.at[lineID, 'r'] / Zb

                X1 = 0.8 * X
                R1 = X1

                adj_lines = get_adjacent_lines(bus2lines, n, lineID, opposite_side)  # Only the adjacent lines that can be "seen" from a forward looking distance relay
                adj_lines_X = [lines.at[adj_line, 'x'] for adj_line in adj_lines]
                if not adj_lines_X: # is empty
                    adj_lines_X = [0]
                max_adj_X = max(adj_lines_X) / Zb
                min_adj_X = min(adj_lines_X) / Zb

                X2 = max(0.9*(X + 0.85*min_adj_X), 1.15*X)
                R2 = X2

                X3 = (X + 1.15 * max_adj_X)
                R3 = X3

                X4 = X3 * 1.2  # X4 is only used to signal when apparent impedance is close to entering zone 3, not used for actual tripping
                R4 = R3 * 1.2

                if adj_lines_X == [0]: # No adjacent lines
                    X3 = 99  # Zone 2 and zone 3 would be identical -> remove zone 3
                    R3 = 99
                    X4 = 99
                    R4 = 99
                
                Ib = Sb / (3**0.5 * Ub)
                Imax = line_limit / Ib
                reach = 0.85 / (1.5 * Imax)

                measurement_max_error = 0.1  # +/- 10%
                if randomise:
                    rand_measurement_ratio = 1 + random.uniform(-measurement_max_error, measurement_max_error)
                    rand_CB = random.uniform(-CB_max_error, CB_max_error)
                else:
                    rand_measurement_ratio = 1
                    rand_CB = 0

                if not args.special:
                    par_attribs = [
                        {'type':'DOUBLE', 'name':'distance_T1', 'value':'999999'},
                        # {'type':'DOUBLE', 'name':'distance_T1', 'value':'{}'.format(0.02 + CB_time + rand_CB)},
                        {'type':'DOUBLE', 'name':'distance_T2', 'value':'{}'.format(0.3 + CB_time + rand_CB)},
                        {'type':'DOUBLE', 'name':'distance_T3', 'value':'{}'.format(0.6 + CB_time + rand_CB)},
                        {'type':'DOUBLE', 'name':'distance_T4', 'value':'999999'},
                        {'type':'DOUBLE', 'name':'distance_R1Pu', 'value': '{}'.format(R1 * rand_measurement_ratio)},
                        {'type':'DOUBLE', 'name':'distance_X1Pu', 'value': '{}'.format(X1 * rand_measurement_ratio)},
                        {'type':'DOUBLE', 'name':'distance_R2Pu', 'value': '{}'.format(R2 * rand_measurement_ratio)},
                        {'type':'DOUBLE', 'name':'distance_X2Pu', 'value': '{}'.format(X2 * rand_measurement_ratio)},
                        {'type':'DOUBLE', 'name':'distance_R3Pu', 'value': '{}'.format(R3 * rand_measurement_ratio)},
                        {'type':'DOUBLE', 'name':'distance_X3Pu', 'value': '{}'.format(X3 * rand_measurement_ratio)},
                        {'type':'DOUBLE', 'name':'distance_R4Pu', 'value': '{}'.format(R4 * rand_measurement_ratio)},
                        {'type':'DOUBLE', 'name':'distance_X4Pu', 'value': '{}'.format(X4 * rand_measurement_ratio)},
                        # {'type':'DOUBLE', 'name':'distance_BlinderAnglePu', 'value': '{}'.format(30 * (pi/180))},  # Load blinder taken as 1.5 times the nominal current at 0.85pu voltage with power factor of 30 degrees following NERC recommandations
                        # {'type':'DOUBLE', 'name':'distance_BlinderReachPu', 'value': '{}'.format(reach)},
                        {'type':'DOUBLE', 'name':'distance_CircuitBreakerTime', 'value': '{}'.format(CB_time + rand_CB)},
                    ]
                else:
                    par_attribs = [
                        {'type':'DOUBLE', 'name':'distance_T1', 'value':'{}'.format(0.3 + CB_time - CB_max_error)},
                        {'type':'DOUBLE', 'name':'distance_T2', 'value':'{}'.format(0.3 + CB_time + CB_max_error)},
                        {'type':'DOUBLE', 'name':'distance_T3', 'value':'{}'.format(0.6 + CB_time + CB_max_error)},
                        {'type':'DOUBLE', 'name':'distance_T4', 'value':'{}'.format(0.6 + CB_time - CB_max_error)},
                        {'type':'DOUBLE', 'name':'distance_R1Pu', 'value': '{}'.format(R2 * (1 + measurement_max_error))},
                        {'type':'DOUBLE', 'name':'distance_X1Pu', 'value': '{}'.format(X2 * (1 + measurement_max_error))},
                        {'type':'DOUBLE', 'name':'distance_R2Pu', 'value': '{}'.format(R2 * (1 - measurement_max_error))},
                        {'type':'DOUBLE', 'name':'distance_X2Pu', 'value': '{}'.format(X2 * (1 - measurement_max_error))},
                        {'type':'DOUBLE', 'name':'distance_R3Pu', 'value': '{}'.format(R3 * (1 - measurement_max_error))},
                        {'type':'DOUBLE', 'name':'distance_X3Pu', 'value': '{}'.format(X3 * (1 - measurement_max_error))},
                        {'type':'DOUBLE', 'name':'distance_R4Pu', 'value': '{}'.format(R3 * (1 + measurement_max_error))},
                        {'type':'DOUBLE', 'name':'distance_X4Pu', 'value': '{}'.format(X3 * (1 + measurement_max_error))},
                        # {'type':'DOUBLE', 'name':'distance_BlinderAnglePu', 'value': '{}'.format(30 * (pi/180))},  # Load blinder taken as 1.5 times the nominal current at 0.85pu voltage with power factor of 30 degrees following NERC recommandations
                        # {'type':'DOUBLE', 'name':'distance_BlinderReachPu', 'value': '{}'.format(reach)},
                        {'type':'DOUBLE', 'name':'distance_SlowCircuitBreakerTime', 'value': '{}'.format(CB_time + CB_max_error)},
                        {'type':'DOUBLE', 'name':'distance_FastCircuitBreakerTime', 'value': '{}'.format(CB_time - CB_max_error)},
                    ]
                for par_attrib in par_attribs:
                    etree.SubElement(dist_par_set, etree.QName(namespace, 'par'), par_attrib)

        # Write the modified dyd and par files
        with open(os.path.join(output_dir, name + '.dyd'), 'wb') as doc:
            doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
        with open(os.path.join(output_dir, name + '.par'), 'wb') as doc:
            doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
        
        # Copy the non-modified files
        shutil.copy(full_name + '.iidm', output_dir)
        shutil.copy(full_name + '.jobs', output_dir)
        if os.path.isfile(full_name + '.crv'):
            shutil.copy(full_name + '.crv', output_dir)
            if randomise:
                print('Warning: Generating curves in batch simulations can take a lot of space')
        if os.path.isfile(full_name + '.crt'):
            shutil.copy(full_name + '.crt', output_dir)
