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


def add_gen_speed_protection_and_params(dyd_root, par_root, namespace, network_name, genID, CB_time, randomise, CB_max_error, omega_max_error=0.01):
    protectionID = genID + '_Speed'
    speed_attrib = {'id': protectionID, 'lib': 'SpeedProtection', 'parFile': network_name + '.par', 'parId': protectionID}
    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), speed_attrib)

    connect_attribs = [
        {'id1': protectionID, 'var1': 'speedProtection_omegaMonitoredPu', 'id2': genID, 'var2': 'generator_omegaPu_value'},
        {'id1': protectionID, 'var1': 'speedProtection_switchOffSignal', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
    ]
    for connect_attrib in connect_attribs:
        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

    if randomise:
        rand_omegaMax = random.uniform(-omega_max_error, omega_max_error)
        rand_omegaMin = random.uniform(-omega_max_error, omega_max_error)
        rand_CB = random.uniform(-CB_max_error, CB_max_error)
    else:
        rand_omegaMax = 0
        rand_omegaMin = 0
        rand_CB = 0

    speed_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : protectionID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'speedProtection_OmegaMaxPu', 'value':str(1.05 + rand_omegaMax)},
        {'type':'DOUBLE', 'name':'speedProtection_OmegaMinPu', 'value':str(0.95 + rand_omegaMin)},
        {'type':'DOUBLE', 'name':'speedProtection_tLagAction', 'value':str(0.02 + CB_time + rand_CB)}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(speed_par_set, etree.QName(namespace, 'par'), par_attrib)


def add_gen_UVA_protection_and_params(dyd_root, par_root, namespace, network_name, genID, CB_time, randomise, CB_max_error):
    protectionID = genID + '_UVA'
    uva_attrib = {'id': protectionID, 'lib': 'UnderVoltageAutomaton', 'parFile': network_name + '.par', 'parId': protectionID}
    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), uva_attrib)

    connect_attribs = [
        {'id1': protectionID, 'var1': 'underVoltageAutomaton_UMonitoredPu', 'id2': 'NETWORK', 'var2': '@' + genID + '@@NODE@_Upu_value'},
        {'id1': protectionID, 'var1': 'underVoltageAutomaton_switchOffSignal', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
    ]
    for connect_attrib in connect_attribs:
        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

    if randomise:
        rand_UMin = random.uniform(-0.05, 0)
        rand_CB = random.uniform(-CB_max_error, CB_max_error)
    else:
        rand_UMin = 0
        rand_CB = 0

    uva_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : protectionID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'underVoltageAutomaton_UMinPu', 'value':str(0.85 + rand_UMin)},
        {'type':'DOUBLE', 'name':'underVoltageAutomaton_tLagAction', 'value':str(1.5 + rand_CB)}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(uva_par_set, etree.QName(namespace, 'par'), par_attrib)


def add_gen_OOS_protection_and_params(dyd_root, par_root, namespace, network_name, genID, CB_time, randomise, CB_max_error, angle_max_error=10):
    protectionID = genID + '_InternalAngle'
    oos_attrib = {'id': protectionID, 'lib': 'LossOfSynchronismProtection', 'parFile': network_name + '.par', 'parId': protectionID}
    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), oos_attrib)

    connect_attribs = [
        {'id1': protectionID, 'var1': 'lossOfSynchronismProtection_thetaMonitored',  'id2': genID, 'var2': 'generator_theta'},
        {'id1': protectionID, 'var1': 'lossOfSynchronismProtection_switchOffSignal', 'id2': genID, 'var2': 'generator_switchOffSignal2'}
    ]
    for connect_attrib in connect_attribs:
        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

    if randomise:
        rand_thetaMin = random.uniform(-angle_max_error, angle_max_error) * pi/180
        rand_CB = random.uniform(-CB_max_error, CB_max_error)
    else:
        rand_thetaMin = 0
        rand_CB = 0

    oos_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : protectionID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'lossOfSynchronismProtection_ThetaMax', 'value':str(2 * pi + rand_thetaMin)},
        {'type':'DOUBLE', 'name':'lossOfSynchronismProtection_tLagAction', 'value':str(0.02 + CB_time + rand_CB)}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(oos_par_set, etree.QName(namespace, 'par'), par_attrib)


def add_centralised_UFLS_and_params(dyd_root, par_root, namespace, network_name, network):
    protectionID = 'UFLS'
    ufls_attrib = {'id': protectionID, 'lib': 'UFLS10Steps', 'parFile': network_name + '.par', 'parId': protectionID}
    etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), ufls_attrib)

    connect_attribs = [
        {'id1': protectionID, 'var1': 'ufls_omegaMonitoredPu',  'id2': 'OMEGA_REF', 'var2': 'omegaRef_0_value'}
    ]
    for loadID in network.get_loads().index:
        if 'Dummy' not in loadID: # Skip dummy loads
            connect_attribs += [
                {'id1': protectionID, 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaP'},
                {'id1': protectionID, 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaQ'}
            ]
    for connect_attrib in connect_attribs:
        etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

    # UFLS parameters
    ufls_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : protectionID})
    par_attribs = [
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_0_', 'value':'0.1'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_1_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_2_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_3_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_4_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_5_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_6_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_7_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_8_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_UFLSStep_9_', 'value':'0.05'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_0_', 'value':'0.98'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_1_', 'value':'0.978'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_2_', 'value':'0.976'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_3_', 'value':'0.974'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_4_', 'value':'0.972'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_5_', 'value':'0.97'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_6_', 'value':'0.968'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_7_', 'value':'0.966'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_8_', 'value':'0.964'},
        {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_9_', 'value':'0.962'},
        {'type':'DOUBLE', 'name':'ufls_tLagAction', 'value':'0.1'}
    ]
    for par_attrib in par_attribs:
        etree.SubElement(ufls_par_set, etree.QName(namespace, 'par'), par_attrib)


def add_decentralised_UFLS_and_params(dyd_root, par_root, namespace, network_name, network):
    raise NotImplementedError('Not yet tested')

    for loadID in network.get_loads().index:
        if 'Dummy' in loadID: # Skip dummy loads
            continue
        protectionID = loadID + '_UFLS'
        ufls_attrib = {'id': protectionID, 'lib': 'UFLS10Steps', 'parFile': network_name + '.par', 'parId': protectionID}
        etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), ufls_attrib)

        connect_attribs = [
            {'id1': protectionID, 'var1': 'ufls_omegaMonitored', 'id2': 'OMEGA_REF', 'var2': 'omegaRef_0_value'},
            {'id1': protectionID, 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaP'},
            {'id1': protectionID, 'var1': 'ufls_deltaPQfiltered', 'id2': loadID, 'var2': 'load_deltaQ'}
        ]
        for connect_attrib in connect_attribs:
            etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

        # UFLS parameters
        ufls_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : protectionID})
        par_attribs = [
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_0_', 'value':'0.1'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_1_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_2_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_3_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_4_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_5_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_6_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_7_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_8_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_UFLSStep_9_', 'value':'0.05'},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_0_', 'value': str(0.99 + random.gauss(0, 0.0005))}, # TODO: put realistic values
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_1_', 'value': str(0.988 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_2_', 'value': str(0.986 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_3_', 'value': str(0.984 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_4_', 'value': str(0.982 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_5_', 'value': str(0.98 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_6_', 'value': str(0.978 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_7_', 'value': str(0.976 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_8_', 'value': str(0.974 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_omegaThresholdPu_9_', 'value': str(0.972 + random.gauss(0, 0.0005))},
            {'type':'DOUBLE', 'name':'ufls_tLagAction', 'value':'0.1'}
        ]
        for par_attrib in par_attribs:
            etree.SubElement(ufls_par_set, etree.QName(namespace, 'par'), par_attrib)


def add_line_dist_protection_and_params(dyd_root, par_root, namespace, network_name, network, bus2lines, lineID, CB_time, randomise, special,
    CB_max_error, measurement_max_error = 0.1):
    for side in [1,2]:
        opposite_side = 3-side  # 2 if side == 1, 1 if side == 2
        protectionID = lineID + '_side{}'.format(side) + '_Distance'
        if special:
            protectionIDs = [protectionID + 'Slow', protectionID + 'Fast']
        else:
            protectionIDs = [protectionID]
        lib = 'DistanceProtectionLineFourZones'
        for protectionID in protectionIDs:
            dist_attrib = {'id': protectionID, 'lib': lib, 'parFile': network_name + '.par', 'parId': protectionID}
            etree.SubElement(dyd_root, etree.QName(namespace, 'blackBoxModel'), dist_attrib)

            connect_attribs = [
                {'id1': protectionID, 'var1': 'distance_UMonitoredPu', 'id2': 'NETWORK', 'var2': lineID + '_U{}_value'.format(side)},
                {'id1': protectionID, 'var1': 'distance_PMonitoredPu', 'id2': 'NETWORK', 'var2': lineID + '_P{}_value'.format(side)},
                {'id1': protectionID, 'var1': 'distance_QMonitoredPu', 'id2': 'NETWORK', 'var2': lineID + '_Q{}_value'.format(side)},
                {'id1': protectionID, 'var1': 'distance_lineState', 'id2': 'NETWORK', 'var2': lineID + '_state'}
            ]
            for connect_attrib in connect_attribs:
                etree.SubElement(dyd_root, etree.QName(namespace, 'connect'), connect_attrib)

            # Parameters
            dist_par_set = etree.SubElement(par_root, etree.QName(namespace, 'set'), {'id' : protectionID})

            voltage_level = lines.at[lineID, 'voltage_level1_id']
            Ub = float(network.get_voltage_levels().at[voltage_level, 'nominal_v']) * 1000
            Sb = 100e6  # 100MW is the default base in Dynawo
            Zb = Ub**2/Sb

            X = lines.at[lineID, 'x'] / Zb
            # R = lines.at[lineID, 'r'] / Zb

            X1 = 0.8 * X
            R1 = X1

            adj_lines = get_adjacent_lines(bus2lines, network, lineID, opposite_side)  # Only the adjacent lines that can be "seen" from a forward looking distance relay
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

            if randomise:
                rand_measurement_ratio = 1 + random.uniform(-measurement_max_error, measurement_max_error)
                rand_CB = random.uniform(-CB_max_error, CB_max_error)
            else:
                rand_measurement_ratio = 1
                rand_CB = 0

            if 'Fast' in protectionID:
                par_attribs = [
                    {'type':'INT', 'name':'distance_LineSide', 'value':str(side)},
                    {'type':'DOUBLE', 'name':'distance_tZone_0_', 'value':'999999'},
                    {'type':'DOUBLE', 'name':'distance_tZone_1_', 'value':str(0.3)},
                    {'type':'DOUBLE', 'name':'distance_tZone_2_', 'value':str(0.6)},
                    {'type':'DOUBLE', 'name':'distance_tZone_3_', 'value':'999999'},
                    {'type':'DOUBLE', 'name':'distance_RPu_0_', 'value': str(R2 * (1 + measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_0_', 'value': str(X2 * (1 + measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_RPu_1_', 'value': str(R2 * (1 + measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_1_', 'value': str(X2 * (1 + measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_RPu_2_', 'value': str(R3 * (1 + measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_2_', 'value': str(X3 * (1 + measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_RPu_3_', 'value': str(R3 * (1 + measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_3_', 'value': str(X3 * (1 + measurement_max_error))},
                    # {'type':'DOUBLE', 'name':'distance_BlinderAnglePu', 'value': str(30 * (pi/180))},  # Load blinder taken as 1.5 times the nominal current at 0.85pu voltage with power factor of 30 degrees following NERC recommandations
                    # {'type':'DOUBLE', 'name':'distance_BlinderReachPu', 'value': str(reach)},
                    {'type':'DOUBLE', 'name':'distance_CircuitBreakerTime', 'value': str(CB_time - CB_max_error)},
                    {'type':'BOOL', 'name':'distance_TrippingZone_0_', 'value': 'false'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_1_', 'value': 'false'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_2_', 'value': 'false'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_3_', 'value': 'false'},
                ]
            elif 'Slow' in protectionID:
                par_attribs = [
                    {'type':'INT', 'name':'distance_LineSide', 'value':str(side)},
                    {'type':'DOUBLE', 'name':'distance_tZone_0_', 'value':'999999'},
                    {'type':'DOUBLE', 'name':'distance_tZone_1_', 'value':str(0.3)},
                    {'type':'DOUBLE', 'name':'distance_tZone_2_', 'value':str(0.6)},
                    {'type':'DOUBLE', 'name':'distance_tZone_3_', 'value':'999999'},
                    {'type':'DOUBLE', 'name':'distance_RPu_0_', 'value': str(R2 * (1 - measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_0_', 'value': str(X2 * (1 - measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_RPu_1_', 'value': str(R2 * (1 - measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_1_', 'value': str(X2 * (1 - measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_RPu_2_', 'value': str(R3 * (1 - measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_2_', 'value': str(X3 * (1 - measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_RPu_3_', 'value': str(R3 * (1 - measurement_max_error))},
                    {'type':'DOUBLE', 'name':'distance_XPu_3_', 'value': str(X3 * (1 - measurement_max_error))},
                    # {'type':'DOUBLE', 'name':'distance_BlinderAnglePu', 'value': str(30 * (pi/180))},  # Load blinder taken as 1.5 times the nominal current at 0.85pu voltage with power factor of 30 degrees following NERC recommandations
                    # {'type':'DOUBLE', 'name':'distance_BlinderReachPu', 'value': str(reach)},
                    {'type':'DOUBLE', 'name':'distance_CircuitBreakerTime', 'value': str(CB_time + CB_max_error)},
                    {'type':'BOOL', 'name':'distance_TrippingZone_0_', 'value': 'true'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_1_', 'value': 'true'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_2_', 'value': 'true'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_3_', 'value': 'true'},
                ]
            else:
                par_attribs = [
                    {'type':'INT', 'name':'distance_LineSide', 'value':str(side)},
                    {'type':'DOUBLE', 'name':'distance_tZone_0_', 'value':'999999'},
                    {'type':'DOUBLE', 'name':'distance_tZone_1_', 'value':str(0.3)},
                    {'type':'DOUBLE', 'name':'distance_tZone_2_', 'value':str(0.6)},
                    {'type':'DOUBLE', 'name':'distance_tZone_3_', 'value':'999999'},
                    {'type':'DOUBLE', 'name':'distance_RPu_0_', 'value': str(R1 * rand_measurement_ratio)},
                    {'type':'DOUBLE', 'name':'distance_XPu_0_', 'value': str(X1 * rand_measurement_ratio)},
                    {'type':'DOUBLE', 'name':'distance_RPu_1_', 'value': str(R2 * rand_measurement_ratio)},
                    {'type':'DOUBLE', 'name':'distance_XPu_1_', 'value': str(X2 * rand_measurement_ratio)},
                    {'type':'DOUBLE', 'name':'distance_RPu_2_', 'value': str(R3 * rand_measurement_ratio)},
                    {'type':'DOUBLE', 'name':'distance_XPu_2_', 'value': str(X3 * rand_measurement_ratio)},
                    {'type':'DOUBLE', 'name':'distance_RPu_3_', 'value': str(R4 * rand_measurement_ratio)},
                    {'type':'DOUBLE', 'name':'distance_XPu_3_', 'value': str(X4 * rand_measurement_ratio)},
                    # {'type':'DOUBLE', 'name':'distance_BlinderAnglePu', 'value': str(30 * (pi/180))},  # Load blinder taken as 1.5 times the nominal current at 0.85pu voltage with power factor of 30 degrees following NERC recommandations
                    # {'type':'DOUBLE', 'name':'distance_BlinderReachPu', 'value': str(reach)},
                    {'type':'DOUBLE', 'name':'distance_CircuitBreakerTime', 'value': str(CB_time + rand_CB)},
                    {'type':'BOOL', 'name':'distance_TrippingZone_0_', 'value': 'true'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_1_', 'value': 'true'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_2_', 'value': 'true'},
                    {'type':'BOOL', 'name':'distance_TrippingZone_3_', 'value': 'true'},
                ]
            for par_attrib in par_attribs:
                etree.SubElement(dist_par_set, etree.QName(namespace, 'par'), par_attrib)


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
    full_network_name = os.path.join(working_dir, args.name)
    network_name = args.name
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

        dyd_root = etree.parse(full_network_name + '.dyd', XMLparser).getroot()
        par_root = etree.parse(full_network_name + '.par', XMLparser).getroot()

        # Read iidm
        network = pp.network.load(full_network_name + ".iidm")
        lines = network.get_lines()
        gens = network.get_generators()

        # Generator protections
        dyd_root.append(etree.Comment('Generator protections'))
        for genID in gens.index:
            # Over-/under-speed protection
            add_gen_speed_protection_and_params(dyd_root, par_root, namespace, network_name, genID, CB_time, randomise, CB_max_error)
            # Under-voltage protection
            add_gen_UVA_protection_and_params(dyd_root, par_root, namespace, network_name, genID, CB_time, randomise, CB_max_error)
            # Out-of-step protection
            add_gen_OOS_protection_and_params(dyd_root, par_root, namespace, network_name, genID, CB_time, randomise, CB_max_error)


        # UFLS
        dyd_root.append(etree.Comment('Under-frequency load shedding'))
        centralised = True
        if centralised: # All UFLSs have the same parameters, and read the same frequency -> only one model to save variables
            add_centralised_UFLS_and_params(dyd_root, par_root, namespace, network_name, network)
        else:
            add_decentralised_UFLS_and_params(dyd_root, par_root, namespace, network_name, network)

        # Line/distance protection
        dyd_root.append(etree.Comment('Line protection'))
        bus2lines = get_buses_to_lines(network)

        for lineID in lines.index:
            add_line_dist_protection_and_params(dyd_root, par_root, namespace, network_name, network, bus2lines, lineID, CB_time, randomise, args.special, CB_max_error)


        # Write the modified dyd and par files
        with open(os.path.join(output_dir, network_name + '.dyd'), 'wb') as doc:
            doc.write(etree.tostring(dyd_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
        with open(os.path.join(output_dir, network_name + '.par'), 'wb') as doc:
            doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

        # Copy the non-modified files
        shutil.copy(full_network_name + '.iidm', output_dir)
        shutil.copy(full_network_name + '.jobs', output_dir)
        if os.path.isfile(full_network_name + '.crv'):
            shutil.copy(full_network_name + '.crv', output_dir)
            if randomise:
                print('Warning: Generating curves in batch simulations can take a lot of space')
        if os.path.isfile(full_network_name + '.crt'):
            shutil.copy(full_network_name + '.crt', output_dir)
        if os.path.isfile(full_network_name + '.fsv'):
            shutil.copy(full_network_name + '.fsv', output_dir)
