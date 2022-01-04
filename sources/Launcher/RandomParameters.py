from lxml import etree
import csv
import random
import math
import argparse


"""
Randomise an .iidm file (containing the static parameters and topology of a grid) by sampling the parameters listed in a .csv according to the probability
density functions given in the same .csv, then performs a load flow and outputs the resulting .iidm file

Does the same to a .par file (containing the dynamic parameters of a grid) (except for the load flow)


Supported distributions are:

gaussian(sigma)
gaussian_percent(percent_sigma)
uniform_minmax(min, max)
uniform_delta(delta)

The average of the distribution (except for uniform_minmax) is taken as the original value of the parameter

"""


# Instantiate the parser
parser = argparse.ArgumentParser(description='Optional app description')

# Optional argument
parser.add_argument('--input_par', type=str,
                    help='Input file containing the dynamic parameters')
parser.add_argument('--output_par', type=str,
                    help='Output file containing the randomised dynamic parameters')
parser.add_argument('--csv_par', type=str,
                    help='Csv file containing the list of dynamic parameters to be randomised and associated distributions')
parser.add_argument('--seed_par', type=int, default=1,
                    help='Seed used to ramdomise the dynamic parameters')
parser.add_argument('--input_iidm', type=str,
                    help='Intput file containing the static parameters of the grid in XIIDM format')
parser.add_argument('--output_iidm', type=str,
                    help='Output file containing the randomised static parameters (in XIIDM format)')
parser.add_argument('--csv_iidm', type=str,
                    help='Csv file containing the list of static parameters to be randomised and associated distributions')
parser.add_argument('--seed_iidm', type=int, default=2,
                    help='Seed used to ramdomise the static parameters')
args = parser.parse_args()


randomiseStaticParams = False
randomiseDynamicParams = False

if args.input_par != None:
    randomiseDynamicParams = True
    if args.output_par == None:
        parser.error('Missing output_par')
    if args.csv_par == None:
        parser.error('Missing csv_arg')
if args.input_iidm != None:
    randomiseStaticParams = True
    if args.output_iidm == None:
        parser.error('Missing output_iidm')
    if args.csv_iidm == None:
        parser.error('Missing csv_iidm')

if randomiseStaticParams:
    import pypowsybl as pp
if randomiseDynamicParams:
    import pandas as pd

def getRandom(value, type_, distribution, params):
    value = float(value)
    output = 0
    if type_ == "BOOL":
        raise Exception('Cannot randomise boolean variables')
    else:
        if distribution == 'gaussian':
            params = [float(i) for i in params[0:1]] # Only cast the necessary parameters (so that unused parameters can be left blank)
            output = random.gauss(value, params[0])
        elif distribution == 'gaussian_percent':
            params = [float(i) for i in params[0:1]]
            output = value * random.gauss(1, params[0]/100)
        elif distribution == 'uniform_minmax':
            params = [float(i) for i in params[0:2]]
            output = random.uniform(params[0], params[1])
        elif distribution == 'uniform_delta':
            params = [float(i) for i in params[0:1]]
            output = random.uniform(value - params[0], value + params[0])
        else:
            raise Exception('Distribution type: "' + distribution + '" is not supported')
    if type_ == "DOUBLE":
        return float(output)
    elif type_ == "INT":
        return int(output)
    else:
        raise Exception("Unknown type: %s" % type_)

def findParameterSet(root, parameterSetId):
    for parameterSet in list(root):
        if parameterSet.get('id') == parameterSetId:
            return parameterSet
    raise Exception("'%s' has no parameterSet with id = '%s'" % (root.tag, parameterSetId))

def findParameter(parameterSet, parameterId):
    for parameter in list(parameterSet):
        if parameter.get('name') == parameterId:
            return parameter
    raise Exception("ParameterSet '%s' has no parameter with id = '%s'" % (parameterSet.get('id'), parameterId))


if randomiseDynamicParams:
    random.seed(args.seed_par)

    parser = etree.XMLParser(remove_blank_text=True) # Necessary to get the pretty print working after adding new elements
    par = etree.parse(args.input_par, parser)
    par_root = par.getroot()
    par_namespace = par_root.nsmap
    par_prefix_root = par_root.prefix
    par_namespace_uri = par_namespace[par_prefix_root]
    if par_prefix_root is None:
        par_prefix_root_string = ''
    else:
        par_prefix_root_string = par_prefix_root + ':'


    with open(args.csv_par) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        row = spamreader.__next__()
        if row != ['ParamSet_id', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
            raise Exception("Incorrect format of %s" % args.csv_par)
        
        modifiedParameters = []
        for row in spamreader:
            # print(', '.join(row))
            paramSetId = row[0]
            paramId = row[1]
            distribution = row[2]
            params = row[3:7]
            
            # Check for duplicate parameters in the params.csv
            for [modified_paramSetId, modified_paramId] in modifiedParameters:
                if modified_paramSetId == paramSetId or modified_paramSetId == '*' or paramSetId == '*':
                    if modified_paramId == paramId:
                        raise Exception('Duplicate parameters (would be randomised twice in the current implementation. Can be modified if needed')
            modifiedParameters.append([paramSetId, paramId])
            
            # Randomise the parameters
            if paramSetId == "*": # Wildcard => look in all paramSet for param
                # './/' means find search in the whole tree (instead of only in one level)
                # 'par[@name] means searching for an element par with attribute name
                for parameter in par_root.findall('.//' + par_prefix_root_string + 'par[@name="' + paramId + '"]', par_namespace):
                    parameter.set('value', str(getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, params)))
            else:
                parameterSet = findParameterSet(par_root, paramSetId)
                parameter = findParameter(parameterSet, paramId)
                parameter.set('value', str(getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, params)))

    with open('Test_new_value' + '.par', 'wb') as doc:
        doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))


# Randomise static parameters (then run a load flow using powsybl and write the results)
if randomiseStaticParams:
    random.seed(args.seed_iidm)

    n = pp.network.load(args.input_iidm)
    P_init = sum(n.get_loads().p0)
    Q_init = sum(n.get_loads().q0)
    with open(args.csv_iidm) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        row = spamreader.__next__()
        if row != ['Component_type', 'Component_name', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
            print(row)
            raise Exception("Incorrect format of %s" % args.input_iidm)
        
        modifiedParameters = []
        for row in spamreader:
            # print(', '.join(row))
            componentType = row[0]
            componentName = row[1]
            paramId = row[2]
            distribution = row[3]
            params = row[4:8]
            
            if componentType == '*':
                raise Exception('Wildcard not implemented for Component_type (and is not really useful)')

            # Check for duplicate parameters in the params.csv
            for [modified_componentType, modified_componentName, modified_paramId] in modifiedParameters:
                if modified_componentType == componentType:
                    if modified_componentName == componentName or modified_componentName == '*' or componentName == '*':
                        if modified_paramId == paramId:
                            raise Exception('Duplicate parameters (would be randomised twice in the current implementation. Can be modified if needed')
            modifiedParameters.append([componentType, componentName, paramId])


            if componentType == 'Load':
                components = n.get_loads()
            elif componentType == 'Line':
                components = n.get_lines()
            elif componentType == 'Bus':
                components = n.get_buses()
            else:
                raise Exception("Component type '%s' not considered" % componentType)
            
            random_params = []
            if componentName == '*':
                for index in components.index:
                    init_value = components.at[index, paramId]
                    if init_value == None:
                        raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (componentName, componentType, paramId))
                    random_param = getRandom(init_value, "DOUBLE", distribution, params) # All parameters in .iidm are of type double
                    random_params.append({paramId : random_param})
                random_params_df = pd.DataFrame(random_params, components.index)

            else:
                init_value = components.get(paramId).get(componentName)
                if init_value == None:
                    raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (componentName, componentType, paramId))
                random_param = getRandom(init_value, "DOUBLE", distribution, params) # All parameters in .iidm are of type double
                random_params.append({paramId : random_param})
                random_params_df = pd.DataFrame(random_params, [componentName])

            if componentType == 'Load':
                n.update_loads(random_params_df)
            elif componentType == 'Line':
                n.update_lines(random_params_df)
            elif componentType == 'Bus':
                n.update_buses(random_params_df)
            else:
                raise Exception("Component type '%s' not considered" % componentType)

    # Modify the first load to keep the balance.
    delta_P = P_init - sum(n.get_loads().p0)
    delta_Q = Q_init - sum(n.get_loads().q0)
    
    slack_load_index = n.get_loads().index[0] #TODO: specify which load is used as the slack
    slack = {'p0' : delta_P + n.get_loads().get('p0').get(slack_load_index), 'q0' : delta_Q + n.get_loads().get('q0').get(slack_load_index)}
    n.update_loads(pd.DataFrame(slack, index = [slack_load_index]))

    pp.loadflow.run_ac(n)
    n.dump('IEEE14' + '.xiidm', 'XIIDM', {'iidm.export.xml.version' : '1.4'}) # Latest version supported by dynawo

    # Trying to modify the static parameters into the .par -> Global initialisation of dynawo suceeds even for large modifications, but lead to steady-state
    # discrepencies compared to modifying the .iidm and running a load flow to compute the initial state
        # for loadIndex in loads.index[:-1]: # Last load is not randomised but used to keep the balance
        #     newParamSet = etree.SubElement(par_root, 'set')
        #     newParamSet.set('id', loadIndex)
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_alpha')
        #     newParam.set('value', '1.5')
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_beta')
        #     newParam.set('value', '2.5')
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_P0Pu')
        #     P_init += loads.p0[loadIndex] / 100
        #     P_load = loads.p0[loadIndex] / 100 * random.uniform(0, 2)
        #     P += P_load
        #     newParam.set('value', str(P_load))
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_Q0Pu')
        #     Q_init = loads.q0[loadIndex] / 100
        #     Q_load = loads.q0[loadIndex] / 100 * random.uniform(0, 2)
        #     Q += Q_load
        #     newParam.set('value', str(Q_load))
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_U0Pu')
        #     u0 = n.get_buses().v_mag[loads.bus_id[loadIndex]]
        #     uNom = n.get_voltage_levels().nominal_v[loads.voltage_level_id[loadIndex]]
        #     newParam.set('value', str(u0/uNom))
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_UPhase0')
        #     newParam.set('value', str(n.get_buses().v_angle[loads.bus_id[loadIndex]] * math.pi/180))

        # for loadIndex in loads.index[-1:]: # Only last load
        #     newParamSet = etree.SubElement(par_root, 'set')
        #     newParamSet.set('id', loadIndex)
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_alpha')
        #     newParam.set('value', '1.5')
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_beta')
        #     newParam.set('value', '2.5')
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_P0Pu')
        #     P_init += loads.p0[loadIndex] / 100
        #     newParam.set('value', str(P_init-P))
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_Q0Pu')
        #     Q_init += loads.q0[loadIndex] / 100
        #     newParam.set('value', str(Q_init-Q))
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_U0Pu')
        #     u0 = n.get_buses().v_mag[loads.bus_id[loadIndex]]
        #     uNom = n.get_voltage_levels().nominal_v[loads.voltage_level_id[loadIndex]]
        #     newParam.set('value', str(u0/uNom))
            
        #     newParam = etree.SubElement(newParamSet, 'par')
        #     newParam.set('type', 'DOUBLE')
        #     newParam.set('name', 'load_UPhase0')
        #     newParam.set('value', str(n.get_buses().v_angle[loads.bus_id[loadIndex]] * math.pi/180))
