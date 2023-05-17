from lxml import etree
import csv
import random
import argparse
import pypowsybl as pp
import pandas as pd
import os
import math
import shutil


def getRandom(value, type_, distribution, distribution_parameters):
    """
    Return a random value

    @param value Average of the distribution (not used for uniform_minmax)
    @param type_ Type of the value to return
    @param distribution Distribution of the random value
    @param distribution_parameters Parameters of the distribution

    Supported distributions are:

    gaussian(sigma)
    gaussian_percent(percent_sigma)
    uniform_minmax(min, max)
    uniform_delta(delta)
    """
    value = float(value)
    output = 0
    if type_ == "BOOL":
        raise Exception('Cannot randomise boolean variables')
    elif distribution == 'gaussian':
        distribution_parameters = [float(i) for i in distribution_parameters[0:1]]  # Only cast the necessary parameters (so that unused parameters can be left blank)
        output = random.gauss(value, distribution_parameters[0])
    elif distribution == 'gaussian_percent':
        distribution_parameters = [float(i) for i in distribution_parameters[0:1]]
        output = value * random.gauss(1, distribution_parameters[0]/100)
    elif distribution == 'uniform_minmax':
        distribution_parameters = [float(i) for i in distribution_parameters[0:2]]
        output = random.uniform(distribution_parameters[0], distribution_parameters[1])
    elif distribution == 'uniform_delta':
        distribution_parameters = [float(i) for i in distribution_parameters[0:1]]
        output = random.uniform(value - distribution_parameters[0], value + distribution_parameters[0])
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

def findParameterSets(root, filter):
    out = []
    for parameterSet in list(root):
        if filter in parameterSet.get('id'):
            out.append(parameterSet)
    if len(out) == 0:
        raise Exception("'%s' has no parameterSet with id containing '%s'" % (root.tag, filter))
    return out

def findParameter(parameterSet, parameterId):
    for parameter in list(parameterSet):
        if parameter.get('name') == parameterId:
            return parameter
    raise Exception("ParameterSet '%s' has no parameter with id = '%s'" % (parameterSet.get('id'), parameterId))


class DynamicParameter:
    def __init__(self, set_id, id, value):
        self.set_id = set_id
        self.id = id
        self.value = value
    
    def __eq__(self, other):
        if self.set_id == other.set_id:
            if self.id == other.id:
                if self.value != other.value:
                    print('Warning: comparison of instances of the same parameter with different values')
                return True
        return False


class DynamicParameterList:
    def __init__(self):
        self.l = []

    def __getitem__(self, item):
        return self.l[item]
    
    def append(self, new_parameter: DynamicParameter):
        """
        Append a static parameter to self, throws in case of duplicate
        """
        for parameter in self.l:
            if parameter == new_parameter:
                raise Exception('Duplicate parameters (would be randomised twice in the current implementation. Can be modified if needed)')
        self.l.append(new_parameter)


def randomiseDynamicParams(input_par, input_csv):
    """
    Generate random dynamic parameters for a parametric systematic analysis

    @param input_par Full path of the par file for which the random dynamic data has to be generated
    @param input_csv Full path of the csv file containing the information on how to randomise (i.e. randomise what and according to which pdf)

    @output dyd_parameters_dic
        @key input_par Full path of the par's file (linked from fic_MULTIPLE) for which random data has been generated
        @value[0] par_set_ids List of the component names of the randomised parameters
        @value[1] param_ids List of the id's of the randomised parameters
        @value[2] param_values List of the randomised parameter values
    """
    random_dynamic_parameters = DynamicParameterList()

    par_root = etree.parse(input_par).getroot()
    par_namespace = par_root.nsmap
    par_prefix = par_root.prefix
    if par_prefix is None:
        par_prefix = ''
    else:
        par_prefix = par_prefix + ':'

    with open(input_csv) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        row = spamreader.__next__()
        if row != ['ParamSet_id', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
            raise Exception("Incorrect format of %s" % input_csv)

        for row in spamreader:
            paramSetId = row[0]
            paramId = row[1]
            distribution = row[2]
            distribution_parameters = row[3:7]

            # Randomise the parameters
            if paramSetId == "*": # Wildcard => look in all paramSet for param
                for parameter in par_root.findall('.//' + par_prefix + 'par[@name="' + paramId + '"]', par_namespace):
                    set_id = parameter.getparent().get('id')
                    value = getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, distribution_parameters)
                    random_dynamic_parameters.append(DynamicParameter(set_id, paramId, value))
            
            elif paramSetId[-1] == "*":
                filter = paramSetId[:-1]
                found_parameter_sets = findParameterSets(par_root, filter)
                for found_parameter_set in found_parameter_sets:
                    found_parameter = findParameter(found_parameter_set, paramId)
                    value = getRandom(found_parameter.attrib['value'], found_parameter.attrib['type'], distribution, distribution_parameters)
                    random_dynamic_parameters.append(DynamicParameter(found_parameter_set.attrib['id'], paramId, value))

            else:
                parameterSet = findParameterSet(par_root, paramSetId)
                parameter = findParameter(parameterSet, paramId)
                value = getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, distribution_parameters)
                random_dynamic_parameters.append(DynamicParameter(paramSetId, paramId, value))

    return random_dynamic_parameters


def writeDynamicParams(dynamic_parameter_list, input_par, output_dir):
    """
    Set new values of parameters in the input par and write the resulting par to output_dir.

    @param dynamic_parameter_list List of dynamic parameters
    @param input_par Full path of the par file to modify
    @param out_dir Output directory for the output par
    """
    par_root = etree.parse(input_par).getroot()

    for parameter in dynamic_parameter_list.l:
        found_parameter_set = findParameterSet(par_root, parameter.set_id)
        found_parameter = findParameter(found_parameter_set, parameter.id)
        found_parameter.set('value', str(parameter.value))

    output_par = os.path.join(output_dir, os.path.basename(input_par))
    with open(output_par, 'xb') as doc:
        doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))


class StaticParameter:
    def __init__(self, component_type, component_name, id, value):
        self.component_type = component_type
        self.component_name = component_name
        self.id = id
        self.value = value
    
    def __eq__(self, other):
        if self.component_type == other.component_type:
            if self.component_name == other.component_name:
                if self.id == other.id:
                    if self.value != other.value:
                        print('Warning: comparison of instances of the same parameter with different values')
                    return True
        return False


class StaticParameterList:
    def __init__(self):
        self.l = []
    
    def __getitem__(self, item):
        return self.l[item]

    def append(self, new_parameter: StaticParameter):
        """
        Append a static parameter to self, throws in case of duplicate
        """
        for parameter in self.l:
            if parameter == new_parameter:
                raise Exception('Duplicate parameters (would be randomised twice in the current implementation. Can be modified if needed)')
        self.l.append(new_parameter)


def randomiseStaticParams(input_iidm, input_csv):
    """
    Randomise static parameters

    @param input_iidm Full path of the iidm file for which the random static data has to be generated
    @param input_csv full path of the csv file containing the information on how to randomise (i.e. randomise what and according to which pdf)

    @output random_static_parameters List of the randomised static parameters
    """
    random_static_parameters = StaticParameterList()

    n = pp.network.load(input_iidm)
    with open(input_csv) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        row = spamreader.__next__()
        if row != ['Component_type', 'Component_name', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
            print(row)
            raise Exception("Incorrect format of %s" % input_csv)

        for row in spamreader:
            componentType = row[0]
            componentName = row[1]
            paramId = row[2]
            distribution = row[3]
            distribution_parameters = row[4:8]
            max_value_ref = row[9]
            
            if componentType == '*':
                raise NotImplementedError('Wildcard not implemented for Component_type (and is not really useful)')

            if componentType == 'Load':
                components = n.get_loads()
            elif componentType == 'Line':
                components = n.get_lines()
            elif componentType == 'Bus':
                components = n.get_buses()
            elif componentType == 'Generator':
                components = n.get_generators()
            else:
                raise Exception("Component type '%s' not considered" % componentType)

            if componentName[-1] == '*':
                for index in components.index:
                    if componentName[:-1] not in index:  # (always false if componentName == '*')
                        continue
                    init_value = components.at[index, paramId]
                    if init_value == None:
                        raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (componentName, componentType, paramId))
                    random_value = getRandom(init_value, "DOUBLE", distribution, distribution_parameters)  # All parameters in .iidm are of type double

                    if max_value_ref != '':
                        max = components.at[index, max_value_ref]
                        nb_it = 0
                        while random_value > max:
                            random_value = getRandom(init_value, "DOUBLE", distribution, distribution_parameters)
                            nb_it += 1
                            if nb_it > 1000:
                                raise Exception()

                    random_static_parameters.append(StaticParameter(componentType, index, paramId, random_value))
            else:
                init_value = components.get(paramId).get(componentName)
                if init_value == None:
                    raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (componentName, componentType, paramId))
                random_value = getRandom(init_value, "DOUBLE", distribution, distribution_parameters)  # All parameters in .iidm are of type double

                if max_value_ref != '':
                    max = components.at[index, max_value_ref]
                    nb_it = 0
                    while random_value > max:
                        random_value = getRandom(init_value, "DOUBLE", distribution, distribution_parameters)
                        nb_it += 1
                        if nb_it > 1000:
                            raise Exception()

                random_static_parameters.append(StaticParameter(componentType, componentName, paramId, random_value))

    return random_static_parameters


def writeStaticParams(static_parameters, input_iidm, output_dir, slack_load_id, target_Q=None, slack_gen_id=None, slack_gen_type=None):
    """
    Set new values of parameters in the input iidm, restore the load/generation balance (and target_Q if given), and write the resulting iidm to output_dir.

    The balance is restored through a slack load

    @param random_static_parameters List of random static parameters
    @param input_iidm Iidm file to modify
    @param output_dir Output directory for the output iidm's
    @param slack_load_id Id of the slack load (used for the active (and reactive if target_Q is specified) balance)
    @param target_Q Amount of reactive power that should be produced at the slack bus
    @param slack_gen_id Id of the generator that emulates an infinite bus, should be connected to the slack bus. Mandatory argument if target_Q is specified.
    """
    n = pp.network.load(input_iidm)
    for parameter in static_parameters.l:
        component_type = parameter.component_type
        component_name = parameter.component_name
        param_id = parameter.id
        param_value = parameter.value

        if component_type == 'Load':
            components = n.get_loads()
        elif component_type == 'Line':
            components = n.get_lines()
        elif component_type == 'Bus':
            components = n.get_buses()
        elif component_type == 'Generator':
            components = n.get_generators()
        else:
            raise Exception("Component type '%s' not considered" % component_type)

        if components.get(param_id).get(component_name) == None:
            raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (component_name, component_type, param_id))
        new_param_df = pd.DataFrame({param_id : param_value}, [component_name])

        if component_type == 'Load':
            n.update_loads(new_param_df)
        elif component_type == 'Line':
            n.update_lines(new_param_df)
        elif component_type == 'Bus':
            n.update_buses(new_param_df)
        elif component_type == 'Generator':
            n.update_generators(new_param_df)
        else:
            raise Exception("Component type '%s' not considered" % component_type)
    
    # Restore the load/generation balance and target_Q
    lf_parameters = pp.loadflow.Parameters(distributed_slack=False)    
    lf_results = pp.loadflow.run_ac(n, lf_parameters)

    if target_Q != None:
        if slack_gen_id == None or slack_gen_type == None:
            raise

    while True:
        delta_P = -lf_results[0].slack_bus_active_power_mismatch
        if math.isnan(delta_P):
            print('Warning: load flow did not converge')
            break
        if target_Q != None:
            if slack_gen_type == 'Line':
                delta_Q = target_Q + n.get_lines().get('q2').get(slack_gen_id)
            elif slack_gen_type == 'Generator':
                delta_Q = target_Q + n.get_generators().get('q').get(slack_gen_id)
            else:
                raise NotImplementedError()
            if (abs(delta_Q) < 1e-6 and abs(delta_P) < 1e-6):
                break
            slack = {'p0' : delta_P + n.get_loads().get('p0').get(slack_load_id), 'q0' : delta_Q + n.get_loads().get('q0').get(slack_load_id)}
        else:
            if (abs(delta_P) < 1e-6):
                break
            slack = {'p0' : delta_P + n.get_loads().get('p0').get(slack_load_id)}

        n.update_loads(pd.DataFrame(slack, index = [slack_load_id]))
        lf_results = pp.loadflow.run_ac(n, lf_parameters)

    output_iidm = os.path.join(output_dir, os.path.basename(input_iidm))
    if os.path.exists(output_iidm):
        raise Exception('')
    else:
        n.dump(output_iidm, 'XIIDM', {'iidm.export.xml.version' : '1.4'})  # Latest version supported by dynawo
        # Replace xiidm extension with the one of output_iidm
        [file, ext] = output_iidm.rsplit('.', 1)
        if ext != 'xiidm':
            os.rename(file + '.xiidm', output_iidm)

 
def addSuffix(s, suffix):
    """
    Add suffix at the end of filename, but before extension
    """
    if '.' in s:
        return (suffix + '.').join(s.rsplit('.', 1))
    else: # File has no extension
        return s + suffix


def fileIsInList(file, lst):
    for f in lst:
        if os.path.samefile(f, file):
            return True
    return False


def writeParametricSAInputs(working_dir, fic_MULTIPLE, network_name, output_dir_name, static_parameters, dyn_data_dic, run_id, target_Q=None, slack_load_id=None, slack_gen_id=None, slack_gen_type=None, disturbance_ids=None):
    """
    Write the inputs files necessary to perform a "parametric" systematic analysis (SA), i.e. a systematic analysis where the parameters are modified

    Hypothesis: 
        - All files needed to perform the SA already exist (altough the values are not randomised yet), are in working_dir,
        and are named network_name + standard extension (the "event" dyd files don't need to follow a specific naming convention)
        - All parameters are in the network_name.par file

    @param working_dir Working directory
    @param fic_MULTIPLE fic_MULTIPLE.xml file that would be used to perform the "classical" SA
    @param output_dir_name Name of the output directory, i.e. output_dir = working_dir + '/' + output_dir_name
    @param static_parameters List of modified static parameters
    @param dyn_data_dic Dictionary containing the modified values of the dynamic parameters, see e.g. randomiseDynamicParams():dyn_parameters_dic
    @param run_id Id of the run
    @param target_Q Amount of reactive power that should be produced at the slack bus (for distribution systems where the slack bus is infinite)
    @param slack_load_id Id of the slack load that should be connected to the slack bus (used to balance the reactive power). Mandatory argument if target_Q is specified
    @param slack_gen_id Id of the generator that emulates an infinite bus, should be connected to the slack bus. Mandatory argument if target_Q is specified.
    """
    output_dir = os.path.join(working_dir, output_dir_name)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)  # Clean the output directory
    os.makedirs(output_dir)
    full_network_name = os.path.join(working_dir, network_name)

    # Simply copy the files that are not modified
    shutil.copy(full_network_name + '.dyd', output_dir)
    shutil.copy(full_network_name + '.jobs', output_dir)

    if os.path.isfile(full_network_name + '.crv'):
        shutil.copy(full_network_name + '.crv', output_dir)
    if os.path.isfile(full_network_name + '.crt'):
        shutil.copy(full_network_name + '.crt', output_dir)
    if os.path.isfile(full_network_name + '.fsv'):
        shutil.copy(full_network_name + '.fsv', output_dir)

    # Write new fic_MULTIPLE.xml
    fic_root = etree.parse(fic_MULTIPLE).getroot()
    scenarios = fic_root[0]
    disturb_id = 0
    for scenario in list(scenarios):
        scenario.set('id', scenario.get('id') + "_%03d" % run_id)  # Adds run_id to scenario id to more easily merge the ouput files later on
        if disturbance_ids is not None:
            if disturb_id not in disturbance_ids:
                scenario.getparent().remove(scenario)
        disturb_id += 1
    with open(os.path.join(output_dir, 'fic_MULTIPLE.xml'), 'xb') as doc:
        doc.write(etree.tostring(fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Copy the "event" dyd files
    dyd_files = []
    for scenario in scenarios:
        dyd = scenario.get('dydFile')
        if dyd != None:
            if dyd not in dyd_files:
                dyd_files.append(dyd)
    for dyd_file in dyd_files:
        shutil.copy(os.path.join(working_dir, dyd_file), output_dir)

    # Write new .iidm's
    writeStaticParams(static_parameters, full_network_name + '.iidm', output_dir, slack_load_id, target_Q, slack_gen_id, slack_gen_type)      

    # Write new .par's
    writeDynamicParams(dyn_data_dic, full_network_name + '.par', output_dir)


if __name__ == "__main__":
    random.seed(1)
    parser = argparse.ArgumentParser('Takes as input the same files as a systematic analysis (SA), and generates a randomised version of those files. '
    'Those can then be used to perform another SA. The ouput files are all written to working_dir/RandomisedInputs/')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--fic_MULTIPLE', type=str, required=True,
                        help='Input file containing the different scenarios to run')
    parser.add_argument('--name', type=str, required=True,
                        help='Name of the network')
    parser.add_argument('--csv_par', type=str, required=True,
                        help='Csv file containing the list of dynamic parameters to be randomised and associated distributions')
    parser.add_argument('--csv_iidm', type=str, required=True,
                        help='Csv file containing the list of static parameters to be randomised and associated distributions')
    parser.add_argument('--nb_runs', type=int, required=True,
                        help='Number of randomised copies of the base scenarios to create')
    args = parser.parse_args()

    working_dir = args.working_dir
    fic_MULTIPLE = os.path.join(working_dir, args.fic_MULTIPLE)
    csv_par = os.path.join(working_dir, args.csv_par)
    csv_iidm = os.path.join(working_dir, args.csv_iidm)
    network_name = args.name
    full_network_name = os.path.join(working_dir, network_name)
    
    for run_id in range(args.nb_runs):
        output_dir_name = "RandomisedInputs_%03d" % run_id
        output_dir = os.path.join(working_dir, output_dir_name)

        random_static_parameters = randomiseStaticParams(full_network_name + '.iidm', args.csv_iidm)
        dyn_data_dic = randomiseDynamicParams(full_network_name + '.par', args.csv_par)

        writeParametricSAInputs(working_dir, fic_MULTIPLE, network_name, output_dir_name, random_static_parameters, dyn_data_dic, run_id)
