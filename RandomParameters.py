from lxml import etree
import csv
import random
import argparse
import pypowsybl as pp
import pandas as pd
import os
import shutil
from copy import deepcopy


def getRandom(value, type_, distribution, params):
    """
    Return a random value

    @param value Average of the distribution (not used for uniform_minmax)
    @param type_ Type of the value to return
    @param distribution Distribution of the random value
    @param params Parameters of the distribution

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


def ficGetPars(fic_MULTIPLE):
    """
    Return the list of the par files that are necessary to perform the systematic analysis described by fic_MULTIPLE
    """
    fic = etree.parse(fic_MULTIPLE)
    fic_root = fic.getroot()
    fic_namespace_uri = fic_root.nsmap[fic_root.prefix]

    if len(fic_root) == 0:
        raise Exception('fic_MULTIPLE file is empty')
    scenarios = fic_root[0]
    if scenarios.tag != '{' + fic_namespace_uri+ '}' + 'scenarios':
        raise Exception('Invalid fic_MULTIPLE file')

    jobsFile = scenarios.get('jobsFile')
    working_dir = os.path.dirname(fic_MULTIPLE)

    jobs = etree.parse(os.path.join(working_dir, jobsFile))
    jobs_root = jobs.getroot()
    jobs_namespace_uri = jobs_root.nsmap[jobs_root.prefix]

    jobs_dyd = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'dynModels[@dydFile]')
    if len(jobs_dyd) != 1:
        raise NotImplementedError("Jobs file should contain exactly one dyd entry (0 would be possible with few changes), %s found" % (len(jobs_dyd)))
    base_dyd = os.path.join(working_dir, jobs_dyd[0].get('dydFile'))

    dyd_files = [base_dyd]
    for scenario in scenarios:
        dyd = scenario.get('dydFile')
        if dyd != None:
            dyd = os.path.join(working_dir, dyd)
            if not fileIsInList(dyd, dyd_files):
                dyd_files.append(dyd)
    
    par_files = []
    for dyd_file in dyd_files:
        dyd_root = etree.parse(dyd_file).getroot()
        dyd_namespace = dyd_root.nsmap
        dyd_prefix_root = dyd_root.prefix
        if dyd_prefix_root is None:
            dyd_prefix_root_string = ''
        else:
            dyd_prefix_root_string = dyd_prefix_root + ':'

        for blackBox in dyd_root.findall('.//' + dyd_prefix_root_string + 'blackBoxModel', dyd_namespace):
            dyd_par = os.path.join(working_dir, blackBox.attrib['parFile'])
            if not fileIsInList(dyd_par, par_files):
                par_files.append(dyd_par)
            blackBox.set('parFile', os.path.basename(dyd_par))

        if len(dyd_root.findall('.//' + dyd_prefix_root_string + 'unitDynamicModel', dyd_namespace)) != 0:
            raise NotImplementedError('Consider using precompiled models for systematic analyses')
    return par_files


def ficGetIidms(fic_MULTIPLE):
    """
    Return the list of the iidm files that are necessary to perform the systematic analysis described by fic_MULTIPLE
    """
    fic = etree.parse(fic_MULTIPLE)
    fic_root = fic.getroot()
    fic_namespace_uri = fic_root.nsmap[fic_root.prefix]

    if len(fic_root) == 0:
        raise Exception('fic_MULTIPLE file is empty')
    scenarios = fic_root[0]
    if scenarios.tag != '{' + fic_namespace_uri+ '}' + 'scenarios':
        raise Exception('Invalid fic_MULTIPLE file')

    jobsFile = scenarios.get('jobsFile')
    working_dir = os.path.dirname(fic_MULTIPLE)

    jobs = etree.parse(os.path.join(working_dir, jobsFile))
    jobs_root = jobs.getroot()
    jobs_namespace_uri = jobs_root.nsmap[jobs_root.prefix]

    # Find base .iidm
    networks = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'network[@iidmFile]')
    if len(networks) != 1:
        raise Exception("Jobs file should contain exactly one iidm entry, %s found" % (len(networks)))
    base_iidm = os.path.join(working_dir, networks[0].get('iidmFile'))

    iidm_files = [base_iidm]
    for scenario in scenarios:
        iidm = scenario.get('iidmFile')
        if iidm != None:
            iidm = os.path.join(working_dir, iidm)
            if not fileIsInList(iidm, iidm_files):
                iidm_files.append(iidm)
    return iidm_files


def randomiseDynamicParams(fic_MULTIPLE, input_csv):
    """
    Generate random dynamic parameters for a parametric systematic analysis

    @param fic_MULTIPLE Full path of the fic_MULTIPLE.xml file for which the random dynamic data has to be generated
    @param input_csv Full path of the csv file containing the information on how to randomise (i.e. randomise what and according to which pdf)

    @output dyd_parameters_dic
        @key input_par Full path of the par's file (linked from fic_MULTIPLE) for which random data has been generated
        @value[0] par_set_ids List of the component names of the randomised parameters
        @value[1] param_ids List of the id's of the randomised parameters
        @value[2] param_values List of the randomised parameter values
    """

    input_pars = ficGetPars(fic_MULTIPLE)
    dyd_parameters_dic = dict()

    pars_root = []
    pars_namespace = []
    pars_prefix_root = []
    pars_prefix_root_string = []

    for input_par in input_pars:
        pars_root.append(etree.parse(input_par).getroot())
        pars_namespace.append(pars_root[-1].nsmap)
        pars_prefix_root.append(pars_root[-1].prefix)
        if pars_prefix_root[-1] is None:
            pars_prefix_root_string.append('')
        else:
            pars_prefix_root_string.append(pars_prefix_root[-1] + ':')

    with open(input_csv) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        row = spamreader.__next__()
        if row != ['parFile', 'ParamSet_id', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
            raise Exception("Incorrect format of %s" % input_csv)
        
        modifiedParameters = []
        for row in spamreader:
            working_dir = os.path.dirname(fic_MULTIPLE)
            parFile = os.path.join(working_dir, row[0])
            paramSetId = row[1]
            paramId = row[2]
            distribution = row[3]
            params = row[4:8]

            par_set_ids = []
            param_ids = []
            param_values = []
            
            # Check for duplicate parameters in the par_csv
            for [modified_paramSetId, modified_paramId] in modifiedParameters:
                if modified_paramSetId == paramSetId or modified_paramSetId == '*' or paramSetId == '*':
                    if modified_paramId == paramId:
                        raise Exception('Duplicate parameters (would be randomised twice in the current implementation. Can be modified if needed)')
            modifiedParameters.append([paramSetId, paramId])
            
            # Randomise the parameters
            if paramSetId == "*": # Wildcard => look in all paramSet for param
                # './/' means find search in the whole tree (instead of only in one level)
                # 'par[@name] means searching for an element par with attribute name
                try:
                    index = input_pars.index(parFile)
                except ValueError:
                    raise Exception("Input par %s given in %s is not used in the SA" % (parFile, input_csv))
                for parameter in pars_root[index].findall('.//' + pars_prefix_root_string[index] + 'par[@name="' + paramId + '"]', pars_namespace[index]):
                    par_set_ids.append(parameter.getparent().get('id'))
                    param_ids.append(paramId)
                    param_values.append(getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, params))
            else:
                try:
                    index = input_pars.index(parFile)
                except ValueError:
                    raise Exception("Input par %s given in %s is not used in the SA" % (parFile, input_csv))
                parameterSet = findParameterSet(pars_root[index], paramSetId)
                parameter = findParameter(parameterSet, paramId)

                par_set_ids.append(paramSetId)
                param_ids.append(paramId)
                param_values.append(getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, params))
            
            value = dyd_parameters_dic.setdefault(parFile, [[],[],[]])
            for i in range(len(par_set_ids)):
                value[0].append(par_set_ids[i])
                value[1].append(param_ids[i])
                value[2].append(param_values[i])

    return dyd_parameters_dic


def writeDynamicParams(dyd_parameters_dic, output_dir):
    """
    Set new values of parameters in the input par's and write the resulting par's to output_dir.

    @param dyd_parameters_dic
        @key input_par Full path of the par's file (linked from fic_MULTIPLE) for which random data has been generated
        @value[0] par_set_ids List of the component names of the randomised parameters
        @value[1] param_ids List of the id's of the randomised parameters
        @value[2] param_values List of the randomised parameter values
    @param out_dir Output directory for the output par's
    """

    for input_par, value in dyd_parameters_dic.items():
        par_set_ids = value[0]
        param_ids = value[1]
        param_values = value[2]

        par_root = etree.parse(input_par).getroot()

        for i in range(len(param_ids)):
            parameterSet = findParameterSet(par_root, par_set_ids[i])
            parameter = findParameter(parameterSet, param_ids[i])
            parameter.set('value', str(param_values[i]))

        output_par = os.path.join(output_dir, os.path.basename(input_par))
        with open(output_par, 'xb') as doc:
            doc.write(etree.tostring(par_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))


def randomiseStaticParams(fic_MULTIPLE, input_csv):
    """
    Randomise static parameters

    @param fic_MULTIPLE Full path of the fic_MULTIPLE.xml file for which the random static data has to be generated
    @param input_csv full path of the csv file containing the information on how to randomise (i.e. randomise what and according to which pdf)

    @output static_parameters_dic
        @key input_iidm Full path of the iidm's file (linked from fic_MULTIPLE) for which random data has been generated
        @value[0] component_types List of the component types of the randomised parameters
        @value[1] component_names List of the component names of the randomised parameters
        @value[2] param_ids List of the id's of the randomised parameters
        @value[3] param_values List of the randomised parameter values
    """

    input_iidms = ficGetIidms(fic_MULTIPLE)
    static_parameters_dic = dict()

    for input_iidm in input_iidms:
        component_types = []
        component_names = []
        param_ids = []
        param_values = []

        n = pp.network.load(input_iidm)
        slack_load_index = n.get_loads().index[0] #TODO: allow to specify which load is used as the slack
        P_load_init = sum(n.get_loads().p0)
        Q_load_init = sum(n.get_loads().q0)
        P_gen_init = sum(n.get_generators().target_p)
        Q_gen_init = sum(n.get_generators().target_q)
        with open(input_csv) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            row = spamreader.__next__()
            if row != ['Component_type', 'Component_name', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
                print(row)
                raise Exception("Incorrect format of %s" % input_csv)
            
            modifiedParameters = []
            for row in spamreader:
                # print(', '.join(row))
                componentType = row[0]
                componentName = row[1]
                paramId = row[2]
                distribution = row[3]
                params = row[4:8]
                
                if componentType == '*':
                    raise NotImplementedError('Wildcard not implemented for Component_type (and is not really useful)')

                # Check for duplicate parameters in the params.csv
                for [modified_componentType, modified_componentName, modified_paramId] in modifiedParameters:
                    if modified_componentType == componentType:
                        if modified_componentName == componentName or modified_componentName == '*' or componentName == '*':
                            if modified_paramId == paramId:
                                raise Exception('Duplicate parameters (would be randomised twice in the current implementation. Can be modified if needed)')
                modifiedParameters.append([componentType, componentName, paramId])


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
                
                random_params = []
                if componentName == '*':
                    for index in components.index:
                        if component_types != 'Load' or componentName != slack_load_index: # The slack load is not randomised but used at the end to keep the balance
                            init_value = components.at[index, paramId]
                            if init_value == None:
                                raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (componentName, componentType, paramId))
                            random_param_value = getRandom(init_value, "DOUBLE", distribution, params) # All parameters in .iidm are of type double
                            random_params.append({paramId : random_param_value})

                            component_types.append(componentType)
                            component_names.append(index)
                            param_ids.append(paramId)
                            param_values.append(random_param_value)

                    random_params_df = pd.DataFrame(random_params, components.index)

                else:
                    if component_types != 'Load' or componentName != slack_load_index: # The slack load is not randomised but used at the end to keep the balance
                        init_value = components.get(paramId).get(componentName)
                        if init_value == None:
                            raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (componentName, componentType, paramId))
                        random_param_value = getRandom(init_value, "DOUBLE", distribution, params) # All parameters in .iidm are of type double
                        random_params.append({paramId : random_param_value})

                        component_types.append(componentType)
                        component_names.append(componentName)
                        param_ids.append(paramId)
                        param_values.append(random_param_value)

                        random_params_df = pd.DataFrame(random_params, [componentName])

                if componentType == 'Load':
                    n.update_loads(random_params_df) # Updates the network only to more easily compute the balance (no .iidm is written/modified)
                elif componentType == 'Line':
                    n.update_lines(random_params_df)
                elif componentType == 'Bus':
                    n.update_buses(random_params_df)
                elif componentType == 'Generator':
                    n.update_generators(random_params_df)
                else:
                    raise Exception("Component type '%s' not considered" % componentType)

        # Modify the first load to keep the balance.
        delta_P = (P_load_init - sum(n.get_loads().p0)) - (P_gen_init - sum(n.get_generators().target_p))
        delta_Q = (Q_load_init - sum(n.get_loads().q0)) - (Q_gen_init - sum(n.get_generators().target_q))
        
        # slack = {'p0' : delta_P + n.get_loads().get('p0').get(slack_load_index), 'q0' : delta_Q + n.get_loads().get('q0').get(slack_load_index)}
        # n.update_loads(pd.DataFrame(slack, index = [slack_load_index]))

        component_types.append('Load')
        component_names.append(slack_load_index)
        param_ids.append('p0')
        param_values.append(delta_P + n.get_loads().get('p0').get(slack_load_index))

        component_types.append('Load')
        component_names.append(slack_load_index)
        param_ids.append('q0')
        param_values.append(delta_Q + n.get_loads().get('q0').get(slack_load_index))

        #TODO: target_q is usually not used (voltage is regulated), change the load at the slack iteratively until the power flow results match
        # the measurements at the point of common coupling
        #TODO: also add toggle to the balance check
        
        static_parameters_dic[input_iidm] = [component_types, component_names, param_ids, param_values]

    return static_parameters_dic


def writeStaticParams(static_parameters_dic, output_dir):
    """
    Set new values of parameters in the input iidm's and write the resulting iidm to output_dir.

    @param static_parameters_dic
        @key input_iidm Full path of the iidm's file (linked from fic_MULTIPLE) for which random data has been generated
        @value[0] component_types List of the component types of the randomised parameters
        @value[1] component_names List of the component names of the randomised parameters
        @value[2] param_ids List of the id's of the randomised parameters
        @value[3] param_values List of the randomised parameter values
    @param output_dir Output directory for the output iidm's
    """
    for input_iidm, value in static_parameters_dic.items():
        component_types = value[0]
        component_names = value[1]
        param_ids = value[2]
        param_values = value[3]

        n = pp.network.load(input_iidm)
        P_load_init = sum(n.get_loads().p0)
        Q_load_init = sum(n.get_loads().q0)
        P_gen_init = sum(n.get_generators().target_p)
        Q_gen_init = sum(n.get_generators().target_q)
        for i in range(len(param_ids)):
            if component_types[i] == 'Load':
                components = n.get_loads()
            elif component_types[i] == 'Line':
                components = n.get_lines()
            elif component_types[i] == 'Bus':
                components = n.get_buses()
            elif component_types[i] == 'Generator':
                components = n.get_generators()
            else:
                raise Exception("Component type '%s' not considered" % component_types[i])

            if components.get(param_ids[i]).get(component_names[i]) == None:
                raise Exception("No component '%s' of type '%s' with a parameter '%s' found" % (component_names[i], component_types[i], param_ids[i]))
            new_param = [{param_ids[i] : param_values[i]}]
            new_param_df = pd.DataFrame(new_param, [component_names[i]])

            if component_types[i] == 'Load':
                n.update_loads(new_param_df)
            elif component_types[i] == 'Line':
                n.update_lines(new_param_df)
            elif component_types[i] == 'Bus':
                n.update_buses(new_param_df)
            elif component_types[i] == 'Generator':
                n.update_generators(new_param_df)
            else:
                raise Exception("Component type '%s' not considered" % component_types[i])

        # Modify the first load to keep the balance.
        delta_P = (P_load_init - sum(n.get_loads().p0)) - (P_gen_init - sum(n.get_generators().target_p))
        delta_Q = (Q_load_init - sum(n.get_loads().q0)) - (Q_gen_init - sum(n.get_generators().target_q))
        
        slack_load_index = n.get_loads().index[0] #TODO: allow to specify which load is used as the slack
        slack = {'p0' : delta_P + n.get_loads().get('p0').get(slack_load_index), 'q0' : delta_Q + n.get_loads().get('q0').get(slack_load_index)}
        n.update_loads(pd.DataFrame(slack, index = [slack_load_index]))
        parameters = pp.loadflow.Parameters(read_slack_bus=True, distributed_slack=False, write_slack_bus=False, balance_type=pp.loadflow.BalanceType.PROPORTIONAL_TO_LOAD)
        pp.loadflow.run_ac(n, parameters)

        output_iidm = os.path.join(output_dir, os.path.basename(input_iidm))
        if os.path.exists(output_iidm):
            raise Exception('')
        else:
            n.dump(output_iidm, 'XIIDM', {'iidm.export.xml.version' : '1.4'}) # Latest version supported by dynawo
            # Set back original extension (powsybl always set it to XIIDM)
            [file, ext] = output_iidm.rsplit('.', 1)
            if ext != 'xiidm':
                os.rename(file + '.xiidm', output_iidm)
            
            # Removes the slack extension that powsybl writes even though write_slack_bus=False
            # By doing so, we also remove all other extensions
            iidm = etree.parse(output_iidm)
            iidm_root = iidm.getroot()
            iidm_namespace = iidm_root.nsmap
            iidm_prefix_root = iidm_root.prefix
            if iidm_prefix_root is None:
                iidm_prefix_root_string = ''
            else:
                iidm_prefix_root_string = iidm_prefix_root + ':'

            for extension in iidm_root.findall('.//' + iidm_prefix_root_string + 'extension', iidm_namespace):
                extension.getparent().remove(extension)

            with open(output_iidm, 'wb') as doc:
                doc.write(etree.tostring(iidm_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))
 
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


def copyNoReplace (f, dest):
    """
    Copy file to dest, raise exception if a file/dir/... with the same name already exists in destination
    """
    final_dest = dest + '/' + os.path.basename(f)
    if os.path.exists(final_dest):
        raise Exception("Cannot copy to %s, another file already exists" % final_dest)
    shutil.copy(f, dest)


def writeParametricSAInputs(working_dir, fic_MULTIPLE, output_dir_name, static_data_dic, dyn_data_dic, run_id):
    """
    Write the inputs files necessary to perform a "parametric" systematic analysis (SA), i.e. a systematic analysis where the parameters are modified

    @param working_dir Working directory
    @param fic_MULTIPLE fic_MULTIPLE.xml file that would be used to perform the "classical" SA
    @param output_dir_name Name of the output directory, i.e. output_dir = working_dir + '/' + output_dir_name
    @param static_data_dic Dictionary containing the modified values of the static parameters, see e.g. randomiseStaticParams():static_parameters_dic
    @param dyn_data_dic Dictionary containing the modified values of the dynamic parameters, see e.g. randomiseDynamicParams():dyn_parameters_dic
    @param run_id Id of the run
    """
    output_dir = os.path.join(working_dir, output_dir_name)
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir) # First clean the output directory for easier collision checking
    os.makedirs(output_dir)

    fic = etree.parse(fic_MULTIPLE)
    fic_root = fic.getroot()
    fic_namespace = fic_root.nsmap
    fic_prefix_root = fic_root.prefix
    fic_namespace_uri = fic_namespace[fic_prefix_root]

    if len(fic_root) == 0:
        raise Exception('fic_MULTIPLE file is empty')
    scenarios = fic_root[0]
    if scenarios.tag != '{' + fic_namespace_uri+ '}' + 'scenarios':
        raise Exception('Invalid fic_MULTIPLE file')

    rootName = etree.QName(fic_namespace_uri, 'multipleJobs')
    output_fic_root = etree.Element(rootName, nsmap=fic_namespace)
    output_scenarios = etree.SubElement(output_fic_root, 'scenarios')
    jobsFile = scenarios.get('jobsFile')
    output_scenarios.set('jobsFile', os.path.basename(jobsFile))

    jobs = etree.parse(os.path.join(working_dir, jobsFile))
    jobs_root = jobs.getroot()
    jobs_namespace = jobs_root.nsmap
    jobs_prefix_root = jobs_root.prefix
    jobs_namespace_uri = jobs_namespace[jobs_prefix_root]

    # Copy .iidm
    networks = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'network[@iidmFile]')
    if len(networks) != 1:
        raise Exception("Jobs file should contain exactly one iidm entry, %s found" % (len(networks)))
    base_iidm = os.path.join(working_dir, networks[0].get('iidmFile'))
    networks[0].set('iidmFile', os.path.basename(base_iidm))

    # Copy .dyd
    jobs_dyd = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'dynModels[@dydFile]')
    if len(jobs_dyd) != 1:
        raise NotImplementedError("Jobs file should contain exactly one dyd entry (0 would be possible with few changes), %s found" % (len(jobs_dyd)))
    base_dyd = os.path.join(working_dir, jobs_dyd[0].get('dydFile'))
    # The dyd file given in the initial .jobs has to be replaced by different ones that correctly links to the random .par
    jobs_dyd[0].getparent().remove(jobs_dyd[0])

    # Copy .crt
    jobs_crt = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'criteria[@criteriaFile]')
    if len(jobs_crt) > 1:
        raise Exception("Jobs file should contain at most one .crt entry, %s found" % (len(jobs_crt)))
    if len(jobs_crt) > 0:
        crt = os.path.join(working_dir, jobs_crt[0].get('criteriaFile'))
        jobs_crt[0].set('criteriaFile', os.path.basename(crt))
        copyNoReplace(crt, output_dir)
    
    # Copy .crv
    jobs_crv = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'curves[@inputFile]')
    if len(jobs_crv) > 1:
        raise Exception("Jobs file should contain at most one .crv entry, %s found" % (len(jobs_crv)))
    if len(jobs_crv) > 0:
        crv = os.path.join(working_dir, jobs_crv[0].get('inputFile'))
        jobs_crv[0].set('inputFile', os.path.basename(crv))
        copyNoReplace(crv, output_dir)

    # Write the modified .jobs file
    with open(os.path.join(output_dir, os.path.basename(jobsFile)), 'xb') as doc:
        doc.write(etree.tostring(jobs_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    par_files = []
    dyd_files = []
    iidm_files = [base_iidm]
    for scenario in scenarios:
        dyd = scenario.get('dydFile')
        if dyd != None:
            dyd = os.path.join(working_dir, dyd)
            if not fileIsInList(dyd, dyd_files):
                dyd_files.append(dyd)
        iidm = scenario.get('iidmFile')
        if iidm != None:
            iidm = os.path.join(working_dir, iidm)
            if not fileIsInList(iidm, iidm_files):
                iidm_files.append(iidm)

    # Write new fic_MULTIPLE.xml
    for scenario in scenarios:
        output_scenario = etree.SubElement(output_scenarios, 'scenario')
        output_scenario.set('id', scenario.get('id') + "_%03d" % run_id)
        
        iidm = scenario.get('iidmFile')
        if iidm == None:
            iidm = base_iidm
        output_scenario.set('iidmFile', os.path.basename(iidm))
        
        dyd = scenario.get('dydFile')
        if dyd == None:
            dyd = base_dyd
        output_scenario.set('dydFile', os.path.basename(dyd))

        networkPar = scenario.get('networkParFile')
        if networkPar != None:
            output_scenario.set('networkParFile', networkPar)
        networkParId = scenario.get('networkParId')
        if networkParId != None:
            output_scenario.set('networkParId', networkParId)
    with open(os.path.join(output_dir, 'fic_MULTIPLE.xml'), 'xb') as doc:
        doc.write(etree.tostring(output_fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Write new .iidm's
    writeStaticParams(static_data_dic, output_dir)        

    # Write new .dyd's (copy the .dyd files, but changing the .par they reference)
    for dyd_file in [base_dyd] + dyd_files:
        dyd_root = etree.parse(dyd_file).getroot()
        dyd_namespace = dyd_root.nsmap
        dyd_prefix_root = dyd_root.prefix
        if dyd_prefix_root is None:
            dyd_prefix_root_string = ''
        else:
            dyd_prefix_root_string = dyd_prefix_root + ':'
        
        # Merge the base dyd into the "event" dyd
        if dyd_file is not base_dyd:
            base_dyd_root = etree.parse(base_dyd).getroot()
            for element in base_dyd_root:
                dyd_root.append(element)

        root = deepcopy(dyd_root)
        for blackBox in root.findall('.//' + dyd_prefix_root_string + 'blackBoxModel', dyd_namespace):
            dyd_par = os.path.join(working_dir, blackBox.attrib['parFile'])
            if not fileIsInList(dyd_par, par_files):
                par_files.append(dyd_par)
            blackBox.set('parFile', os.path.basename(dyd_par))

        if len(root.findall('.//' + dyd_prefix_root_string + 'unitDynamicModel', dyd_namespace)) != 0:
            raise NotImplementedError('Consider using precompiled models for systematic analyses')

        output_dyd = os.path.join(output_dir, os.path.basename(dyd_file))
        with open(output_dyd, 'xb') as doc:
            doc.write(etree.tostring(root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Write new .par's
    writeDynamicParams(dyn_data_dic, output_dir)
    for par_file in par_files:
        if not fileIsInList(par_file, list(dyn_data_dic.keys())):
            copyNoReplace(par_file, output_dir)
    
    # Copy solver parFile
    jobs_solverPar = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'solver[@parFile]')
    if len(jobs_solverPar) != 1:
        raise Exception("Jobs file should contain exactly one solverFile entry, %s found" % (len(jobs_solverPar)))
    solverPar = os.path.join(working_dir, jobs_solverPar[0].get('parFile'))
    jobs_solverPar[0].set('parFile', os.path.basename(solverPar))
    if not fileIsInList(solverPar, par_files):
        copyNoReplace(solverPar, output_dir)

    # Copy network parFile
    jobs_networkPar = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'network[@parFile]')
    if len(jobs_networkPar) != 1:
        raise Exception("Jobs file should contain exactly one networkParFile entry, %s found" % (len(jobs_networkPar)))
    networkParFile = os.path.join(working_dir, jobs_networkPar[0].get('parFile'))
    jobs_networkPar[0].set('parFile', os.path.basename(networkParFile))
    if not fileIsInList(networkParFile, par_files):
        copyNoReplace(networkParFile, output_dir)


if __name__ == "__main__":
    random.seed(1)
    parser = argparse.ArgumentParser('Takes as input the same files as a systematic analysis (SA), and generates a randomised version of those files. '
    'Those can then be used to perform another SA. The ouput files are all written to working_dir/RandomisedInputs/')

    parser.add_argument('--working_dir', type=str, required=True,
                        help='Working directory')
    parser.add_argument('--fic_MULTIPLE', type=str, required=True,
                        help='Input file containing the different scenarios to run')
    parser.add_argument('--csv_par', type=str, required=True,
                        help='Csv file containing the list of dynamic parameters to be randomised and associated distributions')
    parser.add_argument('--csv_iidm', type=str, required=True,
                        help='Csv file containing the list of static parameters to be randomised and associated distributions')
    parser.add_argument('--nb_runs', type=int, required=True,
                        help='Number of randomised copies of the base scenarios to create')
    args = parser.parse_args()

    args.fic_MULTIPLE = os.path.join(args.working_dir, args.fic_MULTIPLE)
    args.csv_par = os.path.join(args.working_dir, args.csv_par)
    args.csv_iidm = os.path.join(args.working_dir, args.csv_iidm)
    
    for run_id in range(args.nb_runs):
        output_dir_name = "RandomisedInputs_%03d" % run_id
        output_dir = os.path.join(args.working_dir, output_dir_name)

        static_data_dic = randomiseStaticParams(args.fic_MULTIPLE, args.csv_iidm)
        dyn_data_dic = randomiseDynamicParams(args.fic_MULTIPLE, args.csv_par)

        writeParametricSAInputs(args.working_dir, args.fic_MULTIPLE, output_dir_name, static_data_dic, dyn_data_dic, run_id)
