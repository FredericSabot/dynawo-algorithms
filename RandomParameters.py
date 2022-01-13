from lxml import etree
import csv
import random
import math
import argparse
import pypowsybl as pp
import pandas as pd
import os
import shutil
from copy import deepcopy


def getRandom(value, type_, distribution, params):
    """
   Supported distributions are:

    gaussian(sigma)
    gaussian_percent(percent_sigma)
    uniform_minmax(min, max)
    uniform_delta(delta)

    The average of the distribution (except for uniform_minmax) is taken as the original value of the parameter
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


def writeRandomDynamicParams(input_pars, input_csv, working_dir, output_dir, seed, nb_runs):
    """
    Randomise dynamic parameters of an input .par file and outputs another file

    @param input_pars List of full paths of the par files containing the dynamic data of the network to be randomised
    @param input_csv Full path of the csv file containing the information on how to randomise (i.e. randomise what and according to which pdf)
    @param working_dir Working directory
    @param out_dir Output directory for the output par's
    @param seed Seed used by the random generator
    @param nb_runs Number of randomised copies of the par files to create
    """
    random.seed(seed)

    parser = etree.XMLParser(remove_blank_text=True) # remove_blank_test Necessary to get the pretty print working after adding new elements
    pars = []
    pars_root = []
    pars_namespace = []
    pars_prefix_root = []
    pars_prefix_root_string = []

    output_pars_root = []

    for input_par in input_pars:
        pars.append(etree.parse(input_par, parser))
        pars_root.append(pars[-1].getroot())
        pars_namespace.append(pars_root[-1].nsmap)
        pars_prefix_root.append(pars_root[-1].prefix)
        if pars_prefix_root[-1] is None:
            pars_prefix_root_string.append('')
        else:
            pars_prefix_root_string.append(pars_prefix_root[-1] + ':')
        
        out_pars_root = []
        for i in range(nb_runs):
            out_pars_root.append(deepcopy(pars_root[-1]))
        output_pars_root.append(out_pars_root)


    with open(input_csv) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        row = spamreader.__next__()
        if row != ['parFile', 'ParamSet_id', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
            raise Exception("Incorrect format of %s" % input_csv)
        
        modifiedParameters = []
        for row in spamreader:
            # print(', '.join(row))
            parFile = working_dir + row[0]
            paramSetId = row[1]
            paramId = row[2]
            distribution = row[3]
            params = row[4:8]
            
            # Check for duplicate parameters in the par_csv
            for [modified_paramSetId, modified_paramId] in modifiedParameters:
                if modified_paramSetId == paramSetId or modified_paramSetId == '*' or paramSetId == '*':
                    if modified_paramId == paramId:
                        raise Exception('Duplicate parameters (would be randomised twice in the current implementation. Can be modified if needed)')
            modifiedParameters.append([paramSetId, paramId])
            
            # Randomise the parameters
            for nb_run in range(nb_runs):
                if paramSetId == "*": # Wildcard => look in all paramSet for param
                    # './/' means find search in the whole tree (instead of only in one level)
                    # 'par[@name] means searching for an element par with attribute name
                    for i in range(len(pars)):
                        for parameter in output_pars_root[i][nb_run].findall('.//' + pars_prefix_root_string[i] + 'par[@name="' + paramId + '"]', pars_namespace[i]):
                            parameter.set('value', str(getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, params)))
                else:
                    found = None
                    for i in range(len(input_pars)):
                        if os.path.samefile(parFile, input_pars[i]):
                            found = i
                            break
                    if found == None:
                        raise Exception("Input par %s given in %s is not used in the SA" % (parFile, input_csv))
                        
                    parameterSet = findParameterSet(output_pars_root[i][nb_run], paramSetId)
                    parameter = findParameter(parameterSet, paramId)
                    parameter.set('value', str(getRandom(parameter.attrib['value'], parameter.attrib['type'], distribution, params)))


    for i in range(len(input_pars)):
        for nb_run in range(nb_runs):
            suffix = "_%03d" % (nb_run)
            output_par = output_dir + addSuffix(getShortPath(input_pars[i]), suffix)
            with open(output_par, 'xb') as doc:
                doc.write(etree.tostring(output_pars_root[i][nb_run], pretty_print = True, xml_declaration = True, encoding='UTF-8'))


def writeRandomStaticParams(input_iidms, input_csv, output_dir, seed, nb_runs):
    """
    Randomise static parameters (then run a load flow using powsybl and write the results)

    @param input_iidms List of full paths of the iidm files containing the static data of the network to be randomised
    @param input_csv full path of the csv file containing the information on how to randomise (i.e. randomise what and according to which pdf)
    @param output_dir Output directory for the output iidm's
    @param seed seed used by the random generator
    @param nb_runs Number of randomised copies of the iidm files to create
    """
    random.seed(seed)

    for input_iidm in input_iidms:
        for nb_run in range(nb_runs):
            n = pp.network.load(input_iidm)
            P_init = sum(n.get_loads().p0)
            Q_init = sum(n.get_loads().q0)
            with open(input_csv) as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',')
                row = spamreader.__next__()
                if row != ['Component_type', 'Component_name', 'Param_id', 'Distribution', 'Param1', 'Param2', 'Param3', 'Param4', 'Param5']:
                    print(row)
                    raise Exception("Incorrect format of %s" % input_iidm)
                
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
                    elif componentType == 'Generator':
                        n.update_generators(random_params_df)
                    else:
                        raise Exception("Component type '%s' not considered" % componentType)

            # Modify the first load to keep the balance.
            delta_P = P_init - sum(n.get_loads().p0)
            delta_Q = Q_init - sum(n.get_loads().q0)
            
            slack_load_index = n.get_loads().index[0] #TODO: allow to specify which load is used as the slack
            slack = {'p0' : delta_P + n.get_loads().get('p0').get(slack_load_index), 'q0' : delta_Q + n.get_loads().get('q0').get(slack_load_index)}
            n.update_loads(pd.DataFrame(slack, index = [slack_load_index]))

            pp.loadflow.run_ac(n)

            suffix = "_%03d" % (nb_run)
            output_iidm = output_dir + addSuffix(getShortPath(input_iidm), suffix)
            if os.path.exists(output_iidm):
                raise Exception('')
            else:
                n.dump(output_iidm, 'XIIDM', {'iidm.export.xml.version' : '1.4'}) # Latest version supported by dynawo
                # Set back original extension (powsybl always set it to XIIDM)
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


def getShortPath(s):
    """
    Return short path from full path, e.g. foo/bar -> bar
    """
    return s.split('/')[-1]


def fileIsInList(file, lst):
    for f in lst:
        if os.path.samefile(f, file):
            return True
    return False


def copyNoReplace (f, dest):
    """
    Copy file to dest, raise exception if a file/dir/... with the same name already exists in destination
    """
    final_dest = dest + '/' + getShortPath(f)
    if os.path.exists(final_dest):
        raise Exception("Cannot copy to %s, another file already exists" % final_dest)
    shutil.copy(f, dest)


if __name__ == "__main__":
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

    args.working_dir += '/'
    args.fic_MULTIPLE = args.working_dir + args.fic_MULTIPLE
    args.csv_par = args.working_dir + args.csv_par
    args.csv_iidm = args.working_dir + args.csv_iidm
    
    output_dir = args.working_dir + 'RandomisedInputs/'
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir) # First clean the output directory for easier collision checking
    os.makedirs(output_dir)

    fic = etree.parse(args.fic_MULTIPLE)
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
    output_scenarios.set('jobsFile', getShortPath(jobsFile))

    jobs = etree.parse(args.working_dir + jobsFile)
    jobs_root = jobs.getroot()
    jobs_namespace = jobs_root.nsmap
    jobs_prefix_root = jobs_root.prefix
    jobs_namespace_uri = jobs_namespace[jobs_prefix_root]

    # Copy .iidm
    networks = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'network[@iidmFile]')
    if len(networks) != 1:
        raise Exception("Jobs file should contain exactly one iidm entry, %s found" % (len(networks)))
    base_iidm = args.working_dir + networks[0].get('iidmFile')
    networks[0].set('iidmFile', getShortPath(base_iidm))
    copyNoReplace(base_iidm, output_dir) # The iidm file given in the .jobs will be overriden by the random ones, but it is still necessary to have a valid one in the .jobs

    # Copy .dyd
    jobs_dyd = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'dynModels[@dydFile]')
    if len(jobs_dyd) != 1:
        raise NotImplementedError("Jobs file should contain exactly one dyd entry (0 would be possible with few changes), %s found" % (len(jobs_dyd)))
    base_dyd = args.working_dir + jobs_dyd[0].get('dydFile')
    # The dyd file given in the initial .jobs has to be replaced by different ones that correctly links to the random .par
    jobs_dyd[0].getparent().remove(jobs_dyd[0])

    # Copy network parFile
    jobs_networkPar = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'network[@parFile]')
    if len(jobs_networkPar) != 1:
        raise Exception("Jobs file should contain exactly one networkParFile entry, %s found" % (len(jobs_networkPar)))
    networkParFile = args.working_dir + jobs_networkPar[0].get('parFile')
    jobs_networkPar[0].set('parFile', getShortPath(networkParFile))
    copyNoReplace(networkParFile, output_dir)

    # Copy solver parFile
    jobs_solverPar = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'solver[@parFile]')
    if len(jobs_solverPar) != 1:
        raise Exception("Jobs file should contain exactly one solverFile entry, %s found" % (len(jobs_solverPar)))
    solverPar = args.working_dir + jobs_solverPar[0].get('parFile')
    jobs_solverPar[0].set('parFile', getShortPath(solverPar))
    copyNoReplace(solverPar, output_dir)

    # Copy .crt
    jobs_crt = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'criteria[@criteriaFile]')
    if len(jobs_crt) > 1:
        raise Exception("Jobs file should contain at most one .crt entry, %s found" % (len(jobs_crt)))
    if len(jobs_crt) > 0:
        crt = args.working_dir + jobs_crt[0].get('criteriaFile')
        jobs_crt[0].set('criteriaFile', getShortPath(crt))
        copyNoReplace(crt, output_dir)
    
    # Copy .crv curves inputFile=
    jobs_crv = jobs_root.findall('.//' + '{' + jobs_namespace_uri + '}' + 'curves[@inputFile]')
    if len(jobs_crv) > 1:
        raise Exception("Jobs file should contain at most one .crv entry, %s found" % (len(jobs_crv)))
    if len(jobs_crv) > 0:
        crv = args.working_dir + jobs_crv[0].get('inputFile')
        jobs_crv[0].set('inputFile', getShortPath(crv))
        copyNoReplace(crv, output_dir)

    # Write the modified .jobs file
    with open(output_dir + getShortPath(jobsFile), 'xb') as doc:
        doc.write(etree.tostring(jobs_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    par_files = []
    dyd_files = []
    iidm_files = [base_iidm]
    for scenario in scenarios:
        dyd = scenario.get('dydFile')
        if dyd != None:
            dyd = args.working_dir + dyd
            if not fileIsInList(dyd, dyd_files):
                dyd_files.append(dyd)
        iidm = scenario.get('iidmFile')
        if iidm != None:
            iidm = args.working_dir + iidm
            if not fileIsInList(iidm, iidm_files):
                iidm_files.append(iidm)

    # Write new fic_MULTIPLE.xml
    for i in range(args.nb_runs):
        suffix = "_%03d" % (i)
        for scenario in scenarios:
            output_scenario = etree.SubElement(output_scenarios, 'scenario')
            output_scenario.set('id', addSuffix(scenario.get('id'), suffix))
            
            iidm = scenario.get('iidmFile')
            if iidm == None:
                iidm = base_iidm
            output_scenario.set('iidmFile', addSuffix(getShortPath(iidm), suffix))
            
            dyd = scenario.get('dydFile')
            if dyd == None:
                dyd = base_dyd
            output_scenario.set('dydFile', addSuffix(getShortPath(dyd), suffix))

            networkPar = scenario.get('networkParFile')
            if networkPar != None:
                output_scenario.set('networkParFile', networkPar)
            networkParId = scenario.get('networkParId')
            if networkParId != None:
                output_scenario.set('networkParId', networkParId)
    with open(output_dir + 'fic_MULTIPLE' + '.xml', 'xb') as doc:
        doc.write(etree.tostring(output_fic_root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Write new .iidm's
    writeRandomStaticParams(iidm_files, args.csv_iidm, output_dir, 1, args.nb_runs)

    # Write new .dyd's (copy the .dyd files, but changing the .par they reference)
    for dyd_file in [base_dyd] + dyd_files:
        dyd_root = etree.parse(dyd_file).getroot()
        dyd_namespace = dyd_root.nsmap
        dyd_prefix_root = dyd_root.prefix
        dyd_namespace_uri = dyd_namespace[dyd_prefix_root]
        if dyd_prefix_root is None:
            dyd_prefix_root_string = ''
        else:
            dyd_prefix_root_string = dyd_prefix_root + ':'
        
        # Merge the base dyd into the "event" dyd
        if dyd_file is not base_dyd:
            base_dyd_root = etree.parse(base_dyd).getroot()
            for element in base_dyd_root:
                dyd_root.append(element)

        for nb_run in range(args.nb_runs):
            suffix = "_%03d" % (nb_run)
            root = deepcopy(dyd_root)
            for blackBox in root.findall('.//' + dyd_prefix_root_string + 'blackBoxModel', dyd_namespace):
                dyd_par = args.working_dir + blackBox.attrib['parFile']
                if not fileIsInList(dyd_par, par_files):
                    par_files.append(dyd_par)
                blackBox.set('parFile', addSuffix(getShortPath(dyd_par), suffix))

            if len(root.findall('.//' + dyd_prefix_root_string + 'unitDynamicModel', dyd_namespace)) != 0:
                raise NotImplementedError('Consider using precompiled models for systematic analyses')

            output_dyd = output_dir + addSuffix(getShortPath(dyd_file), suffix)
            with open(output_dyd, 'xb') as doc:
                doc.write(etree.tostring(root, pretty_print = True, xml_declaration = True, encoding='UTF-8'))

    # Write new .par's
    # Use a different seed than in writeRandomStaticPar
    writeRandomDynamicParams(par_files, args.csv_par, args.working_dir, output_dir, 1e9, args.nb_runs)
