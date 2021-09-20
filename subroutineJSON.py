import json
import os
from datetime import datetime

import git

def get_date_time_string():
    now = datetime.now()
    return now.strftime("%d_%m_%H_%M")

def retrive_json_options(filename, options, results):
    JsonOptions = {
        'commit': None,
        'file': filename,
        'options': options,
        'results_tot': results,
        'description': None,
        'varianti': check_varianti(options)
    }

    instances = []
    optimizers = []
    for item in JsonOptions['options']['quantum_instance']:
        instances.append(item[1])
    for item in JsonOptions['options']['optimizer']:
        optimizers.append(item[1])
    JsonOptions['options']['quantum_instance'] = instances
    JsonOptions['options']['optimizer'] = optimizers
    del JsonOptions['options']['converter']
    JsonOptions['options']['dists'] = JsonOptions['options']['dists'].tolist()
    JsonOptions['options']['series']['itermax'] = JsonOptions['options']['series']['itermax'].tolist()
    JsonOptions['options']['series']['step'] = JsonOptions['options']['series']['step'].tolist()
    JsonOptions['options']['series']['lamb'] = JsonOptions['options']['series']['lamb'].tolist()

    total_results = {}
    for result_tot in JsonOptions['results_tot']:
        results = []
        for idx in range(len(JsonOptions['results_tot'][result_tot])):
            single_res = from_electronic_res_to_dict(JsonOptions['results_tot'][result_tot][idx])
            results.append(single_res)
        total_results[result_tot] = results

    JsonOptions['results_tot'] = total_results

    return JsonOptions

def from_electronic_res_to_dict(result_old):
    result_new = {
        'energy': result_old.total_energies[0],
        'auxiliary': {
            'particles': result_old.num_particles,
            'spin-z': result_old.spin,
            'spin-sq': result_old.total_angular_momentum
        }
    }
    return result_new

def check_varianti(opt):

    varianti = []
    if len(opt['molecule']['basis']) > 1:
        varianti.append('base')
    if len(opt['varforms']) > 1:
        varianti.append('varform')
    if len(opt['quantum_instance']) > 1:
        varianti.append('quantum_instance')
    if len(opt['optimizer']) > 1:
        varianti.append('optimizer')
    if len(opt['lagrange']['active']) > 1:
        varianti.append('lagrange activity')
    if len(opt['lagrange']['operators']) > 1:
        varianti.append('lagrange operator')
    return varianti

def write_json(JsonOptions):
    filename = "data/exp_"+get_date_time_string()
    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object.hexsha
    JsonOptions['commit'] = commit

    description = input("Inserisci una minima descrizione: ")
    JsonOptions['description'] = description

    json_obj = json.dumps(JsonOptions, indent=4)

    while os.path.exists(filename):
        print('Esiste già un file con questo nome!!')
        filename = input('Inserisci un nuovo nome: ')

    with open(filename, "w") as out:
        out.write(json_obj)
