import json
import os
from datetime import datetime

import git
import numpy as np

def get_date_time_string():
    now = datetime.now()
    return now.strftime("%d_%m_%H_%M")

def retrive_json_options(filename, options, results):
    json_options = {
        'commit': None,
        'file': filename,
        'options': options,
        'results_tot': results,
        'description': None,
        'varianti': check_varianti(options)
    }

    instances = []
    optimizers = []
    for item in json_options['options']['quantum_instance']:
        instances.append(item[1])
    for item in json_options['options']['optimizer']:
        optimizers.append(item[1])
    json_options['options']['quantum_instance'] = instances
    json_options['options']['optimizer'] = optimizers
    del json_options['options']['converter']
    json_options['options']['dists'] = json_options['options']['dists'].tolist()
    json_options['options']['series']['itermax'] = json_options['options']['series']['itermax'].tolist()
    json_options['options']['series']['step'] = json_options['options']['series']['step'].tolist()
    json_options['options']['series']['lamb'] = json_options['options']['series']['lamb'].tolist()

    total_results = {}
    for result_tot in json_options['results_tot']:
        results = []
        for idx in range(len(json_options['results_tot'][result_tot])):
            single_res = from_electronic_res_to_dict(json_options['results_tot'][result_tot][idx])
            results.append(single_res)
        total_results[result_tot] = results

    json_options['results_tot'] = total_results

    return json_options

def from_electronic_res_to_dict(result_old):
    energy_val = result_old.total_energies[0]
    if np.iscomplexobj(energy_val):
        energy_val = energy_val.real
    result_new = {
        'energy': float(energy_val),
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

def write_json(json_options):
    filename = "data/exp_"+get_date_time_string()

    print('DATA: ', filename)

    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object.hexsha
    json_options['commit'] = commit

    try:
        description = input("Inserisci una minima descrizione: ")
    except:
        print('errore nella codifica della descrizione')
        description = "dummy"

    json_options['description'] = description

    json_obj = json.dumps(json_options, indent=4)

    while os.path.exists(filename):
        print('Esiste già un file con questo nome!!')
        filename = input('Inserisci un nuovo nome: ')

    with open(filename, "w", encoding='utf-8') as out:
        out.write(json_obj)
