import json
import git
import os

from datetime import datetime

def getDateTimeString():
    now = datetime.now()
    return now.strftime("%d_%m_%H_%M")


def retriveJSONOptions(filename,dist,energies):
    JsonOptions = {
        'file': filename,
        'algo': {
            'name': 'vqe',
            'type': 'lagrangianvqe'
        },
        'backend': {
            'type': 'statevector_simulator',
            'noisy': 'None'
        },
        'dist': dist,
        'energies' : energies
    }
    return JsonOptions

def writeJson(JsonOptions):
    filename = "data/exp_"+getDateTimeString()
    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object.hexsha
    JsonOptions['commit'] = commit

    description = input("Inserisci una minima descrizione: ")
    JsonOptions['description'] = description

    json_obj = json.dumps(JsonOptions, indent = 4)

    while os.path.exists(filename):
        print('Esiste già un file con questo nome!!')
        filename = input('Inserisci un nuovo nome: ')

    with open(filename, "w") as out:
        out.write(json_obj)
