import copy

import numpy as np


def iterator_item_to_string(item):
    name = item[0]
    name += "_" + item[1]
    name += "_" + item[2][1]
    name += "_" + item[3][1]
    if item[4] == "True":
        if item[5][0][0] != "dummy":
            name += "_Lag_"
            for operator in item[5]:
                name += operator[0] + "(" + str(np.round(operator[2], 2)) + ")"
    elif item[4] == "Series":
        if item[5][0][0] != "dummy":
            name += "_LagSeries_"
            for operator in item[5]:
                name += operator[0] + "_" + str(item[6]) + "x" + str(np.round(item[7], 2)) + "_"
    elif item[4] == "AUGSeries":
        if item[5][0][0] != "dummy":
            name += "_LagAUGSeries_"
            for operator in item[5]:
                name += operator[0] + "_" + str(item[6]) + "x" + str(np.round(item[7], 2))
                name += "_lamb" + str(np.round(item[8], 2)) + "_"

    return name


def from_item_iter_to_option(options, item):
    option = copy.deepcopy(options)
    option["molecule"]["basis"] = item[0]
    option["var_form_type"] = item[1]
    option["quantum_instance"] = item[2][0]
    option["optimizer"] = item[3][0]

    if "hardware" in item[2][1]:
        option["hardware"] = True
    else:
        option["hardware"] = False

    if item[4] == "True" or item[4] == "False":
        if item[4] == "True":
            lag_bool = True
        else:
            lag_bool = False
        option["lagrange"]["active"] = lag_bool
        option["lagrange"]["series"] = False
        option["lagrange"]["augmented"] = False
        option["init_point"] = [None]
    elif item[4] == "Series":
        option["lagrange"]["active"] = True
        option["lagrange"]["series"] = True
        option["lagrange"]["augmented"] = False
    elif item[4] == "AUGSeries":
        option["lagrange"]["active"] = True
        option["lagrange"]["series"] = True
        option["lagrange"]["augmented"] = True

    option["lagrange"]["operators"] = item[5]

    if item[5][0][0] == "dummy":
        option["lagrange"]["active"] = False

    option["series"]["itermax"] = item[6]
    option["series"]["step"] = item[7]
    option["series"]["lamb"] = item[8]

    return option
