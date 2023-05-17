import json
import os.path

from infrastructure_layer.basic_utils import mkdir


def save_args_config(args):
    result_path = "./results/{}/".format(args.save_folder_name)
    argsDict = args.__dict__
    with open(os.path.join(result_path, '{}_setting.txt'.format()), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def save2json(fpt, data, name):
    mkdir(fpt)
    with open(fpt + '/' + name + '.json', 'w+') as jsonf:
        json.dump(data, jsonf)