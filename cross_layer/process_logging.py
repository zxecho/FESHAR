import json
import os.path
import logging
import datetime
from infrastructure_layer.basic_utils import mkdir


def save_args_config(args):
    result_path = "./results/{}/".format(args.save_folder_name)
    mkdir(result_path)
    argsDict = args.__dict__
    with open(os.path.join(result_path, 'setting.txt'), 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')


def save2json(fpt, data, name):
    mkdir(fpt)
    with open(fpt + '/' + name + '.json', 'w+') as jsonf:
        json.dump(data, jsonf)


def get_logger(dir_pth):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(os.path.join(dir_pth, 'log_{}.txt'.format(current_time)), mode='a')
    fileHandler.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    return logger