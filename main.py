import torch
print (torch.__version__)


#TODO:
#1. move all hyperparameters to a config file (use yaml)
#2. shuffle train data after every epoch
#3. add residual blocks
#4. add data augmentation
#5. add early stopping
#6. add tensorboard to show learning curves (loss, accuracy, error)
#7. add visualization for each images with actual and predicted
#8. try GlobalAveragePooling instead of flatten + dense layers


#Optional TODO
# add learning rate decay, dropout
# adopt Tensorflow 2.4 best practices
# play with different architectures and hyperparameters to see what works best and why
# load data from disk for every iterations
# code cleanup

import os, sys, errno, shutil
from pytorch_model import PytorchModel

from datetime import datetime
from dataset import Dataset

def createdir_safe(dirname):
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname, 0o700)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def get_script_path():
    if hasattr(sys, 'ps1') or sys.flags.interactive:  #python -i
        return os.getcwd()
    else:
        return os.path.dirname(os.path.realpath(__file__))

def create_log_dir():
    path = get_script_path()
    logdir = os.path.join(path, 'logs', str(datetime.now().strftime("%Y%m%d_%H%M%S")))
    createdir_safe(logdir)
    return logdir

def remove_previous_tests():
    path = get_script_path()
    dirpath = os.path.join(path, 'logs')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)


if __name__ == '__main__':
    dataset = Dataset()
    dataset.prepare_dataset()
    pytorch_model = PytorchModel(dataset)
    remove_previous_tests()
    pytorch_model.train(create_log_dir())
    pytorch_model.predict_dataset()