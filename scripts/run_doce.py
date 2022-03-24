
import doce
from time import sleep
from pandas import DataFrame
import time
import numpy as np
from pathlib import Path

import fire
import os
import gin

from train_cnn import run_train

if __name__ == "__main__":
  doce.cli.main()

def set(args):
  experiment = doce.Experiment(
    name = 'jtfs-cnn',
    purpose = 'Medley DB Solos Classification with Hybrid Time-Frequency Scattering Networks',
    author = 'Cyrus Vahidi',
    address = 'c.vahidi@qmul.ac.uk',
    version = '0.1'
  )

  experiment.setPath('output', 'results/'+experiment.name+'/')

  experiment.addPlan('cqt',
                     feature = ['cqt'])
  experiment.addPlan('scattering',
                     feature = ['scat1d', 'jtfs'],
                     c = np.array([1e-1, 1e-2, 1e-3]),
                     learn_adalog = [0, 1])
  experiment.setMetrics(
    accuracy = ['mean*%', 'std%'],
  )

  experiment.nb_runs = 3
  experiment._display.metricPrecision = 4

  experiment._display.bar = False

  return experiment

def step(setting, experiment):
  if os.path.exists(experiment.path.output+setting.id()+'_acc.npy'):
    return
  setting_acc = np.zeros((len(experiment.nb_runs)))
  setting_acc_instruments = np.zeros((experiment.n_instruments, len(experiment.nb_runs)))

  tic = time.time()
  print(setting.id())

  preprocess_gin_file(setting)

  results = run_train()
  setting_acc[i] = results['acc']
  for i, r in enumerate(results['acc_instruments']):
    setting_acc_instruments[i] = r


  np.save(experiment.path.output+setting.id()+'_acc.npy', setting_acc)
  np.save(experiment.path.output+setting.id()+'_acc_instruments.npy', setting_acc_instruments)
  duration = time.time()-tic
  np.save(experiment.path.output+setting.id()+'_duration.npy', duration)


def preprocess_gin_file(setting, 
                        gin_base ='gin/doce/config.gin',
                        gin_temp = 'gin/doce/setting.gin'):
  gin_base = os.path.join(os.getcwd(), gin_base)
  gin_temp = os.path.join(os.getcwd(), gin_temp)
  
  config = [f'MedleySolosDB.feature = \'{setting.feature}\'',
            f'MedleySolosClassifier.feature = \'{setting.feature}\'']

  open(temp, 'w').close() # clear temp
  with open(gin_base,'r') as f_template, open(gin_temp,'a') as f_temp:
    # write template to temp
    for line in f_template:
      f_temp.write(line + '\n')
    # write config
    for line in config:
      f_temp.write(line + '\n')

  gin_config_path = os.path.join(os.getcwd(), f'gin/doce/{setting.feature}.gin')
  gin.register_and_parse(gin_config_path)