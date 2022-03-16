
import doce
from time import sleep
from pandas import DataFrame
import time
import numpy as np
from pathlib import Path

import fire
import os
import gin

from timbremetrics.metrics import TripletKNNAgreement, TripletKNNPenaltyAgreement

from train_cnn import run_train

if __name__ == "__main__":
  doce.run.run()

def set(args):
  experiment = doce.Experiment(
    name = 'jtfs-cnn',
    purpose = 'Medley DB Solos Classification with Hybrid Time-Frequency Scattering Networks',
    author = 'Cyrus Vahidi',
    address = 'c.vahidi@qmul.ac.uk',
    version = '0.1'
  )

  experiment.setPath('output', 'results/'+experiment.name+'/')

  experiment.addPlan('plan',
                     feature = ['cqt', 'scat1d', 'jtfs'],
                     c = np.array([1e-1, 1e-2, 1e-3]),
                     learn_c = [True, False])
  experiment.setMetrics(
    accuracy = ['mean*%', 'std%'],
  )
  experiment.instruments = []
  experiment._display.metricPrecision = 4

  experiment._display.bar = False

  return experiment

def step(setting, experiment):
  if os.path.exists(experiment.path.output+setting.id()+'_accuracy.npy'):
    return
  setting_triplet = np.zeros((len(experiment.datasets)))
  tic = time.time()
  print(setting.id())

  preprocess_gin_file(setting)

  results = run_metric_learning_ptl()
  for i, r in enumerate(results):
    setting_triplet[i] = r[TripletKNNAgreement.__name__]

  np.save(experiment.path.output+setting.id()+'_accuracy.npy', setting_triplet)
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