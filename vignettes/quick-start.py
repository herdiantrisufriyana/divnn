# Import modules and set GPU off
from  divnn import *
import os
import pickle
import pandas as pd
import numpy as np
import scipy as sp
import regex as re
import tensorflow as tf
import math
from dfply import *
from progressbar import ProgressBar
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from tensorflow import keras
from tensorflow.keras import layers, activations
from tensorflow.keras.models import model_from_json

# Create input example
input=input_example()

# Create TidySet
tidy_set=create_tidy_set(
    value=input['value']
    ,outcome=input['outcome']
    ,similarity=input['similarity']
    ,mapping=input['mapping']
    ,ontology=input['ontology']
  )

write_ts_tar_gz(tidy_set,'vignettes/quick-start-py/tidy_set_py')

tidy_set=read_ts_tar_gz('vignettes/quick-start-py/tidy_set_py.ts.tar.gz')

ontonet=ontonet_generator(tidy_set,path='vignettes/quick-start-py/ontonet_py')

json_file=open('vignettes/quick-start-py/ontonet_py.json','r')
ontonet=json_file.read()
json_file.close()
ontonet=model_from_json(ontonet)

ontonet.compile(
    optimizer=keras.optimizers.SGD(
      learning_rate=2**-6
      ,momentum=0.9
    )
    ,loss='mean_squared_error'
    ,loss_weights=np.repeat(0.3,len(ontonet.outputs)-1).tolist()+[1]
    ,metrics=['accuracy']
  )

def cb_lr_reduction(epoch,lr):
  lr_factor=0.1
  if epoch in [29,59,79]: lr=lr*lr_factor
  return lr
cb_lr_reduction=keras.callbacks.LearningRateScheduler(cb_lr_reduction)

cb_early_stopping=keras.callbacks.EarlyStopping(
    monitor='val_loss'
    ,mode='min'
    ,min_delta=0.001
    ,patience=30
    ,restore_best_weights=True
  )

np.random.seed(33)
index=np.random.choice(
    np.arange(exprs(tidy_set).shape[1])
    ,size=exprs(tidy_set).shape[1]
    ,replace=False
  ).tolist()

test_i=np.random.choice(
    index
    ,size=np.round(0.2*len(index)).astype(int)
    ,replace=False
  ).tolist()

val_i=[]
for i in index:
  if not i in [index[j] for j in test_i]:
    val_i.append(i)
val_i=np.random.choice(
    val_i
    ,size=np.round(0.2*len(val_i)).astype(int)
    ,replace=False
  ).tolist()

train_i=[]
for i in index:
  if not i in [index[j] for j in (test_i+val_i)]:
    train_i.append(i)

history=ontonet.fit_generator(
    generator=ontoarray_generator(
        tidy_set
        ,[index[i] for i in train_i]
        ,batch_size=32
      )
    ,steps_per_epoch=math.ceil(len(train_i)/32)
    ,validation_data=ontoarray_generator(
        tidy_set
        ,[index[i] for i in val_i]
        ,batch_size=32
      )
    ,validation_steps=math.ceil(len(val_i)/32)
    ,epochs=100
    ,callbacks=[cb_lr_reduction,cb_early_stopping]
    ,verbose=1
  )

ontonet.save_weights('vignettes/quick-start-py/ontonet_py.h5')

ontonet.load_weights('vignettes/quick-start-py/ontonet_py.h5')

with open('vignettes/quick-start-py/history_py.pkl','wb') as f:
  pickle.dump(history.history,f)

with open('vignettes/quick-start-py/history_py.pkl','rb') as f:
  history=pickle.load(f)

np.random.seed(33)
evaluation={}
for i in np.arange(30):
  test_i_boot=np.random.choice(
    test_i
    ,size=len(test_i)
    ,replace=True
  ).tolist()
  evaluation[i]=ontonet.evaluate_generator(
    generator=ontoarray_generator(
        tidy_set
        ,[index[i] for i in test_i_boot]
        ,batch_size=32
      )
    ,steps=math.ceil(len(test_i)/32)
  )

with open('vignettes/quick-start-py/evaluation_py.pkl','wb') as f:
  pickle.dump(evaluation,f)

with open('vignettes/quick-start-py/evaluation_py.pkl','rb') as f:
  evaluation=pickle.load(f)

I=[]
for i in history.keys():
  if 'val' in i:
    I.append(re.sub('val','test',i))

Y=pd.DataFrame()
for i in evaluation.keys():
  Y=Y.append(
      pd.DataFrame.from_dict({'boot':i,'metric':I,'result':evaluation[i]})
    )

Z=pd.DataFrame()
for i in I:
  K=Y.result[Y.metric==i].to_list()
  Z=Z.append(
      pd.DataFrame.from_dict({
        'mean':[np.mean(K)]
        ,'lb':[np.mean(K)-1.96*np.std(K)/np.sqrt(30)]
        ,'ub':[np.mean(K)+1.96*np.std(K)/np.sqrt(30)]
      })
    )
Z=Z.reset_index(inplace=False)
del Z['index']
results=Z
del i, I, Y, Z

results
