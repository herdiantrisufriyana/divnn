import numpy as np
import pandas as pd
import keras

def ontoarray_generator(tidy_set,index,batch_size):
  
  """
  Make an ontoarray generator for visible neural network (VNN) modeling
  
  This function create a function that generate a batch of ontoarray for
  training or testing a Keras Convolutional Neural Network (CNN) model using
  fit_generator, evaluate_generator, or predict_generator function from Keras R
  package.
  
  :param tidy_set: TidySet, an ExpressionSet with three tables.
  :param index: An integer vector of index to select which ontoarray will be
  used for training or testing.
  :param batch_size: An integer of how much samples are generated everytime
  this function runs. If all samples are generated,this function will loop over
  the samples.
  :return: output sample generator, a function for argument of generator
  in fit_generator, evaluate_generator, or predict_generator function from Keras
  R package.
  """
  
  # Recall ontomap
  ontomap=notes(tidy_set.experimentData)['ontomap']
  
  # Recall ontotype
  ontotype=notes(tidy_set.experimentData)['ontotype']
  
  # Recall outcome
  outcome=pData(tidy_set).outcome
  
  # Build a generator function to load a batch of ontoarray
  ontomap=ontomap[index]
  I=[]
  Y=[]
  for i in index:
    I.append(outcome.index.values.tolist()[i])
    Y.append(outcome.to_list()[i])
  outcome_names=I
  outcome=Y
  
  I=()
  for i in np.arange(len(ontomap.shape)):
    if i==0:
      I+=(1,)
    else:
      I+=(ontomap.shape[i],)
  ontofilter={}
  for i in ontotype.keys():
    Y=np.zeros(np.prod(ontomap.shape[1:])).reshape(I)
    for j in np.arange(ontotype[i].shape[0]):
      Z=ontotype[i].astype(int)
      Z=Z.iloc[j,:]-1
      Y[:,Z.x,Z.y,Z.z]=1
    ontofilter[i+'_input']=Y
  del I, Y, Z
  
  i=0
  
  while True:
    if ((i+batch_size)>(ontomap.shape[0]-1)): i=0
    rows=np.arange(i,np.min([i+batch_size,ontomap.shape[0]-1])).tolist()
    i=i+batch_size
    
    x_array={}
    for k in ontofilter.keys(): x_array[k]=ontomap[rows] * ontofilter[k]
    x_array=list(x_array.values())
    
    y_vector={}
    for k in ontotype.keys(): y_vector[k]=[outcome[j] for j in rows]
    y_vector=list(y_vector.values())
    
    batch=(x_array,y_vector)
    
    yield(batch)
