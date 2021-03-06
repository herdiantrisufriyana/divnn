import math
import regex as re
from divnn.ExpressionSet import *
from dfply import X, mutate, mask, select, rename, left_join
from tensorflow import keras
from tensorflow.keras import layers, activations
from progressbar import ProgressBar

def ontonet(TidySet,path=None,init_seed=888,init2_seed=9999,l2_norm=0):
  
  """
  Make an ontonet generator for visible neural network (VNN) modeling
  
  This function create a function that generate a Keras Convolutional Neural
  Network (CNN) model with a specific layer architecture for each path in the
  hierarchy of the given ontology.
  
  :param TidySet: TidySet, an ExpressionSet with three tables.
  :param path: A character of file path if the model json file is saved.
  :param init_seed: An integer of random seed for ReLU initializer.
  :param init2_seed: An integer of random seed for tanh initializer.
  :param l2_norm: A floating number of L2-norm regularization factor.
  :return: output Keras model object, a pointer to Keras model object in python
  environment, which will be an input to train VNN model using Keras R package.
  """
  
  # Recall ontomap
  ontomap=notes(TidySet.experimentData)['ontomap']
  
  # Recall ontotype
  ontotype=notes(TidySet.experimentData)['ontotype']
  
  # Recall ontology
  ontology=notes(TidySet.experimentData)['ontology']
  
  # Build a function to insert an inception module along with a pre-activation residual unit
  def layer_inception_resnet(object
                             ,residue
                             ,filters
                             ,kernel_initializer
                             ,name=None):
    
    def namer(name,suffix):
      if not name:
        return suffix
      else:
        return name+suffix
    
    pre_activation=layers.BatchNormalization(
        name=namer(name,'_pre_bn')
      )(object)
    pre_activation=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_pre_ac')
      )(pre_activation)
    
    tower_1=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(3,3)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower1_cv')
      )(pre_activation)
    tower_1=layers.BatchNormalization(
        name=namer(name,'_tower1_bn')
      )(tower_1)
    tower_1=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_towe1_ac')
      )(tower_1)
    
    tower_2=layers.MaxPool2D(
        pool_size=(3,3)
        ,strides=(1,1)
        ,padding='same'
        ,name=namer(name,'_tower2a_mp')
      )(pre_activation)
    tower_2=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(1,1)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower2b_cv')
      )(tower_2)
    tower_2=layers.BatchNormalization(
        name=namer(name,'_tower2b_bn')
      )(tower_2)
    tower_2=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower2b_ac')
      )(tower_2)
    
    tower_3a=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(1,1)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower3a_cv')
      )(pre_activation)
    tower_3a=layers.BatchNormalization(
        name=namer(name,'_tower3a_bn')
      )(tower_3a)
    tower_3a=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower3a_ac')
      )(tower_3a)
    
    tower_3b1=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(1,3)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower3b1_cv')
      )(tower_3a)
    tower_3b1=layers.BatchNormalization(
        name=namer(name,'_tower3b1_bn')
      )(tower_3b1)
    tower_3b1=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower3b1_ac')
      )(tower_3b1)
    
    tower_3b2=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(3,1)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower3b2_cv')
      )(tower_3a)
    tower_3b2=layers.BatchNormalization(
        name=namer(name,'_tower3b2_bn')
      )(tower_3b2)
    tower_3b2=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower3b2_ac')
      )(tower_3b2)
    
    tower_4b=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(1,1)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower4a_cv')
      )(pre_activation)
    tower_4b=layers.BatchNormalization(
        name=namer(name,'_tower4a_bn')
      )(tower_4b)
    tower_4b=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower4a_ac')
      )(tower_4b)
    tower_4b=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(3,3)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower4b_cv')
      )(tower_4b)
    tower_4b=layers.BatchNormalization(
        name=namer(name,'_tower4b_bn')
      )(tower_4b)
    tower_4b=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower4b_ac')
      )(tower_4b)
    
    tower_4c1=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(1,3)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower4c1_cv')
      )(tower_4b)
    tower_4c1=layers.BatchNormalization(
        name=namer(name,'_tower4c1_bn')
      )(tower_4c1)
    tower_4c1=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower4c1_ac')
      )(tower_4c1)
    
    tower_4c2=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(3,1)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_tower4c2_cv')
      )(tower_4b)
    tower_4c2=layers.BatchNormalization(
        name=namer(name,'_tower4c2_bn')
      )(tower_4c2)
    tower_4c2=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_tower4c2_ac')
      )(tower_4c2)
    
    towers=layers.Concatenate(
        axis=-1
        ,name=namer(name,'_dc')
      )([
        tower_1
        ,tower_2
        ,tower_3b1,tower_3b2
        ,tower_4c1,tower_4c2
      ])
    
    scaling=layers.SeparableConv2D(
        filters=residue.shape[3]
        ,kernel_size=(1,1)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=namer(name,'_sc_cv')
      )(towers)
    
    inception_resnet=layers.Add(
        name=name
      )([scaling,residue])
    
    return inception_resnet
  
  # Build a function to insert auxiliary output layers
  def layer_aux_output(object
                       ,filters
                       ,units
                       ,kernel_initializer
                       ,kernel_initializer2
                       ,activity_regularizer
                       ,name=None):
    
    def namer(name,suffix):
      if not name:
        return suffix
      else:
        return name+suffix
    
    aux_output=layers.AveragePooling2D(
        pool_size=(5,5)
        ,strides=(3,3)
        ,padding='valid'
        ,name=name+'_ap'
      )(object)
    
    aux_output=layers.SeparableConv2D(
        filters=filters
        ,kernel_size=(1,1)
        ,strides=(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=name+'_hl1_cv'
      )(aux_output)
    aux_output=layers.BatchNormalization(
        name=namer(name,'_hl1_bn')
      )(aux_output)
    aux_output=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_hl1_ac')
      )(aux_output)
    
    aux_output=layers.Dense(
      units=units
      ,kernel_initializer=kernel_initializer
      ,name=namer(name,'_hl2_de')
    )(aux_output)
    aux_output=layers.BatchNormalization(
        name=namer(name,'_hl2_bn')
      )(aux_output)
    aux_output=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_hl2_ac')
      )(aux_output)
    
    aux_output=layers.Dense(
      units=units
      ,kernel_initializer=kernel_initializer
      ,name=namer(name,'_hl3_de')
    )(aux_output)
    aux_output=layers.BatchNormalization(
        name=namer(name,'_hl3_bn')
      )(aux_output)
    aux_output=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_hl3_ac')
      )(aux_output)
    
    aux_output=layers.Dense(
      units=2
      ,activation='tanh'
      ,kernel_initializer=kernel_initializer2
      ,name=namer(name,'_ao_tn')
    )(aux_output)
    aux_output=layers.BatchNormalization(
        name=namer(name,'_ao_bn')
      )(aux_output)
    aux_output=layers.Flatten(
      name=namer(name,'_ao_fl')
    )(aux_output)
    aux_output=layers.Dense(
      units=1
      ,activation='sigmoid'
      ,kernel_initializer=kernel_initializer2
      ,activity_regularizer=activity_regularizer
      ,name=name
    )(aux_output)
    
    return aux_output
  
  # Build a function to insert output layers
  def layer_output(object
                       ,units
                       ,kernel_initializer
                       ,kernel_initializer2
                       ,activity_regularizer
                       ,name=None):
    
    def namer(name,suffix):
      if not name:
        return suffix
      else:
        return name+suffix
    
    output=layers.AveragePooling2D(
        pool_size=(7,7)
        ,strides=(1,1)
        ,padding='valid'
        ,name=namer(name,'_ap')
      )(object)
    
    output=layers.Dense(
      units=units
      ,kernel_initializer=kernel_initializer
      ,name=namer(name,'_hl1_de')
    )(output)
    output=layers.BatchNormalization(
        name=namer(name,'_hl1_bn')
      )(output)
    output=layers.Activation(
        activation=activations.relu
        ,name=namer(name,'_hl1_ac')
      )(output)
    
    output=layers.Dense(
      units=2
      ,activation='tanh'
      ,kernel_initializer=kernel_initializer2
      ,name=namer(name,'_mo_tn')
    )(output)
    output=layers.BatchNormalization(
        name=namer(name,'_mo_bn')
      )(output)
    output=layers.Flatten(
      name=namer(name,'_mo_fl')
    )(output)
    output=layers.Dense(
      units=1
      ,activation='sigmoid'
      ,kernel_initializer=kernel_initializer2
      ,activity_regularizer=activity_regularizer
      ,name=name
    )(output)
    
    return output
  
  # Build a generator function to construct ontonet
  keras.backend.clear_session()
  init=keras.initializers.he_uniform(seed=init_seed)
  init2=keras.initializers.glorot_uniform(seed=init2_seed)
  
  reg=keras.regularizers.l2(l2_norm)
  
  feature=ontology
  while any('ONT' in s for s in feature.source):
    Z=pd.DataFrame()
    for i in np.arange(feature.shape[0]):
      if 'ONT' in feature.source[i]:
        Y=feature >> mask(X.target==feature.source[i], X.relation=='feature')
        if Y.shape[0]>0:
          Z=Z.append(pd.DataFrame({
              'source':Y.source
              ,'target':feature.target[i]
              ,'similarity':feature.similarity[i]
              ,'relation':'feature'
            }))
        else:
          Z=Z.append(feature.iloc[i,])
      else:
        Z=Z.append(feature.iloc[i,])
    Z=Z >> select(X.source,X.target,X.similarity,X.relation)
    feature=Z
    feature=feature.reset_index(inplace=False)
  
  Y=feature >> select(X.source,X.target)
  Z=feature >> select(X.source)
  Z=Z.drop_duplicates()
  Z=Z >> mutate(target='root')
  feature=Y.append(Z)
  
  Y=feature.target.drop_duplicates()
  Z=pd.DataFrame()
  for i in np.arange(Y.shape[0]):
    K=feature >> mask(X.target==Y.iloc[i])
    Z=Z.append(pd.DataFrame.from_dict({'target':[Y.iloc[i]],'n':[K.shape[0]]}))
  Z.index=np.arange(Y.shape[0])
  Y=[]
  for i in Z.target.to_numpy():
    Y.append(re.sub('ONT\\:','ONT',i))
  feature=Z >> mutate(oid=Y) >> select(X.oid,X.n)
  del Y, Z, K
  
  I=[]
  for i in np.arange(ontology.shape[0]):
    if 'ONT:' in ontology.source[i]: I.append(i)
  Y=[]
  for i in ontology.iloc[I,].source.to_numpy():
    Y.append(re.sub('ONT\\:','ONT',i))
  Z=[]
  for i in ontology.iloc[I,].target.to_numpy():
    Z.append(re.sub('ONT\\:','ONT',i))
  
  Y=pd.DataFrame.from_dict({'from':Y,'to':Z})
  I=[]
  for i in Y['to']:
    if not i in ' '.join(Y['from']): I.append(i)
  Z=pd.DataFrame.from_dict({'from':I,'to':'root'})
  Z=Z.drop_duplicates()
  hierarchy=Y.append(Z)
  hierarchy.index=np.arange(hierarchy.shape[0])
  I=feature >> rename(to=X.oid)
  hierarchy=hierarchy >> left_join(I,by='to')
  del I
  
  terminal_nodes=[]
  for i in hierarchy['from'].drop_duplicates():
    if not i in ' '.join(hierarchy['to']): terminal_nodes.append(i)
  
  non_terminal_nodes=hierarchy['to'].drop_duplicates().to_list()
  
  pb=ProgressBar(3+len(terminal_nodes)+len(non_terminal_nodes))
  tick=0
  pb.start()
  
  inputs=dict()
  for i in ontotype.keys():
    inputs[i]=keras.Input(
        shape=ontomap.shape[1:4]
        ,dtype='float32'
        ,name=i+'_input'
      )
  
  hiddens=dict()
  outputs=dict()
  
  for i in np.arange(len(terminal_nodes)):
    tick+=1
    pb.update(tick)
    
    A=terminal_nodes[i]
    B=A
    C=feature.n[feature.oid==A].values.tolist()[0]
    
    hiddens[A]=layer_inception_resnet(
      object=inputs[B]
      ,residue=inputs[A]
      ,filters=np.max([20,math.ceil(0.3*C)])
      ,kernel_initializer=init
      ,name=A+'_hidden'
    )
    
    outputs[A]=layer_aux_output(
        object=hiddens[A]
        ,filters=np.max([20,math.ceil(0.3*C)])
        ,units=np.max([20,math.ceil(0.3*C)])
        ,kernel_initializer=init
        ,kernel_initializer2=init2
        ,activity_regularizer=reg
        ,name=A
      )
  
  for i in np.arange(len(non_terminal_nodes)):
    tick+=1
    pb.update(tick)
    
    A=non_terminal_nodes[i]
    B=hierarchy['from'][hierarchy['to']==A].values.tolist()
    C=feature.n[feature.oid==A].values.tolist()[0]
    
    if len(B)==1:
      hiddens[A]=layer_inception_resnet(
          object=[hiddens[i] for i in B][0]
          ,residue=inputs[A]
          ,filters=np.max([20,math.ceil(0.3*C)])
          ,kernel_initializer=init
          ,name=A+'_hidden'
        )
    else:
      hiddens[A]=layers.Concatenate(
          axis=-1
          ,name=A+'_dc'
        )([hiddens[i] for i in B])
      hiddens[A]=layer_inception_resnet(
          object=hiddens[A]
          ,residue=inputs[A]
          ,filters=np.max([20,math.ceil(0.3*C)])
          ,kernel_initializer=init
          ,name=A+'_hidden'
        )
    
    if A!=non_terminal_nodes[len(non_terminal_nodes)-1]:
      outputs[A]=layer_aux_output(
          object=hiddens[A]
          ,filters=np.max([20,math.ceil(0.3*C)])
          ,units=np.max([20,math.ceil(0.3*C)])
          ,kernel_initializer=init
          ,kernel_initializer2=init2
          ,activity_regularizer=reg
          ,name=A
        )
    else:
      outputs[A]=layer_output(
          object=hiddens[A]
          ,units=np.max([20,math.ceil(0.3*C)])
          ,kernel_initializer=init
          ,kernel_initializer2=init2
          ,activity_regularizer=reg
          ,name=A
        )
  
  del i, A, B, C
  
  tick+=1
  pb.update(tick)
  model=keras.Model(inputs=inputs,outputs=outputs)
  
  tick+=1
  pb.update(tick)
  if not path:
    pass
  else:
    model_json=model.to_json()
    with open(path+'.json','w') as json_file:
      json_file.write(model_json)
  
  keras.backend.clear_session()
  
  if not path:
    pass
  else:
    print('\nOntonet has been saved to '+path+'.json')
  
  return model


def ontoarray(TidySet,index,batch_size):
  
  """
  Make an ontoarray generator for visible neural network (VNN) modeling
  
  This function create a function that generate a batch of ontoarray for
  training or testing a Keras Convolutional Neural Network (CNN) model using
  fit_generator, evaluate_generator, or predict_generator function from Keras R
  package.
  
  :param TidySet: TidySet, an ExpressionSet with three tables.
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
  ontomap=notes(TidySet.experimentData)['ontomap']
  
  # Recall ontotype
  ontotype=notes(TidySet.experimentData)['ontotype']
  
  # Recall outcome
  outcome=pData(TidySet).outcome
  
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
