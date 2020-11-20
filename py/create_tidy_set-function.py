import pandas as pd
import numpy as np
import regex as re
import math
from dfply import *
from progressbar import ProgressBar

def create_tidy_set(value
                    ,outcome
                    ,similarity
                    ,mapping
                    ,ontology
                    ,ranked=True
                    ,dims=7
                    ,decreasing=False
                    ,seed_num=33):
  
  """
  Make a TidySet for visible neural network (VNN) modeling
  
  This function create a TidySet, an ExpressionSet class to orchestrate five
  data into a single set of three tables.
  
  :param value: Instance-feature value, a pandas data frame with rows for
  instances and columns for features. All rows in value should have names. All
  values should be floating numbers.
  :param outcome: Outcome, a single-column pandas data frame of binary integers
  with the same rows as the instances. The row numbers and the order of outcome
  should be the same with those of value. Value  of 0 and 1 should refer to
  non-event and event outcome, respectively.
  :param similarity: Feature similarity, a square pandas data frame of floating
  numbers containing feature-feature similarity measures.
  :param mapping: Feature three-dimensional mapping, a pandas data frame of
  floating numbers with rows for features and three columns for three dimensions
  where the features are mapped onto.
  :param ontology: Ontology, a pandas data frame with rows for ontologies and
  four columns for source, target, similarity, and relation. Feature (source)-
  ontology (target) relation should be annotated as 'feature', while ontology-
  ontology relation should be annotated as 'is_a'. To differentiate between
  feature and ontology names, a prefix of 'ONT:' precedes an ontology name. All
  columns except similarity in ontology should be strings. Similarity (a
  floating number) is a minimum threshold by which either features or ontologies
  (source) belong to an ontology (target).
  :return: output TidySet, an ExpressionSet with three tables. Instance-feature
  value and outcome pandas data frame are compiled as a phenotype pandas data
  frame with rows for instances and columns for features and outcome. Instance-
  feature value and feature three-dimensional mapping pandas data frame are
  compiled as an expression two-dimensional array with rows for positions of
  features and columns for instances. The mapping, similarity, and ontology
  pandas data frame are compiled as a feature pandas data frame with rows for
  positions of features and columns for feature names and ontological relations.
  For easier access, the similarity pandas data frame, ontomap four-dimensional
  numpy array, ontotype dictionary of pandas data frame, and ontology pandas
  data frame are included in experiment notes that can be called using function
  of notes.
  """
  
  pb=ProgressBar(8)
  tick=0
  pb.start()
  
  # Leibniz formula for pi
  # https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80
  # pi=1
  # for i in range(1,int(10e+6)):
  #   pi+=((-1)**i)*(1/(2*i+1))
  # pi=pi*4
  
  tick+=1
  pb.update(tick) #1
  
  def rotate_2_col_mat(X,angle):
    angle=(math.pi/180*angle)*-1
    M=np.array([
        math.cos(angle)
        ,math.sin(angle)
        ,-math.sin(angle)
        ,math.cos(angle)
      ])
    M=M.reshape(2,2)
    M=np.dot(X.to_numpy(),M)
    M=pd.DataFrame(
      M
      ,index=X.index.values.tolist()
      ,columns=X.columns.values.tolist()
    )
    return M
  
  def create_fmap(mapping,similarity,angle,ranked=ranked,dims=dims):
    data=pd.DataFrame.from_dict(
       {'feature':similarity.index.values.tolist()
        ,'dim1':mapping.iloc[:,0].tolist()
        ,'dim2':mapping.iloc[:,1].tolist()
        ,'dim3':mapping.iloc[:,2].tolist()}
      )
    
    if ranked:
      data=data >> arrange(X.dim1) >> mutate(dim1=np.arange(data.shape[0])+1)
      data=data >> arrange(X.dim2) >> mutate(dim2=np.arange(data.shape[0])+1)
    
    data=data >> arrange(X.dim1)
    data=data >> mutate(resize_x=np.round_(np.linspace(1,dims,data.shape[0])))
    data.resize_x=data.resize_x.astype(int)
    data=data >> arrange(X.dim2)
    data=data >> mutate(resize_y=np.round_(np.linspace(1,dims,data.shape[0])))
    data.resize_y=data.resize_y.astype(int)
    
    data2=pd.DataFrame.from_dict({'rot_x':data.resize_x,'rot_y':data.resize_y})
    data2=rotate_2_col_mat(data2,angle)
    data=data >> bind_cols(data2)
    del data2
    
    data=data >> arrange(X.rot_x)
    data=data >> mutate(x=np.round_(np.linspace(1,dims,data.shape[0])))
    data.x=data.x.astype(int)
    
    data=data >> arrange(X.rot_y)
    data=data >> mutate(y=np.round_(np.linspace(1,dims,data.shape[0])))
    data.y=data.y.astype(int)
    
    data=data >> arrange(X.dim3)
    
    data2={}
    data2['X']=data >> select(X.x,X.y)
    data2['Y']=data2['X'].drop_duplicates()
    data2['X']=np.arange(data2['Y'].shape[0])
    data2['Z']=data
    for i in data2['X']:
      data2['result']=data2['Z'] >> mask(X.x==data2['Z'].x[i])
      data2['result']=data2['result'] >> mask(X.y==data2['Z'].y[i])
      data2['result.z']=np.arange(data2['result'].shape[0])+1
      data2['result.z']=data2['result.z'].tolist()
      data2['result']=data2['result'] >> mutate(z=data2['result.z'])
      if i==0:
        data2['results']=data2['result']
      else:
        data2['results']=pd.DataFrame.append(data2['results'],data2['result'])
    data=data2['results']
    del data2
    
    data2a=similarity.index.values
    data2b=data >> mask(data.feature.isin(data2a))
    data2a=pd.DataFrame.from_dict({'feature':data2a})
    data2a=data2a >> mask(data2a.feature.isin(data2b['feature'].to_numpy()))
    data=data2a >> left_join(data2b,by='feature')
    del data2a, data2b
    
    data=data.set_index('feature')
    data=data >> select(X.x,X.y,X.z)
    data=data >> arrange(X.z,X.y,X.x)
    
    return data
  
  def order_angle_by_channel(mapping
                             ,similarity
                             ,ranked=ranked
                             ,dims=dims
                             ,decreasing=False):
    angles=np.arange(360)+1
    for i in angles:
      if i==1:
        data_=create_fmap(mapping,similarity,i,ranked,dims)
        data=[np.max(data_['z'])]
      else:
        data_=create_fmap(mapping,similarity,i,ranked,dims)
        data.append(np.max(data_['z']))
    
    data=pd.DataFrame.from_dict({'angle':angles,'channel':np.array(data)})
    data=data >> arrange(X.channel,ascending=decreasing==False)
    return data
  
  tick+=1
  pb.update(tick) #2
  np.random.seed(seed_num)
  angle=order_angle_by_channel(mapping,similarity,ranked,dims,decreasing)
  angle=angle >> mask(X.channel==np.min(angle['channel']))
  angle=angle['angle'].values
  angle=np.random.choice(np.arange(angle.shape[0]).tolist(),1,False)
  
  tick+=1
  pb.update(tick) #3
  fmap=create_fmap(mapping,similarity,angle,ranked,dims)
  
  fval=value[fmap.index.values].to_numpy()
  fval=pd.DataFrame(fval,index=value.index.values,columns=value.columns.values)
  
  fboth=fmap >> summarize_each([np.max],X.x,X.y,X.z)
  fboth=fboth.to_numpy()
  data=[]
  for i in np.arange(fboth.shape[1]):
    data_=np.arange(fboth[:,i])+1
    data.append(data_.tolist())
    del data_
  
  fboth=np.meshgrid(data[0],data[1],data[2])
  del data
  fboth=np.array(fboth).T.reshape(-1,3)
  fboth=pd.DataFrame(fboth,columns=fmap.columns.values)
  fboth=fboth >> arrange(X.z,X.y,X.x)
  
  fboth=fboth >> left_join(fmap.reset_index(inplace=False),by=['x','y','z'])
  
  idx=[]
  for i in fboth['feature'].values.tolist():
    idx.append(str(i)!='nan')
  
  fval=fval[fboth['feature'][idx]].to_numpy()
  fval=np.matrix.transpose(fval)
  fval=pd.DataFrame(fval,index=fboth['feature'][idx],columns=value.index.values)
  
  fboth=fboth >> left_join(fval.reset_index(inplace=False),by='feature')
  
  fboth=fboth >> mutate(x_='x') >> unite('x',['x_','x'],remove=False,sep='')
  fboth=fboth >> select(~X.x_)
  fboth=fboth >> unite('pos_id',['x','y'],remove=True,sep='y')
  fboth=fboth >> unite('pos_id',['pos_id','z'],remove=False,sep='z')
  fboth=fboth >> select(~X.z)
  
  ori_ontology=ontology
  
  def str_detect(string,pattern):
    match=[]
    for i in string:
      match.append('ONT:' in i)
    return match
  
  while np.sum(str_detect(ontology['source'],'ONT:'))>0:
    
    data=ontology >> mask(X.relation=='feature')
    for i in np.arange(ontology.shape[0]):
      if 'ONT:' in ontology['source'][i]:
        data2=data >> mask(X.target==ontology['source'][i])
        if data2.shape[0]>0:
          data_=pd.DataFrame.from_dict({
              'source':data2['source']
              ,'target':ontology['target'][i]
              ,'similarity':ontology['similarity'][i]
              ,'relation':'feature'
            })
        else:
          data_=ontology.iloc[i,:]
      else:
        data_=ontology.iloc[i,:]
      
      if i==0:
        data2=data_
      else:
        data2=data.append(data_)
    ontology=data2
  del data_, data, data2
  
  tick+=1
  pb.update(tick) #4
  adata=fboth >> select(~X.feature)
  adata=adata.set_index('pos_id')
  adata=adata.fillna(0)
  
  pdata=value >> mutate(outcome=outcome.astype(int))
  pdata=pdata >> select(X.outcome,fmap.index.values.tolist())
  
  fdata=fboth >> select(X.pos_id,X.feature)
  fdata2=ontology >> select(X.source,X.target)
  fdata2=fdata2.drop_duplicates()
  fdata2=fdata2 >> separate(X.target,['t1','t2'])
  fdata2=fdata2 >> mutate(t1='ONT') >>unite('target',['t1','t2'],sep='')
  fdata2=fdata2 >> mutate(included=1) >> spread(X.target,X.included)
  fdata2=fdata2 >> rename(feature=X.source)
  fdata=fdata >> left_join(fdata2,by='feature')
  del fdata2
  fdata=fdata.set_index('pos_id')
  
  tick+=1
  pb.update(tick) #5
  ontomap=adata.transpose()
  for i in np.arange(ontomap.columns.values.shape[0]):
    dim=re.split('x|y|z',ontomap.columns.values[i])
    if i>0:
      dim[1]=np.max([int(dim[1]),int(dim_[1])])
      dim[2]=np.max([int(dim[2]),int(dim_[2])])
      dim[3]=np.max([int(dim[3]),int(dim_[3])])
    dim_=dim
  del dim_
  ontomap=ontomap.to_numpy()
  ontomap=ontomap.reshape(ontomap.shape[0]*ontomap.shape[1])
  ontomap=np.array(ontomap)
  ontomap=ontomap.reshape(adata.shape[1],dim[2],dim[1],dim[3])
  
  tick+=1
  pb.update(tick) #6
  ontotype={}
  for i in np.arange(fdata.shape[1]-1):
    data0=fdata >> select(~X.feature)
    data=data0.iloc[:,i]
    data=data.reset_index(inplace=False)
    data=data.rename(columns={data.columns.values[1]:'ontotype'})
    data=data >> mask(X.ontotype==1)
    data=data >> left_join(fdata.reset_index(inplace=False),var='pos_id')
    data=data  >> select(X.pos_id,X.feature)
    
    data2=data >> separate(X.pos_id,['a','x','y','z'],sep='x|y|z')
    data2=data2[['feature','x','y','z']].set_index('feature')
    
    ontotype[data0.columns.values[i]]=data2
  
  ontotype['root']=fdata.reset_index(inplace=False) >> select(X.pos_id,X.feature)
  ontotype['root']=ontotype['root'] >> mutate(f_str=X.feature.astype(str))
  ontotype['root']=ontotype['root'] >> mask(X.f_str!='nan') >> select(~X.f_str)
  ontotype['root']=ontotype['root'] >> separate(
    X.pos_id
    ,['a','x','y','z']
    ,sep='x|y|z'
  )
  ontotype['root']=ontotype['root'][['feature','x','y','z']].set_index('feature')
  
  data2a=fmap.reset_index(inplace=False)
  data2b=similarity[data2a['feature'].to_numpy().tolist()]
  data2b=data2b.reset_index(inplace=False)
  data2a=data2a >> rename(index=X.feature)
  similarity=data2a >> left_join(data2b,by='index')
  del data2a, data2b
  similarity=similarity >> select(~X.x,~X.y,~X.z)
  similarity=similarity.set_index('index')
  similarity.index.name=None
  
  tick+=1
  pb.update(tick) #7
  adata.index.name=None
  fdata.index.name=None
  ori_ontology.index=pd.Index(np.arange(ori_ontology.shape[0]))
  output=ExpressionSet(
      assayData=adata.to_numpy()
      ,phenoData=AnnotatedDataFrame(pdata)
      ,featureData=AnnotatedDataFrame(fdata)
      ,experimentData=
        MIAME(
          other={
              'similarity':similarity
              ,'ontomap':ontomap
              ,'ontotype':ontotype
              ,'ontology':ori_ontology
            }
        )
    )
  
  tick+=1
  pb.update(tick) #8
  return output
