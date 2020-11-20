import os
import tarfile
import regex as re
from divnn.ExpressionSet import *
from dfply import *

def read_ts_tar_gz(path):
  
  """
  Read a .ts.tar.gz file to a TidySet
  
  This function read multiple files archived by tar with gzip compression
  to a TidySet.
  
  :param path: A character of .ts.tar.gz file path (include file extension).
  :return: output A TidySet, an ExpressionSet with three tables. Function of
  write_ts_tar_gz can write this file from the TidySet.
  """
  
  filename=path
  path=re.sub('.ts.tar.gz','',filename)
  os.mkdir(path)
  tar=tarfile.open(filename)
  tar.extractall(path)
  tar.close()
  
  f=open(path+'/others.txt','r')
  others=f.read()
  f.close()
  
  other=re.split('\n',others)
  elements=[]
  for i in np.arange(len(other)):
    if re.search('^>>',other[i]):
      elements.append(i)
  
  XX=np.arange(len(elements)).tolist()
  Y=elements
  Z=np.arange(len(other)).tolist()
  K=[]
  for i in XX:
    if i<(len(Y)-1):
      L=Z[(Y[i]+1):Y[i+1]]
    else:
      L=Z[(Y[i]+1):]
    K.append(L)
  
  XX=np.arange(len(K))
  Y=K
  Z=other
  K=[]
  for i in elements:
    K.append(re.sub('>>','',other[i]))
  M=dict()
  for i in XX:
    L=[]
    for j in Y[i]:
      L.append(Z[j])
    L=' '.join(L)
    M[K[i]]=L
  others=M
  del XX, Y, Z, K, L, M, i, j, f, other, elements
  
  adata=pd.read_csv(
    path+'/exprs.csv'
    ,names=re.split('\\s',others['sampleNames'])
  )
  adata.index=re.split('\\s',others['featureNames'])
  
  pdata_names=re.split('\\s',others['varLabels'])
  pdata_dtype=re.split('\\s',others['varClass'])
  pdata=dict()
  for i in np.arange(len(pdata_dtype)):
    if pdata_dtype[i]=='numeric':
      pdata[pdata_names[i]]='float64'
    elif pdata_dtype[i]=='integer':
      pdata[pdata_names[i]]='int64'
    elif pdata_dtype[i]=='factor':
      pdata[pdata_names[i]]='category'
    else:
      pdata[pdata_names[i]]='object'
  pdata=pd.read_csv(path+'/pData.csv',names=pdata_names,dtype=pdata)
  pdata.index=re.split('\\s',others['sampleNames'])
  string=re.split('\\s',others['varMetadata'])
  i=0
  for c in string:
    if string[i]=='NA': string[i]=np.NaN
    i+=1
  pmetadata=pd.DataFrame(
      string
      ,index=re.split('\\s',others['varLabels'])
      ,columns=['labelDescription']
    )
  pdata=AnnotatedDataFrame(pdata,pmetadata)
  
  fdata_names=re.split('\\s',others['fvarLabels'])
  fdata_dtype=re.split('\\s',others['fvarClass'])
  fdata=dict()
  for i in np.arange(len(fdata_dtype)):
    if fdata_dtype[i]=='numeric':
      fdata[fdata_names[i]]='float64'
    elif fdata_dtype[i]=='integer':
      fdata[fdata_names[i]]='int64'
    elif fdata_dtype[i]=='factor':
      fdata[fdata_names[i]]='category'
    else:
      fdata[fdata_names[i]]='object'
  fdata=pd.read_csv(path+'/fData.csv',names=fdata_names,dtype=fdata)
  fdata.index=re.split('\\s',others['featureNames'])
  fdata.index.name='pos_id'
  
  sim_names=re.split('\\s',others['simNames'])
  sim_dtype=dict()
  for i in np.arange(len(sim_names)):
    sim_dtype[sim_names[i]]='float64'
  similarity=pd.read_csv(path+'/similarity.csv',names=sim_names,dtype=sim_dtype)
  similarity.index=sim_names
  
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
  
  string=re.split('\\s',others['fvarMetadata'])
  i=0
  for c in string:
    if string[i]=='NA': string[i]=np.NaN
    i+=1
  fmetadata=pd.DataFrame(
      string
      ,index=re.split('\\s',others['fvarLabels'])
      ,columns=['labelDescription']
    )
  fdata.index.name=None
  fdata=AnnotatedDataFrame(fdata,fmetadata)
  
  notes(tidy_set.experimentData)['ontology']
  
  ontology_names=re.split('\\s',others['ontoNames'])
  ontology_dtype=re.split('\\s',others['ontoClass'])
  ontology=dict()
  for i in np.arange(len(ontology_dtype)):
    if ontology_dtype[i]=='numeric':
      ontology[ontology_names[i]]='float64'
    elif ontology_dtype[i]=='integer':
      ontology[ontology_names[i]]='int64'
    elif ontology_dtype[i]=='factor':
      ontology[ontology_names[i]]='category'
    else:
      ontology[ontology_names[i]]='object'
  ontology=pd.read_csv(path+'/ontology.csv',names=ontology_names,dtype=ontology)
  
  xData=MIAME(
      name=others['name']
      ,lab=others['lab']
      ,contact=others['contact']
      ,title=others['title']
      ,abstract=others['abstract']
      ,url=others['url']
      ,pubMedIds=others['pubMedIds']
      ,other={
          'similarity':similarity
          ,'ontomap':ontomap
          ,'ontotype':ontotype
          ,'ontology':ontology
        }
    )
  
  for i in os.listdir(path):
    os.remove(path+'/'+i)
  os.rmdir(path)
  
  if re.match('^( +)',others['annotation']):
    annot=''
  else:
    annot=re.sub(' +',' ',others['annotation'])
  
  eset=ExpressionSet(
      assayData=adata.to_numpy()
      ,phenoData=pdata
      ,featureData=fdata
      ,experimentData=xData
      ,annotation=annot
    )
  
  return eset
