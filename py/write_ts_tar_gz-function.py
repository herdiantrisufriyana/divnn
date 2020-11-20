import os
import tarfile
import pandas as pd
import numpy as np
import regex as re

def write_ts_tar_gz(tidy_set,path):
  
  """
  Write a .ts.tar.gz file from a TidySet
  
  This function write multiple files archived by tar with gzip compression
  from a TidySet.
  
  :param tidy_set: TidySet, an ExpressionSet with three tables.
  :param path: A character of .ts.tar.gz file path (do not include file
  extension).
  :return: output A .ts.tar.gz file containing exprs.csv, pData.csv,
  fData.csv, similarity.csv, ontology.csv, and others.txt. Function of
  read_ts_tar_gz can read this file back to a TidySet.
  """
  
  os.mkdir(path)
  os.chdir(path)
  
  pd.DataFrame(exprs(tidy_set)).to_csv(
    'exprs.csv'
    ,header=False
    ,index=False
  )
  
  pd.DataFrame(pData(tidy_set)).to_csv(
    'pData.csv'
    ,header=False
    ,index=False
  )
  
  pd.DataFrame(fData(tidy_set)).to_csv(
    'fData.csv'
    ,header=False
    ,index=False
  )
  
  pd.DataFrame(notes(tidy_set.experimentData)['similarity']).to_csv(
    'similarity.csv'
    ,header=False
    ,index=False
  )
  
  pd.DataFrame(notes(tidy_set.experimentData)['ontology']).to_csv(
    'ontology.csv'
    ,header=False
    ,index=False
  )
  
  def write(string,ncolumns,file):
    while len(string)>0:
      print(' '.join(string[0:ncolumns]),file=file)
      if len(string)>=ncolumns:
        i=ncolumns
      else:
        i=len(string)
      string=string[i:]
  
  with open('others.txt','a') as f:
    print('>>sampleNames',file=f)
    write(colnames(tidy_set),ncolumns=1000,file=f)
    
    print('>>varLabels',file=f)
    write(varLabels(tidy_set.phenoData),ncolumns=1000,file=f)
    
    print('>>varMetadata',file=f)
    string=varMetadata(tidy_set.phenoData).labelDescription.values.tolist()
    i=0
    for c in string:
      if np.isnan(string[i]): string[i]='NA'
      i+=1
    write(string,ncolumns=1000,file=f)
    
    print('>>varClass',file=f)
    string=[]
    for i in pData(tidy_set).dtypes.values.tolist():
      if 'float' in str(i):
        string.append('numeric')
      elif 'int' in str(i):
        string.append('integer')
      elif 'category' in str(i):
        string.append('factor')
      else:
        string.append('character')
    write(string,ncolumns=1000,file=f)
    
    print('>>featureNames',file=f)
    write(rownames(tidy_set),ncolumns=1000,file=f)
    
    print('>>fvarLabels',file=f)
    write(varLabels(tidy_set.featureData),ncolumns=1000,file=f)
    
    print('>>fvarMetadata',file=f)
    string=varMetadata(tidy_set.featureData).labelDescription.values.tolist()
    i=0
    for c in string:
      if np.isnan(string[i]): string[i]='NA'
      i+=1
    write(string,ncolumns=1000,file=f)
    
    print('>>fvarClass',file=f)
    string=[]
    for i in fData(tidy_set).dtypes.values.tolist():
      if 'float' in str(i):
        string.append('numeric')
      elif 'int' in str(i):
        string.append('integer')
      elif 'category' in str(i):
        string.append('factor')
      else:
        string.append('character')
    write(string,ncolumns=1000,file=f)
    
    print('>>simNames',file=f)
    string=notes(tidy_set.experimentData)['similarity'].index.values.tolist()
    write(string,ncolumns=1000,file=f)
    
    print('>>ontoNames',file=f)
    string=notes(tidy_set.experimentData)['ontology'].columns.values.tolist()
    write(string,ncolumns=1000,file=f)
    
    print('>>ontoClass',file=f)
    string=[]
    for i in notes(tidy_set.experimentData)['ontology'].dtypes.values.tolist():
      if 'float' in str(i):
        string.append('numeric')
      elif 'int' in str(i):
        string.append('integer')
      elif 'category' in str(i):
        string.append('factor')
      else:
        string.append('character')
    write(string,ncolumns=1000,file=f)
    
    for i in ['name','lab','contact','title','abstract','url','pubMedIds']:
      print('>>'+i,file=f)
      if i=='name': print(tidy_set.experimentData.name,file=f)
      elif i=='lab': print(tidy_set.experimentData.lab,file=f)
      elif i=='contact': print(tidy_set.experimentData.contact,file=f)
      elif i=='title': print(tidy_set.experimentData.title,file=f)
      elif i=='abstract': print(tidy_set.experimentData.abstract,file=f)
      elif i=='url': print(tidy_set.experimentData.url,file=f)
      elif i=='pubMedIds': print(tidy_set.experimentData.pubMedIds,file=f)
    
    print('>>annotation',file=f)
    print(annotation(tidy_set),file=f,end='')
  
  path2=re.split('/',path)
  path2=path2[len(path2)-1]
  with tarfile.open(path2+'.ts.tar.gz','w:gz') as tar:
    for i in os.listdir():
      if i==path2+'.ts.tar.gz':
        pass
      else:
        tar.add(i)
        os.remove(i)
  tar.close()
  
  os.rename(path2+'.ts.tar.gz','../'+path2+'.ts.tar.gz')
  os.chdir('../')
  os.rmdir(path2)
  for i in np.arange(len(re.split('/',path))-1):
    os.chdir('../')
