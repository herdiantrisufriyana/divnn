import pandas as pd
import numpy as np

class AnnotatedDataFrame(object):
  
  """
  Class Containing Measured Variables and Their Meta-Data Description
  
  An AnnotatedDataFrame consists of two parts. There is a collection of\
  protocols, samples, or features and the values of variables measured on those
  attributes. There is also a description of each variable measured. The
  components of an AnnotatedDataFrame can be accessed with: (1) protocolData,
  pData or fData for the values; (2) varMetadata or fvarMetadata for the
  attribute informations; and (3) varLabels ir fvarLabels for the column names
  of the value table. This class and the documentation are adapted from the
  original AnnotatedDataFrame class in Bioconductor by V.J. Carey, after initial
  design by R. Gentleman.
  
  Use attribute of .desc() to see summary of an AnnotatedDataFrame object. For
  meta-coding purpose, developer can use arguments of annotate, rowNames,
  varLabels, and varMetadata to substitute the corresponding text in the
  summary.
  
  :param data: A data.frame of the protocols, samples, or features
  (rows) and measured variables (columns). This argument can be missing.
  :param varMetadata: A data.frame with the number of rows equal to the number
  of columns of the data argument. varMetadata describes aspects of each
  measured variable. This argument can be missing.
  """
  
  __slots__=['data','varMetadata']
  
  def __init__(self,data=pd.DataFrame(),varMetadata=pd.DataFrame()):
    if not isinstance(data,pd.DataFrame):
      raise TypeError('data should be a pandas data frame')
    if not isinstance(varMetadata,pd.DataFrame):
      raise TypeError('varMetadata should be a pandas data frame')
    if varMetadata.empty:
      varMetadata=pd.DataFrame(
          np.repeat(np.nan,data.shape[1])
          ,index=data.columns.values
          ,columns=['labelDescription']
        )
    else:
      if len(data)>0:
        if np.sum(varMetadata.index.values!=data.columns.values)>0:
          raise ValueError(
            'index of varMetadata should be the same with columns of data'
          )
    self.data=data
    self.varMetadata=varMetadata
  
  def desc(self,annotate=None,rowNames=None,varLabels=None,varMetadata=None):
    if not rowNames:
      rowNames='rowNames'
    if not varLabels:
      varLabels='varLabels'
    if not varMetadata:
      varMetadata='varMetadata'
    if not annotate:
      print('An object of class "AnnotatedDataFrame":',end='')
    else:
      print(str(annotate),end='')
    if self.data.empty & self.varMetadata.empty:
      print(': none')
    else:
      print('')
      print('  '+rowNames+':',end=' ')
      msg=self.data.index.values
      msg2=self.varMetadata.index.values
      if len(msg)>0:
        if len(msg)<=4:
          for i in msg:
            print(str(i),end=' ')
        else:
          j=0
          for i in msg:
            j+=1
            if j in [1,2,4,len(msg)]:
              if j==4:
                print('...',end=' ')
              else:
                print(str(i),end=' ')
        print('(',str(len(msg)),' total)',sep='')
      elif len(msg2)>0:
        if len(msg2)<=3:
          for i in msg2:
            print(str(i),end=' ')
        else:
          j=0
          for i in msg2:
            j+=1
            if j in [1,2,3,4,len(msg2)]:
              if j==4:
                print('...',end=' ')
              else:
                print(str(i),end=' ')
        print('')
      print('  '+varLabels+':',end=' ')
      msg=self.data.columns.values
      msg2=self.varMetadata.index.values
      if len(msg)>0:
        if len(msg)<=4:
          for i in msg:
            print(str(i),end=' ')
        else:
          j=0
          for i in msg:
            j+=1
            if j in [1,2,3,4,len(msg)]:
              if j==4:
                print('...',end=' ')
              else:
                print(str(i),end=' ')
        print('')
      elif len(msg2)>0:
        print('labelDescription')
      print('  '+varMetadata+': labelDescription',end=' ')
