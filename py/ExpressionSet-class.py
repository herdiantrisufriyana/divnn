import pandas as pd
import numpy as np

class ExpressionSet(object):
  
  """
  Class to Contain and Describe High-Throughput Expression Level Assays
  
  Container for high-throughput assays and experimental metadata. ExpressionSet
  class is derived from eSet, and requires a matrix named exprs as assayData
  member. This class and the documentation are adapted from the original
  ExpressionSet class in Bioconductor by Biocore team.
  
  Use attribute of .desc() to see summary of an ExpressionSet object.
  
  :param assayData: A two-dimensional numpy array matrix of expression values.
  The rows represent probe sets (‘features’ in ExpressionSet parlance). Columns
  represent samples. When present, row names identify features and column names
  identify samples. Row and column names must be unique, and consistent with row
  names of featureData and phenoData, respectively. The assay data can be
  retrieved with exprs().
  :param phenoData: An optional AnnotatedDataFrame containing information about
  each sample. The number of rows in phenoData must match the number of columns
  in assayData. Row names of phenoData must match column names of the array in
  assayData.
  :param featureData: An optional AnnotatedDataFrame containing information
  about each feature. The number of rows in featureData must match the number of
  rows in assayData. Row names of featureData must match row names of the array
  in assayData.
  :param experimentData: An optional MIAME instance with meta-data (e.g., the
  lab and resulting publications from the analysis) about the experiment.
  :param annotation: A character describing the platform on which the samples
  were assayed. This is often the name of a Bioconductor chip annotation
  package, which facilitated down-stream analysis.
  :param protocolData: An optional AnnotatedDataFrame containing equipment-
  generated information about protocols. The number of rows and row names of
  protocolData must agree with the dimension and column names of assayData.
  """
  
  __slots__=[
      'assayData'
      ,'phenoData'
      ,'featureData'
      ,'experimentData'
      ,'annotation'
      ,'protocolData'
    ]
  
  def __init__(self
               ,assayData=np.array([]).reshape(0,0)
               ,phenoData=AnnotatedDataFrame()
               ,featureData=AnnotatedDataFrame()
               ,experimentData=MIAME()
               ,annotation=str()
               ,protocolData=AnnotatedDataFrame()):
    if not isinstance(assayData,np.ndarray):
      raise TypeError('assayData should be a numpy array')
    if not isinstance(featureData,AnnotatedDataFrame):
      raise TypeError('featureData should be an AnnotatedDataFrame')
    if not isinstance(experimentData,MIAME):
      raise TypeError('experimentData should be a python dictionary')
    if not isinstance(annotation,str):
      raise TypeError('annotation should be a python string')
    if not isinstance(protocolData,AnnotatedDataFrame):
      raise TypeError('protocolData should be an AnnotatedDataFrame')
    if phenoData.data.empty:
      phenoData=AnnotatedDataFrame(
        pd.DataFrame(index=np.arange(assayData.shape[1]))
      )
    if featureData.data.empty:
      featureData=AnnotatedDataFrame(
        pd.DataFrame(index=np.arange(assayData.shape[0]))
      )
    if protocolData.data.empty:
      protocolData=AnnotatedDataFrame(
        pd.DataFrame(index=np.arange(assayData.shape[1]))
      )
    if not phenoData.data.empty:
      if phenoData.data.shape[0]!=assayData.shape[1]:
        raise ValueError(
          'Index of phenoData and assayData should be the same'
        )
      if protocolData.data.empty:
        protocolData=AnnotatedDataFrame(
          pd.DataFrame(index=phenoData.data.index.values)
        )
    if not protocolData.data.empty:
      if protocolData.data.shape[0]!=assayData.shape[1]:
        raise ValueError(
          'Index of protocolData and assayData should be the same'
        )
      if phenoData.data.empty:
        phenoData=AnnotatedDataFrame(
          pd.DataFrame(index=protocolData.data.index.values)
        )
    if not (phenoData.data.empty | protocolData.data.empty):
      if np.sum(
        phenoData.data.index.values!=protocolData.data.index.values
      )>0:
        raise ValueError(
          'Index of phenoData and protocolData should be the same'
        )
    if not featureData.data.empty:
      if featureData.data.shape[0]!=assayData.shape[0]:
        raise ValueError(
          'Index of featureData and assayData should be the same'
        )
    self.assayData=assayData
    self.phenoData=phenoData
    self.featureData=featureData
    self.experimentData=experimentData
    self.annotation=annotation
    self.protocolData=protocolData
  
  def desc(self):
    print('ExpressionSet (storageMode: lockedEnvironment)')
    print('assayData: ',end='')
    print(self.assayData.shape[0],end='')
    print(' features, ',end='')
    print(self.assayData.shape[1],end='')
    print(' samples')
    print('  element names: exprs')
    if self.protocolData.data.empty:
      print('protocolData: none')
    else:
      self.protocolData.desc(annotate='protocolData',rowNames='sampleNames')
      print('')
    if self.phenoData.data.empty:
      print('phenoData: none')
    else:
      self.phenoData.desc(annotate='phenoData',rowNames='sampleNames')
      print('')
    if self.featureData.data.empty:
      print('featureData: none')
    else:
      self.featureData.desc(
        annotate='featureData'
        ,rowNames='featureNames'
        ,varLabels='fvarLabels'
        ,varMetadata='fvarMetadata'
      )
      print('')
    print('experimentData: use ".experimentData.desc()"')
    print('Annotation:',end=' ')
    print(self.annotation)
