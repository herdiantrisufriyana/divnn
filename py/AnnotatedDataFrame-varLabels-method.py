def varLabels(AnnotatedDataFrame):
  """
  Retrieve information from AnnotatedDataFrame recorded in ExpressionSet
  
  These generic functions access the variable names data stored in an object
  derived from the eSet-class. This method and the documentation are adapted
  from the original method in Bioconductor by Biocore team.
  
  :param AnnotatedDataFrame: An AnnotatedDataFrame object.
  :return: A list of strings of measured variable names.
  """
  
  return AnnotatedDataFrame.data.columns.values.tolist()
