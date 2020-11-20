def varMetadata(AnnotatedDataFrame):
  """
  Retrieve information from AnnotatedDataFrame recorded in ExpressionSet
  
  These generic functions access the variable meta-data stored in an object
  derived from the eSet-class. This method and the documentation are adapted
  from the original method in Bioconductor by Biocore team.
  
  :param AnnotatedDataFrame: An AnnotatedDataFrame object.
  :return: A pandas data frame with variable names as rows, description tags
  (e.g., unit of measurement) as columns.
  """
  
  return AnnotatedDataFrame.varMetadata

