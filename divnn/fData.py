def fData(ExpressionSet):
  """
  Retrieve information on features recorded in eSet-derived classes
  
  These generic functions access feature data (experiment specific information
  about features). This method and the documentation are adapted from the
  original method in Bioconductor by Biocore team.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: A pandas data frame with features as rows, variables as columns.
  """
  
  return ExpressionSet.featureData.data
