def pData(ExpressionSet):
  """
  Retrieve information on experimental phenotypes recorded in eSet and
  ExpressionSet-derived classes
  
  These generic functions access the phenotypic data (e.g., covariates)
  associated with an experiment. This method and the documentation are adapted
  from the original method in Bioconductor by Biocore team.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: A pandas data frame with samples as rows, variables as columns.
  """
  
  return ExpressionSet.phenoData.data
