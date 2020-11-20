def exprs(ExpressionSet):
  """
  Retrieve expression data from eSets
  
  These generic functions access the expression of assay data stored in an
  object derived from the eSet-class.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: A two-dimensional numpy array matrix (usually large!) of expression
  values
  """
  
  return ExpressionSet.assayData
