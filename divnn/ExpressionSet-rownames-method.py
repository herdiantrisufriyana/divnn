def rownames(ExpressionSet):
  """
  Row names
  
  Get the row names of an ExpressionSet.
  
  NOTE: This man page is for the rownames S4 generic functions defined in the
  BiocGenerics package. This method and the documentation are adapted from the
  original method in Bioconductor by BiocGenerics team.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: A list containing strings of row names.
  """
  
  return ExpressionSet.featureData.data.index.values.tolist()
