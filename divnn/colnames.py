def colnames(ExpressionSet):
  """
  Column names
  
  Get the column names of an ExpressionSet.
  
  NOTE: This man page is for the colnames S4 generic functions defined in the
  BiocGenerics package. This method and the documentation are adapted from the
  original method in Bioconductor by BiocGenerics team.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: A list containing strings of column names.
  """
  
  return ExpressionSet.phenoData.data.index.values.tolist()
