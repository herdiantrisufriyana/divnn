def annotation(ExpressionSet):
  """
  Accessing annotation information
  
  Get or set the annotation information contained in an object. This method and
  the documentation are adapted from the original method in BiocGenerics by
  BiocGenerics team.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: A string.
  """
  
  return ExpressionSet.annotation
