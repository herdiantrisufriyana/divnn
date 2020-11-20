def protocolData(ExpressionSet):
  """
  Protocol Metadata
  
  This generic function handles methods for adding and retrieving protocol
  metadata for the samples in eSets. This method and the documentation are
  adapted from the original method in Bioconductor by Biocore team.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: An AnnotatedDataFrame containing the protocol metadata for the
  samples.
  """
  
  return ExpressionSet.protocolData.data
