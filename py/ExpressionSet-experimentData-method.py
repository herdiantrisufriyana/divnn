def experimentData(ExpressionSet):
  """
  Retrieve Meta-data from eSets and ExpressionSets
  
  These generic functions access generic data, abstracts, PubMed IDs and
  experiment data from instances of the eSet-class or ExpressionSet-class. This
  method and the documentation are adapted from the original method in
  Bioconductor by Biocore team.
  
  :param ExpressionSet: An ExpressionSet object.
  :return: A MIAME object representing the description of an experiment.
  """
  
  return ExpressionSet.experimentData
