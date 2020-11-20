def notes(MIAME):
  """
  Retrieve eSet notes
  
  These generic functions access notes (unstructured descriptive data)
  associated eSet-class. This method and the documentation are adapted from the
  original method in Bioconductor by Biocore team.
  
  :param MIAME: A MIAME object.
  :return: A dictionary.
  """
  
  return MIAME.other
