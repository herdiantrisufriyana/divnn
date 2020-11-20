class MIAME(object):
  
  """
  Class for Storing Microarray Experiment Information
  
  Class MIAME covers MIAME entries that are not covered by other classes in
  Bioconductor. Namely, experimental design, samples, hybridizations,
  normalization controls, and pre-processing information. The MIAME class is
  derived from MIAxE. This class and the documentation are adapted from the
  original MIAME class in Bioconductor by Rafael A. Irizarry.
  
  :param name: Object of class string containing the experimenter name
  :param lab: Object of class string containing the laboratory where the
  experiment was conducted
  :param contact: Object of class string containing contact information for lab
  and/or experimenter
  :param title: Object of class string containing a single-sentence experiment
  title
  :param abstract: Object of class string containing an abstract describing the
  experiment
  :param url: Object of class string containing a URL for the experiment
  :param samples: Object of class dictionary containing information about the
  samples
  :param hybridizations: Object of class dictionary containing information about
  the hybridizations
  :param normControls: Object of class dictionary containing information about
  the controls such as house keeping genes
  :param preprocessing: Object of class dictionary containing information about
  the pre-processing steps used on the raw data from this experiment
  :param pubMedIds: Object of class string listing strings of PubMed identifiers
  of papers relevant to the dataset
  :param other: Object of class dictionary containing other information for
  which none of the above slots does not applies 
  """
  
  __slots__=[
      'name'
      ,'lab'
      ,'contact'
      ,'title'
      ,'abstract'
      ,'url'
      ,'samples'
      ,'hybridizations'
      ,'normControls'
      ,'preprocessing'
      ,'pubMedIds'
      ,'other'
    ]
  
  def __init__(self,name=str()
               ,lab=str()
               ,contact=str()
               ,title=str()
               ,abstract=str()
               ,url=str()
               ,samples=dict()
               ,hybridizations=dict()
               ,normControls=dict()
               ,preprocessing=dict()
               ,pubMedIds=str()
               ,other=dict()):
    if not isinstance(name,str):
      raise TypeError('name should be a string')
    if not isinstance(lab,str):
      raise TypeError('lab should be a string')
    if not isinstance(contact,str):
      raise TypeError('contact should be a string')
    if not isinstance(title,str):
      raise TypeError('title should be a string')
    if not isinstance(abstract,str):
      raise TypeError('abstract should be a string')
    if not isinstance(url,str):
      raise TypeError('url should be a string')
    if not isinstance(samples,dict):
      raise TypeError('samples should be a dictionary')
    if not isinstance(hybridizations,dict):
      raise TypeError('hybridizations should be a dictionary')
    if not isinstance(normControls,dict):
      raise TypeError('normControls should be a dictionary')
    if not isinstance(preprocessing,dict):
      raise TypeError('preprocessing should be a dictionary')
    if not isinstance(pubMedIds,str):
      raise TypeError('pubMedIds should be a string')
    if not isinstance(other,dict):
      raise TypeError('other should be a dictionary')
    self.name=name
    self.lab=lab
    self.contact=contact
    self.title=title
    self.abstract=abstract
    self.url=url
    self.samples=samples
    self.hybridizations=hybridizations
    self.normControls=normControls
    self.preprocessing=preprocessing
    self.pubMedIds=pubMedIds
    self.other=other
  
  def desc(self):
    print('Experiment data')
    print('  Experimenter name: '+self.name)
    print('  Laboratory: '+self.lab)
    print('  Contact information: '+self.contact)
    print('  Title: '+self.title)
    print('  URL: '+self.url)
    print('  PMIDs: '+self.pubMedIds)
    
    if len(self.abstract)==0:
      print('  No abstract available')
    else:
      print('  A '+str(len(self.abstract.split())),end=' ')
      print('word abstract is available. Use "abstract" method.')
    
    if (
      len(self.samples)
      +len(self.hybridizations)
      +len(self.normControls)
      +len(self.preprocessing)
    )>0:
      msg0='  Information is available on:'
      msg1=''
      i=0
      sep=' '
      if len(self.samples)>0:
        i+=1
        if i>1: sep=', '
        msg1+=sep+'samples'
      if len(self.hybridizations)>0:
        i+=1
        if i>1: sep=', '
        msg1+=sep+'hybridizations'
      if len(self.normControls)>0:
        i+=1
        if i>1: sep=', '
        msg1+=sep+'normalization controls'
      if len(self.preprocessing)>0:
        i+=1
        if i>1: sep=', '
        msg1+=sep+'preprocessing'
      print(msg0+msg1)
    
    if (len(self.other)>0):
      print('  notes:')
      for i in self.other.keys():
        print('    ',end='')
        print(i,end='')
        print(':')
        print('    ',end='')
        for j in self.other.values():
          print(j)
