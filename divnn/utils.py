import pandas as pd
import numpy as np
from dfply import X, mutate, select, arrange, left_join, bind_cols
from scipy.stats import pearsonr
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram


def hartigan_K(list_of_tuples, threshold = 12):
  # Original hartigan_k function by Teddy Roland
  # https://github.com/teddyroland/python-hartigan/
  
  # 'list_of_tuples' is a list containing the points you want to cluster
  # 'threshold' optimizes goodness of fit values
  # Hartigan recommends setting threshold to 10, but Chiang & Mirkin confirm
  # up to 12 returns integer, "correct" number of clusters
  
  # initializes for maximum possible clusters
  inertia_list = np.zeros(len(list_of_tuples)+1)
  # counter
  num = 0
  # simply initializes above threshold to meet 'while' condition
  H_Rule = threshold+1
  
  # NOTE:
  # 'inertia' is equivalent to the sum of within-cluster distances to centroids
  while num < len(list_of_tuples) and H_Rule > threshold:
    kmn = KMeans(n_clusters = num+1, random_state = 33)
    kmn.fit(list_of_tuples)
    inertia_list[num+1]+=kmn.inertia_
    if num > 0:
      a = ((float(inertia_list[num])/inertia_list[num+1])-1)
      b = (len(list_of_tuples)-(num)-1)
      H_Rule = a*b
    num+=1
  
  if H_Rule > threshold:
    num+=1
  # NOTE:
  # if while-loop reaches the number of K-Means clusters equal to the length of
  # list_of_tuples without hitting the threshold, then function returns trivial
  # solution that there are N clusters (where N is the number of points under
  # observation)
  
  return num-1


def example():
  
  """
  Make an input example for divnn package
  
  This function create an input example for several function in divnn package.
  
  :return: output A list of inputs: 1) value, a pandas data frame with rows for
  instances and columns for features; 2) outcome, a single-column pandas data
  frame of binary integers with the same rows as the instances; 3) similarity,
  a square pandas data frame of floating numbers containing feature-feature
  similarity measures; 4) mapping, a pandas data frame of floating numbers with
  rows for features and three columns for three dimensions where the features
  are mapped onto; and 5) ontology, a pandas data frame with rows for ontologies
  and four columns for source, target, similarity, and relation. In addition, a
  result of hierarchical clustering is also included for visualization purpose.
  """
  
  ## Set an empty list of inputs and random seed
  input={}
  np.random.seed(33)
  
  ## Create example of instance-feature data frame
  input['value']=np.random.normal(0,1,3000*5).reshape((3000,5))
  input['value']=pd.DataFrame(
      data=input['value']
      ,index=['I'+str(i) for i in range(1,3000+1)]
      ,columns=['F'+str(i) for i in range(1,5+1)]
    )
  
  ## Create example of outcome vector
  ## This example uses k-means to create a classifiable outcome
  numCluster_hartigan=hartigan_K(input['value'].to_numpy())
  kmeans=KMeans(n_clusters=numCluster_hartigan,random_state=33)
  input['outcome']=kmeans.fit(input['value'].to_numpy())
  input['outcome']=input['outcome'].predict(input['value'].to_numpy())
  input['outcome']=input['outcome'].astype(int)
  for i in np.arange(len(input['outcome'])):
    if input['outcome'][i]==set(input['outcome'])[0]:
      input['outcome'][i]=1
    else:
      input['outcome'][i]=0
  input['outcome']=pd.DataFrame(
      data=input['outcome']
      ,index=input['value'].index.values.tolist()
      ,columns=['outcome']
    )
  
  ## Create example of feature similarity matrix using Pearson correlation
  def cor(X):
    sim_mat=np.empty((X.shape[1],X.shape[1]))
    for i in range(X.shape[1]):
      for j in range(X.shape[1]):
        sim_mat[i,j]=pearsonr(X.to_numpy()[i,:],X.to_numpy()[j,:])[0]
    return sim_mat
  input['similarity']=cor(input['value'])
  input['similarity']=pd.DataFrame(
      data=input['similarity']
      ,index=input['value'].columns.values.tolist()
      ,columns=input['value'].columns.values.tolist()
    )
  
  ## Create example of feature three-dimensional mapping matrix
  ## This example uses PCA for simplicity
  ## DeepInsight originally uses t-SNE and kernel PCA
  input['mapping']=PCA(n_components=3,random_state=33)
  input['mapping']=input['mapping'].fit_transform(input['similarity'])
  input['mapping']=pd.DataFrame(
      data=input['mapping']
      ,index=input['value'].columns.values.tolist()
      ,columns=['dimension'+str(i) for i in range(1,4)]
    )
  
  ## Create example of ontology
  ## The similarity is recalculated to express dissimilarity
  ## Because this example uses hierarchical clustering for simplicity
  ## VNN originally uses Clique Extracted Ontology (CLiXO)
  input['hierarchy']=AgglomerativeClustering(
      n_clusters=None
      ,distance_threshold=0
      ,affinity='precomputed'
      ,linkage='complete'
    )
  input['hierarchy']=input['hierarchy'].fit((1-input['similarity'])/2)
  
  ## A function to convert a hierarchy object into an ontology data frame
  
  def ontology_df(hierarchy,value):
    
    def linkage_matrix(hierarchy):
      counts=np.zeros(hierarchy.children_.shape[0])
      n_samples=len(hierarchy.labels_)
      for i, merge in enumerate(hierarchy.children_):
        current_count=0
        for child_idx in merge:
          if child_idx<n_samples:
            current_count+=1
          else:
            current_count+=counts[child_idx-n_samples]
        counts[i]=current_count
      l=[hierarchy.children_,hierarchy.distances_,counts]
      return np.column_stack(l).astype(float)
    
    labels=value.columns.values[hierarchy.labels_]
    linkage=linkage_matrix(hierarchy)
    tree=dendrogram(linkage)
    
    A=pd.DataFrame(labels,columns=['A'])
    A=A >> bind_cols(pd.DataFrame(tree['leaves'],columns=['i']))
    
    B=pd.DataFrame(labels,columns=['B'])
    B=B >> bind_cols(pd.DataFrame(tree['leaves'],columns=['i2']))
    
    linkages=pd.DataFrame(linkage,columns=['i','i2','similarity','count'])
    
    ontology=linkages >> left_join(A,by='i')
    ontology=ontology >> left_join(B,by='i2')
    ontology=ontology >> mutate(similarity=1-X.similarity)
    ontology=ontology >> mutate(
      target=['ONT:'+str(i+1) for i in range(ontology.shape[0])]
    )
    
    ontology=ontology >> mutate(i=X.i-ontology.shape[0])
    ontology=ontology >> mutate(i2=X.i2-ontology.shape[0])
    
    A=ontology['i'].values.astype(int)
    A1=ontology['A'].values
    A2=['ONT:'+str(i) for i in A]
    ontology=ontology >> mutate(A=np.where(A<=0,A1,A2))
    
    B=ontology['i2'].values.astype(int)
    B1=ontology['B'].values
    B2=['ONT:'+str(i) for i in B]
    ontology=ontology >> mutate(B=np.where(B<=0,B1,B2))
    
    ontology=pd.melt(
        ontology
        ,id_vars=['similarity','target','i','i2']
        ,value_vars=['A','B']
        ,var_name='key'
        ,value_name='source'
      )
    
    C=np.where(ontology['key']=='A',ontology['i'],ontology['i2'])
    C=np.where(C<=0,'feature','is_a')
    
    ontology=ontology >> mutate(relation=C)
    ontology=ontology >> select(X.source,X.target,X.similarity,X.relation)
    
    ontology=ontology >> arrange(1-X.similarity,X.relation)
    
    return ontology
  
  ## Create example of ontology data frame
  input['ontology']=ontology_df(input['hierarchy'],input['value'])
  
  ## Return
  return input
