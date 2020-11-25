from setuptools import setup

setup(
  name='divnn'
  ,packages=['divnn']
  ,version='0.1.1'
  ,author='Herdiantri Sufriyana'
  ,author_email='herdiantrisufriyana@unusa.ac.id'
  ,description="""
    This module facilitates application of DeepInsight (DI) and Visible Neural
    Network (VNN) algorithms from Alok Sharma and Michael Ku Yu, respectively.
    The application is intended for supervised machine learning by convolutional
    neural network (CNN). DeepInsight converts non-image data into image-like
    data by dimensionality reduction algorithms. This module maps the data into
    a multi-dimensional array. Meanwhile, VNN determines a neural network
    architecture by hierarchical clustering algorithms, particularly for data-
    driven ontology. This module generate a CNN model based on the ontology
    using the DeepInsight array as the input. However, this module includes
    neither dimensionality reduction nor data-driven ontology inference. A
    comprehensive guide to orchestrate this package and other packages to
    develop the DI-VNN model is described in this module vignette. The inputs
    are instance-feature value data frame, outcome vector, feature similarity
    matrix, feature three-dimensional mapping matrix, and ontology source-
    target-similarity-relation data frame. All of these inputs are pandas data
    frames. The outputs are tidy (expression) set, training array, and Keras CNN
    model.
    """
  ,license='GPL-3'
  ,install_requires=[
      'regex'
      ,'pandas>=1.0.5'
      ,'numpy>=1.18.5'
      ,'scipy>=1.5.0'
      ,'dfply>=0.3.3'
      ,'progressbar>=2.5'
      ,'sklearn>=0.0'
      ,'tensorflow==2.0.0'
      ,'tensorflow-gpu==2.0.0'
    ]
  ,URL='https://github.com/herdiantrisufriyana/divnn'
)
