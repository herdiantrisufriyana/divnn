# An implementation of the DeepInsight Visible Neural Network

This package facilitates application of DeepInsight (DI) and
Visible Neural Network (VNN) algorithms from Alok Sharma and Michael Ku Yu,
respectively. The application is intended for supervised machine learning
by convolutional neural network (CNN). DeepInsight converts non-image
data into image-like data by dimensionality reduction algorithms. This
package maps the data into a multi-dimensional array. Meanwhile, VNN
determines a neural network architecture by hierarchical clustering
algorithms, particularly for data-driven ontology. This package generate a
CNN model based on the ontology using the DeepInsight array as the input.
However, this package includes neither dimensionality reduction nor
data-driven ontology inference. A comprehensive guide to orchestrate this
package and other packages to develop the DI-VNN model is described in this
package vignette. The inputs are instance-feature value data frame, outcome
vector, feature similarity matrix, feature three-dimensional mapping matrix,
and ontology source-target-similarity-relation data frame. The outputs are
tidy (expression) set, training array, and Keras CNN model.
