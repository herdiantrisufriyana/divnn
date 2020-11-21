# An implementation of the DeepInsight Visible Neural Network

This package facilitates application of DeepInsight (DI) and
Visible Neural Network (VNN) algorithms from
<a href="http://www.alok-ai-lab.com/">Alok Sharma</a><sup>1</sup> and
<a href="https://github.com/michaelkyu">Michael Ku Yu</a><sup>2</sup>,
respectively. The application is intended for supervised machine learning by
convolutional neural network (CNN). DeepInsight converts non-image data into
image-like data by dimensionality reduction algorithms. This package maps the
data into a multi-dimensional array. Meanwhile, VNN determines a neural network
architecture by hierarchical clustering algorithms, particularly for data-driven
ontology. This package generate a CNN model based on the ontology using the
DeepInsight array as the input. However, this package includes neither
dimensionality reduction nor data-driven ontology inference. A comprehensive
guide to orchestrate this package and other packages to develop the DI-VNN model
is described in this package vignette. The inputs are instance-feature value
table, outcome vector, feature similarity table, feature three-dimensional
mapping table, and ontology source-target-similarity-relation table. The outputs
are tidy (expression) set, training array, and Keras CNN model.

## Quick Start divnn R

<a href="https://htmlpreview.github.io/?https://github.com/herdiantrisufriyana/divnn/blob/master/vignettes/quick-start-R.html">Read simple example in R</a>

## Quick Start divnn python

<a href="https://htmlpreview.github.io/?https://github.com/herdiantrisufriyana/divnn/blob/master/vignettes/quick-start-py.html">Read simple example in python</a>

## References

[1] Sharma A, Vans E, Shigemizu D, Boroevich KA, Tsunoda T, DeepInsight: a
methodology to transform a non-image data to an image for convolution neural
network architecture, Scientific Reports, 9:11399, pp. 1-7, 2019.
<a href="http://www.alok-ai-lab.com/materials/DeepInsight.pdf">Paper.pdf</a>
<a href="http://www.alok-ai-lab.com/materials/DeepInsight_Supp.pdf">Supplement
.pdf</a> <a href="http://www.alok-ai-lab.com/materials/DeepInsight_Pkg.tar.gz">
DeepInsight Matlab Code: Tested on Ubuntu 18.10.</a>

[2] Ma, J., Yu, M., Fong, S. et al. Using deep learning to model the
hierarchical structure and function of a cell. Nat Methods 15, 290–298 (2018).
<a href="https://doi.org/10.1038/nmeth.4627">Paper</a>
<a href="https://github.com/michaelkyu/">Author GitHub</a>
<a href="https://github.com/idekerlab/DCell/">DCell code</a>