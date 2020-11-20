---
title: "Quick Start"
author:
  - name: Herdiantri Sufriyana
    affiliation:
    - &gibi Graduate Institute of Biomedical Informatics, College of Medical
      Science and Technology, Taipei Medical University, Taipei, Taiwan
    - Department of Medical Physiology, College of Medicine, University of
      Nahdlatul Ulama Surabaya, Surabaya, Indonesia
    email: herdiantrisufriyana@unusa.ac.id
  - name: Yu-Wei Wu
    affiliation:
    - *gibi
    - &tmuh Clinical Big Data Research Center, Taipei Medical University
      Hospital, Taipei, Taiwan
  - name: Emily Chia-Yu Su
    affiliation:
    - *gibi
    - *tmuh
    - Research Center for Artificial Intelligence in Medicine, Taipei Medical
      University, Taipei, Taiwan
package: divnn
abstract: >
  This vignette explains how to use this package using simulated data. However,
  reader may need to read another vignette to apply the DeepInsight Visible
  Neural Network (DI-VNN) model.
output:
  BiocStyle::html_document:
    toc_float: true
vignette: >
  %\VignetteIndexEntry{Quick Start}
  %\VignetteEngine{knitr::knitr}
  %\VignetteEncoding{UTF-8}
---





# Load simulated data

Load simulated data using this code.


```r
input=input_example()
```

The first input is an instance-feature value data frame with rows for instances
and columns for features. All rows in value should have names. All
values should be numerics.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Summary of instance-feature value data frame</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> attribute </th>
   <th style="text-align:left;"> outcome_0 </th>
   <th style="text-align:left;"> outcome_1 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> F1 </td>
   <td style="text-align:left;"> 0.3 ± 0.91 </td>
   <td style="text-align:left;"> -0.31 ± 0.97 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F2 </td>
   <td style="text-align:left;"> -0.57 ± 0.83 </td>
   <td style="text-align:left;"> 0.59 ± 0.79 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F3 </td>
   <td style="text-align:left;"> 0.26 ± 0.95 </td>
   <td style="text-align:left;"> -0.29 ± 0.99 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F4 </td>
   <td style="text-align:left;"> -0.3 ± 0.95 </td>
   <td style="text-align:left;"> 0.26 ± 0.98 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F5 </td>
   <td style="text-align:left;"> 0.35 ± 0.9 </td>
   <td style="text-align:left;"> -0.28 ± 0.94 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> outcome </td>
   <td style="text-align:left;"> 1498 </td>
   <td style="text-align:left;"> 1502 </td>
  </tr>
</tbody>
</table>

The second input is and outcome vector of binary integers with the same length
as the instances. The length and the order of outcome should be the same with
those of value. Value  of 0 and 1 should refer to non-event and event outcome,
respectively.


```
##  I1  I2  I3  I4  I5  I6  I7  I8  I9 I10 
##   1   0   0   1   1   0   0   0   0   1
```

The third input is a similarity matrix of numerics containing feature-feature
similarity measures.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Feature similarity matrix</caption>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> F1 </th>
   <th style="text-align:right;"> F2 </th>
   <th style="text-align:right;"> F3 </th>
   <th style="text-align:right;"> F4 </th>
   <th style="text-align:right;"> F5 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> F1 </td>
   <td style="text-align:right;"> 1.0000000 </td>
   <td style="text-align:right;"> -0.0291670 </td>
   <td style="text-align:right;"> -0.0097115 </td>
   <td style="text-align:right;"> -0.0100418 </td>
   <td style="text-align:right;"> 0.0149212 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F2 </td>
   <td style="text-align:right;"> -0.0291670 </td>
   <td style="text-align:right;"> 1.0000000 </td>
   <td style="text-align:right;"> -0.0100969 </td>
   <td style="text-align:right;"> 0.0069537 </td>
   <td style="text-align:right;"> -0.0548541 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F3 </td>
   <td style="text-align:right;"> -0.0097115 </td>
   <td style="text-align:right;"> -0.0100969 </td>
   <td style="text-align:right;"> 1.0000000 </td>
   <td style="text-align:right;"> -0.0154304 </td>
   <td style="text-align:right;"> 0.0078851 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F4 </td>
   <td style="text-align:right;"> -0.0100418 </td>
   <td style="text-align:right;"> 0.0069537 </td>
   <td style="text-align:right;"> -0.0154304 </td>
   <td style="text-align:right;"> 1.0000000 </td>
   <td style="text-align:right;"> -0.0088595 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F5 </td>
   <td style="text-align:right;"> 0.0149212 </td>
   <td style="text-align:right;"> -0.0548541 </td>
   <td style="text-align:right;"> 0.0078851 </td>
   <td style="text-align:right;"> -0.0088595 </td>
   <td style="text-align:right;"> 1.0000000 </td>
  </tr>
</tbody>
</table>

The fourth input is a feature three-dimensional mapping matrix of integers with
rows for features and three columns for three dimensions where the features
are mapped onto.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Feature three-dimensional mapping matrix</caption>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> dimension 1 </th>
   <th style="text-align:right;"> dimension 2 </th>
   <th style="text-align:right;"> dimension 3 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> F1 </td>
   <td style="text-align:right;"> -0.3445922 </td>
   <td style="text-align:right;"> 0.3823556 </td>
   <td style="text-align:right;"> -0.6741419 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F2 </td>
   <td style="text-align:right;"> 0.6883690 </td>
   <td style="text-align:right;"> -0.1021583 </td>
   <td style="text-align:right;"> -0.2998355 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F3 </td>
   <td style="text-align:right;"> -0.0936063 </td>
   <td style="text-align:right;"> -0.7789270 </td>
   <td style="text-align:right;"> 0.0801490 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F4 </td>
   <td style="text-align:right;"> 0.2840867 </td>
   <td style="text-align:right;"> 0.4861979 </td>
   <td style="text-align:right;"> 0.5848708 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F5 </td>
   <td style="text-align:right;"> -0.5638589 </td>
   <td style="text-align:right;"> 0.0158824 </td>
   <td style="text-align:right;"> 0.3273127 </td>
  </tr>
</tbody>
</table>

The fifth input is an ontology data frame with rows for ontologies and four
columns for source, target, similarity, and relation. Feature (source)-
ontology (target) relation should be annotated as 'feature', while ontology-
ontology relation should be annotated as 'is_a'. To differentiate between
feature and ontology names, a prefix of 'ONT:' precedes an ontology name. All
columns except similarity in ontology should be characters. Similarity
(a numeric) is a minimum threshold by which either features or ontologies
(source) belong to an ontology (target).

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Ontology data frame</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> source </th>
   <th style="text-align:left;"> target </th>
   <th style="text-align:right;"> similarity </th>
   <th style="text-align:left;"> relation </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> F1 </td>
   <td style="text-align:left;"> ONT:1 </td>
   <td style="text-align:right;"> 0.5074606 </td>
   <td style="text-align:left;"> feature </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F5 </td>
   <td style="text-align:left;"> ONT:1 </td>
   <td style="text-align:right;"> 0.5074606 </td>
   <td style="text-align:left;"> feature </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F2 </td>
   <td style="text-align:left;"> ONT:2 </td>
   <td style="text-align:right;"> 0.5034769 </td>
   <td style="text-align:left;"> feature </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F4 </td>
   <td style="text-align:left;"> ONT:2 </td>
   <td style="text-align:right;"> 0.5034769 </td>
   <td style="text-align:left;"> feature </td>
  </tr>
  <tr>
   <td style="text-align:left;"> F3 </td>
   <td style="text-align:left;"> ONT:3 </td>
   <td style="text-align:right;"> 0.4951443 </td>
   <td style="text-align:left;"> feature </td>
  </tr>
  <tr>
   <td style="text-align:left;"> ONT:1 </td>
   <td style="text-align:left;"> ONT:3 </td>
   <td style="text-align:right;"> 0.4951443 </td>
   <td style="text-align:left;"> is_a </td>
  </tr>
  <tr>
   <td style="text-align:left;"> ONT:2 </td>
   <td style="text-align:left;"> ONT:4 </td>
   <td style="text-align:right;"> 0.4725730 </td>
   <td style="text-align:left;"> is_a </td>
  </tr>
  <tr>
   <td style="text-align:left;"> ONT:3 </td>
   <td style="text-align:left;"> ONT:4 </td>
   <td style="text-align:right;"> 0.4725730 </td>
   <td style="text-align:left;"> is_a </td>
  </tr>
</tbody>
</table>

In addition, a result of hierarchical clustering is also shown below for
visualization to get intuition how the features are grouped and how the
connection constructs the VNN model architecture.

![Ontology by hierarchical clustering](figure/figure-1-1.png)

# Create TidySet

Create a TidySet using this code.


```r
tidy_set=
  create_tidy_set(
    value=input$value
    ,outcome=input$outcome
    ,similarity=input$similarity
    ,mapping=input$mapping
    ,ontology=input$ontology
  )
```


TidySet is an ExpressionSet with three tables. Instance-feature
value data frame and outcome vector are compiled as a phenotype data frame
with rows for instances and columns for features and outcome.


```
## ExpressionSet (storageMode: lockedEnvironment)
## assayData: 49 features, 3000 samples 
##   element names: exprs 
## protocolData: none
## phenoData
##   sampleNames: I1 I2 ... I3000 (3000 total)
##   varLabels: outcome F2 ... F1 (6 total)
##   varMetadata: labelDescription
## featureData
##   featureNames: x1y1z1 x2y1z1 ... x7y7z1 (49 total)
##   fvarLabels: feature ONT1 ... ONT4 (5 total)
##   fvarMetadata: labelDescription
## experimentData: use 'experimentData(object)'
## Annotation:
```

Instance-feature value data frame and feature three-dimensional mapping matrix
are compiled as an expression matrix with rows for positions of features and
columns for instances. The mapping and similarity matrices and ontology data
frame are compile as a feature data frame with rows for positions of features
and columns for feature names and ontological relations. For easier access,
the original ontology data frame is included in experiment information at
section 'notes' as a list.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Dimensions of three tables</caption>
 <thead>
  <tr>
   <th style="text-align:left;">   </th>
   <th style="text-align:right;"> dimension 1 </th>
   <th style="text-align:right;"> dimension 2 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> Phenotype data frame </td>
   <td style="text-align:right;"> 3000 </td>
   <td style="text-align:right;"> 6 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Feature data frame </td>
   <td style="text-align:right;"> 49 </td>
   <td style="text-align:right;"> 5 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> Expression matrix </td>
   <td style="text-align:right;"> 49 </td>
   <td style="text-align:right;"> 3000 </td>
  </tr>
</tbody>
</table>

Recall experiment notes such similarity, ontomap, ontotype, and ontology using
this code.


```r
# Recall a similarity matrix
notes(tidy_set)$similarity

# Recall an ontomap four-dimensional array
notes(tidy_set)$ontomap

# Recall an ontotype list of two-dimensional matrices
notes(tidy_set)$ontotype

# Recall an ontology data frame
notes(tidy_set)$ontology
```

# Save or load a TidySet

Use this code to save a TidySet by writing a .ts.tar.gz file containing
exprs.csv, pData.csv, fData.csv, similarity.csv, ontology.csv, and others.txt.


```r
write_ts_tar_gz(tidy_set,'quick-start/tidy_set')
```

Function of read_ts_tar_gz can read this file back to a TidySet. Use
this code to load a TidySet by reading a .ts.tar.gz file.


```r
tidy_set=read_ts_tar_gz('quick-start/tidy_set.ts.tar.gz')
```


# Create ontonet

Let's create a function that generate a Keras Convolutional Neural Network (CNN)
model with a specific layer architecture for each path in the hierarchy of the
given ontology. The model architecture can be saved to JSON for later use by
specifying the file path; otherwise (by let it NULL), no JSON will be created.


```r
ontonet=
  tidy_set %>%
  ontonet_generator(path='quick-start/ontonet')
```



# Set up hyperparameters

The model is compiled with stochastic gradient decent (SGD)
using learning rate (LR) of 2^-6, momentum of 0.9, and decay 10^-4. The loss
function is mean squared error (MSE) with weight of 1 for main output and 0.3
fir auxiliary output. The evaluation metric is accuracy.

The LR will be reduce by factor of 0.1 at iteration 30, 60, and 80. Early
stopping will also happen if validation loss does not decrease >0.001 after
30 iteration. The best weight at minimum validation loss will be applied.


```r
ontonet %>%
  compile(
    optimizer=optimizer_sgd(lr=2^-6,momentum=0.9,decay=10^-4)
    ,loss='mean_squared_error'
    ,loss_weights=c(rep(0.3,length(.$outputs)-1),1)
    ,metrics='accuracy'
  )

cb_lr_reduction=
  callback_learning_rate_scheduler(
    function(epoch,lr){
      lr_factor=0.1
      if(epoch %in% c(30,60,80)) lr=lr*lr_factor
      lr
    }
  )

cb_early_stopping=
  callback_early_stopping(
    monitor='val_loss'
    ,mode='min'
    ,min_delta=0.001
    ,patience=30
    ,restore_best_weights=T
  )
```

# Data partition

We hold out ~20% dataset for test set, while ~20% of the remaining will be a
validation set. Training set

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Data partition</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> set </th>
   <th style="text-align:right;"> outcome_0 </th>
   <th style="text-align:right;"> outcome_1 </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> train </td>
   <td style="text-align:right;"> 965 </td>
   <td style="text-align:right;"> 955 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> validation </td>
   <td style="text-align:right;"> 229 </td>
   <td style="text-align:right;"> 251 </td>
  </tr>
  <tr>
   <td style="text-align:left;"> test </td>
   <td style="text-align:right;"> 304 </td>
   <td style="text-align:right;"> 296 </td>
  </tr>
</tbody>
</table>

# Model training

We use this code to train the model. The sample generator for ontoarray will
take 32 samples for each batch. This step is repeated until all samples of
training set are used per epoch, up to 100 epochs, but can be stopped earlier.
For each epoch, validation performance is also measured with the same batch
size.


```r
history=
  ontonet %>%
  fit_generator(
    generator=
      ontoarray_generator(
        tidy_set
        ,index[train_i]
        ,batch_size=32
      )
    ,steps_per_epoch=ceiling(length(train_i)/32)
    ,validation_data=
      ontoarray_generator(
        tidy_set
        ,index[val_i]
        ,batch_size=32
      )
    ,validation_steps=ceiling(length(val_i)/32)
    ,epochs=100
    ,callbacks=c(cb_lr_reduction,cb_early_stopping)
    ,view_metrics=F
    ,verbose=1
  )
```


Here below is the training result. We can estimate the iteration of which the
weights are optimum for prediction.

![Plot training and validation losses](figure/figure-2-1.png)

# Model evaluation

The test set is used for model evaluation by bootstrapping. The sample generator
for ontoarray will take 32 samples for each batch. This step is repeated until
all samples of training set are used per bootstrap, until 30 times.


```r
set.seed(33)
evaluation=
  pblapply(X=1:30,function(X){
    test_i=sample(test_i,length(test_i),T)
    ontonet %>%
      evaluate_generator(
        generator=
          ontoarray_generator(
            tidy_set
            ,index[test_i]
            ,batch_size=32
          )
        ,steps=ceiling(length(test_i)/32)
      )
  })
```


The point and interval (95% CI) estimates of accuracy can be computed by
the bootstrapping.



Therefore, the predictive performance of this model can be
generalized for future, unobserved samples.

<table class="table" style="margin-left: auto; margin-right: auto;">
<caption>Accuracy estimates (95% CI) on test set</caption>
 <thead>
  <tr>
   <th style="text-align:left;"> metric </th>
   <th style="text-align:right;"> mean </th>
   <th style="text-align:right;"> lb </th>
   <th style="text-align:right;"> ub </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:left;"> root_accuracy </td>
   <td style="text-align:right;"> 0.7523026 </td>
   <td style="text-align:right;"> 0.7470964 </td>
   <td style="text-align:right;"> 0.7575088 </td>
  </tr>
</tbody>
</table>

# Session info


```r
sessionInfo()
```

```
## R version 4.0.2 (2020-06-22)
## Platform: x86_64-w64-mingw32/x64 (64-bit)
## Running under: Windows 10 x64 (build 18363)
## 
## Matrix products: default
## 
## locale:
## [1] LC_COLLATE=English_United States.1252  LC_CTYPE=English_United States.1252    LC_MONETARY=English_United States.1252
## [4] LC_NUMERIC=C                           LC_TIME=English_United States.1252    
## 
## attached base packages:
## [1] parallel  stats     graphics  grDevices utils     datasets  methods   base     
## 
## other attached packages:
##  [1] divnn_0.1.1         testthat_2.3.2      keras_2.3.0.0       matrixStats_0.56.0  pbapply_1.4-3       Biobase_2.48.0      BiocGenerics_0.34.0
##  [8] ggplot2_3.3.2       tibble_3.0.3        readr_1.3.1         stringr_1.4.0       tidyr_1.1.0         dplyr_1.0.0        
## 
## loaded via a namespace (and not attached):
##  [1] httr_1.4.2          pkgload_1.1.0       jsonlite_1.7.0      viridisLite_0.3.0   splines_4.0.2       assertthat_0.2.1    BiocManager_1.30.10
##  [8] highr_0.8           yaml_2.2.1          remotes_2.2.0       sessioninfo_1.1.1   pillar_1.4.6        backports_1.1.8     lattice_0.20-41    
## [15] glue_1.4.2          reticulate_1.16     digest_0.6.27       rvest_0.3.6         colorspace_1.4-1    htmltools_0.5.0     Matrix_1.2-18      
## [22] pkgconfig_2.0.3     devtools_2.3.1      purrr_0.3.4         scales_1.1.1        webshot_0.5.2       processx_3.4.3      whisker_0.4        
## [29] mgcv_1.8-31         generics_0.0.2      farver_2.0.3        usethis_1.6.1       ellipsis_0.3.1      withr_2.2.0         cli_2.0.2          
## [36] magrittr_1.5        crayon_1.3.4        memoise_1.1.0       evaluate_0.14       ps_1.3.3            fs_1.4.2            fansi_0.4.1        
## [43] nlme_3.1-148        xml2_1.3.2          pkgbuild_1.1.0      tools_4.0.2         prettyunits_1.1.1   BiocStyle_2.16.1    hms_0.5.3          
## [50] lifecycle_0.2.0     munsell_0.5.0       callr_3.4.3         kableExtra_1.3.1    compiler_4.0.2      rlang_0.4.8         grid_4.0.2         
## [57] rstudioapi_0.11     rappdirs_0.3.1      base64enc_0.1-3     labeling_0.3        rmarkdown_2.5       gtable_0.3.0        R6_2.4.1           
## [64] tfruns_1.4          knitr_1.30          tensorflow_2.2.0    zeallot_0.1.0       rprojroot_1.3-2     desc_1.2.0          stringi_1.4.6      
## [71] Rcpp_1.0.5          vctrs_0.3.4         tidyselect_1.1.0    xfun_0.16
```
