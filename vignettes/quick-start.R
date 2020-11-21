# Load packages and set GPU off
library(divnn)
library(tidyverse)
library(BiocGenerics)
library(Biobase)
library(pbapply)
library(matrixStats)
Sys.setenv('CUDA_VISIBLE_DEVICES'=-1)
library(keras)
use_backend('tensorflow')

# Load simulated data
input=utils.example()

# Compile input to a TidySet
tidy_set=
  TidySet.create(
    value=input$value
    ,outcome=input$outcome
    ,similarity=input$similarity
    ,mapping=input$mapping
    ,ontology=input$ontology
  )

# Recall a similarity matrix
notes(tidy_set)$similarity

# Recall an ontomap four-dimensional array
notes(tidy_set)$ontomap

# Recall an ontotype list of two-dimensional matrices
notes(tidy_set)$ontotype

# Recall an ontology data frame
notes(tidy_set)$ontology

# Save a TidySet
TidySet.write(tidy_set,'vignettes/quick-start-R/tidy_set_R')

# Load a TidySet
tidy_set=TidySet.read('vignettes/quick-start-R/tidy_set_R.ts.tar.gz')

# Create ontonet
ontonet=
  tidy_set %>%
  generator.ontonet(path='vignettes/quick-start-R/ontonet_R')

# Set up hyperparameters
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

# Data partition
set.seed(33)
index=sample(1:dim(tidy_set)[2],dim(tidy_set)[2],F)

test_i=
  index %>%
  sample(round(0.2*length(.)),F)

val_i=
  which(!index %in% index[test_i]) %>%
  sample(round(0.2*length(.)),F)

train_i=
  which(!index %in% index[c(test_i,val_i)])

# Model training
history=
  ontonet %>%
  fit_generator(
    generator=
      generator.ontoarray(
        tidy_set
        ,index[train_i]
        ,batch_size=32
      )
    ,steps_per_epoch=ceiling(length(train_i)/32)
    ,validation_data=
      generator.ontoarray(
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

saveRDS(history,'vignettes/quick-start-R/history_R.rds')
save_model_weights_hdf5(ontonet,'vignettes/quick-start-R/ontonet_R.h5')

# Model evaluation
set.seed(33)
evaluation=
  pblapply(X=1:30,function(X){
    test_i=sample(test_i,length(test_i),T)
    ontonet %>%
      evaluate_generator(
        generator=
          generator.ontoarray(
            tidy_set
            ,index[test_i]
            ,batch_size=32
          )
        ,steps=ceiling(length(test_i)/32)
      )
  })

saveRDS(evaluation,'vignettes/quick-start-R/evaluation_R.rds')

results=
  evaluation %>%
  lapply(X=seq(length(.)),Y=.,function(X,Y){
    as.data.frame(Y[[X]]) %>%
      gather(metric,value) %>%
      mutate(b=X)
  }) %>%
  do.call(rbind,.) %>%
  group_by(metric) %>%
  summarize(
    mean=mean(value)
    ,lb=mean(value)-qnorm(0.975)*sd(value)/sqrt(n())
    ,ub=mean(value)+qnorm(0.975)*sd(value)/sqrt(n())
  )
