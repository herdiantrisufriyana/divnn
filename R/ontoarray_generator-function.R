#' Make an ontoarray generator for visible neural network (VNN) modeling
#'
#' This function create a function that generate a batch of ontoarray for
#' training or testing a Keras Convolutional Neural Network (CNN) model using
#' \code{fit_generator}, \code{evaluate_generator}, or \code{predict_generator}
#' function from Keras R package.
#'
#' @param tidy_set TidySet, an ExpressionSet with three tables.
#' @param index An integer vector of index to select which ontoarray will be
#' used for training or testing.
#' @param batch_size An integer of how much samples are generated everytime
#' this function runs. If all samples are generated,this function will loop over
#' the samples.
#'
#' @return output sample generator, a function for argument of \code{generator}
#' in \code{fit_generator}, \code{evaluate_generator}, or
#' \code{predict_generator} function from Keras R package.
#'
#' @keywords sample generator
#'
#' @export
#'
#' @examples
#'
#' ## Create input example
#' input=input_example()
#'
#' ## Create a TidySet
#' tidy_set=
#'   create_tidy_set(
#'     value=input$value
#'     ,outcome=input$outcome
#'     ,similarity=input$similarity
#'     ,mapping=input$mapping
#'     ,ontology=input$ontology
#'   )
#'
#' ## Create ontonet (Keras model object) generator function
#' ontonet=ontonet_generator(tidy_set)
#'
#' ## Randomize sample and split indices for train and test set
#' set.seed(33)
#' index=sample(1:dim(tidy_set)[2],dim(tidy_set)[2],F)
#' test_i=1:round(0.2*length(index))
#' train_i=!index %in% index[test_i]
#'
#' ## Fit the model
#' history=
#'   ontonet %>%
#'   compile(
#'     loss='mean_squared_error'
#'     ,loss_weights=c(rep(0.3,length(.$outputs)-1),1)
#'     ,metrics='accuracy'
#'   ) %>%
#'   fit_generator(
#'     generator=
#'       ontoarray_generator(
#'         tidy_set
#'         ,index=index[train_i]
#'         ,batch_size=4
#'       )
#'     ,steps_per_epoch=24
#'     ,validation_data=
#'       ontoarray_generator(
#'         tidy_set
#'         ,index=index[test_i]
#'         ,batch_size=4
#'       )
#'     ,validation_steps=6
#'     ,epochs=30
#'     ,verbose=1
#'   )

ontoarray_generator=function(tidy_set,index,batch_size) {

  # Recall ontomap
  ontomap=
    tidy_set %>%
    experimentData() %>%
    notes() %>%
    .$ontomap

  # Recall ontotype
  ontotype=
    tidy_set %>%
    experimentData() %>%
    notes() %>%
    .$ontotype

  # Recall outcome
  outcome=tidy_set$outcome

  # Build a generator function to load a batch of ontoarray

  ontomap=ontomap[index,,,,drop=F]
  outcome=outcome[index]

  ontofilter=
    ontotype %>%
    lapply(X=seq(length(.)),Y=.,Z=ontomap[1,,,,drop=F]*0,function(X,Y,Z){
      Z[,Y[[X]][,1],Y[[X]][,2],Y[[X]][,3]]=1
      Z
    }) %>%
    setNames(paste0(names(ontotype),'_input'))

  i<-1

  function() {

    if((i+batch_size-1)>dim(ontomap)[1]) i<<-1
    rows<-c(i:min(i+batch_size-1,dim(ontomap)[1]))
    i<<-i+batch_size

    x_array=
      ontomap %>%
      .[rows,,,,drop=F] %>%
      lapply(X=seq(length(ontofilter)),Y=ontofilter,Z=.,function(X,Y,Z){
        sweep(Z,2:4,Y[[X]],FUN='*')
      }) %>%
      setNames(names(ontofilter))

    y_vector=
      outcome %>%
      .[rows] %>%
      lapply(X=seq(length(ontotype)),Y=.,function(X,Y)Y) %>%
      setNames(names(ontotype))

    list(x_array, y_vector)
  }

}
