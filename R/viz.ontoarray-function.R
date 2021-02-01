#' Visualize an ontoarray for visible neural network (VNN) modeling
#'
#' This function visualizes an ontoarray using the average values representing 
#' all ontoarrays. The average value of event is substracted by that of 
#' non-event. The visualization demonstrates a representation layer per 
#' ontology as relative values between event and non-event.
#'
#' @param tidy_set TidySet, an ExpressionSet with three tables.
#' @param ontonet A Keras model object, a pointer to Keras model object in 
#' python environment, which will be an input to train VNN model using Keras R 
#' package.
#' @param batch_size An integer of how much samples are generated everytime
#' this function runs to get the output of the representation layers. If all 
#' samples are generated, this function will loop over the samples. But, only 
#' the same sample size will be generated.
#' @param verbose Verbosity, a logical indicating whether progress should be
#' shown.
#' @param pal A vector of two characters for colors representing the minimum 
#' and maximum values of the metric.
#' @param label A logical whether feature label are shown.
#' @param label.family A character of font family for feature label.
#' @param label.size A numeric of font size for feature label.
#' @param label.color A character of font color for feature label.
#' @param grid_col A integer of the column number of grid for showing for 
#' ontoarrays for all ontologies per channel (z).
#'
#' @return viz.ontoarray object, a list of ontoarrays containing differences of
#' summarized values between event and non-event. For each ontology, there are 
#' an output of a layer representing the ontology, and an ontotype information 
#' about features grouped within the ontology. Visualization are shown by 
#' \code{plot()}.
#'
#' @keywords ontoarray visualization
#'
#' @export
#'
#' @examples
#'
#' ## Create input example
#' input=utils.example()
#'
#' ## Compile input to a TidySet
#' tidy_set=
#'   TidySet.compile(
#'     value=input$value
#'     ,outcome=input$outcome
#'     ,similarity=input$similarity
#'     ,mapping=input$mapping
#'     ,ontology=input$ontology
#'   )
#' ## Create ontonet (Keras model object) generator function
#' ontonet=generator.ontonet(tidy_set)
#' 
#' ## Visualize ontoarray
#' viz.ontoarray(tidy_set,ontonet)

viz.ontoarray=function(tidy_set,ontonet,batch_size=32,verbose=T){
  
  k_clear_session()
  layers=
    ontonet %>%
    get_config() %>%
    .$layers %>%
    sapply(function(x)x$name)
  
  rep_suffix='_hidden_sc_cv$'
  rep_layers=layers[str_detect(layers,rep_suffix)]
  
  if(!verbose) pblapply=lapply
  
  representation=
    rep_layers %>%
    pblapply(X=seq(length(.))
             ,Y=.
             ,Z=ontonet
             ,K=tidy_set
             ,L=batch_size
             ,function(X,Y,Z,K,L){
      
      M=keras_model(
          inputs=Z$inputs
          ,outputs=get_layer(Z,Y[[X]])$output
        ) %>%
        compile(
          optimizer=optimizer_sgd(lr=2^-6,momentum=0.9,decay=10^-4)
          ,loss='mean_squared_error'
        ) %>%
        predict_generator(
          generator=generator.ontoarray(K,seq(ncol(K)),batch_size=L)
          ,steps=ceiling(ncol(K)/L)
          ,verbose=0
        ) %>%
        .[seq(ncol(K)),,,,drop=F] %>%
        `dimnames<-`(list(colnames(K),NULL,NULL,NULL))
      
      N=M %>%
        .[K$outcome==0,,,,drop=F] %>%
        apply(2:4,mean)
      O=M %>%
        .[K$outcome==1,,,,drop=F] %>%
        apply(2:4,mean)
      
      P=notes(K)$ontotype[[X]] %>%
        as.data.frame() %>%
        rownames_to_column(var='feature')
      
      list(output=O-N,ontotype=P)
      
    }) %>%
    setNames(
      rep_layers %>%
        str_remove_all(rep_suffix) %>%
        str_replace_all('ONT','ONT:')
    )
  
  output=representation
  
  class(output)='viz.ontoarray'
  
  assign(x='print.viz.ontoarray',envir=baseenv(),value=function(x){
    cat(paste0(
      '
    Method: viz.ontoarray

    A representation layer per ontology as relative values between event and 
    non-event for visualization purpose.
    
    Total: ',length(x),' ontolog',ifelse(length(x)==1,'y','ies'),'
    '
    ))
  })
  
  assign(x='plot.viz.ontoarray'
         ,envir=baseenv()
         ,value=function(x
                         ,pal=c('red','green')
                         ,label=F
                         ,label.family='sans'
                         ,label.size=3
                         ,label.color='white'
                         ,grid_col=3){
    
    p=x %>%
      lapply(X=names(.),Y=.,function(X,Y){
        ontoarray=Y[[X]]$output
        ontoarray %>%
          lapply(
            X=seq(dim(.)[3]),
            Y=.,
            Z=X,
            FUN=function(X,Y,Z){
              Y[,,X] %>%
                matrix()
            }
          ) %>%
          do.call(rbind,.) %>%
          as.data.frame() %>%
          setNames('fill') %>%
          mutate(
            x=rep(1:dim(ontoarray)[1]
                  ,dim(ontoarray)[2]*dim(ontoarray)[3])
            ,y=rep(1:dim(ontoarray)[2],dim(ontoarray)[1]) %>% sort() %>%
              rep(dim(ontoarray)[3])
            ,z=rep(1:dim(ontoarray)[3]
                   ,dim(ontoarray)[1]*dim(ontoarray)[2]) %>% sort()
          ) %>%
          left_join(Y[[X]]$ontotype,by=c('x','y','z')) %>%
          mutate(ontology=X)
      }) %>%
      do.call(rbind,.) %>%
      mutate(z=paste0('z=',z))
    
    if(!label) p=p %>% mutate(feature=NA)
    
    max_range=p$fill %>% abs() %>% max(na.rm=T)
    
    p %>%
      ggplot(aes(x=y,y=x,fill=fill)) +
      geom_tile() +
      geom_text(
        aes(label=feature)
        ,family=label.family
        ,size=label.size
        ,color=label.color
        ,na.rm=T
      ) +
      facet_wrap(ontology~z,ncol=grid_col) +
      coord_equal() +
      scale_fill_gradientn(
        '0 < outcome > 1'
        ,colors=c(pal[1],'black',pal[2])
        ,na.value=NA
        ,limit=c(-max_range,max_range)
      ) +
      theme_void() +
      theme(
        strip.text=element_text(family=label.family,size=label.size*3)
        ,legend.title=element_text(family=label.family,size=label.size*3)
        ,legend.text=element_text(family=label.family,size=label.size*3)
        ,legend.position='bottom'
      ) +
      guides(fill=guide_colorbar(title.position="top",title.hjust=0.5))
    
  })
  
  output
  
}
