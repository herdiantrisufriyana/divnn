#' Make an ontonet generator for visible neural network (VNN) modeling
#'
#' This function create a function that generate a Keras Convolutional Neural
#' Network (CNN) model with a specific layer architecture for each path in the
#' hierarchy of the given ontology.
#'
#' @param tidy_set TidySet, an ExpressionSet with three tables.
#' @param path A character of file path if the model json file is saved.
#' @param init_seed An integer of random seed for ReLU initializer.
#' @param init2_seed An integer of random seed for tanh initializer.
#'
#' @return output Keras model object, a pointer to Keras model object in python
#' environment, which will be an input to train VNN model using Keras R package.
#'
#' @keywords ontonet, Keras model object
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

ontonet_generator=function(tidy_set
                           ,path=NULL
                           ,init_seed=888
                           ,init2_seed=9999){

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

  # Recall ontology
  ontology=
    tidy_set %>%
    experimentData() %>%
    notes() %>%
    .$ontology

  # Build a function to insert an inception module along with a pre-activation residual unit
  layer_inception_resnet=function(object
                                  ,residue
                                  ,filters
                                  ,kernel_initializer
                                  ,name=NULL){

    pre_activation=object %>%
      layer_batch_normalization(name=paste0(name,'_pre_bn')) %>%
      layer_activation_relu(name=paste0(name,'_pre_ac'))

    tower_1=pre_activation %>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(1,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower1_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower1_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower1_ac'))

    tower_2=pre_activation %>%
      layer_max_pooling_2d(
        pool_size=c(3,3)
        ,strides=c(1,1)
        ,padding='same'
        ,name=paste0(name,'_tower2a_mp')
      ) %>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(1,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower2b_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower2b_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower2b_ac'))

    tower_3a=pre_activation %>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(1,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower3a_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower3a_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower3a_ac'))

    tower_3b1=tower_3a%>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(1,3)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower3b1_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower3b1_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower3b1_ac'))

    tower_3b2=tower_3a%>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(3,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower3b2_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower3b2_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower3b2_ac'))

    tower_4b=pre_activation %>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(1,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower4a_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower4a_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower4a_ac')) %>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(3,3)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower4b_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower4b_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower4b_ac'))

    tower_4c1=tower_4b%>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(1,3)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower4c1_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower4c1_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower4c1_ac'))

    tower_4c2=tower_4b %>%
      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(3,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_tower4c2_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_tower4c2_bn')) %>%
      layer_activation_relu(name=paste0(name,'_tower4c2_ac'))

    towers=layer_concatenate(
      c(tower_1
        ,tower_2
        ,tower_3b1,tower_3b2
        ,tower_4c1,tower_4c2),
      axis=-1,
      name=paste0(name,'_dc')
    )

    scaling=towers%>%
      layer_separable_conv_2d(
        filters=dim(residue)[[4]]
        ,kernel_size=c(1,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_sc_cv')
      )

    inception_resnet=c(scaling,residue) %>%
      layer_add(name=name)

    inception_resnet
  }

  # Build a function to insert auxiliary output layers
  layer_aux_output=function(object
                            ,filters
                            ,units
                            ,kernel_initializer
                            ,kernel_initializer2
                            ,name=NULL){

    object %>%
      layer_average_pooling_2d(
        pool_size=c(5,5)
        ,strides=c(3,3)
        ,padding='valid'
        ,name=paste0(name,'_ap')
      ) %>%

      layer_separable_conv_2d(
        filters=filters
        ,kernel_size=c(1,1)
        ,strides=c(1,1)
        ,padding='same'
        ,depthwise_initializer=kernel_initializer
        ,pointwise_initializer=kernel_initializer
        ,name=paste0(name,'_hl1_cv')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_hl1_bn')) %>%
      layer_activation_relu(name=paste0(name,'_hl1_ac')) %>%

      layer_dense(
        units=units
        ,kernel_initializer=kernel_initializer
        ,name=paste0(name,'_hl2_de')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_hl2_bn')) %>%
      layer_activation_relu(name=paste0(name,'_hl2_ac')) %>%

      layer_dense(
        units=units
        ,kernel_initializer=kernel_initializer
        ,name=paste0(name,'_hl3_de')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_hl3_bn')) %>%
      layer_activation_relu(name=paste0(name,'_hl3_ac')) %>%

      layer_dense(
        units=2
        ,activation='tanh'
        ,kernel_initializer=kernel_initializer2
        ,name=paste0(name,'_ao_tn')
      ) %>%
      layer_batch_normalization(name=name)

  }

  # Build a function to insert output layers
  layer_output=function(object
                        ,units
                        ,kernel_initializer
                        ,kernel_initializer2
                        ,name=NULL){

    object %>%
      layer_average_pooling_2d(
        pool_size=c(7,7)
        ,strides=c(1,1)
        ,padding='valid'
        ,name=paste0(name,'_ap')
      ) %>%

      layer_dense(
        units=units
        ,kernel_initializer=kernel_initializer
        ,name=paste0(name,'_hl1_de')
      ) %>%
      layer_batch_normalization(name=paste0(name,'_hl1_bn')) %>%
      layer_activation_relu(name=paste0(name,'_hl1_ac')) %>%

      layer_dense(
        units=2
        ,activation='tanh'
        ,kernel_initializer=kernel_initializer2
        ,name=paste0(name,'_mo_tn')
      ) %>%
      layer_batch_normalization(name=name)

  }

  # Build a generator function to construct ontonet

  k_clear_session()
  init=initializer_he_uniform(seed=init_seed)
  init2=initializer_glorot_uniform(seed=init2_seed)

  feature=ontology
  while(sum(str_detect(feature$source,'ONT'))>0){
    feature=
      feature %>%
      lapply(X=1:nrow(.),Y=.,function(X,Y){
        if(str_detect(Y$source[X],'ONT')){
          Z=filter(Y,target==Y$source[X] & relation=='feature')
          if(nrow(Z)>0){
            data.frame(
              source=Z$source
              ,target=Y$target[X]
              ,similarity=Y$similarity[X]
              ,relation='feature'
            )
          }else{
            Y[X,]
          }
        }else{
          Y[X,]
        }
      }) %>%
      do.call(rbind,.)
  }

  feature=
    feature %>%
    select(source,target) %>%
    rbind(
      select(.,source) %>%
        .[!duplicated(.),,drop=F] %>%
        mutate(target='root')
    ) %>%
    group_by(target) %>%
    summarise(n=n()) %>%
    ungroup() %>%
    mutate(oid=str_replace_all(target,'ONT\\:','ONT')) %>%
    select(oid,n)

  hierarchy=
    ontology %>%
    filter(str_detect(source,'ONT\\:')) %>%
    mutate(
      from=str_replace_all(source,'ONT\\:','ONT')
      ,to=str_replace_all(target,'ONT\\:','ONT')
    ) %>%
    select(from,to) %>%
    rbind(
      data.frame(
        from=filter(.,!to%in%.$from)$to %>% .[!duplicated(.)]
        ,to='root'
      )
    ) %>%
    left_join(rename(feature,to=oid),by='to')

  terminal_nodes=length(unique(hierarchy$from) %>% .[!.%in%hierarchy$to])
  non_terminal_nodes=length(unique(hierarchy$to))

  pb=startpb(0,5+terminal_nodes+non_terminal_nodes)
  on.exit(closepb(pb))

  setpb(pb,0)
  inputs=
    ontotype %>%
    lapply(
      X=seq(length(.))
      ,Y=.
      ,Z=ontomap
      ,FUN=function(X,Y,Z){
        layer_input(
          shape=dim(Z)[2:4]
          ,dtype='float32'
          ,name=paste0(names(Y)[X],'_input')
        )
      }) %>%
    setNames(names(ontotype))

  hiddens=list()
  outputs=list()

  setpb(pb,1)
  for(i in seq(terminal_nodes)){

    setpb(pb,1+i)
    A=unique(hierarchy$from) %>% .[!.%in%hierarchy$to] %>% .[i]
    B=A
    C=feature$n[feature$oid==A]

    hiddens[[A]]=
      layer_inception_resnet(
        object=inputs[[B]]
        ,residue=inputs[[A]]
        ,filters=max(20,ceiling(0.3*C))
        ,kernel_initializer=init
        ,name=paste0(A,'_hidden')
      )

    outputs[[A]]=
      hiddens[[A]] %>%
      layer_aux_output(
        filters=max(20,ceiling(0.3*C))
        ,units=max(20,ceiling(0.3*C))
        ,kernel_initializer=init
        ,kernel_initializer2=init2
        ,name=A
      )
  }
  j=i

  setpb(pb,2+j)
  for(i in seq(non_terminal_nodes)){

    setpb(pb,2+j+i)
    A=unique(hierarchy$to)[i]
    B=hierarchy$from[hierarchy$to==A]
    C=feature$n[feature$oid==A]

    if(length(B)==1){
      hiddens[[A]]=
        layer_inception_resnet(
          object=hiddens[[B]]
          ,residue=inputs[[A]]
          ,filters=max(20,ceiling(0.3*C))
          ,kernel_initializer=init
          ,name=paste0(A,'_hidden')
        )
    }else{
      hiddens[[A]]=
        hiddens[B] %>%
        unlist(use.names=F) %>%
        layer_concatenate(
          axis=-1
          ,name=paste0(A,'_dc')
        ) %>%
        layer_inception_resnet(
          object=.
          ,residue=inputs[[A]]
          ,filters=max(20,ceiling(0.3*C))
          ,kernel_initializer=init
          ,name=paste0(A,'_hidden')
        )
    }

    if(A!=unique(hierarchy$to)[length(unique(hierarchy$to))]){
      outputs[[A]]=
        hiddens[[A]] %>%
        layer_aux_output(
          filters=max(20,ceiling(0.3*C))
          ,units=max(20,ceiling(0.3*C))
          ,kernel_initializer=init
          ,kernel_initializer2=init2
          ,name=A
        )
    }else{
      outputs[[A]]=
        hiddens[[A]] %>%
        layer_output(
          units=max(20,ceiling(0.3*C))
          ,kernel_initializer=init
          ,kernel_initializer2=init2
          ,name=A
        )
    }

  }
  j=j+i

  rm(i,A,B,C)

  setpb(pb,3+j)
  model=keras_model(inputs=inputs,outputs=outputs)

  setpb(pb,4+j)
  if(!is.null(path)){
    model %>%
      model_to_json() %>%
      writeLines(paste0(path,'.json'))
  }

  setpb(pb,5+j)
  k_clear_session()
  if(!is.null(path)) cat('\nOntonet has been saved to',paste0(path,'.json'))

  model

}
