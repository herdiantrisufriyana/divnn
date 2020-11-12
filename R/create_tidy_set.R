#' Make a TidySet for visible neural network (VNN) modeling
#'
#' This function create a TidySet, an ExpressionSet class to orchestrate five
#' data into a single set of three tables.
#'
#' @param value Instance-feature value, a data frame with rows for instances
#' and columns for features. All rows in \code{value} should have names. All
#' values should be numerics.
#' @param outcome Outcome, a vector of binary integers with the same length as
#' the instances. The length and the order of \code{outcome} should be the same
#' with those of \code{value}. Value  of 0 and 1 should refer to non-event and
#' event outcome, respectively.
#' @param similarity Feature similarity, a square matrix of numerics containing
#' feature-feature similarity measures.
#' @param mapping Feature three-dimensional mapping, a matrix of integers with
#' rows for features and three columns for three dimensions where the features
#' are mapped onto.
#' @param ontology Ontology, a data frame with rows for ontologies and four
#' columns for source, target, similarity, and relation. Feature (source)-
#' ontology (target) relation should be annotated as 'feature', while ontology-
#' ontology relation should be annotated as 'is_a'. To differentiate between
#' feature and ontology names, a prefix of 'ONT:' precedes an ontology name. All
#' columns except similarity in \code{ontology} should be characters. Similarity
#' (a numeric) is a minimum threshold by which either features or ontologies
#' (source) belong to an ontology (target).
#'
#' @return output TidySet, an ExpressionSet with three tables. Instance-feature
#' value data frame and outcome vector are compiled as a phenotype data frame
#' with rows for instances and columns for features and outcome. Instance-
#' feature value data frame and feature three-dimensional mapping matrix are
#' compiled as an expression matrix with rows for positions of features and
#' columns for instances. The mapping and similarity matrices and ontology data
#' frame are compile as a feature data frame with rows for positions of features
#' and columns for feature names and ontological relations. For easier access,
#' the original ontology data frame is included in experiment information at
#' section 'notes' as a list.
#'
#' @keywords TidySet, ExpressionSet
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
#' ## The TidySet
#' tidy_set
#'
#' ## The phenotype data frame
#' pData(tidy_set)
#'
#' ## The feature data frame
#' fData(tidy_set)
#'
#' ## The expression data frame
#' exprs(tidy_set)
#'
#' ## The original ontology data frame
#' tidy_set %>%
#'   experimentData() %>%
#'   notes() %>%
#'   .$ontology

create_tidy_set=function(value
                         ,outcome
                         ,similarity
                         ,mapping
                         ,ontology
                         ,ranked=T
                         ,dims=7
                         ,decreasing=F
                         ,seed_num=33){

  pb=startpb(0,4)
  on.exit(closepb(pb))
  setpb(pb,0)

  rotate_2_col_mat=function(X,angle){
    angle=(pi/180*angle)*-1
    M=matrix(c(cos(angle),-sin(angle),sin(angle),cos(angle)),2,2)
    M=X %*% M
    dimnames(M)=dimnames(X)
    M
  }

  create_fmap=function(mapping,similarity,angle,ranked=T,dims=7){
    data=
      data.frame(
        feature=rownames(similarity)
        ,dim1=mapping[,1]
        ,dim2=mapping[,2]
        ,dim3=mapping[,3]
      )

    if(ranked){
      data=
        data %>%
        arrange(dim1) %>%
        mutate(dim1=seq(nrow(.))) %>%
        arrange(dim2) %>%
        mutate(dim2=seq(nrow(.)))
    }

    data %>%
      arrange(dim1) %>%
      mutate(resize_x=seq(1,dims,length.out=nrow(.)) %>% round()) %>%
      arrange(dim2) %>%
      mutate(resize_y=seq(1,dims,length.out=nrow(.)) %>% round()) %>%
      cbind(
        data.frame(rot_x=.$resize_x,rot_y=.$resize_y) %>%
          as.matrix() %>%
          rotate_2_col_mat(angle)
      ) %>%
      arrange(rot_x) %>%
      mutate(x=seq(1,dims,length.out=nrow(.)) %>% round()) %>%
      arrange(rot_y) %>%
      mutate(y=seq(1,dims,length.out=nrow(.)) %>% round()) %>%
      arrange(dim3) %>%
      lapply(
        X=seq(nrow(select(.,x,y) %>% .[!duplicated(.),])),
        Y=select(.,x,y) %>% .[!duplicated(.),],
        Z=.,
        FUN=function(X,Y,Z){
          Z %>%
            filter(x==Y$x[X] & y==Y$y[X]) %>%
            mutate(z=seq(nrow(.)))
        }
      ) %>%
      do.call(rbind,.) %>%
      column_to_rownames(var='feature') %>%
      .[match(rownames(similarity),rownames(.)),] %>%
      select(x,y,z) %>%
      arrange(z,y,x)
  }

  order_angle_by_channel=function(mapping
                                  ,similarity
                                  ,ranked=T
                                  ,dims=7
                                  ,decreasing=F){
    lapply(
        1:360
        ,FUN=create_fmap
        ,mapping=mapping
        ,similarity=similarity
        ,ranked=ranked
        ,dims=dims
      ) %>%
      sapply(function(x)max(x$z)) %>%
      setNames(1:360) %>%
      .[order(.,decreasing=decreasing)]
  }

  setpb(pb,1)
  set.seed(seed_num)
  angle=
    order_angle_by_channel(mapping,similarity,ranked,dims,decreasing) %>%
    lapply(X=1,Y=.,function(X,Y)as.integer(names(Y)[Y==min(Y)])) %>%
    .[[1]] %>%
    .[sample(seq(length(.)),1,F)]

  setpb(pb,2)
  fmap=
    create_fmap(mapping,similarity,angle,ranked,dims)

  fval=
    value[,rownames(fmap)] %>%
    t() %>%
    as.data.frame()

  fboth=
    fmap %>%
    summarize_all(max) %>%
    as.list() %>%
    lapply(seq) %>%
    expand.grid() %>%
    setNames(c('x','y','z')) %>%
    arrange(z,y,x) %>%
    left_join(rownames_to_column(fmap,var='feature'),by=c('x','y','z')) %>%
    cbind(fval[.$feature,]) %>%
    mutate(x=paste0('x',x)) %>%
    unite(pos_id,x,y,sep='y') %>%
    unite(pos_id,pos_id,z,sep='z')

  ori_ontology=ontology

  while(sum(str_detect(ontology$source,'ONT\\:'))>0){
    ontology=
      ontology %>%
      lapply(X=1:nrow(.),Y=.,function(X,Y){
        if(str_detect(Y$source[X],'ONT\\:')){
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

  setpb(pb,3)
  adata=
    fboth %>%
    select(-feature) %>%
    column_to_rownames(var='pos_id') %>%
    t() %>%
    t()

  adata[is.na(adata)]=0

  pdata=
    value %>%
    .[,rownames(fmap)] %>%
    as.data.frame() %>%
    mutate(outcome=outcome) %>%
    select(outcome,everything()) %>%
    `rownames<-`(colnames(adata))

  fdata=
    fboth %>%
    select(pos_id,feature) %>%
    left_join(
      ontology %>%
        select(source,target) %>%
        .[!duplicated(.),] %>%
        mutate(included=1) %>%
        spread(target,included) %>%
        rename_all(str_replace_all,'ONT\\:','ONT') %>%
        rename(feature=source)
      ,by='feature'
    ) %>%
    column_to_rownames(var='pos_id')

  output=
    ExpressionSet(
      assayData=assayData(ExpressionSet(adata))
      ,phenoData=AnnotatedDataFrame(pdata)
      ,featureData=AnnotatedDataFrame(fdata)
      ,experimentData=MIAME(other=list(ontology=ori_ontology))
    )

  gc()
  setpb(pb,4)
  output

}
