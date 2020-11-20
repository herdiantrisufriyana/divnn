#' Read a .ts.tar.gz file to a TidySet
#'
#' This function read multiple files archived by tar with gzip compression
#' to a TidySet.
#'
#' @param path A character of .ts.tar.gz file path (include file extension).
#'
#' @return output A TidySet, an ExpressionSet with three tables. Function of
#' \code{write_ts_tar_gz} can write this file from the TidySet.
#'
#' @keywords .ts.tar.gz, TidySet
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
#' ## Write a .ts.tar.gz file from a TidySet
#' write_tidy_set(tidy_set,'example')
#'
#' ## Read a .ts.tar.gz file to a TidySet
#' read_ts_tar_gz('example.ts.tar.gz')

read_ts_tar_gz=function(path){

  filename=path
  path=str_remove_all(filename,'.ts.tar.gz')
  dir.create(path)
  untar(filename,exdir=path)

  others=
    readLines(paste0(path,'/others.txt')) %>%
    lapply(X=1,other=.,function(X,other){
      elements=
        other %>%
        str_detect('^>>') %>%
        which()

      elements %>%
        lapply(
          X=seq(length(.))
          ,Y=.
          ,Z=seq(length(other))
          ,function(X,Y,Z){
            if(X<length(Y)){
              Z %>% .[.>Y[X] & .<Y[X+1]]
            }else{
              Z %>% .[.>Y[X]]
            }
          }
        ) %>%
        lapply(
          X=seq(length(.))
          ,Y=.
          ,Z=other
          ,function(X,Y,Z){
            paste(Z[Y[[X]]],collapse=' ')
          }
        ) %>%
        setNames(str_remove_all(other[elements],'>>'))
    }) %>%
    .[[1]]

  adata=
    paste0(path,'/exprs.csv') %>%
    read_csv(
      col_names=str_split(others$sampleNames,'\\s')[[1]]
      ,col_types=
        str_split(others$sampleNames,'\\s')[[1]] %>%
        length() %>%
        rep(x='d',times=.) %>%
        setNames(str_split(others$sampleNames,'\\s')[[1]])
    ) %>%
    as.data.frame() %>%
    `rownames<-`(str_split(others$featureNames,'\\s')[[1]]) %>%
    as.matrix()

  pdata=
    paste0(path,'/pData.csv') %>%
    read_csv(
      col_names=str_split(others$varLabels,'\\s')[[1]]
      ,col_types=
        str_split(others$varClass,'\\s')[[1]] %>%
        sapply(function(x){
          case_when(
            x=='numeric'~'d'
            ,x=='integer'~'i'
            ,x=='factor'~'f'
            ,x=='logical'~'l'
            ,TRUE~'c'
          )
        }) %>%
        setNames(str_split(others$varLabels,'\\s')[[1]])
    ) %>%
    as.data.frame()

  pdata=
    pdata %>%
    mutate(
      outcome=
        setNames(
          pdata$outcome
          ,str_split(others$sampleNames,'\\s')[[1]]
        )
    ) %>%
    `rownames<-`(str_split(others$sampleNames,'\\s')[[1]]) %>%
    `attributes<-`(attributes(.)[c('names','row.names','class')])

  pdata=
    pdata %>%
    AnnotatedDataFrame(
      varMetadata=
        str_split(others$varMetadata,'\\s')[[1]] %>%
        data.frame(labelDescription=.) %>%
        mutate(
          labelDescription=
            ifelse(labelDescription=='NA',NA,labelDescription)
        ) %>%
        `rownames<-`(str_split(others$varLabels,'\\s')[[1]])
    )

  fdata=
    paste0(path,'/fData.csv') %>%
    read_csv(
      col_names=str_split(others$fvarLabels,'\\s')[[1]]
      ,col_types=
        str_split(others$fvarClass,'\\s')[[1]] %>%
        sapply(function(x){
          case_when(
            x=='numeric'~'d'
            ,x=='integer'~'i'
            ,x=='factor'~'f'
            ,x=='logical'~'l'
            ,TRUE~'c'
          )
        }) %>%
        setNames(str_split(others$fvarLabels,'\\s')[[1]])
    ) %>%
    as.data.frame() %>%
    `rownames<-`(str_split(others$featureNames,'\\s')[[1]]) %>%
    `attributes<-`(attributes(.)[c('names','row.names','class')])

  similarity=
    paste0(path,'/similarity.csv') %>%
    read_csv(
      col_names=str_split(others$simNames,'\\s')[[1]]
      ,col_types=
        str_split(others$simNames,'\\s')[[1]] %>%
        length() %>%
        rep(x='d',times=.) %>%
        setNames(str_split(others$simNames,'\\s')[[1]])
    ) %>%
    as.data.frame() %>%
    `rownames<-`(str_split(others$simNames,'\\s')[[1]]) %>%
    as.matrix()

  ontomap=
    adata %>%
    t() %>%
    array(
      dim=
        c(dim(.)[1]
          ,colnames(.) %>%
            lapply(str_split_fixed,'x|y|z',4) %>%
            sapply(as.integer) %>%
            t() %>%
            .[,2:4] %>%
            colMaxs()
        )
      ,dimnames=list(rownames(.),NULL,NULL,NULL)
    )

  ontotype=
    fdata %>%
    lapply(X=seq(ncol(.)-1),Y=.,function(X,Y){
      Z=Y %>%
        select(-feature) %>%
        .[,X,drop=F] %>%
        rownames_to_column(var='pos_id') %>%
        setNames(c('pos_id','ontotype')) %>%
        filter(ontotype==1) %>%
        left_join(
          rownames_to_column(Y,var='pos_id') %>%
            select(pos_id,feature),by='pos_id'
        )

      K=Z %>%
        pull(pos_id) %>%
        str_split_fixed('x|y|z',4) %>%
        .[,2:4]

      matrix(
        as.integer(K)
        ,ncol=3
        ,byrow=F
        ,dimnames=list(Z$feature,c('x','y','z'))
      )
    }) %>%
    setNames(fdata %>% colnames(.) %>% .[.!='feature']) %>%
    c(list(
      root=
        fdata  %>%
        rownames_to_column(var='pos_id') %>%
        select(pos_id,feature) %>%
        filter(!is.na(feature)) %>%
        lapply(X=1,Y=.,function(X,Y){
          Z=Y %>%
            pull(pos_id) %>%
            str_split_fixed('x|y|z',4) %>%
            .[,2:4]
          matrix(
            as.integer(Z)
            ,ncol=3
            ,byrow=F
            ,dimnames=list(Y$feature,c('x','y','z'))
          )
        }) %>%
        .[[1]]
    ))

  fdata=
    fdata %>%
    AnnotatedDataFrame(
      varMetadata=
        str_split(others$fvarMetadata,'\\s')[[1]] %>%
        data.frame(labelDescription=.) %>%
        mutate(
          labelDescription=
            ifelse(labelDescription=='NA',NA,labelDescription)
        ) %>%
        `rownames<-`(str_split(others$fvarLabels,'\\s')[[1]])
    )

  ontology=
    paste0(path,'/ontology.csv') %>%
    read_csv(
      col_names=str_split(others$ontoNames,'\\s')[[1]]
      ,col_types=
        str_split(others$ontoClass,'\\s')[[1]] %>%
        sapply(function(x){
          case_when(
            x=='numeric'~'d'
            ,x=='integer'~'i'
            ,x=='factor'~'f'
            ,x=='logical'~'l'
            ,TRUE~'c'
          )
        }) %>%
        setNames(str_split(others$ontoNames,'\\s')[[1]])
    ) %>%
    as.data.frame() %>%
    `attributes<-`(attributes(.)[c('names','row.names','class')])

  xData=
    new(
      'MIAME'
      ,name=others$name
      ,lab=others$lab
      ,contact=others$contact
      ,title=others$title
      ,abstract=others$abstract
      ,url=others$url
      ,pubMedIds=others$pubMedIds
      ,other=
        list(
          similarity=similarity
          ,ontomap=ontomap
          ,ontotype=ontotype
          ,ontology=ontology
        )
    )

  unlink(path,T)

  eset=
    ExpressionSet(
      assayData=adata
      ,phenoData=pdata
      ,featureData=fdata
      ,experimentData=xData
    )

  if(paste0(str_split(others$annotation,'\\s')[[1]],collapse='')==''){
    annotation(eset)=''
  }

  eset
}
