#' Write a .ts.tar.gz file from a TidySet
#'
#' This function write multiple files archived by tar with gzip compression
#' from a TidySet.
#'
#' @param tidy_set TidySet, an ExpressionSet with three tables.
#' @param path A character of .ts.tar.gz file path (do not include file
#' extension).
#'
#' @return output A .ts.tar.gz file containing exprs.csv, pData.csv,
#' fData.csv, similarity.csv, ontology.csv, and others.txt. Function of
#' \code{read_ts_tar_gz} can read this file back to a TidySet.
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

write_ts_tar_gz=function(tidy_set,path){

  dir.create(path)

  as.data.frame(exprs(tidy_set)) %>%
    write_csv(paste0(path,'/exprs.csv'),col_names=F)

  pData(tidy_set) %>%
    write_csv(paste0(path,'/pData.csv'),col_names=F)

  fData(tidy_set) %>%
    write_csv(paste0(path,'/fData.csv'),col_names=F)

  as.data.frame(notes(tidy_set)$similarity) %>%
    write_csv(paste0(path,'/similarity.csv'),col_names=F)

  notes(tidy_set)$ontology %>%
    write_csv(paste0(path,'/ontology.csv'),col_names=F)

  path=paste0(path,'/others')
  write('>>sampleNames',paste0(path,'.txt'),append=F)
  sampleNames(tidy_set) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>varLabels',paste0(path,'.txt'),append=T)
  varLabels(tidy_set) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>varMetadata',paste0(path,'.txt'),append=T)
  varMetadata(tidy_set)$labelDescription %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>varClass',paste0(path,'.txt'),append=T)
  sapply(pData(tidy_set),class) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>featureNames',paste0(path,'.txt'),append=T)
  featureNames(tidy_set) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>fvarLabels',paste0(path,'.txt'),append=T)
  fvarLabels(tidy_set) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>fvarMetadata',paste0(path,'.txt'),append=T)
  fvarMetadata(tidy_set)$labelDescription %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>fvarClass',paste0(path,'.txt'),append=T)
  sapply(fData(tidy_set),class) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>simNames',paste0(path,'.txt'),append=T)
  rownames(notes(tidy_set)$similarity) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>ontoNames',paste0(path,'.txt'),append=T)
  colnames(notes(tidy_set)$ontology) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  write('>>ontoClass',paste0(path,'.txt'),append=T)
  sapply(as.list(notes(tidy_set)$ontology),class) %>%
    write(paste0(path,'.txt'),append=T,ncolumns=1000)

  experimentData(tidy_set) %>%
    attributes() %>%
    .[names(.)%in%c(
      'name','lab','contact','title','abstract','url','pubMedIds'
    )] %>%
    .[!sapply(.,function(x)length(attributes(x))>0)] %>%
    c(list(annotation=annotation(tidy_set))) %>%
    lapply(X=seq(length(.)),Y=.,Z=paste0(path,'.txt'),function(X,Y,Z){
      write(paste0('>>',names(Y)[X]),Z,append=T)
      write(Y[[X]],Z,append=T)
    })

  path=str_remove_all(path,'/others')
  setwd(paste(c(getwd(),path),collapse='/'))
  path_=str_split(path,'\\/')[[1]] %>% .[length(.)]

  tar(
    paste0('../',path_,'.ts.tar.gz')
    ,list.files()
    ,'gzip'
    ,tar='tar'
  )
  setwd(str_remove_all(getwd(),paste0('/',path)))
  unlink(path,T)
}
