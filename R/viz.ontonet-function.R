#' Visualize an ontonet for visible neural network (VNN) modeling
#'
#' This function visualizes an ontonet, including the evaluation results for 
#' each ontology.
#'
#' @param tidy_set TidySet, an ExpressionSet with three tables.
#' @param feature A logical whether feature nodes are shown; otherwise, only 
#' ontology nodes are shown.
#' @param eval.results A list of resampled results or a single result from 
#' \code{evaluate_generator()} of \code{keras} package.
#' @param eval.metric A character of metric name to visualize.
#' @param eval.pal A vector of two characters for colors representing the 
#' minimum and maximum values of the metric.
#' @param eval.gradient An integer of the number of possible colors between 
#' ones for the minimum and maximum values of the metric.
#' @param node.color A character of color for bordering node.
#' @param node.fill A character of color for filling node.
#' @param node.shape A character of node shape. Run \code{vertex.shapes()} from 
#' \code{igraph} package to find the options.
#' @param node.size A numeric of node size.
#' @param label A logical whether the nodes are labeled.
#' @param label.family A character of font family for node label.
#' @param label.cex A numeric of font size for node label.
#' @param label.color A character of font color for node label.
#' @param edge.color A character of color for edge.
#' @param seed_num A integer of random seed for network construction algorithm.
#'
#' @return viz.ontonet object, a list of node and edge data frames with the 
#' visualization parameters. Visualization are shown by \code{plot()}.
#'
#' @keywords ontonet visualization
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
#'
#' ## Visualize ontonet
#' viz.ontonet(tidy_set)

viz.ontonet=function(tidy_set
                     ,feature=F
                     ,eval.results=NULL
                     ,eval.metric='loss'
                     ,eval.pal=c('darkred','darkgreen')
                     ,eval.gradient=100){
  
  edge=
    notes(tidy_set)$ontology %>%
    filter(!target%in%source) %>%
    select(target) %>%
    filter(!duplicated(target)) %>%
    rename(source=target) %>%
    mutate(
      target='root'
      ,similarity=NA
      ,relation='is_a'
    ) %>%
    rbind(notes(tidy_set)$ontology) %>%
    slice(c(2:nrow(.),1))
  
  if(!feature) edge=edge %>% filter(relation!='feature')
  
  if(is.null(eval.results)){
    node=
      edge %>%
      graph_from_data_frame(directed=TRUE) %>%
      V() %>%
      names() %>%
      data.frame(node=.) %>%
      mutate(avg=NA,color=NA)
  }else{
    node=
      eval.results %>%
      sapply(unlist) %>%
      t() %>%
      as.data.frame() %>%
      gather() %>%
      group_by(key) %>%
      summarize(
        avg=mean(value)
        ,lb=mean(value)-qnorm(0.975)*sd(value)/sqrt(n())
        ,ub=mean(value)+qnorm(0.975)*sd(value)/sqrt(n())
        ,.groups='drop'
      ) %>%
      mutate(
        node=
          case_when(
            str_detect(key,'ONT') & !str_detect(key,'lr')~'ONT'
            ,str_detect(key,'root') & !str_detect(key,'lr')~'root'
            ,str_detect(key,'lr')~'lr'
            ,TRUE~'total'
          ) %>%
          paste0(str_remove_all(key,'[:alpha:]|[:punct:]'))
        ,key=str_remove_all(key,'root|val|ONT|[:digit:]|[:punct:]')
      ) %>%
      group_by(key) %>%
      mutate(norm.avg=(avg-min(avg))/(max(avg)-min(avg))) %>%
      ungroup() %>%
      mutate(
        color=
          sapply(norm.avg,function(x){
            data.frame(
              avg=
                cut(seq(0,1,len=eval.gradient),eval.gradient) %>%
                as.character() %>%
                str_remove_all('\\(|\\]')
              ,color=
                colorRampPalette(eval.pal)(eval.gradient)
            ) %>%
              separate(avg,c('from','to'),',') %>%
              mutate_at(c('from','to'),as.numeric) %>%
              filter(from<x & to>=x) %>%
              pull(color)
          })
      ) %>%
      mutate(node=str_replace_all(node,'ONT','ONT:')) %>%
      filter(key==eval.metric)
    
    node=
      edge %>%
      graph_from_data_frame(directed=TRUE) %>%
      V() %>%
      names() %>%
      data.frame(node=.) %>%
      left_join(node,by='node')
  }
  
  output=
    list(
      node=node
      ,edge=edge
      ,feature=feature
      ,eval.results=eval.results
      ,eval.metric=eval.metric
      ,eval.pal=eval.pal
      ,eval.gradient=eval.gradient
    )
  
  class(output)='viz.ontonet'
  
  assign(x='print.viz.ontonet',envir=baseenv(),value=function(x){
    cat(paste0(
      '
    Method: viz.ontonet

    Ontonet node and edge tables for visualization purpose
    
    Total:
      ',nrow(x$node),' node',ifelse(nrow(x$node)==1,'','s'),'
      ',nrow(x$edge),' edge',ifelse(nrow(x$edge)==1,'','s'),'
    '
    ))
  })
  
  assign(x='plot.viz.ontonet'
         ,envir=baseenv()
         ,value=function(x
                         ,node.color=NA
                         ,node.fill='black'
                         ,node.shape='circle'
                         ,node.size=20
                         ,label=F
                         ,label.family='sans'
                         ,label.cex=0.5
                         ,label.color='white'
                         ,edge.color='black'
                         ,seed_num=33
                         ,...){
    
    node=
      node %>%
      mutate(color=ifelse(is.na(color),node.fill,color))
    
    set.seed(seed_num)
    ontograph=
      x$edge %>%
      graph_from_data_frame(directed=TRUE)
    
    V(ontograph)$color=node$color
    V(ontograph)$shape=node.shape
    if(label){
      V(ontograph)$label=
        ifelse(
          is.na(x$node$avg)
          ,names(V(ontograph))
          ,paste0(names(V(ontograph)),'\n',round(x$node$avg,3))
        )
      V(ontograph)$label.family=label.family
      V(ontograph)$label.cex=label.cex
      V(ontograph)$label.color=label.color
    }else{
      V(ontograph)$label=NA
    }
    
    ontograph %>%
      plot.igraph(
        layout=layout_as_tree(.,mode='in')
        ,vertex.frame.color=node.color
        ,vertex.size=node.size
        ,edge.color=edge.color
        ,...
      )
    
  })
  
  output
  
}
