#' Make an input example for divnn package
#'
#' This function create an input example for several function in divnn package.
#'
#' @return output A list of inputs: 1) value, a data frame with rows for
#' instances and columns for features; 2) outcome, a vector of binary integers
#' with the same length as the instances; 3) similarity, a square matrix of
#' numerics containing feature-feature similarity measures; 4) mapping, a matrix
#' of numerics with rows for features and three columns for three dimensions
#' where the features are mapped onto; and 5) ontology, a data frame with rows
#' for ontologies and four columns for source, target, similarity, and relation.
#' In addition, a result of hierarchical clustering is also included for
#' visualization purpose.
#'
#' @keywords example data
#'
#' @export
#'
#' @examples
#'
#' ## Create input example
#' input=utils.example()
#'
#' ## Show output and visualize the ontology by hierarchical clustering
#' input
#' plot(input$hierarchy)

utils.example=function(){

  ## Set an empty list of inputs and random seed
  input=list()
  set.seed(33)

  ## Create example of instance-feature data frame
  input$value=
    rnorm(3000*5) %>%
    matrix(3000,5,T,list(paste0('I',1:3000),paste0('F',1:5))) %>%
    as.data.frame()

  ## Create example of outcome vector
  ## This example uses k-means to create a classifiable outcome
  input$outcome=
   kmeans(input$value,centers=2,algorithm='Hartigan-Wong')$cluster-1

  input$outcome=setNames(as.integer(input$outcome),names(input$outcome))

  ## Create example of feature similarity matrix using Pearson correlation
  input$similarity=
    input$value %>%
    cor(method='pearson')

  ## Create example of feature three-dimensional mapping matrix
  ## This example uses PCA for simplicity
  ## DeepInsight originally uses t-SNE and kernel PCA
  input$mapping=
    prcomp(input$similarity) %>%
    .$rotation %>%
    .[,1:3] %>%
    `colnames<-`(NULL)

  ## Create example of ontology
  ## The similarity is recalculated to express dissimilarity
  ## Because this example uses hierarchical clustering for simplicity
  ## VNN originally uses Clique Extracted Ontology (CLiXO)
  input$hierarchy=
    ((1-input$similarity)/2) %>%
    as.dist() %>%
    hclust(method='complete')

  ## A function to convert a hierarchy object into an ontology data frame
  ontology_df=function(hierarchy){
    d=hierarchy$merge %>%
      as.data.frame() %>%
      mutate(target=abs(V1)) %>%
      mutate_all(function(x)ifelse(x<0,hierarchy$labels[abs(x)],paste0('ONT:',x)))

    e=filter(d,target==V1)
    f=d %>%
      filter(target!=V1) %>%
      pull(target) %>%
      str_remove_all('ONT\\:') %>%
      as.integer() %>%
      max()

    g=data.frame(target=unique(e$target)) %>%
      mutate(target2=paste0('ONT:',seq(f+1,f+nrow(.)))) %>%
      right_join(e,by='target') %>%
      mutate(target2=ifelse(is.na(target2),target,target2)) %>%
      select(-target)

    d %>%
      left_join(g,by=c('V1','V2')) %>%
      mutate(target=ifelse(is.na(target2),target,target2)) %>%
      select(-target2) %>%
      mutate(similarity=1-hierarchy$height) %>%
      gather(key,source,-target,-similarity) %>%
      select(-key) %>%
      mutate(relation=ifelse(str_detect(source,'ONT\\:'),'is_a','feature')) %>%
      arrange(desc(similarity),source) %>%
      select(source,everything())
  }

  ## Create example of ontology data frame
  input$ontology=ontology_df(input$hierarchy)

  ## Return
  input
}
