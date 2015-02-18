pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("submissions/problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

final_predictions <- predict(m_rf, testing)
pml_write_files(as.character(final_predictions))