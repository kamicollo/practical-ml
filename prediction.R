library(caret)
library(Hmisc)
library(randomForest)
library(doMC)
library(data.table)
registerDoMC(cores=6)
set.seed(135790)

#split data into training set, cross-validation set and test set
trainIndex <- createDataPartition(training$classe, p = 0.7, list=F)
trcv1_data <- training[trainIndex[, 1]]
cv2_data <- training[-trainIndex[, 1]]
trainIndex <- createDataPartition(trcv1_data$classe, p = 0.7, list=F)
cv1_data <- trcv1_data[-trainIndex[, 1]]
tr_data <- trcv1_data[trainIndex[, 1]]

#define helper function for seeding
setSeeds <- function (seed) {
    seeds <- vector(mode="list", length=26)
    for (i in 1:25) {
        seeds[[i]] <- c(seed, seed, seed)
    } 
    seeds[[26]] <- seed 
    seeds
}

#train three different models
setnames(tr_data, "classe", ".outcome")
m_rpart <- train(.outcome ~ ., data=tr_data, method="rpart", trControl=trainControl(seeds=setSeeds(135790)))
m_rf <- train(.outcome ~ ., data=tr_data, method="rf", trControl=trainControl(seeds=setSeeds(135790)))
m_gbm <- train(.outcome ~ ., data=tr_data, method="gbm", trControl=trainControl(seeds=setSeeds(135790)))
save(m_rpart, file="n~/Coursera/predmachlearn-011/project/data/new_rpart.RData")
save(m_rf, file="~/Coursera/predmachlearn-011/project/data/new_rf.RData")
save(m_gbm, file="~/Coursera/predmachlearn-011/project/data/new_gbm.RData")

#load("~/Coursera/predmachlearn-011/project/data/new_rpart.RData")
#load("~/Coursera/predmachlearn-011/project/data/new_rf.RData")
#load("~/Coursera/predmachlearn-011/project/data/new_gbm.RData")

#get training data predictions
tr_pred_rpart <- predict(m_rpart, tr_data)
tr_pred_rf <- predict(m_rf, tr_data)
tr_pred_gbm <- predict(m_gbm, tr_data)

cm_tr_rpart <- confusionMatrix(tr_pred_rpart, tr_data$.outcome)
cm_tr_rf <- confusionMatrix(tr_pred_rf, tr_data$.outcome)
cm_tr_gbm <- confusionMatrix(tr_pred_gbm, tr_data$.outcome)

#get accuracy on cross-validation set
cv1_pred_rpart <- predict(m_rpart, cv1_data)
cv1_pred_rf <- predict(m_rf, cv1_data)
cv1_pred_gbm <- predict(m_gbm, cv1_data)

cm_cv1_rpart <- confusionMatrix(cv1_pred_rpart, cv1_data$classe)
cm_cv1_rf <- confusionMatrix(cv1_pred_rf, cv1_data$classe)
cm_cv1_gbm <- confusionMatrix(cv1_pred_gbm, cv1_data$classe)

#get out-of-sample prediction accuracy
cv2_pred_rf <- predict(m_rf, cv2_data)
cm_cv2_rf <- confusionMatrix(cv2_pred_rf, cv2_data$classe)

