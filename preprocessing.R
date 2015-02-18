library(caret)
library(Hmisc)
library(randomForest)
library(doMC)
library(data.table)
registerDoMC(cores=6)
set.seed(135790)

setwd("~/Coursera/predmachlearn-011/project/data")
training <- data.table(read.csv("pml-training.csv"))
testing <- data.table(read.csv("pml-testing.csv"))

o_training <- copy(training)

#convert variables to numeric
factors <- names(Filter( function(x) { x == "factor"}, sapply(training, class)))
num_factors <- factors[!(factors %in% c("user_name", "classe", "new_window", "cvtd_timestamp"))]
for (i in 1:length(num_factors)) {
    training[, num_factors[i] := lapply(training[, num_factors[i], with=F], as.numeric), with=F]
    testing[, num_factors[i] := lapply(testing[, num_factors[i], with=F], as.numeric), with=F]
}

#convert timestamps
training[, cvtd_timestamp := as.Date(training[, cvtd_timestamp], "%d/%m/%Y %H:%M")]
testing[, cvtd_timestamp := as.Date(testing[, cvtd_timestamp], "%d/%m/%Y %H:%M")]

#delete features where more than 50% of data is missing
missingp <- sapply(training, function(a) { sum(is.na(a))}) / dim(training)[1]
missing_features <- names(missingp[missingp > 0.5])
training[, (missing_features) := NULL]
testing[, (missing_features) := NULL]

#delete irrelevant features
irrelevant <- c("X", "user_name", 
                "raw_timestamp_part_1", "raw_timestamp_part_2", "num_window",
                "cvtd_timestamp", "new_window")
training[, (irrelevant) := NULL]
testing[, (irrelevant) := NULL]

##Pre-process numeric features
numeric <- names(Filter( function(x) { x == "numeric" | x == "integer"}, sapply(training, class)))

#delete zero_var features
zerovar <- nzv(training[, (numeric), with=F])
training[, (zerovar) := NULL, with=F]
testing[, (zerovar) := NULL, with=F]

#scale and center features
numeric <- names(Filter( function(x) { x == "numeric" | x == "integer"}, sapply(training, class)))
for (i in 1:length(numeric)) {
    preProcValues <- preProcess(training[, numeric[i], with=F]*1, method = c("center", "scale"))
    training[, numeric[i] :=predict(preProcValues, training[, numeric[i], with=F]*1), with=F]
    testing[, numeric[i] :=predict(preProcValues, testing[, numeric[i], with=F]*1), with=F]
}
