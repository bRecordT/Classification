#boosting
rm(list=ls())
gc()
library(caret)
library(dplyr)
setwd("G:/*****")

#load data
data <- read.csv("data/wine quality red.csv", header = T, stringsAsFactors = T)
data$quality <- factor(data$quality, ordered = T)
summary(data)
#aggregate into binary classification problem
data2 <- data %>% mutate(quality = factor(ifelse(quality <= 5, "below_avg", "above_avg")))
summary(data2)

#split data by 80 - 20
set.seed(1234)
trainindex <- createDataPartition(data2$quality, p = 0.8, list = F)
trainset <- data2[trainindex,]
testset <- data2[-trainindex,]

#10-fold cv on trainset
ctrl <- trainControl(method = "cv", number = 10)
#adaptive boosting
grid1 = expand.grid(nIter = c(1:20) *10, method = "adaboost")
model1 <- train(quality ~ ., data = trainset, method = "adaboost", preProcess = c("center", "scale"),
                trControl = ctrl, tuneGrid = grid1)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)

#gradient boosting
grid2 = expand.grid(n.trees = c(1:20) *10, interaction.depth = 2, shrinkage = 0.1, n.minobsinnode = 10)
model2 <- train(quality ~ ., data = trainset, method = "gbm", preProcess = c("center", "scale"),
                trControl = ctrl, tuneGrid = grid2)
model2
plot(model2)
pred2<-predict(model2, testset)
confusionMatrix(pred2, testset$quality)

#extreme gradient boosting
grid3 = expand.grid(nrounds = c(1:20) *10, max_depth = 2, eta = 0.1, gamma = 0,
                    colsample_bytree = 0.5, min_child_weight = 1, subsample = 0.5)
model3 <- train(quality ~ ., data = trainset, method = "xgbTree", preProcess = c("center", "scale"),
                trControl = ctrl, tuneGrid = grid3)
model3
plot(model3)
pred3<-predict(model3, testset)
confusionMatrix(pred3, testset$quality)

#detailed application of complex model can be found in other files



