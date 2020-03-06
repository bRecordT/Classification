#SVM
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
#be careful about categorical var
#standardization first, normalization also works
#for large dataset, searching on exponential grid first, then fine tune
#linear kernel
grid1 = expand.grid(C = seq(0.1, 5, 0.1))
model1 <- train(quality ~ ., data = trainset, method = "svmLinear", preProcess = c("center", "scale"),
                trControl = ctrl, tuneGrid = grid1)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)

#polynomial
grid2 = expand.grid(C = 10 ^ c(-1:2), degree = c(1:5), scale = c(0.5, 1, 2))
model2 <- train(quality ~ ., data = trainset, method = "svmPoly", preProcess = c("center", "scale"),
                trControl = ctrl, tuneGrid = grid2)
model2
plot(model2)
pred2<-predict(model2, testset)
confusionMatrix(pred2, testset$quality)

#Gaussian radial basis function (RBF)
#sigma is gamma in the applied function
grid3 = expand.grid(C = 10 ^ c(-1:2), sigma = 10 ^ c(-1:2))
model3 <- train(quality ~ ., data = trainset, method = "svmRadial", preProcess = c("center", "scale"),
                trControl = ctrl, tuneGrid = grid3)
model3
plot(model3)
pred3<-predict(model3, testset)
confusionMatrix(pred3, testset$quality)

#there are other kernel functions that might be useful for specific scenarios

