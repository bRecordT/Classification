#KNN
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
grid <- data.frame(k = 1:10)
#be careful about categorical var when calculating distance
#without standardization
model1 <- train(quality ~ ., data = trainset, method = "knn", trControl = ctrl, tuneGrid = grid)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)
#with standardization
model2 <- train(quality ~ ., data = trainset, method = "knn", preProcess = c("center", "scale"), 
                trControl = ctrl, tuneGrid = grid)
model2
plot(model2)
pred2<-predict(model2, testset)
confusionMatrix(pred2, testset$quality)
#with normalization
model3 <- train(quality ~ ., data = trainset, method = "knn", preProcess = c("range"), 
                trControl = ctrl, tuneGrid = grid)
model3
plot(model3)
pred3<-predict(model3, testset)
confusionMatrix(pred3, testset$quality)


