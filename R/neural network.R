#neural network
rm(list=ls())
gc()
library(caret)
library(keras)
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
#standardization first
#simple 1 layer example from nnet package
grid1 = expand.grid(size = c(1:10), decay = c(1:10) * 0.1)
model1 <- train(quality ~ ., data = trainset, method = "nnet", preProcess = c("center", "scale"),
                trControl = ctrl, tuneGrid = grid1)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)

#application of complex network can be found in other files


