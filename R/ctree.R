#RF
rm(list=ls())
gc()
library(caret)
library(dplyr)
library(party)
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
#try different paramerters, although significant level is pre-defined in most cases
grid <- data.frame(mincriterion = seq(0.01, 1, 0.01))
model1 <- train(quality ~ ., data = trainset, method = "ctree", trControl = ctrl, tuneGrid = grid)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)


