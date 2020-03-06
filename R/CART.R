#CART
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
#repeat 5 times to cancel out tree's instability nature
#use Breiman's 1-SE rule to select cp
ctrl1 <- trainControl(method = "repeatedcv", number = 10, repeats = 5, selectionFunction = "oneSE")
grid <- data.frame(cp = seq(0, 100, 5) * 0.001)
#tree model is not sensitive to var scaling
model1 <- train(quality ~ ., data = trainset, method = "rpart", trControl = ctrl1, tuneGrid = grid)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)
#alternatively, use rpart1SE method
ctrl2 <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
#tree model is not sensitive to var scaling
model2 <- train(quality ~ ., data = trainset, method = "rpart1SE", trControl = ctrl2)
model2
pred1<-predict(model2, testset)
confusionMatrix(pred2, testset$quality)

