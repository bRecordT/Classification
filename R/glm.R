#glm: binary logistic regression
rm(list=ls())
gc()
library(caret)
library(dplyr)
library(car)
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
#full var list
model1 <- train(quality ~ ., data = trainset, method = "glm", trControl = ctrl)
model1
#check var coef and p-value
summary(model1)
vif(model1$finalModel)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)

#filter based on significance
model2 <- train(quality ~ volatile_acidity + chlorides + free_sulfur_dioxide +
                  total_sulfur_dioxide + sulphates + alcohol, data = trainset, method = "glm", trControl = ctrl)
model2
summary(model2)
vif(model2$finalModel)
pred2<-predict(model2, testset)
confusionMatrix(pred2, testset$quality)


