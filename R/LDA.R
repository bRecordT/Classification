#discriminant analysis
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
#no need to scale for DA
#be careful about var types: continuous independent var and categorical dependent var
#contrast to ANOVA
#LDA
model1 <- train(quality ~ ., data = trainset, method = "lda", trControl = ctrl)
model1
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)

#QDA
model2 <- train(quality ~ ., data = trainset, method = "qda", trControl = ctrl)
model2
pred2<-predict(model2, testset)
confusionMatrix(pred2, testset$quality)




