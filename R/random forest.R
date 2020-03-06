#RF
rm(list=ls())
gc()
library(caret)
library(dplyr)
setwd("G:/Archive")

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
grid <- data.frame(mtry = 1:11)
#rf model is not sensitive to var scaling
model1 <- train(quality ~ ., data = trainset, method = "rf", trControl = ctrl, tuneGrid = grid)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)


