#regularization
rm(list=ls())
gc()
library(caret)
library(dplyr)
library(glmnet)
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
#standardization before regularization
#LASSO
grid1 <- expand.grid(alpha = 1, lambda = 10 ^ seq(0, -2, -0.1))
model1 <- train(quality ~ ., data = trainset, method = "glmnet", preProcess = c("center", "scale"), 
                trControl = ctrl, tuneGrid = grid1)
model1
plot(model1)
pred1<-predict(model1, testset)
confusionMatrix(pred1, testset$quality)

#ridge
grid2 <- expand.grid(alpha = 0, lambda = 10 ^ seq(0, -2, -0.1))
model2 <- train(quality ~ ., data = trainset, method = "glmnet", preProcess = c("center", "scale"), 
                trControl = ctrl, tuneGrid = grid2)
model2
plot(model2)
pred2<-predict(model2, testset)
confusionMatrix(pred2, testset$quality)

#elastic net
grid3 <- expand.grid(alpha = seq(0, 100, 5) * 0.01, lambda = 10 ^ seq(0, -2, -0.1))
model3 <- train(quality ~ ., data = trainset, method = "glmnet", preProcess = c("center", "scale"), 
                trControl = ctrl, tuneGrid = grid3)
model3
plot(model3)
pred3<-predict(model3, testset)
confusionMatrix(pred3, testset$quality)

#fit final model to check coefficient regularization
scaling <- preProcess(trainset[,-12], method = c("center", "scale"))
trainX <- as.matrix(predict(scaling, trainset[,-12]))
testX <- predict(scaling, testset[,-12])

fitmodel1<-glmnet(x=trainX, y=trainset[,12], family = c("binomial"), alpha = 1, lambda = model1$bestTune[2])
coef(fitmodel1)
fitmodel2<-glmnet(x=trainX, y=trainset[,12], family = c("binomial"), alpha = 0, lambda = model2$bestTune[2])
coef(fitmodel2)
fitmodel3<-glmnet(x=trainX, y=trainset[,12], family = c("binomial"), alpha = model3$bestTune[1], 
                  lambda = model3$bestTune[2])
coef(fitmodel3)

#parameter tuning can also be achieved directly using cv.glmnet function


