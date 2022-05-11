#clear all variables
rm(list=ls())

dataset <-read.csv('breast-cancer-wisconsin.csv')
cols <- colnames(dataset[,2:11])
dataset$Class<- factor(dataset$Class , levels = c("2","4") , labels = c("Benign","Malignant"))
dataset[cols] <- lapply(dataset[cols], factor)
str(dataset)
dataset <- dataset[,2:11]

library('randomForest')
set.seed(111)
index<-sort(sample(nrow(dataset),round(0.3*nrow(dataset))))
train<-dataset[-index,]
test<-dataset[index,]

fit <- randomForest( Class~., data=train, importance=TRUE, ntree=1000)
importance(fit)
varImpPlot(fit)
Prediction <- predict(fit, test)
table(actual=test[,4],Prediction)
cm = as.matrix(table(Actual = test[,10], Predicted = Prediction))
cm
accuracy<-sum(diag(cm))/length(test[,10])
accuracy
err <-(1 - accuracy)
err