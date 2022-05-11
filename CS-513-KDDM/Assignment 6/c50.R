#clear all variables
rm(list=ls())
dataset <-read.csv('breast-cancer-wisconsin.csv')


cols <- colnames(dataset[,2:11])
dataset$Class<- factor(dataset$Class , levels = c("2","4") , labels = c("Benign","Malignant"))
dataset[cols] <- lapply(dataset[cols], factor)
str(dataset)
dataset <- dataset[,2:11]

library('C50')
set.seed(111)
index<-sort(sample(nrow(dataset),round(0.3*nrow(dataset))))
train<-dataset[-index,]
test<-dataset[index,]
c50_class<-C5.0(Class~.,data=train)
plot(c50_class)
c50_cls<-predict(c50_class, test, type='class')
summary(c50_class)
c50_cls<-predict(c50_class, test, type='class')
cm = as.matrix(table(Actual = test[,10], Predicted = c50_cls))
cm
accuracy<-sum(diag(cm))/length(test[,10])
accuracy
err <-(1 - accuracy)
err