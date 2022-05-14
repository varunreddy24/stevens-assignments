rm(list=ls())
dataset = read.csv('breast-cancer-wisconsin.csv',header=TRUE)
num <- as.numeric(as.character(dataset$F6))

dataset$F6 <- num
dataset <- na.omit(dataset)
dataset$Class<- factor(dataset$Class , levels = c("2","4") , labels = c("Benign","Malignant"))
size <- sample(1:nrow(dataset), 0.7 * nrow(dataset))
nr <-function(x){(x -min(x))/(max(x)-min(x))}
norm <- as.data.frame(lapply(dataset[,c(2,3,4,5,6,7,8,9,10)], nr))
dataset = dataset['Class']
train <- norm[size,]
new_train <- dataset[size,]
test <- norm[-size,] 
new_test <- dataset[-size,]

library(class)

k_values = c(3,5,10)

for (k in k_values){
  print(paste("K Value: ",k))
  clf <- knn(train,test,cl=new_train,k=k)
  confusion_matrix <- table(clf, new_test)
  print(confusion_matrix)
  cat("\n")
}
