
############ Q2
rm(list = ls())
dataset <- read.csv("IBM_Attrition_v3.csv", na.strings = "?")

summary(dataset)

sum(is.na(dataset))

#columns with missing values
col_with_na <- colnames(dataset)[apply(dataset, 2, anyNA)]
col_with_na

#replacing missing column values with column mean
for (i in seq_len(ncol(dataset))) {
  dataset[is.na(dataset[, i]), i] <- mean(dataset[, i], na.rm = TRUE)
}

#frequency table of Class vs F6
table(dataset[, "Class"], dataset[, "F6"])

#scatter plot of F1 to F6
pairs(dataset[, c("Age", "MonthlyIncome", "YearsAtCompany")])

#boxplot for F7 to F9
boxplot(dataset[, c("Age", "MonthlyIncome", "YearsAtCompany")])


# Clearing all variables from the memory
rm(list = ls())





############ Q4

# loading the libraries
library(e1071)
library(class) 
library("caret")
library("lattice")
library("ggplot2")

dataSet <- read.csv("IBM_Attrition_v2.csv", na.strings = c("?"))

dataSet[,1:4][dataSet[,1:4]=="?"] <- NA

dataSet <- na.omit(dataSet[,1:4])
View(dataSet)


# discretize Monthly Income
for(i in 1:length(dataSet$MonthlyIncome)){
  if(as.numeric(dataSet[i,"MonthlyIncome"])>8500){
    dataSet[i,"MonthlyIncome"]<-c("8500 or more")
  }
  else if(as.numeric(dataSet[i,"MonthlyIncome"])>3000&as.numeric(dataSet[i,"MonthlyIncome"])<=5000){
    dataSet[i,"MonthlyIncome"]<-c("3000 to 5000")
  }
  else if(as.numeric(dataSet[i,"MonthlyIncome"])>5000&as.numeric(dataSet[i,"MonthlyIncome"])<=8500){
    dataSet[i,"MonthlyIncome"]<-"5000 to 8500"
  }
  else if(as.numeric(dataSet[i,"MonthlyIncome"])<=3000){
    dataSet[i,"MonthlyIncome"]<-c("up to 3,000")
  }
}

# discretize age 
dataSet$Age<-cut(dataSet$Age , c(0,30,37,47,100))
dataSet$Age<- factor(dataSet$Age, levels = c("(0,30]","(30,37]","(37,47]","(47,100]") , labels = c("less than 31","31 up to 38","38 up to 48","48 or over"))


View(dataSet)


# dividing dataset into test(30%) and train set
idx<-sample(nrow(dataSet),0.7*nrow(dataSet))
training<-dataSet[idx,]
testing<-dataSet[-idx,]



# applying naiveBayes() 
nb <- naiveBayes(Attrition ~., data =training)
category_all<-predict(nb,testing)
table(nb_class=category_all,Attrition=testing$Attrition)

# calculating the accuracy of the model 
nb_wrong<-sum(category_all!=testing$Attrition)
NB_error_rate<-nb_wrong/length(category_all)
cat(paste0("the error rate is: ",NB_error_rate,"\n"))

cm <- table(testing$Attrition, category_all)
prop.table(cm) 
cm

# printing out confusion matrix
confusionMatrix(cm)




############ Q5
rm(list=ls())


library(rpart)
library(rpart.plot)
library(rattle)      
library(RColorBrewer)
library(rpart)

dataSet <- read.csv("IBM_Attrition_v2.csv", na.strings = c("?"))

dataSet[,1:4][dataSet[,1:4]=="?"] <- NA

dataSet <- na.omit(dataSet[,1:4])
View(dataSet)


# discretize Monthly Income
for(i in 1:length(dataSet$MonthlyIncome)){
  if(as.numeric(dataSet[i,"MonthlyIncome"])>8500){
    dataSet[i,"MonthlyIncome"]<-c("8500 or more")
  }
  else if(as.numeric(dataSet[i,"MonthlyIncome"])>3000&as.numeric(dataSet[i,"MonthlyIncome"])<=5000){
    dataSet[i,"MonthlyIncome"]<-c("3000 to 5000")
  }
  else if(as.numeric(dataSet[i,"MonthlyIncome"])>5000&as.numeric(dataSet[i,"MonthlyIncome"])<=8500){
    dataSet[i,"MonthlyIncome"]<-"5000 to 8500"
  }
  else if(as.numeric(dataSet[i,"MonthlyIncome"])<=3000){
    dataSet[i,"MonthlyIncome"]<-c("up to 3,000")
  }
}

# discretize age 
dataSet$Age<-cut(dataSet$Age , c(0,30,37,47,100))
dataSet$Age<- factor(dataSet$Age, levels = c("(0,30]","(30,37]","(37,47]","(47,100]") , labels = c("less than 31","31 up to 38","38 up to 48","48 or over"))


View(dataSet)


# dividing dataset into test(30%) and train set
idx<-sample(nrow(dataSet),0.7*nrow(dataSet))
training<-dataSet[idx,]
testing<-dataSet[-idx,]

CART_class<-rpart(Attrition ~. ,data=training)
rpart.plot(CART_class)
CART_predict2<-predict(CART_class,testing, type="class") 
table(Actual=testing[,4],CART=CART_predict2)

table(Actual=testing[,4],CART=CART_predict_cat)
CART_wrong<-sum(testing[,4]!=CART_predict_cat)
CART_error_rate<-CART_wrong/length(testing[,4])
CART_error_rate
CART_predict2<-predict(CART_class,testing, type="class")
CART_wrong2<-sum(testing[,4]!=CART_predict2)
CART_error_rate2<-CART_wrong2/length(testing[,4])
CART_error_rate2 

library(rpart.plot)
prp(CART_class)


# much fancier graph
fancyRpartPlot(CART_class)


############ Q6
rm(list=ls())

library(kknn)

dataset = read.csv('IBM_Attrition_v2.csv',header=TRUE)
dataset_clean<-na.omit(dataset)


index<-sort(sample(nrow(dataset_clean),round(.30*nrow(dataset_clean ))))
training<- dataset_clean[-index,]
test<- dataset_clean[index,]


predict_k1 <- kknn(formula= as.factor(Attrition)~., training[,c(-1)] , test[,c(-1)], k=3, kernel='rectangular')

fit <- fitted(predict_k1)
table(test$Attrition,fit)

#Performance of knn
wrong<- ( test$Attrition!=fit)
rate<-sum(wrong)/length(wrong)
rate


## Clearing all variables
rm(list=ls())
