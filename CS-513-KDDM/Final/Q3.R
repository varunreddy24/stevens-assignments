
# Running without clearing variables

View(Attrition_Modifed)

# Test
train_inds <- seq(1, nrow(Attrition_Modifed), 3)
test <- Attrition_Modifed[train_inds, ]
train <- Attrition_Modifed[-train_inds, ]

View(train)
View(test)

Attrition_Modifed$Attrition = factor(Attrition_Modifed$Attrition)

library(C50)

train$Attrition<-as.factor(train$Attrition)
str(train$Attrition)

C50_class <- C5.0(Attrition~.,data=train)
C50_predict<-predict( C50_class ,test , type="class" )

table(actual=test$Attrition,C50=C50_predict)
c50_wrong<- (test$Attrition!=C50_predict)
c50_errorRate<-sum(c50_wrong)/length(test$Attrition)
c50_errorRate

c50_accuracy <- 1-c50_errorRate
c50_accuracy


