#NAME: Varun Reddy Doddipalli
#CWID: 10474196
#Q1


# Clear  the environment
rm(list=ls())

Attrition_DataSet <- read.csv("IBM_Attrition_v3.csv", na.string = "?")


which(is.na(Attrition_DataSet$Age))
MissingAge<-Attrition_DataSet[which(is.na(Attrition_DataSet$Age)),]
View(MissingAge)

which(is.na(Attrition_DataSet$JobSatisfaction))
MissingJobSatisfaction<-Attrition_DataSet[which(is.na(Attrition_DataSet$JobSatisfaction)),]
View(MissingJobSatisfaction)

which(is.na(Attrition_DataSet$MaritalStatus))
MissingMaritalStatus<-Attrition_DataSet[which(is.na(Attrition_DataSet$MaritalStatus)),]
View(MissingMaritalStatus)

which(is.na(Attrition_DataSet$MonthlyIncome))
MissingMonthlyIncome<-Attrition_DataSet[which(is.na(Attrition_DataSet$MonthlyIncome)),]
View(MissingMonthlyIncome)

which(is.na(Attrition_DataSet$YearsAtCompany))
MissingYearsAtCompany<-Attrition_DataSet[which(is.na(Attrition_DataSet$YearsAtCompany)),]
View(MissingYearsAtCompany)

which(is.na(Attrition_DataSet$Attrition))
MissingAttrition<-Attrition_DataSet[which(is.na(Attrition_DataSet$Attrition)),]
View(MissingAge)

Attrition_Modifed <- na.omit(Attrition_DataSet)

View(Attrition_Modifed)


Attrition_Modifed$MonthlyIncome_Categorized <- as.factor(ifelse(Attrition_Modifed$MonthlyIncome<2900, 'income1',
                                     ifelse(Attrition_Modifed$MonthlyIncome<5000, 'income2',
                                     ifelse(Attrition_Modifed$MonthlyIncome<8500, 'income3', 'income4'))))
Attrition_Modifed$YearsAtCompany_Categorized <- as.factor(ifelse(Attrition_Modifed$YearsAtCompany<7, 'not-senior','senior'))
Attrition_Modifed$Age_Categorized <- as.factor(ifelse(Attrition_Modifed$Age<38, 'young','mature'))
Attrition_Modifed <- Attrition_Modifed[ , c("JobSatisfaction","MaritalStatus","Attrition","MonthlyIncome_Categorized","YearsAtCompany_Categorized","Age_Categorized")]

# View the final data set 

View(Attrition_Modifed)


