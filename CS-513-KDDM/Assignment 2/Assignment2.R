rm(list = ls())
dataset <- read.csv("breast-cancer-wisconsin.csv", na.strings = "?")

#Q1

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
pairs(dataset[, 2:7])

#boxplot for F7 to F9
boxplot(dataset[, 8:10])

#histogram for F7 to F9
hist(dataset[, "F7"])
hist(dataset[, "F8"])
hist(dataset[, "F9"])

#Q2
#clear all variables
rm(list = ls())

#load dataset
dataset <- read.csv("breast-cancer-wisconsin.csv", na.strings = "?")

#number of missing values before removing rows
sum(is.na(dataset))

#remove rows with missing values
dataset <- na.omit(dataset)

#number of missing values after removing rows
sum(is.na(dataset))


# Clearing all variables from the memory
rm(list = ls())