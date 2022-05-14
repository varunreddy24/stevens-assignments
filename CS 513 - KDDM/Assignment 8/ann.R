rm(list=ls())
data_raw <-read.csv('wisc_bc_ContinuousVar.csv')
data <-data_raw[,-1]
idx <- seq (1,nrow(data), by=5)
train<-data[-idx,]
test<-data[idx,]
library("neuralnet")
result <- neuralnet(diagnosis~., train, hidden = 5, threshold = 0.01)
plot(result)
ann <- compute(result, test[,-1])
ann$net.result
