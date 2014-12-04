install.packages("nnet")
install.packages("e1071")
install.packages("svmpath")

# Code for evaluation
source("ROC.R")
# Code for test sets
source("TestSet.R")
# scatter plot matrix
source("SPM_Panel.R")

#****************************************************
#   Basic Statistics
#****************************************************
bank.train <- read.table("bank-full.csv", sep = ";", header =T)
bank.test <- read.table("bank.csv", sep = ";", header =T)
bank <- rbind(btrain,btest)
# remove NA values
bank.train<-na.omit(btrain)
bank.test<-na.omit(btest)

# test technique test and train data set
b<- test.set(bank,.33)
btrain<-b$train
btest<-b$test

# The training set is btrain and the test set is btest
# comparing quantitative variables for the 3 data sets
par(mfrow = c(2,4))
for(i in c(1,6, 10,12:15))
{
	boxplot(bank[,i], at = 1, xaxt = "n", xlim = c(0, 4), main = colnames(bank)[i])
	boxplot(bank.test[,i], at = 2, xaxt = "n", add = TRUE)
	boxplot(bank.train[,i], at = 3, xaxt = "n", add = TRUE)
	axis(1, at = 1:3, labels = c("Original", "Test", "Train"), tick = TRUE)
}
par(mfrow = c(1,1))

# Subsample of bank data matrix subsample_size<-rows of observations
subsample <- function(x, subsample_size) {
	x[sample(x = seq_len(nrow(x)), size = subsample_size), ]
}


#******************************
#
#  Neural Nets
#
# *****************************
#Load the nnets package
library(nnet)
#First we do 5 hidden nodes
bank.nn1 = nnet(y~., data = bank.train, size = 5, decay=1e-3,  maxit = 1000)
b.nn1<-nnet(y~.,data=btrain,size=5,decay=1e-3,maxit=1000)
#The weights on the neural network
summary(bank.nn1)
# Plot of training set performance
plot(bank.nn1$fit~as.factor(bank.train$y), pch = 21, main = "NN 5 Training Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(b.nn1$fit~as.factor(btrain$y), pch = 21, main = "NN 5 Training Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Test set predictions
bank.nn1.pred <- predict(bank.nn1, newdata = bank.test, type = "raw")
b.nn1.pred <- predict(b.nn1, newdata = btest, type = "raw")
plot(bank.nn1.pred~as.factor(bank.test$y), pch = 21, main = "NN 5 Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(b.nn1.pred~as.factor(btest$y), pch = 21, main = "NN 5 Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Confusion matrix for test set
score.table(bank.nn1.pred, bank.test$y)
score.table(b.nn1.pred, btest$y)


# 10 nodes in the hidden layer
bank.nn2 = nnet(y~., data = bank.train, size = 10, decay=1e-3,  maxit = 1000)
b.nn2<-nnet(y~.,data=btrain,size=10,decay=1e-3,maxit=1000)
#The weights on the neural network
summary(bank.nn2)
# Plot of training set performance
plot(bank.nn2$fit~as.factor(bank.train$y), pch = 21, main = "NN 10 Training Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Test set predictions
b.nn2.pred <- predict(b.nn2, newdata = btest, type = "raw")
plot(bank.nn2.pred~as.factor(bank.test$y), pch = 21, main = "NN 10 Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Confusion matrix for test set
score.table(bank.nn2.pred, bank.test$y)
#**********************************************************
#
#			SVM
#
#*********************************************************
#  Two packages for SVM
library(e1071)
library(svmpath)
sub1<-subsample(bank.train,2000)
sub2<-subsample(bank.train,1000)
sub3<-subsample(bank.train,500)
sub4<-subsample(bank.train,100)
sub1<-subsample(btrain,2000)
sub2<-subsample(btrain,1000)
sub3<-subsample(btrain,500)
sub4<-subsample(btrain,100)


# We will use e1071
#		Training set model
# Radial Basis - the default
bank.svm1 <- svm(as.factor(y)~., data = sub1,  cost = 100, gamma = 1, probability = TRUE)
bank.svm2 <- svm(as.factor(y)~., data = sub2,  cost = 100, gamma = 1, probability = TRUE)
bank.svm3 <- svm(as.factor(y)~., data = sub3,  cost = 100, gamma = 1, probability = TRUE)
bank.svm4 <- svm(as.factor(y)~., data = sub4,  cost = 100, gamma = 1, probability = TRUE)
# Plot of training set performance
bank.svm1.fit <- predict(bank.svm1, bank.train, probability = T)
bank.svm2.fit <- predict(bank.svm2, bank.train, probability = T)
bank.svm3.fit <- predict(bank.svm3, btrain, probability = T)
bank.svm4.fit <- predict(bank.svm4, bank.train, probability = T)
bank.svm1.fit <- predict(bank.svm1, btrain, probability = T)
bank.svm2.fit <- predict(bank.svm2, btrain, probability = T)
bank.svm3.fit <- predict(bank.svm3, btrain, probability = T)
bank.svm4.fit <- predict(bank.svm4, btrain, probability = T)
bank.svm.fit1 <- attr(bank.svm1.fit, "probabilities")
bank.svm.fit2 <- attr(bank.svm2.fit, "probabilities")
bank.svm.fit3 <- attr(bank.svm3.fit, "probabilities")
bank.svm.fit4 <- attr(bank.svm4.fit, "probabilities")
par(mfrow=c(2,2))
plot(bank.svm.fit1[,1]~as.factor(bank.train$y), pch = 21, main = "SVM RBF Training Results for Bank Data 2000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit2[,1]~as.factor(bank.train$y), pch = 21, main = "SVM RBF Training Results for Bank Data 1000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit3[,1]~as.factor(bank.train$y), pch = 21, main = "SVM RBF Training Results for Bank Data 500", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit4[,1]~as.factor(bank.train$y), pch = 21, main = "SVM RBF Training Results for Bank Data 100", xlab = "Class", ylab = "Posterior", col = "steelblue" )

# Test set fit
svm1.pred <- predict(bank.svm1, newdata = btest, decision.values = T, probability = TRUE)
svm2.pred <- predict(bank.svm2, newdata = btest, decision.values = T, probability = TRUE)
svm3.pred <- predict(bank.svm3, newdata = btest, decision.values = T, probability = TRUE)
svm4.pred <- predict(bank.svm4, newdata = btest, decision.values = T, probability = TRUE)
svmr1.pr <- attr(svm1.pred, "probabilities")
svmr2.pr <- attr(svm2.pred, "probabilities")
svmr3.pr <- attr(svm3.pred, "probabilities")
svmr4.pr <- attr(svm4.pred, "probabilities")
plot(svmr1.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM RBF Test Set Results for Bank Data 2000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svmr2.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM RBF Test Set Results for Bank Data 1000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svmr3.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM RBF Test Set Results for Bank Data 500", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svmr4.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM RBF Test Set Results for Bank Data 100", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Confusion matrix for test set
score.table(svm1.pr[,1], bank.test$y)

#ROC Curve
par(mfrow=c(1,1))
plot.roc(svmr1.pr[,2], btest[,17]) # Radial basis functions 
lines.roc(svmr2.pr[,1],btest[,17],col="red") # Polynomial (degree 3) 		
lines.roc(svmr3.pr[,2],btest[,17],col="green") # Linear 
lines.roc(svmr4.pr[,1],btest[,17],col="orange") # Sigmoid
legend(.65, 0.45, legend = c("SVM RBF2000","SVM RBF1000","SVM RBF500","SVM RBF100"), lwd=2,col = c("blue","red","green","orange"))


# Polynomial degree 3 
bank.svmp1 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub1, cost = 10, gamma = 1, probability = TRUE)
bank.svmp2 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub2, cost = 10, gamma = 1, probability = TRUE)
bank.svmp3 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub3, cost = 10, gamma = 1, probability = TRUE)
bank.svmp4 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub4, cost = 10, gamma = 1, probability = TRUE)

bank.svmp1 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub1, cost = 10, gamma = 1, probability = TRUE)
bank.svmp2 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub2, cost = 10, gamma = 1, probability = TRUE)
bank.svmp3 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub3, cost = 10, gamma = 1, probability = TRUE)
bank.svmp4 <- svm(as.factor(y)~., kernel = "polynomial", degree = 3,  data = sub4, cost = 10, gamma = 1, probability = TRUE)
# Plot of training set performance
bank.svmp1.fit <- predict(bank.svmp1,btrain, probability = T)
bank.svmp2.fit <- predict(bank.svmp2, sub3, probability = T)
bank.svmp3.fit <- predict(bank.svmp3, sub3, probability = T)
bank.svmp4.fit <- predict(bank.svmp4, btrain, probability = T)
bank.svm.fit1 <- attr(bank.svmp1.fit, "probabilities")
bank.svm.fit2 <- attr(bank.svmp2.fit, "probabilities")
bank.svm.fit3 <- attr(bank.svmp3.fit, "probabilities")
bank.svm.fit4 <- attr(bank.svmp4.fit, "probabilities")
par(mfrow=c(2,2))
plot(bank.svm.fit1[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Polynomial Training Results for Bank Data 2000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit2[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Polynomial Training Results for Bank Data 1000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit3[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Polynomial Training Results for Bank Data 500", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit4[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Polynomial Training Results for Bank Data 100", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Test set fit
svmp1.pred <- predict(bank.svmp1, newdata = btest, decision.values = T, probability = TRUE)
svmp2.pred <- predict(bank.svmp2, newdata = btest, decision.values = T, probability = TRUE)
svmp3.pred <- predict(bank.svmp3, newdata = btest, decision.values = T, probability = TRUE)
svmp4.pred <- predict(bank.svmp4, newdata = btest, decision.values = T, probability = TRUE)
svmp1.pr <- attr(svmp1.pred, "probabilities")
svmp2.pr <- attr(svmp2.pred, "probabilities")
svmp3.pr <- attr(svmp3.pred, "probabilities")
svmp4.pr <- attr(svmp4.pred, "probabilities")
plot(svmp1.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Polynomial Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svmp2.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Polynomial Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svmp3.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Polynomial Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svmp4.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Polynomial Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Confusion matrix for test set
score.table(svm2.pr[,1], bank.test$y)

#ROC Curve
par(mfrow=c(1,1))
plot.roc(svmp1.pr[,2], btest[,17]) # Radial basis functions 
lines.roc(svmp2.pr[,2],btest[,17],col="red") # Polynomial (degree 3) 		
lines.roc(svmp3.pr[,2],btest[,17],col="green") # Linear 
lines.roc(svmp4.pr[,2],btest[,17],col="orange") # Sigmoid
legend(.65, 0.45, legend = c("SVM Poly2000","SVM Poly1000","SVM Poly500","SVM Poly100"), lwd=2,col = c("blue","red","green","orange"))


# Linear degree 1 
bank.svml1 <- svm(as.factor(y)~., kernel = "linear", data = sub1, cost = 100, gamma = 1, probability = TRUE)
bank.svml2 <- svm(as.factor(y)~., kernel = "linear", data = sub2, cost = 100, gamma = 1, probability = TRUE)
bank.svml3 <- svm(as.factor(y)~., kernel = "linear", data = sub3, cost = 100, gamma = 1, probability = TRUE)
bank.svml4 <- svm(as.factor(y)~., kernel = "linear", data = sub4, cost = 100, gamma = 1, probability = TRUE)
# Plot of training set performance
bank.svml1.fit <- predict(bank.svml1, btrain, probability = T)
bank.svml2.fit <- predict(bank.svml2, btrain, probability = T)
bank.svml3.fit <- predict(bank.svml3, sub3, probability = T)
bank.svml4.fit <- predict(bank.svml4, btrain, probability = T)
bank.svm.fit1 <- attr(bank.svml1.fit, "probabilities")
bank.svm.fit2 <- attr(bank.svml2.fit, "probabilities")
bank.svm.fit3 <- attr(bank.svml3.fit, "probabilities")
bank.svm.fit4 <- attr(bank.svml4.fit, "probabilities")
par(mfrow=c(2,2))
plot(bank.svm.fit1[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Linear Training Results for Bank Data 2000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit2[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Linear Training Results for Bank Data 1000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit3[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Linear Training Results for Bank Data 500", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit4[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Linear Training Results for Bank Data 100", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Test set fit
svml1.pred <- predict(bank.svml1, newdata = btest, decision.values = T, probability = TRUE)
svml2.pred <- predict(bank.svml2, newdata = btest, decision.values = T, probability = TRUE)
svml3.pred <- predict(bank.svml3, newdata = btest, decision.values = T, probability = TRUE)
svml4.pred <- predict(bank.svml4, newdata = btest, decision.values = T, probability = TRUE)
svml1.pr <- attr(svml1.pred, "probabilities")
svml2.pr <- attr(svml2.pred, "probabilities")
svml3.pr <- attr(svml3.pred, "probabilities")
svml4.pr <- attr(svml4.pred, "probabilities")
par(mfrow=c(2,2))
plot(svml1.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Linear Test Set Results for Bank Data 2000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svml2.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Linear Test Set Results for Bank Data 1000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svml3.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Linear Test Set Results for Bank Data 500", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svml4.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Linear Test Set Results for Bank Data 100", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Confusion matrix for test set
score.table(svm3.pr[,1], bank.test$y)

#ROC Curve
par(mfrow=c(1,1))
plot.roc(svml1.pr[,1], btest[,17]) # Radial basis functions 
lines.roc(svml2.pr[,2],btest[,17],col="red") # Polynomial (degree 3) 		
lines.roc(svml3.pr[,2],btest[,17],col="green") # Linear 
lines.roc(svml4.pr[,2],btest[,17],col="orange") # Sigmoid
legend(.65, 0.45, legend = c("SVM Linear2000","SVM Linear1000","SVM Linear500","SVM Linear100"), lwd=2,col = c("blue","red","green","orange"))

# Sigmoid
bank.svmsig1 <- svm(as.factor(y)~., data = sub1, probability = TRUE,  cost = 100, gamma = 1, kernal = "sigmoid")
bank.svmsig2 <- svm(as.factor(y)~., data = sub2, probability = TRUE,  cost = 100, gamma = 1, kernal = "sigmoid")
bank.svmsig3 <- svm(as.factor(y)~., data = sub3, probability = TRUE,  cost = 100, gamma = 1, kernal = "sigmoid")
bank.svmsig4 <- svm(as.factor(y)~., data = sub4, probability = TRUE,  cost = 100, gamma = 1, kernal = "sigmoid")
# Plot of training set performance
bank.svmsig1.fit <- predict(bank.svmsig1, btrain, probability = T)
bank.svmsig2.fit <- predict(bank.svmsig2, btrain, probability = T)
bank.svmsig3.fit <- predict(bank.svmsig3, sub3, probability = T)
bank.svmsig4.fit <- predict(bank.svmsig4, btrain, probability = T)
bank.svm.fit1 <- attr(bank.svmsig1.fit, "probabilities")
bank.svm.fit2 <- attr(bank.svmsig2.fit, "probabilities")
bank.svm.fit3 <- attr(bank.svmsig3.fit, "probabilities")
bank.svm.fit4 <- attr(bank.svmsig4.fit, "probabilities")
par(mfrow=c(2,2))
plot(bank.svm.fit1[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Sigmoid Training Results for Bank Data 2000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit2[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Sigmoid Training Results for Bank Data 1000", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit3[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Sigmoid Training Results for Bank Data 500", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(bank.svm.fit4[,1]~as.factor(bank.train$y), pch = 21, main = "SVM Sigmoid Training Results for Bank Data 100", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Test set fit
svms1.pred <- predict(bank.svmsig1, newdata = btest, decision.values = T, probability = TRUE)
svms2.pred <- predict(bank.svmsig2, newdata = btest, decision.values = T, probability = TRUE)
svms3.pred <- predict(bank.svmsig3, newdata = btest, decision.values = T, probability = TRUE)
svms4.pred <- predict(bank.svmsig4, newdata = btest, decision.values = T, probability = TRUE)
svms1.pr <- attr(svms1.pred, "probabilities")
svms2.pr <- attr(svms2.pred, "probabilities")
svms3.pr <- attr(svms3.pred, "probabilities")
svms4.pr <- attr(svms4.pred, "probabilities")
plot(svms1.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Sigmoid Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svms2.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Sigmoid Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svms3.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Sigmoid Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
plot(svms4.pr[,1]~as.factor(bank.test$y), pch = 21, main = "SVM Sigmoid Test Set Results for Bank Data", xlab = "Class", ylab = "Posterior", col = "steelblue" )
# Confusion matrix for test set
score.table(svm4.pr[,1], bank.test$y)

#ROC Curve
par(mfrow=c(1,1))
plot.roc(svms1.pr[,1], btest[,17]) # Radial basis functions 
lines.roc(svms2.pr[,2],btest[,17],col="red") # Polynomial (degree 3) 		
lines.roc(svms3.pr[,2],btest[,17],col="green") # Linear 
lines.roc(svms4.pr[,1],btest[,17],col="orange") # Sigmoid
legend(.65, 0.45, legend = c("SVM Sigmoid2000","SVM Sigmoid1000","SVM Sigmoid500","SVM Sigmoid100"), lwd=2,col = c("blue","red","green","orange"))

#ROC Curve
plot.roc(b.nn1.pred,btest[,17])#Neural Nets with 5 hidden units
lines.roc(b.nn2.pred,btest[,17],col="green4")#Neural Nets with 10 hidden units
lines.roc(svmr1.pr[,2], btest[,17],col="black") # Radial basis functions 
lines.roc(svmp1.pr[,2],btest[,17],col="red") # Polynomial (degree 3) 		
lines.roc(svml1.pr[,2],btest[,17],col="green") # Linear 
lines.roc(svms1.pr[,2],btest[,17],col="orange") # Sigmoid
legend(.6, 0.5, legend = c("Neural Nets5","Neural Nets10","SVM RBF2000","SVM Poly2000","SVM Linear2000","SVM Sigmoid2000"), lwd=2,col = c("blue","green4","black","red","green","orange"))
