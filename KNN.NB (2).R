#Some libraries

library(car)
library(caret)
library(class)
library(devtools)
library(e1071)
library(ggord)
library(ggplot2)
library(Hmisc)
library(klaR)
library(klaR)
library(MASS)
library(nnet)
library(plyr)
library(pROC)
library(psych)
library(scatterplot3d)
library(SDMTools)
library(dplyr)
library(ElemStatLearn)
library(rpart)
library(rpart.plot)
library(randomForest)
library(neuralnet)

setwd( "/Users/bappa/Desktop/BACP")
Loan<-read.csv(file.choose(), header=T)
#na.omit(Loan)

summary(Loan)
str(Loan)

levels(Loan$Purpose)
levels(Loan$Job)
#Define some dummies
Loan$Default<-ifelse(Loan$Status=="Default",1,0)
Loan$Female<-ifelse(Loan$Gender=="Female",1,0)
Loan$Management<-ifelse(Loan$Job=="Management",1,0)
Loan$Skilled<-ifelse(Loan$Job=="skilled",1,0)
Loan$Unskilled<-ifelse(Loan$Job=="unskilled",1,0)


Loan$CH.Poor<-ifelse(Loan$Credit.History=="Poor",1,0)
Loan$CH.critical<-ifelse(Loan$Credit.History=="critical",1,0)
Loan$CH.good<-ifelse(Loan$Credit.History=="good",1,0)
Loan$CH.verygood<-ifelse(Loan$Credit.History=="very good",1,0)

Loan$Purpose.car<-ifelse(Loan$Purpose=="car",1,0)
Loan$Purpose.cd<-ifelse(Loan$Purpose=="consumer.durable",1,0)
Loan$Purpose.education<-ifelse(Loan$Purpose=="education",1,0)
Loan$Purpose.personal<-ifelse(Loan$Purpose=="personal",1,0)


#We will use this throughout so that samples are comparable
# But before that, we will normalize

normalize<-function(x){
  +return((x-min(x))/(max(x)-min(x)))}

Loan$norm.Loan<-normalize(Loan$Loan.Offered)
Loan$norm.Work.Exp<-normalize(Loan$Work.Exp)
Loan$norm.Cred.Score<-normalize(Loan$Credit.Score)
Loan$norm.EMI<-normalize(Loan$EMI.Ratio)
Loan$norm.House<-normalize(Loan$Own.house)
Loan$norm.Dependents<-normalize(Loan$Dependents)
Loan$norm.Female<-normalize(Loan$Female)
Loan$norm.Management<-normalize(Loan$Management)
Loan$norm.Skilled<-normalize(Loan$Skilled)
Loan$norm.CH.Poor<-normalize(Loan$CH.Poor)
Loan$norm.CH.critical<-normalize(Loan$CH.critical)
Loan$norm.CH.good<-normalize(Loan$CH.good)
Loan$norm.car<-normalize(Loan$Purpose.car)
Loan$norm.cd<-normalize(Loan$Purpose.cd)
Loan$norm.edu<-normalize(Loan$Purpose.education)



set.seed(1234)
pd<-sample(2,nrow(Loan),replace=TRUE, prob=c(0.7,0.3))

train<-Loan[pd==1,]
val<-Loan[pd==2,]


sum(Loan$Default)
sum(val$Default)
sum(train$Default)


#Data Frame for KNN
#Normalization is must


train.NB<-train[,c(12,26:39)]
val.NB<-val[,c(12,26:39)]
str(train.NB)

sum(val.NB$Default)
sum(train.NB$Default)

#KNN 
####KNN
#knn3
y_pred.3<-knn(train=train.NB[,-1],test=val.NB[-1], cl=train.NB[,1],k=3)
tab.knn.3<-table(val.NB[,1],y_pred.3)
tab.knn.3

accuracy.knn.3<-sum(diag(tab.knn.3))/sum(tab.knn.3)
accuracy.knn.3
loss.knn.3<-tab.knn.3[2,1]/(tab.knn.3[2,1]+tab.knn.3[1,1])
loss.knn.3

#knn5
y_pred.5<-knn(train=train.NB[,-1],test=val.NB[-1], cl=train.NB[,1],k=5)
tab.knn.5<-table(val.NB[,1],y_pred.5)
tab.knn.5

accuracy.knn.5<-sum(diag(tab.knn.5))/sum(tab.knn.5)
accuracy.knn.5
loss.knn.5<-tab.knn.5[2,1]/(tab.knn.5[2,1]+tab.knn.5[1,1])
loss.knn.5

#knn7
y_pred.7<-knn(train=train.NB[,-1],test=val.NB[-1], cl=train.NB[,1],k=7)
tab.knn.7<-table(val.NB[,1],y_pred.7)
tab.knn.7

accuracy.knn.7<-sum(diag(tab.knn.7))/sum(tab.knn.7)
accuracy.knn.7
loss.knn.7<-tab.knn.7[2,1]/(tab.knn.7[2,1]+tab.knn.7[1,1])
loss.knn.7

#knn9
y_pred.9<-knn(train=train.NB[,-1],test=val.NB[-1], cl=train.NB[,1],k=9)
tab.knn.9<-table(val.NB[,1],y_pred.9)
tab.knn.9

accuracy.knn.9<-sum(diag(tab.knn.9))/sum(tab.knn.9)
accuracy.knn.9
loss.knn.9<-tab.knn.9[2,1]/(tab.knn.9[2,1]+tab.knn.9[1,1])
loss.knn.9



######
#NB
train.NB$Default<-as.factor(train.NB$Default)
val.NB$Default<-as.factor(val.NB$Default)

NB<-naiveBayes(x=train.NB[-1], y=train.NB$Default)
#pedict
y_pred.NB<-predict(NB,newdata=val.NB[-1])
y_pred.NB


#Confusion matrix

tab.NB=table(val.NB[,1],y_pred.NB)
tab.NB



accuracy.NB<-sum(diag(tab.NB))/sum(tab.NB)
accuracy.NB
loss.NB<-tab.NB[2,1]/(tab.NB[2,1]+tab.NB[1,1])
loss.NB
opp.loss.NB<-tab.NB[1,2]/(tab.NB[1,2]+tab.NB[2,2])
opp.loss.NB
tot.loss.NB<-0.95*loss.NB+0.05*opp.loss.NB
tot.loss.NB

#with KNN.3

accuracy.knn.3<-sum(diag(tab.knn.3))/sum(tab.knn.3)
accuracy.knn.3
loss.knn.3<-tab.knn.3[2,1]/(tab.knn.3[2,1]+tab.knn.3[1,1])
loss.knn.3
opp.loss.knn.3<-tab.knn.3[1,2]/(tab.knn.3[1,2]+tab.knn.3[2,2])
opp.loss.knn.3
tot.loss.knn.3<-0.95*loss.knn.3+0.05*opp.loss.knn.3
tot.loss.knn.3
