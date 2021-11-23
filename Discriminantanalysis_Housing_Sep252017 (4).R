rm(list=ls())

library("car")
library("rattle")
library("MASS")
library("gdata")
library("lawstat")
library("psych")
library(OpenMx)
library("compute.es")
library("effects")
library("ggplot2")
library("multcomp")
library("pastecs")
library("nlme")
library("biotools")
library("mvoutlier")
library("mvnormtest")
library("plyr")
library("reshape2")
library("expm")
library("Matrix")


setwd("F:/Sridhar/PGPBA/NewOnlineAS/ANOVA/PPTs/Discriminant analysis/PPT & R & Data")

statesf<-read.table("Housing.txt", header = TRUE)
#attach(statesf)

head(statesf)

location1<-c(rep(1,13),rep(2,13), rep(3,9))
length(location1)
nlevels(location1)

#location2<-c(rep(1,13),rep(2,13), rep(3,9))

location1<-factor(location1, levels = c(1:3), labels = c("PA", "MP", "LA"))
statesf<-cbind(location1, statesf)

##### Box's M-Test: Equality of variance-covariance matrix. 
####Pooling can be done if the matrices are the same. Otherwise use qda

boxM(statesf[, 3:5], statesf$location1)

#scatterplotMatrix(statesf[3:5])

### Raw appication of LDA
region.lda <- lda(location1 ~price+bedrooms+area, data=statesf)
region.lda
region.lda$svd #######gives the ratio of the between- and within-group standard deviations 
####on the linear discriminant variables.

prop = region.lda$svd^2/sum(region.lda$svd^2)

########################################################################################
###Jump to line number 451 without any loss of understanding. 
###The rest of the program derives the standar output dsiplayed in SPSS
########################################################################################
##### Derivation ofLDA####

### Data Preparation
### Number of groups is equal to 3. Two discriminant functions is what we obtain
###Let us create dummies for each group

statesf$PA<- as.numeric(ifelse ( ( statesf$location1 == "PA" ), 1 , 0 ) )
statesf$MP<- as.numeric(ifelse ( ( statesf$location1 == "MP" ), 1 , 0 ))
statesf$LA<- as.numeric(ifelse ( ( statesf$location1 == "LA" ), 1 , 0 ))

length(location1)
levells<-nlevels(location1)
levells
Z1<-c("PA", "MP", "LA")
Z2<-c("price","bedrooms", "area")

H<-as.matrix(data.frame(statesf[,c(Z1)]))
X<-as.matrix(data.frame(statesf[,c(Z2)]))


###Group Centroids
xg<-ginv((t(H)%*%H))%*%t(H)%*%X
xg
###Grand Mean
oness<- matrix(1,35,1)
length(oness)
dim(oness)
grandmeanX<-(as.matrix(t(oness)%*%X))/nrow(oness)
grandmeanX

####Matrix of within group deviations and within group sum of squares

P<-as.matrix(X-H%*%xg) ### Within group deviations
W<-as.matrix(t(P)%*%P) ### Within group sum of squares
withincovar<-as.matrix(W/(nrow(statesf)-levells))
withincorrel1<-as.matrix(vec2diag(sqrt(diag2vec(withincovar)))) ### Matrix of standard Deviations
withincorrel<-matrix(1,nrow=3,ncol=3)

for(i in 1:nrow(withincorrel1)){
for(j in 1:ncol(withincorrel1)){
if(i==j){withincorrel[i,j]=withincovar[i,j]/withincorrel1[i,j]^2}
else{withincorrel[i,j]=withincovar[i,j]/(withincorrel1[i,i]*withincorrel1[j,j])}

}}


####Matrix of between group deviations and between group sum of squares

Q<-as.matrix(H%*%xg-oness%*%grandmeanX) ### betwen group deviations
A<-as.matrix(t(Q)%*%Q) #### between group sum of squares


#### Matrix of Total Sample Deviations and Total Sum of squares

R<-as.matrix(X-oness%*%grandmeanX) ###Total Sample Deviations
T<-as.matrix(t(R)%*%R) #### Total Sum of squares

#### Verification that Total is equal to sum of between and within group sum of squares

Verification<-as.matrix(T-A-W)



#### Eigen vector Values####

S<-as.matrix(ginv(W)%*%A)

lambdas<-eigen(S)
V1 <- as.matrix(lambdas$vectors)
V<-as.matrix(cbind(V1[,1],V1[,2]))
lambdad <- as.matrix((lambdas$values))

gg<-vec2diag(lambdad)

### Canonical Correlations
cc1<-sqrt(as.matrix(lambdad[1,1]/(1+lambdad[1,1])))
cc2<-sqrt(as.matrix(lambdad[2,1]/(1+lambdad[2,1])))

#### Wilk's Lambda and Chi-Square Values: Significance of Discriminant fucntions

wilks1<-as.matrix(1/((1+lambdad[1,1])*(1+lambdad[2,1])))
wilks2<-as.matrix(1/((1+lambdad[2,1])))
chisq1<-(nrow(statesf)-1-0.5*(ncol(X)+ncol(H)))*(log(1+lambdad[1,1])+log(1+lambdad[2,1]))

chisq2<-(nrow(statesf)-1-0.5*(ncol(X)+ncol(H)))*(log(1+lambdad[2,1]))  

#### Standardized coefficients###

#### Using within groups covariance matrix
cw<-as.matrix(1/(nrow(statesf)-ncol(H))*W)
ww<-vec2diag(sqrt(diag2vec(cw)))
coeffstdw<-ww%*%V ### normalized values of coeffstdw are comparable to SPSS standardized coefficient results####
coeffstdw



######################Delete if it does not work

#### Within Group Variance

calcWithinGroupsVariance <- function(variable,groupvariable)
{
  # find out how many values the group variable can take
  groupvariable2 <- as.factor(groupvariable[[1]])
  levels <- levels(groupvariable2)
  numlevels <- length(levels)
  # get the mean and standard deviation for each group:
  numtotal <- 0
  denomtotal <- 0
  for (i in 1:numlevels)
  {
    leveli <- levels[i]
    levelidata <- variable[groupvariable==leveli,]
    levelilength <- length(levelidata)
    # get the standard deviation for group i:
    sdi <- sd(levelidata)
    numi <- (levelilength - 1)*(sdi * sdi)
    denomi <- levelilength
    numtotal <- numtotal + numi
    denomtotal <- denomtotal + denomi
  }
  
  # calculate the within-groups variance
  Vw <- numtotal / (denomtotal - numlevels)
  
  return(Vw)
}

calcWithinGroupsVariance(statesf[5],statesf[1])

##### Between Groups Variance

calcBetweenGroupsVariance <- function(variable,groupvariable)
{
    # find out how many values the group variable can take
  groupvariable2 <- as.factor(groupvariable[[1]])
  levels <- levels(groupvariable2)
  numlevels <- length(levels)
 
   # calculate the overall grand mean:
  
  grandmean <- colMeans(variable, na.rm = TRUE)
  
  # get the mean and standard deviation for each group:
  numtotal <- 0
  denomtotal <- 0
  for (i in 1:numlevels)
  {
    leveli <- levels[i]
    levelidata <- variable[groupvariable==leveli,]
    levelilength <- length(levelidata)
    # get the mean and standard deviation for group i:
    meani <- mean(levelidata)
    meani
    sdi <- sd(levelidata)
    sdi
    numi <- levelilength * ((meani - grandmean)^2)
    denomi <- levelilength
    numtotal <- numtotal + numi
    denomtotal <- denomtotal + denomi
  }
  # calculate the between-groups variance
  Vb <- numtotal / (numlevels - 1)
  Vb <- Vb[[1]]
  return(Vb)
}
calcBetweenGroupsVariance(statesf[4],statesf[1])


##### Calculating separations between group sum of squares divided by within group sum of squares for each variable## 
calcSeparations <- function(variables,groupvariable)
{
  # find out how many variables we have
  variables <- as.data.frame(variables)
  numvariables <- length(variables)
  # find the variable names
  variablenames <- colnames(variables)
  # calculate the separation for each variable
  for (i in 1:numvariables)
  {
    variablei <- variables[i]
    variablename <- variablenames[i]
    Vw <- calcWithinGroupsVariance(variablei, groupvariable)
    Vb <- calcBetweenGroupsVariance(variablei, groupvariable)
    sep <- Vb/Vw
    print(paste("variable",variablename,"Vw=",Vw,"Vb=",Vb,"separation=",sep))
  }
}


##### 
calcWithinGroupsCovariance <- function(variable1,variable2,groupvariable)
{
  # find out how many values the group variable can take
  groupvariable2 <- as.factor(groupvariable[[1]])
  levels <- levels(groupvariable2)
  numlevels <- length(levels)
  # get the covariance of variable 1 and variable 2 for each group:
  Covw <- 0
  for (i in 1:numlevels)
  {
    leveli <- levels[i]
    levelidata1 <- variable1[groupvariable==leveli,]
    levelidata2 <- variable2[groupvariable==leveli,]
    mean1 <- mean(levelidata1,na.rm=TRUE)
    mean2 <- mean(levelidata2,na.rm=TRUE)
    
    levelilength <- length(levelidata1)
    # get the covariance for this group:
    term1 <- 0
    for (j in 1:levelilength)
    {
      term1 <- term1 + ((levelidata1[j] - mean1)*(levelidata2[j] - mean2))
    }
    Cov_groupi <- term1 # covariance for this group
    Covw <- Covw + Cov_groupi
  }
  totallength <- nrow(variable1)
  Covw <- Covw / (totallength - numlevels)
  return(Covw)
}

calcWithinGroupsCovariance(statesf[3],statesf[4],statesf[1])


#### Calculate Between Group Covariance

calcBetweenGroupsCovariance <- function(variable1,variable2,groupvariable)
{
  # find out how many values the group variable can take
  groupvariable2 <- as.factor(groupvariable[[1]])
  levels <- levels(groupvariable2)
  numlevels <- length(levels)
  # calculate the grand means
  variable1mean <- colMeans(variable1,na.rm=TRUE)
  variable2mean <- colMeans(variable2,na.rm=TRUE)
  # calculate the between-groups covariance
  Covb <- 0
  for (i in 1:numlevels)
  {
    leveli <- levels[i]
    levelidata1 <- variable1[groupvariable==leveli,]
    levelidata2 <- variable2[groupvariable==leveli,]
    mean1 <- mean(levelidata1,na.rm=TRUE)
    mean2 <- mean(levelidata2,na.rm=TRUE)
    levelilength <- length(levelidata1)
    term1 <- (mean1 - variable1mean)*(mean2 - variable2mean)*(levelilength)
    Covb <- Covb + term1
  }
  Covb <- Covb / (numlevels - 1)
  Covb <- Covb[[1]]
  return(Covb)
}

calcBetweenGroupsCovariance(statesf[4],statesf[3],statesf[1])

###### Most highly Correlated

mosthighlycorrelated <- function(mydataframe,numtoreport)
{
  # find the correlations
  cormatrix <- cor(mydataframe)
  # set the correlations on the diagonal or lower triangle to zero,
  # so they will not be reported as the highest ones:
  diag(cormatrix) <- 0
  cormatrix[lower.tri(cormatrix)] <- 0
  # flatten the matrix into a dataframe for easy sorting
  fm <- as.data.frame(as.table(cormatrix))
  # assign human-friendly names
  names(fm) <- c("First.Variable", "Second.Variable","Correlation")
  # sort and print the top n correlations
  head(fm[order(abs(fm$Correlation),decreasing=T),],n=numtoreport)
}

mosthighlycorrelated(statesf[3:5], 4)

#####################

##### Standardizing the variables in the data

standardizedstatesf <- as.data.frame(scale(statesf[3:5]))
sapply(standardizedstatesf,mean)
sapply(standardizedstatesf,sd)
lapply(standardizedstatesf,mean)
lapply(standardizedstatesf,sd)

##### Standardized Coefficients in Discriminant Analysis

groupStandardise <- function(variables, groupvariable)
{
  # find out how many variables we have
  variables <- as.data.frame(variables)
  numvariables <- length(variables)
  # find the variable names
  variablenames <- colnames(variables)
  # calculate the group-standardised version of each variable
  for (i in 1:numvariables)
  {
    variablei <- variables[i]
    variablei_name <- variablenames[i]
    variablei_Vw <- calcWithinGroupsVariance(variablei, groupvariable)
    variablei_mean <- colMeans(variablei,na.rm=TRUE)
    variablei_new <- (variablei - variablei_mean)/(sqrt(variablei_Vw))
    data_length <- nrow(variablei)
    if (i == 1) { variables_new <- data.frame(row.names=seq(1,data_length)) }
    variables_new['variablei_name'] <- variablei_new
  }
  return(variables_new)
}


pricestd <- groupStandardise(statesf[3], statesf[1])
colnames(pricestd)<-c("pricestd1")

bedroomsstd <- groupStandardise(statesf[4], statesf[1])
colnames(bedroomsstd)<-c("bedroomstd1")
areastd <- groupStandardise(statesf[5], statesf[1])
colnames(areastd)<-c("areastd1")
statesfstd<-cbind(statesf,pricestd,bedroomsstd,areastd)

####after finding the group standardized coefficients now apply lda

house.lda<-lda(statesfstd$location1~ statesfstd$pricestd1+statesfstd$bedroomstd1+statesfstd$areastd1) ### Standardized Coefficients
house.lda

statesfstd.lda.values <- predict(house.lda, statesfstd[9:11]) ### Prediction using standardized LDA
statesfstd.lda.values$x[,1] 

houseun.lda<-lda(statesfstd$location1~ statesfstd$price+statesfstd$bedrooms+statesfstd$area) ### UnStandardized Coefficients
statesfun.lda.values <- predict(houseun.lda, statesfstd[3:5]) ### Prediction using unstandardized LDA
statesfun.lda.values$x[,1] 

calcSeparations(statesfun.lda.values$x,statesfstd[1])

##### In the above step we see the values to be the same from both the methods

#### Structure Matrix: product of standardized coefficients and pooled within group correlation matrix####
xstdcoeff<-as.matrix(house.lda$scaling)
strucmat<-t(t(xstdcoeff)%*%withincorrel)
strucmat



#### Discriminant Probabilities

######
##########################################################################################################

#Prediction
region.lda.values <- predict(region.lda)
ldahist(data = region.lda.values$x[,1], g=location1)
plot(region.lda.values$x[,1],region.lda.values$x[,2]) # make a scatterplot
text(region.lda.values$x[,1],region.lda.values$x[,2],location1,cex=0.7,pos=4,col="red") # add labels

df<-predict(region.lda,statesf[,c(3,4,5)]) ### Predicting for test data#####
statesf<-cbind(statesf,df)

table(statesf$location,statesf$class)


### Classification
region1.lda <- lda(location1 ~price+bedrooms+area, data=statesf, na.action="na.omit", CV=TRUE)
region1.lda
ct <- table(statesf$location1, region1.lda$class)
diag(prop.table(ct, 1))
# total percent correct
sum(diag(prop.table(ct)))

df<-predict(region.lda,statesf[,c(3,4,5)]) ### Predicting for test data#####
statesf<-cbind(statesf,df)

table(statesf$location,statesf$class)

