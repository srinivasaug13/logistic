#setwd ("/Users/kumar/Dropbox/_Projects/GreatLearning/GL-scripts/")

custSpendData <- read.csv("Cust_Spend_Data.csv", header=TRUE)
View(custSpendData)

## K Means Clustering

#Reading and scaling
custSpendData <- read.csv("Cust_Spend_Data.csv", header=TRUE)


custSpendData.Scaled <- scale(custSpendData[,3:7])
View(custSpendData.Scaled)

## code taken from the R-statistics blog
## http://www.r-statistics.com/2013/08/k-means-clustering-from-r-in-action/

## Identifying the optimal number of clusters form WSS

wssplot <- function(data, nc=15, seed=1234){
  wss <- (nrow(data)-1)*sum(apply(data,2,var))
  for (i in 2:nc){
    set.seed(seed)
    wss[i] <- sum(kmeans(data, centers=i)$withinss)}
  plot(1:nc, wss, type="b", xlab="Number of Clusters",
       ylab="Within groups sum of squares")}

wssplot(custSpendData.Scaled, nc=5)

## Identifying the optimal number of clusters
##install.packages("NbClust")

library(NbClust)
?NbClust

set.seed(1234)
nc <- NbClust(custSpendData[,c(-1,-2)], min.nc=2, max.nc=4, method="kmeans")
table(nc$Best.n[1,])

barplot(table(nc$Best.n[1,]),
          xlab="Numer of Clusters", ylab="Number of Criteria",
          main="Number of Clusters Chosen by 26 Criteria")


?kmeans
kmeans.clus = kmeans(x=custSpendData.Scaled, centers = 3, nstart = 3)
kmeans.clus

## plotting the clusters
##install.packages("fpc")
library(fpc)
plotcluster(custSpendData.Scaled, kmeans.clus$cluster)

# More complex
library(cluster)
?clusplot
clusplot(custSpendData.Scaled, kmeans.clus$cluster, 
         color=TRUE, shade=TRUE, labels=2, lines=1)

## profiling the clusters
KcustSpendData$Clusters <- kmeans.clus$cluster
View(KcustSpendData)
aggr = aggregate(KcustSpendData[,-c(1,2, 8)],list(KcustSpendData$Clusters),mean)
clus.profile <- data.frame( Cluster=aggr[,1],
                            Freq=as.vector(table(KcustSpendData$Clusters)),
                            aggr[,-1])

View(clus.profile)
