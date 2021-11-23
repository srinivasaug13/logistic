#setwd ("/Users/kumar/Dropbox/_Projects/GreatLearning/GL-scripts/")

custSpendData <- read.csv("Cust_Spend_Data.csv", header=TRUE)
View(custSpendData)


#Hierarchical Clustering using hclust

?dist  ## help
distMatrix = dist(x=custSpendData[,3:7], method = "euclidean") 
#distMatrix = dist(x=custSpendData[,3:7], method = "minkowski", p=2)   #equivalent to above line
print(distMatrix, digits = 3)


## scale function standardizes the values
custSpendData.Scaled = scale(custSpendData[,3:7])
View(custSpendData.Scaled)

# Compute distance matrix again with scaled data
distMatrix.Scaled = dist(x=custSpendData.Scaled, method = "euclidean") 
print(distMatrix.Scaled, digits = 3)
cluster <- hclust(distMatrix.Scaled, method = "average")
plot(cluster, labels = as.character(custSpendData[,2]))


# Plot rectagles defining the clusters for any given level of K 
rect.hclust(cluster, k=3, border="red")

# Print cluster combining heights
cluster$height

## Adding the cluster number back to the dataset
custSpendData$Cluster <- cutree(cluster, k=3)

## Aggregate columns 3:7 for each cluster by their means
custProfile = aggregate(custSpendData[,-c(1,2, 8)],list(custSpendData$Cluster),FUN="mean")
custProfile$Frequency = as.vector(table(custSpendData$Cluster))
View(custProfile)