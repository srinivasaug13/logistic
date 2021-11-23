RTxn <- read.table("Market_Basket_Analysis.csv", sep = ",", header=T)
nrow(RTxn)

str(RTxn)
RTxn$Invoice_No <- as.factor(RTxn$Invoice_No)


?split
Agg.RTxn <- split(RTxn$Item_Desc,RTxn$Invoice_No)
class(Agg.RTxn)
Agg.RTxn
## To see specific row number transaction
Agg.RTxn [105]



##install.packages("arules")
library(arules)
## logic to remove duplicate items from the list
Agg.RTxn_DD <- list()
for (i in 1:length(Agg.RTxn)) {
  Agg.RTxn_DD[[i]] <- as.character(Agg.RTxn[[i]][!duplicated(Agg.RTxn[[i]])])
}
## converting transaction items from list format to transaction format
Txns <- as(Agg.RTxn_DD, "transactions")


summary(Txns)


inspect(Txns[10])


freq <- itemFrequency(Txns)
freq <- freq[order(-freq)]
freq["Bread"]
barplot(freq[1:20])
?itemFrequencyPlot
itemFrequencyPlot(
  
  Txns, support = 0.10)

itemFrequencyPlot( Txns, topN = 10)

library("arulesViz")

?apriori
arules1 <- apriori(data = Txns)
summary(arules1)

inspect(arules1)
inspect(sort(arules1,by="lift"))


arules2 <- apriori(
  
  data = Txns, parameter = list(
    support = 0.05, confidence = 0.5, maxlen = 2
  )
)

library(RColorBrewer)
plot ( arules2,control=list(
  col = brewer.pal(11,"Spectral")
),
main="Association Rules Plot"
)


subrules2 <- head(sort(arules2, by="support"), 20)
plot(subrules2, method="grouped" , interactive=TRUE )


rules_df <- as(arules2,"data.frame")
rules_df$lhs_suuport <- rules_df$support/ rules_df$confidence;
rules_df$rhs_support <- rules_df$confidence / rules_df$lift;
View(rules_df)
write.table(rules_df, file = "output/mba_output.csv", sep = "," , append = F, row.names = F)
unlink("mba_output.csv")