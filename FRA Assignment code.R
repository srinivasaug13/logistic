#Read the data file in R
dev <- read.csv("raw-data_assgn_transformed.csv")

# removed rows at tail which were used for calculations of means for default vs non-default
dev <- dev[1:2409,]
attach(dev)

dev$Default = as.factor(dev$Default)

# Execute the model for selected variables
temp <- glm(Default~Total.Income_2.Total.assets_1 +
              Change.in.stock_2.Total.Income_2 +
              Total.expenses_2.Total.Income_2 +
              Profit.after.tax_2.Total.assets_1 + 
              PBT_2.Total.assets_1 +
              PAT.as...of.total.income_2 +
              Cash.profit.as...of.total.income_2 + 
              PBDITA.as...of.total.income_2 + 
              Sales_2.Total.assets_1 +
              Income.from.financial.services_2.Total.Income_2 + 
              Other.income_2.Total.Income_2 +
              Reserves.and.funds_2.Total_Assets +
              Total.capital_2.Total_Assets +
              Borrowings_2.Total_Assets + 
              Current.liabilities...provisions_2.Total_assets +
              Deferred.tax.liability_2.Total_Assets +
              Cumulative.retained.profits_2.Total.Income_2 +
              Contingent.liabilities_2.Total.Assets +
              Net.fixed.assets_2.Total.Assets + 
              Investments_2.Total.Income_2 + 
              Current.assets_2.Total_Assets +
              Current.ratio..times._2 +
              Debt.to.equity.ratio..times._1 +
              EPS_1 + 
              Adjusted.EPS_1, family = "binomial"
)

summary(temp)

# Eliminate the variables in stepwise manner which are highly insignificant
temp <- glm(Default~Total.Income_2.Total.assets_1 +
              Change.in.stock_2.Total.Income_2 +
              Total.expenses_2.Total.Income_2 +
              Profit.after.tax_2.Total.assets_1 + 
              PBT_2.Total.assets_1 +
              PAT.as...of.total.income_2 +
              Cash.profit.as...of.total.income_2 + 
              PBDITA.as...of.total.income_2 + 
              Sales_2.Total.assets_1 +
              Income.from.financial.services_2.Total.Income_2 + 
              Other.income_2.Total.Income_2 +
              Reserves.and.funds_2.Total_Assets +
              Total.capital_2.Total_Assets +
              Borrowings_2.Total_Assets + 
              Current.liabilities...provisions_2.Total_assets +
              Deferred.tax.liability_2.Total_Assets +
              Cumulative.retained.profits_2.Total.Income_2 +
              Contingent.liabilities_2.Total.Assets +
              Net.fixed.assets_2.Total.Assets + 
              Investments_2.Total.Income_2 + 
              Current.assets_2.Total_Assets +
              Current.ratio..times._2 +
              Debt.to.equity.ratio..times._1 +
              EPS_1, family = "binomial"
)
summary(temp)

temp <- glm(Default~Total.Income_2.Total.assets_1 +
              Change.in.stock_2.Total.Income_2 +
              Total.expenses_2.Total.Income_2 +
              Profit.after.tax_2.Total.assets_1 + 
              PBT_2.Total.assets_1 +
              PAT.as...of.total.income_2 +
              Cash.profit.as...of.total.income_2 + 
              PBDITA.as...of.total.income_2 + 
              Sales_2.Total.assets_1 +
              Income.from.financial.services_2.Total.Income_2 + 
              Other.income_2.Total.Income_2 +
              Reserves.and.funds_2.Total_Assets +
              Total.capital_2.Total_Assets +
              Borrowings_2.Total_Assets + 
              Current.liabilities...provisions_2.Total_assets +
              Deferred.tax.liability_2.Total_Assets +
              Cumulative.retained.profits_2.Total.Income_2 +
              Contingent.liabilities_2.Total.Assets +
              Net.fixed.assets_2.Total.Assets + 
              Current.assets_2.Total_Assets +
              Current.ratio..times._2 +
              Debt.to.equity.ratio..times._1 +
              EPS_1, family = "binomial"
)

summary(temp)

# Finally we are left with this model where all are significant
temp <- glm(Default~
              Profit.after.tax_2.Total.assets_1 + 
              PAT.as...of.total.income_2 +
              Cash.profit.as...of.total.income_2 + 
              PBDITA.as...of.total.income_2 + 
              Deferred.tax.liability_2.Total_Assets +
              Current.ratio..times._2 +
              Debt.to.equity.ratio..times._1 , family = "binomial"
)

# Here all values are significant, but if we look th sign for profitablity variable, it is +ve
# we would thus remove such variables

temp <- glm(Default~
              PAT.as...of.total.income_2 +
              Cash.profit.as...of.total.income_2 + 
              Deferred.tax.liability_2.Total_Assets +
              Current.ratio..times._2 +
              Debt.to.equity.ratio..times._1 , family = "binomial"
)
summary(temp)

#Now if we see the variables are significant as well as have correct signs for coefficient
# But in order to balance the equation we might need to add variables for profitablity and size

# create a new variable as below

dev$Cumulative.retained.profits_2.Sales_2 = Cumulative.retained.profits_2/Sales_2
aggregate(Cumulative.retained.profits_2.Sales_2~Default,data =dev,mean)
# this seems to be a good predictor
# for size retested the variable Total.capital_2.Total_Assets (this gives the best result
#when compared with other size variables)

temp = glm(Default ~ PAT.as...of.total.income_2 + Current.ratio..times._2 + 
      Debt.to.equity.ratio..times._1 + Cash.profit.as...of.total.income_2 + 
        Cumulative.retained.profits_2.Sales_2 + Deferred.tax.liability_2.Total_Assets + Total.capital_2.Total_Assets, 
    family = binomial(link = "logit"))
summary(temp)
vif(temp)

# check for the power of model on same training data set
dev$predict = predict(temp,dev)
range(dev$predict)
dev$predictexp = exp(dev$predict)/(1+exp(dev$predict))
range(dev$predictexp)

# read the validation data set 
val <- read.csv("validation_data_assgn.csv")
# this file already contains the transformation for variable used in the model, 
# so no extra data preparation step required

# retain just the original data
val = val[1:715,]

# predict the output
val$predict = predict(temp, val)

range(val$predict)

val$predictexp = exp(val$predict)/(1+exp(val$predict))

range(val$predictexp)

# write the output data in excel for validation
write.csv(dev,"output_devf.csv")
write.csv(val,"output_valf.csv")

