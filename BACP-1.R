
USGDP <- ts(US_GDP[,2], start=c(1929,1), end=c(1991,1), frequency=1)

Shoe <- ts(Shoe_Sales[,3], start=c(2011,1), frequency=12)
plot(Shoe)

Income <- ts(Quarterly_Income[,3], start=c(2000,4), frequency=4)
plot(Income)

Champagne <- ts(ChampagneSales[,2], start=c(1964, 1), frequency=12)
plot(Champagne)

AirPax <- ts(AirPassengerVol[,2], start=c(1949, 1), frequency=12)
plot(AirPax)

monthplot(Champagne)
monthplot(AirPax)

monthplot(Income)

IncDec<-stl(Income[,1], s.window='p') #constant seasonality
plot(IncDec)
IncDec

IncDec7<-stl(Income[,1], s.window=7) #seasonality changes
plot(IncDec7)
IncDec

DeseasonRevenue <- (IncDec7$time.series[,2]+IncDec7$time.series[,3])
ts.plot(DeseasonRevenue, Income, col=c("red", "blue"), main="Comparison of Revenue and Deseasonalized Revenue")

stl(Champagne[,1], s.window=5, robust=T))

logAirPax <- log(AirPax)
logAirPaxDec <- stl(logAirPax[,1], s.window="p")
logAirPaxDec$time.series[1:12,1]
AirPaxSeason <- exp(logAirPaxDec$time.series[1:12,1])
plot(AirPaxSeason, type="l")

IncomeTrain <- window(Income, start=c(2000,4), end=c(2012,4), frequency=4)
IncomeTest <- window(Income, start=c(2013,1), frequency=4)

IncTrn7<-stl(IncomeTrain, s.window=7)
fcst.Inc.stl <- forecast(IncTrn7, method="rwdrift", h=5)

Vec<- cbind(IncomeTest,fcst.Inc.stl$mean)
ts.plot(Vec, col=c("blue", "red"), main="Quarterly Income: Actual vs Forecast")
MAPE <- mean(abs(Vec[,1]-Vec[,2])/Vec[,1])
MAPE

Library(fpp2)
fcoil <- ses(oildata, h=3)

Champ1 <- window(Champagne, start=c(1964,1), end=c(1970,12))
ChampHO <- window(Champagne, start=c(1971,1), end=c(1972,9))

Champ.fc <- hw(Champ1, h=21)

Champ1.stl <- stl(Champ1, s.window = 5)
fcst.Champ1.stl <- forecast(Champ1.stl, method="rwdrift", h=21)

AirPax1 <- window(AirPassengers, start=c(1949,1), end=c(1958,12))
AirPaxHO <- window(AirPassengers, start=c(1959,1), end=c(1960,12))

Library(tseries)
adf.test(BMW) 

acf(BASF, lag = 50)

BASF.arima.fit <- arima(BASF, c(1, 1, 0))
BASF.arima.fit

Box.test(BASFTr.arima.fit$residuals, lag=30, type="Ljung-Box")

Plot(forecast(BASF.arima.fit, h=6))

