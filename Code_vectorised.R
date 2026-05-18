################################################################################
###################### chapter 1, libraries and data ###########################
################################################################################

# The set-up

# Note to reader; it is adviced to install the packages by console rather 
# tham the RScript here. 

install.packages("ggplot2")
install.packages("quantmod")
install.packages("tidyverse")
install.packages("stats")
install.packages("e1071")
install.packages("tseries")
install.packages("FinTS")

require(ggplot2)
require(quantmod)
require(tidyverse)
require(e1071)
require(tseries)
require(FinTS)
require(stats)

# Run analysis from here to the bottom

# Data, Gold futures data from yahoo finance, 2022/01/01 to 17/04/2026:

GC_F_uncleaned <- getSymbols("GC=F", from = "2022-01-01",
                   to = "2026-04-17", auto.assign = FALSE)

GC_F  <- na.omit(GC_F_uncleaned$`GC=F.Close`) 

# Repeat but for longer-time span, 01/01/2002 to 17/04/2026:

GC_F_hist_uncleaned <- getSymbols("GC=F", from = "2002-01-01",
                        to = "2026-04-17", auto.assign = FALSE) 

GC_F_hist <- na.omit(GC_F_hist_uncleaned$`GC=F.Close`)

# na.omit() removes NAs for Gold closing prices. 

# Time plots:

chartSeries(GC_F$`GC=F.Close`) 
chartSeries(GC_F_hist$`GC=F.Close`) 

# Log-return transformation. Useful for tractibility and supporting Stationarity  

log_r <- diff(log(GC_F))
log_r_hist <- diff(log(GC_F_hist))

chartSeries(log_r) # Heteroskedasticity (Violation of Uniformity of Risk)
chartSeries(log_r_hist) # unclear how dramatic in the 20 year data, this is why formal assessment is needed

# Run the following once, if GOLD_close < 1076.  Re-run from beggining  

GOLD_close <- as.numeric(GC_F$`GC=F.Close`) 
GOLD_close <- GOLD_close[-1]
GOLD_close <- as.numeric(GOLD_close)

log_r <- as.numeric(log_r)
log_r <- na.omit(log_r)
log_r <- as.numeric(log_r) # Should match GOLD_close at 1076 obs

# Write closing prices and log-returns as a dataset to save as '

df <- data.frame(GOLD_close)
df$log_r <- log_r

write.csv(df, "GoldF_2022.csv")

# Limitation of data/analysis: Treats weekends (or Public holidays) 
# the same as single-day lags

data <- read.csv("GoldF_2022.csv")

# Read in dataset as data

################################################################################
####### Chapter 2: Data/log-returns, distributive properties and VaR ###########
################################################################################

# ADF test to assess data stationarity 
# Looking at charts is useful here

adf.test(data$GOLD_close) # p-value 0.9368. Cannot reject H0, non-stationarity 

adf.test(log_r) # p-value sig at 0.05. Sufficient evidence to rejecct H0 

# set axes for ecdf, may require adjustment if applied to different data

x_for_ecdf <- sort(log_r)
y_for_ecdf <- (1:length(log_r)) / length(log_r)

# Set visualisation to 1R : 2C in base R

par(mfrow=c(1,2))

# Plot density 

plot(density(log_r), col = "darkgreen", lwd = 3,
     xlim = c(-0.15, 0.15), main = "Density Plot of log_r",
     xlab = "Log-returns of Gold (GC=F)")

# xlim can set based on summary() results 

plot(x_for_ecdf,y_for_ecdf,
     type = "step", col = "red", lwd = "3",
     xlim = range(x_for_ecdf), ylim = c(0,1),
     x_for_ecdfaxs = "i",
     y_for_ecdfaxs = "i",
     main = "ECDF of log-returns",
     xlab = "Log Returns of GOLD (GC=F)")

# ECDF shows assymetry amd tendancy to more 'moderate' loss days

# Natural next step is to assess statistical properties

mean <- mean(log_r)
mean

med <- median(log_r)
med

var <- var(log_r)
var

skew <- skewness(log_r)
skew 

kurt <- kurtosis(log_r)
exc_kurt <- kurt - 3
exc_kurt

# Non-parametric since normality failed as below: 

shapiro.test(log_r)

jarque.bera.test(log_r)

# VaR with visualisation 

par(mfrow=c(1,1))

plot(density(log_r), col = "red", lwd = 3.5,
     xlim = c(-0.15, 0.15), main = "Density Plot of log_r",
     xlab = "Log-returns of Gold (GC=F)")

# Improve with histogram 

# Below is code for VaR table

quantile(log_r, 0.05) # 95%
quantile(log_r, 0.025) 
quantile(log_r, 0.01) # 99%
quantile(log_r, 0.001) 
quantile(log_r, 0) # min
quantile(log_r, 1) # max

# PACF and ACF 

par(mfrow=c(1,1))
acf(log_r)

pacf(log_r) # lag 28 seems odd, or like noise, more formal test below

Box.test(log_r, type = 'Ljung-Box') # Can only reject at 10%
Box.test(log_r, type = 'Box-Pierce')

# 28 seems more plausible to be noise. No compelling evidence for Autocorrelation
# Fits financial concept of no-arbitrage, market appears random and past information
# does not provide an insight to tomorrow.

################################################################################
################### Chapter 3, ARIMA and AIC selection #########################
################################################################################

# Better fit penalised by more parameters (for robustness,  avoid overfitting)

fit <- arima(log_r, order = c(0,0,0))

fit_1 <- arima(log_r, order = c(0,0,1)) 
fit_2 <- arima(log_r, order = c(1,0,0)) 
fit_3 <- arima(log_r, order = c(0,0,0)) 

# Using Alkaike's information criterion (fit, generalisation trade-off)

AIC(fit_1, fit_2, fit_3)

# MA(1) BEST BUT NEGLIGEABLE DIFFERENCE
# AR(1) SECOND BEST
# WN worst, but by negligible margin 

################################################################################
####### Chapter 5, GARCH(1,1). Volatility when Risk Uniformity violated ######## 
################################################################################

GARCH <- garch(log_r, order = c(1,1))

plot(GARCH$residuals)

qqnorm(GARCH$residuals); qqline(GARCH$residuals, col=2, main = "Normal Q-Q Plot of GARCH(1,1) residuals")

(GARCH$residuals)

summary(GARCH) # For GARCH(1,1) interpretation a0, a1 and b1. 

