#CLEAR R ENVIRONMENT

rm(list=ls())

#INSTALLING REQUIRED PACKAGES AND LIBRARIES

install.packages('randomForest')
install.packages('dplyr')
install.packages('corrgram')
install.packages('sm')
require(randomForest)
library(dplyr)
library(rpart)
library(corrgram)
library(sm)
library(caTools)

# SETTING SEED FOR UNIFORMITY
set.seed(123)

#SET WORKING DIRECTORY
setwd("E:/Project Data")


#IMPORTING DATASET

bikeData_Original = read.csv("day.csv")

#CREATING A DUPLICATE DATASET FROM THE ORIGINAL DATASET

bikeData = bikeData_Original

#DENORMALISING THE NUMERICAL VALUES

denorm_temp <- function(x) x*(47) - 8
denorm_atemp <- function(x) x*(66) - 16
denorm_hum <- function(x) x * 100
denorm_wind <- function(x) x * 67

# ADDING THE DENOMRALIZED VALUES TO THE DATASET
bikeData$denorm_temp = unlist(lapply(bikeData$temp, denorm_temp))
bikeData$denorm_atemp = unlist(lapply(bikeData$atemp, denorm_atemp))
bikeData$denorm_hum = unlist(lapply(bikeData$hum, denorm_hum))
bikeData$denorm_wind = unlist(lapply(bikeData$windspeed, denorm_wind))


#CREATING BOXPLOTS TO CHECK FOR OUTLIERS

boxplot(cnt~season, data=bikeData, main="Boxplot for Season", xlab="Season", ylab="Bike Count")
boxplot(cnt~holiday, data=bikeData, main="Boxplot for Holiday", xlab="Holiday", ylab="Bike Count")
boxplot(cnt~mnth, data=bikeData, main="Boxplot for Month", xlab="Month", ylab="Bike Count")
boxplot(cnt~weathersit, data=bikeData, main="Boxplot for Weather Situation", xlab="WeatherSit.", ylab="Bike Count")
boxplot(cnt~weekday, data=bikeData, main="Boxplot for Weekday", xlab="WeekDay", ylab="Bike Count")
boxplot(cnt~workingday, data=bikeData, main="Boxplot for Working Day", xlab="WorkingDay", ylab="Bike Count")


#CREATING DENSITY PLOTS OF CATEGORICAL VARIABLES TO CHECK THE DISTRIBUTION

season_factors <- factor(season, levels= c(1,2,3,4),
                         labels = c("Spring", "Summer", "Fall", "Winter"))
sm.density.compare(bikeData$cnt, bikeData$season, xlab="Count of Bikes")
title(main="Count of Bikes Distribution by Season values")
colfill<-c(2:(2+length(levels(season_factors)))) 
legend(locator(1), levels(season_factors), fill=colfill)

weathersit_factors <- factor(weathersit, levels= c(1,2,3,4),
                             labels = c("Clear", "Cloudy", "Light Snow/Rain", "Heavy Rain/Snow"))
sm.density.compare(bikeData$cnt, bikeData$weathersit, xlab="Count of Bikes")
title(main="Count of Bikes Distribution by Weather Situation")
colfill<-c(2:(2+length(levels(weathersit_factors)))) 
legend(locator(1), levels(weathersit_factors), fill=colfill)

holiday_factors <- factor(holiday, levels= c(1,2),
                          labels = c("Non-Holiday", "Holiday"))
sm.density.compare(bikeData$cnt, bikeData$holiday, xlab="Count of Bikes")
title(main="Count of Bikes Distribution by Holiday")
colfill<-c(2:(2+length(levels(holiday_factors)))) 
legend(locator(1), levels(holiday_factors), fill=colfill)


# CHECKING DISTRIBUTION OF NUMERICAL VARIABLES

plot(bikeData$denorm_temp, bikeData$cnt ,type = 'h', col= 'green', xlab = 'Denormalized Temperature', ylab = 'Total Bike Rentals')

plot(bikeData$denorm_atemp, bikeData$cnt ,type = 'h', col= 'green', xlab = 'Denormalized Feel Temperature', ylab = 'Total Bike Rentals')

plot(bikeData$denorm_wind, bikeData$cnt ,type = 'h', col= 'green', xlab = 'Denormalized Windspeed', ylab = 'Total Bike Rentals')

plot(bikeData$denorm_hum, bikeData$cnt ,type = 'h', col= 'green', xlab = 'Denormalized Humidity', ylab = 'Total Bike Rentals')


# CREATING TRAINING AND TEST SETS AND CHECK BOTH THE SETS

split = sample.split(bikeData$cnt, SplitRatio = 2/3)
training_set = subset(bikeData, split == TRUE)
test_set = subset(bikeData, split == FALSE)
head(training_set)
head(test_set)

#CHECKING FOR IMPORTANCE OF THE PREDICTORS THROUGH RANDOM FOREST


predictor_importance <- randomForest(cnt ~ season + yr + mnth + holiday + weekday + workingday + temp + atemp + hum + windspeed, data = training_set,
                                     ntree = 100, keep.forest = FALSE, importance = TRUE)
importance(predictor_importance, type = 1)

#SELECTING COLUMNS FOR CORRELATION MATRIX WITH IMPORTANT PREDICTORS

truncated = select(bikeData,season,mnth,weathersit,temp,atemp,hum,windspeed,cnt)

#USING SYMNUM FUNCTION TO DEPICT HIGHEST COLLINEARITY WITH SYMBOLS.
symnum(cor(truncated))

#PLOT THE CORRELATION MATRIX
corrgram(bikeData[3:16], order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt,
         main="Bike Count Data")


#BUILDING LINEAR REGRESSION MODEL ON THE TRAINING SET USING MOST IMPORTANT PREDICTORS 

linearModel <- lm(cnt~season+weathersit+denorm_temp+denorm_hum+denorm_wind+yr, data=training_set)
summary(linearModel)

#USING THE LINEAR REGRESSION MODEL FOR PREDICTIONS ON TEST SET
predictions <- predict(linearModel,test_set)
predictions

#USING RANDOM FOREST REGRESSION MODEL ON TRAINING SET

fit <- rpart(cnt~season+weathersit+denorm_atemp+denorm_hum+denorm_wind+yr, 
             method="anova", data=training_set)

#USING RANDOM FOREST FOR PREDICTIONS ON TEST SET

predictions_DT = predict(fit, test_set)
head(predictions_DT)

# VISULALIZATION OF THE RESULTS

printcp(fit) # display the results 
plotcp(fit) # visualize cross-validation results 
summary(fit) # detailed summary of splits

plot(fit, uniform=TRUE, 
     main="Regression Tree for Count of Bikes ")
text(fit, use.n=FALSE, all=TRUE, cex=.8)

#CREATING FUNCITON FOR MEAN ABOSULTE ERROR(MAE) PERCENTAGE

mape = function(y, yhat){
           mean(abs((y-yhat)/y))*100
}

#CALCULATING MAE PERCENTAGE FOR BOTH MODELS RESPECTIVELY

mape(test_set[,16], predictions)
mape(test_set[,16], predictions_DT)



