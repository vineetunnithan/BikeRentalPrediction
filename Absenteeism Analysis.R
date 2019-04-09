##### Reading data in R environment####
install.packages("readxl", "httr")
library(readxl) 
library(httr)

link = "https://s3-ap-southeast-1.amazonaws.com/edwisor-india-bucket/projects/data/DataN0101/Absenteeism_at_work_Project.xls" 

GET(link, write_disk(dataset1 <- tempfile(fileext = ".xls")))
Absenteeism_at_work <- read_excel(dataset1)

# Checking the structure of the dataset

str(Absenteeism_at_work)
View(Absenteeism_at_work)

#### Cleaning the dataset
# Removing the first column, as it won't participate in modelling
Absenteeism_at_work = Absenteeism_at_work[-1]
data4svm = Absenteeism_at_work
data4svm = na.omit(data4svm)

##### Preprocessing the dataset#######
#Changing attributes to factors as they are taken as integer in R

Absenteeism_at_work$`Reason for absence` = as.factor(Absenteeism_at_work$`Reason for absence`)
Absenteeism_at_work$`Day of the week`= as.factor(Absenteeism_at_work$`Day of the week`)
Absenteeism_at_work$Seasons = as.factor(Absenteeism_at_work$Seasons)
Absenteeism_at_work$`Disciplinary failure`= as.factor(Absenteeism_at_work$`Disciplinary failure`)
Absenteeism_at_work$`Social drinker`= as.factor(Absenteeism_at_work$`Social drinker`)
Absenteeism_at_work$`Social smoker`=as.factor(Absenteeism_at_work$`Social smoker`)

# Again verifying the structure of the dataset
str(Absenteeism_at_work)

####### Cleaning the dataset. Looking for the NA values #########
# We will use the summary function to see the descriptive statistics and it will also check the NA values

any(is.na(Absenteeism_at_work))
sum(is.na(Absenteeism_at_work))#Checking for NA values. 

#Our dataset is not clean
# Removing NA values from the dataset
Absenteeism_at_work = na.omit(Absenteeism_at_work)

####### Descriptive statistics###############

summary(Absenteeism_at_work) # We can see there is no NA values reported here. Hence, our dataset is clean.



###### Graphical visualization of the dataset  ######
# Scatter plot to check the correlation 
plot.new()
pairs(Absenteeism_at_work)

# Find out which season has highest number of absentism
boxplot(Absenteeism_at_work$`Absenteeism time in hours`~Absenteeism_at_work$Seasons, col = "yellow")

# Exploring more with visualization
# Installing the required libraries
library(ggplot2)  
library(grid)  
library(gridExtra)  
plot.new()

p1t1 <- ggplot(Absenteeism_at_work, aes(x = Pet, fill = as.factor(Pet))) + geom_bar() 
s1n1 <- ggplot(Absenteeism_at_work, aes(x = Son, fill = as.factor(Son))) + geom_bar()

S1S1 <- ggplot(Absenteeism_at_work, aes(x = `Social smoker`, fill =`Social drinker`)) + geom_bar() 

Day1.wk1 <- ggplot(Absenteeism_at_work, aes(x = `Day of the week`, fill =  `Day of the week`)) + geom_bar() 
Sn1s1 <- ggplot(Absenteeism_at_work, aes(x =   Seasons,fill = Seasons)) + geom_bar()

grid.arrange(p1t1,s1n1, nrow = 1)
grid.arrange(S1S1,Sn1s1, nrow = 1)
grid.arrange(Day1.wk1, nrow = 1)

# Histograms of numeric attributes of signifiance. Means parameters which are measured not the obvious one like Months
graphics.off()
par("mar")
par(mar = c(2,2,2,2))
plot.new()
par(mfrow=c(4,3))
hist(Absenteeism_at_work$`Transportation expense`, main="Histograms for Transportation Expenses", col="orange")
hist(Absenteeism_at_work$`Distance from Residence to Work`, main="Histograms for Distance from Residence to Work", col="orange")
hist(Absenteeism_at_work$`Service time`, main="Histograms for Service Time", col="orange")
hist(Absenteeism_at_work$Age, main="Histograms for Age", col="orange")
hist(Absenteeism_at_work$`Work load Average/day`, main="Histograms for Work load Average/day", col="orange")
hist(Absenteeism_at_work$`Hit target`, main="Histograms for Hit Target", col="orange")
hist(Absenteeism_at_work$Weight, main="Histograms for Weight", col="orange")
hist(Absenteeism_at_work$Height, main="Histograms for Height", col="orange")
hist(Absenteeism_at_work$`Body mass index`, main="Histograms for BMI", col="orange")
hist(Absenteeism_at_work$`Hit target`, main="Histograms for Hit Target", col="orange")
hist(Absenteeism_at_work$`Absenteeism time in hours`, main="Histograms for Absenteesim time", col="orange")

# We are also interested in knowing the density distribution of the Absentees hours
plot(density(Absenteeism_at_work$`Absenteeism time in hours`), main="Distribution of the Abseneeism hour")
polygon(density(Absenteeism_at_work$`Absenteeism time in hours`), col="red", border="blue") 


####### Planning for the modelling. ########
# As our data has both categorical and numerical data, we cannot do simple Linear regression for this.
# Hence, we choose to do deision tree regression Analysis.

# Making training data and testing data
set.seed(101)
index = sample(seq(nrow(Absenteeism_at_work)), size = 0.8*nrow(Absenteeism_at_work))

train = Absenteeism_at_work[index,]
test = Absenteeism_at_work[-index,]



##### Modelling #######
library(rpart) 

tree.model <- rpart(train$`Absenteeism time in hours`~ ., data = train, method="anova")

summary(tree.model)
tree.model

# Visualize the fitted tree
graphics.off()
par("mar")
par(mar = c(2,2,2,2))
plot.new()
plot(tree.model)
text(tree.model, pretty=0)

par(mfrow=c(1,2)) 
rsq.rpart(tree.model) # So maximum R square is 40% with 7 splits

###### Prediction A ######
pre.val = predict(tree.model, newdata = test, method = "anova")
plot(pre.val, test$`Absenteeism time in hours`)
abline(0,1)

# Calculate RMSE for the predicted values
print(sqrt(mean((pre.val-test$`Absenteeism time in hours`)^2)))


#### Model 2 SVM regression####
#As SVM works on numeric data, we have to use all the numeric value instead of categorical.

# Preparing test and train data for the svm model

set.seed(101)
index = sample(seq(nrow(data4svm)), size = 0.8*nrow(data4svm))

train.svm = data4svm[index,]
test.svm = data4svm[-index,]


library(e1071)

svm.regression <- svm(train.svm$`Absenteeism time in hours` ~ . , data=train.svm)
svm.regression

#Prediction B

predict.svm = predict(svm.regression, newdata = test.svm, method = "anova")
predict.svm

# Calculate RMSE for the predicted values
print(sqrt(mean((predict.svm-test.svm$`Absenteeism time in hours`)^2)))

####Final Comments#####
#The model can be further improved with higher level complex interaction between the numerical variables
#For the simplicity that has not be been included in this study