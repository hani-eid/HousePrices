# setwd("C:\\Users\\angel\\Dropbox\\IE\\1st Assignement\\house-prices")
Sys.setenv(JAVA_HOME='C:\\Program Files\\Java\\jre1.8.0_161')
library(ggplot2)
library(plyr)
library(dplyr)
library(moments)
library(glmnet)
library(caret)
library(FSelector)
library(dummies)


# Introduction
#In this first practical session we will make the first contact with the featuring engineering process and its impact in a ML pipeline.
#Feature engineering is a very important part of the process of developing prediction models. It is considered, by many authors, an art, and it involves human-driven design and intuition. Feature engineering sessions will try to uncover the most relevant issues that must be addressed, and also provide some guidelines to start building sound feature engineering processes for ML problems. 

#The experimental dataset we are going to use is the House Prices Dataset. It includes 79 explanatory variables of residential homes. For more details on the dataset and the competition see <https://www.kaggle.com/c/house-prices-advanced-regression-techniques>.

## What is my goal?
# I have to predict predict the final price of each home (Therefore, this is a regression task)
# I have to use the feature engineering techniques explained in class to transform the dataset.

# Data Reading and preparation
#The dataset is offered in two separated fields, one for the training and another one for the test set. Best way of moving forward is to sign in into Kaggle and download them from the 'Data' section.


training_data = read.csv(file = file.path("Data", "train.csv"))
test_data = read.csv(file = file.path("Data", "test.csv"))


length(unique(training_data$Id)) == nrow(training_data)


#There is no duplicates so we remove the Id column
training_data = training_data[ , -which(names(training_data) %in% c("Id"))]



#Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.

## Hunting NAs
#Our dataset is filled with many missing values, therefore, before we can build any predictive model we'll clean our 
#data by filling in all NA's with appropriate values.

#Counting columns with null values

na.cols <- which(colSums(is.na(training_data)) > 0)
sort(colSums(sapply(training_data[na.cols], is.na)), decreasing = TRUE)
paste('There are', length(na.cols), 'columns with missing values')


#NA imputation:

# Alley : NA means "no alley access"
training_data$Alley = factor(training_data$Alley, levels=c(levels(training_data$Alley), "None"))
training_data$Alley[is.na(training_data$Alley)] = "None"

training_data$BedroomAbvGr[is.na(training_data$BedroomAbvGr)] <- 0

# Bsmt : NA for basement features is "no basement"
training_data$BsmtQual = factor(training_data$BsmtQual, levels=c(levels(training_data$BsmtQual), "No"))
training_data$BsmtQual[is.na(training_data$BsmtQual)] = "No"

training_data$BsmtCond = factor(training_data$BsmtCond, levels=c(levels(training_data$BsmtCond), "No"))
training_data$BsmtCond[is.na(training_data$BsmtCond)] = "No"

training_data$BsmtExposure[is.na(training_data$BsmtExposure)] = "No"

training_data$BsmtFinType1 = factor(training_data$BsmtFinType1, levels=c(levels(training_data$BsmtFinType1), "No"))
training_data$BsmtFinType1[is.na(training_data$BsmtFinType1)] = "No"

training_data$BsmtFinType2 = factor(training_data$BsmtFinType2, levels=c(levels(training_data$BsmtFinType2), "No"))
training_data$BsmtFinType2[is.na(training_data$BsmtFinType2)] = "No"

# Fence : NA means "no fence"
training_data$Fence = factor(training_data$Fence, levels=c(levels(training_data$Fence), "No"))
training_data$Fence[is.na(training_data$Fence)] = "No"

# FireplaceQu : NA means "no fireplace"
training_data$FireplaceQu = factor(training_data$FireplaceQu, levels=c(levels(training_data$FireplaceQu), "No"))
training_data$FireplaceQu[is.na(training_data$FireplaceQu)] = "No"

# Garage : NA for garage features is "no garage"
training_data$GarageType = factor(training_data$GarageType, levels=c(levels(training_data$GarageType), "No"))
training_data$GarageType[is.na(training_data$GarageType)] = "No"

training_data$GarageFinish = factor(training_data$GarageFinish, levels=c(levels(training_data$GarageFinish), "No"))
training_data$GarageFinish[is.na(training_data$GarageFinish)] = "No"

training_data$GarageQual = factor(training_data$GarageQual, levels=c(levels(training_data$GarageQual), "No"))
training_data$GarageQual[is.na(training_data$GarageQual)] = "No"

training_data$GarageCond = factor(training_data$GarageCond, levels=c(levels(training_data$GarageCond), "No"))
training_data$GarageCond[is.na(training_data$GarageCond)] = "No"

# LotFrontage : NA most likely means no lot frontage
#training_data$LotFrontage[is.na(training_data$LotFrontage)] <- 0
training_data$LotFrontage[is.na(training_data$LotFrontage)] <- median(training_data$LotFrontage)

# MasVnrType : NA most likely means no veneer
training_data$MasVnrType[is.na(training_data$MasVnrType)] = "None"
training_data$MasVnrArea[is.na(training_data$MasVnrArea)] <- 0

# MiscFeature : NA = "no misc feature"
training_data$MiscFeature = factor(training_data$MiscFeature, levels=c(levels(training_data$MiscFeature), "No"))
training_data$MiscFeature[is.na(training_data$MiscFeature)] = "No"

# PoolQC : data description says NA means "no pool"
training_data$PoolQC = factor(training_data$PoolQC, levels=c(levels(training_data$PoolQC), "No"))
training_data$PoolQC[is.na(training_data$PoolQC)] = "No"

# Electrical : NA means "UNK"
training_data$Electrical = factor(training_data$Electrical, levels=c(levels(training_data$Electrical), "UNK"))
#training_data$Electrical[is.na(training_data$Electrical)] = "UNK"
training_data$Electrical[is.na(training_data$Electrical)] = "SBrkr"

# GarageYrBlt: It seems reasonable that most houses would build a garage when the house itself was built.
idx <- which(is.na(training_data$GarageYrBlt))
training_data[idx, 'GarageYrBlt'] <- training_data[idx, 'YearBuilt']

na.cols <- which(colSums(is.na(training_data)) > 0)
paste('There are now', length(na.cols), 'columns with missing values')

#We repeat the process for test_data ####

# Alley : data description says NA means "no alley access"
test_data$Alley = factor(test_data$Alley, levels=c(levels(test_data$Alley), "None"))
test_data$Alley[is.na(test_data$Alley)] = "None"

test_data$BedroomAbvGr[is.na(test_data$BedroomAbvGr)] <- 0

# BsmtQual etc : data description says NA for basement features is "no basement"
test_data$BsmtQual = factor(test_data$BsmtQual, levels=c(levels(test_data$BsmtQual), "No"))
test_data$BsmtQual[is.na(test_data$BsmtQual)] = "No"

test_data$BsmtCond = factor(test_data$BsmtCond, levels=c(levels(test_data$BsmtCond), "No"))
test_data$BsmtCond[is.na(test_data$BsmtCond)] = "No"

test_data$BsmtExposure[is.na(test_data$BsmtExposure)] = "No"

test_data$BsmtFinType1 = factor(test_data$BsmtFinType1, levels=c(levels(test_data$BsmtFinType1), "No"))
test_data$BsmtFinType1[is.na(test_data$BsmtFinType1)] = "No"

test_data$BsmtFinType2 = factor(test_data$BsmtFinType2, levels=c(levels(test_data$BsmtFinType2), "No"))
test_data$BsmtFinType2[is.na(test_data$BsmtFinType2)] = "No"

# Fence : data description says NA means "no fence"
test_data$Fence = factor(test_data$Fence, levels=c(levels(test_data$Fence), "No"))
test_data$Fence[is.na(test_data$Fence)] = "No"

# FireplaceQu : data description says NA means "no fireplace"
test_data$FireplaceQu = factor(test_data$FireplaceQu, levels=c(levels(test_data$FireplaceQu), "No"))
test_data$FireplaceQu[is.na(test_data$FireplaceQu)] = "No"

# GarageType etc : data description says NA for garage features is "no garage"
test_data$GarageType = factor(test_data$GarageType, levels=c(levels(test_data$GarageType), "No"))
test_data$GarageType[is.na(test_data$GarageType)] = "No"

test_data$GarageFinish = factor(test_data$GarageFinish, levels=c(levels(test_data$GarageFinish), "No"))
test_data$GarageFinish[is.na(test_data$GarageFinish)] = "No"

test_data$GarageQual = factor(test_data$GarageQual, levels=c(levels(test_data$GarageQual), "No"))
test_data$GarageQual[is.na(test_data$GarageQual)] = "No"

test_data$GarageCond = factor(test_data$GarageCond, levels=c(levels(test_data$GarageCond), "No"))
test_data$GarageCond[is.na(test_data$GarageCond)] = "No"

# LotFrontage : NA most likely means no lot frontage
#test_data$LotFrontage[is.na(test_data$LotFrontage)] <- 0
test_data$LotFrontage[is.na(test_data$LotFrontage)] <- median(test_data$LotFrontage)

# MasVnrType : NA most likely means no veneer
test_data$MasVnrType[is.na(test_data$MasVnrType)] = "None"
test_data$MasVnrArea[is.na(test_data$MasVnrArea)] <- 0

# MiscFeature : data description says NA means "no misc feature"
test_data$MiscFeature = factor(test_data$MiscFeature, levels=c(levels(test_data$MiscFeature), "No"))
test_data$MiscFeature[is.na(test_data$MiscFeature)] = "No"

# PoolQC : data description says NA means "no pool"
test_data$PoolQC = factor(test_data$PoolQC, levels=c(levels(test_data$PoolQC), "No"))
test_data$PoolQC[is.na(test_data$PoolQC)] = "No"



## Factorize features
#Some numerical features are actually really categories. 
#Therefore we transform the feature from numeric to categorical

training_data$MSSubClass <- as.factor(training_data$MSSubClass)
training_data$MoSold <- as.factor(training_data$MoSold)

test_data$MSSubClass <- as.factor(test_data$MSSubClass)
test_data$MoSold <- as.factor(test_data$MoSold)

training_data$YrSold <- as.factor(training_data$YrSold)
test_data$YrSold <- as.factor(test_data$YrSold)

## Skewness

#If we print the histogram of the target value, 
#we obseve a large skewness in the Target value (i.e., the distribution in not normally distributed).
#To solve that we log transform this variable so that it becomes normally distributed. 
#A normally distributed target variable helps in the modeling step 
#(i.e., the finding of the relationship between target and independent variables).

# get data frame of SalePrice and log(SalePrice + 1) for plotting
df <- rbind(data.frame(version="log(price+1)",x=log(training_data$SalePrice + 1)),
            data.frame(version="price",x=training_data$SalePrice))

ggplot(data=df) +
  facet_wrap(~version,ncol=2,scales="free_x") +
  geom_histogram(aes(x=x), bins = 50)




#We therefore transform the target value applying log.
# Log transform the target for official scoring
training_data$SalePrice <- log1p(training_data$SalePrice)


#Dealing with Basement variables, in which we remove the below three variables since we already have TotalBsmtSf ####
# training_data$BsmtFinSF1 = NULL
# training_data$BsmtFinSF2 = NULL
# training_data$BsmtUnfSF = NULL
# 
# test_data$BsmtFinSF1 = NULL
# test_data$BsmtFinSF2 = NULL
# test_data$BsmtUnfSF = NULL
# 
# Creating new TotalSF feature ####
training_data$TotalSF = training_data$TotalBsmtSF + training_data$X1stFlrSF + training_data$X2ndFlrSF
training_data$TotalBsmtSF = NULL
training_data$X1stFlrSF = NULL
training_data$X2ndFlrSF = NULL
# 
test_data$TotalSF = test_data$TotalBsmtSF + test_data$X1stFlrSF + test_data$X2ndFlrSF
test_data$TotalBsmtSF = NULL
test_data$X1stFlrSF = NULL
test_data$X2ndFlrSF = NULL



#Combining both datasets ####
fullDS = rbind(training_data,test_data)

# Create dummies for all factors in the Training and Test ####
training_data = dummy.data.frame(training_data)
test_data = dummy.data.frame(test_data)

#The same "skewness" observed in the target variable also affects other variables. To facilitate the application of the regression model we are going to also eliminate this skewness
#For numeric feature with excessive skewness, perform log transformation

class(training_data$MSSubClass)
column_types <- sapply(names(training_data),function(x){class(training_data[[x]])})
numeric_columns <-names(column_types[column_types != "factor"])

# skew of each variable
skew <- sapply(numeric_columns,function(x){skewness(training_data[[x]],na.rm = T)})

# transform all variables above a threshold skewness.
skew <- skew[skew > 0.75]
for(x in names(skew)) {
  print(x)
  training_data[[x]] <- log(training_data[[x]] + 1)
  #test_data[[x]] = log(test_data[[x]] + 1)
}


#The same for the test data

# column_types <- sapply(names(test_data),function(x){class(test_data[[x]])})
# numeric_columns <-names(column_types[column_types != "factor"])
# 
# skew <- sapply(numeric_columns,function(x){skewness(test_data[[x]],na.rm = T)})
# skew <- skew[skew > 0.75]
# for(x in names(skew)) {
#   test_data[[x]] <- log(test_data[[x]] + 1)
# }


## Pulling the numeric variables for correlation matrix  ####
training_data_numeric = training_data[sapply(training_data, is.numeric)]
corMatrix = as.data.frame(cor(training_data_numeric$SalePrice, training_data_numeric))
corMatrix[order(corMatrix, decreasing = TRUE)]

corMatrixAll = cor(training_data_numeric)

## Ploting the correlation matrix against SalePrice  ####
corrplot(cor(training_data_numeric$SalePrice, training_data_numeric), method = "circle")
corrplot(corMatrixAll,  method = "square", order="hclust")


# Handling OUtliers ####
#Scatter Plots GrLivArea & OverallQual against SalePrice:  ####
ggplot(training_data, aes(x=GrLivArea, y=SalePrice)) +
  geom_point(color="blue")

# Removing 2 outliers in GrLivArea
which(training_data$GrLivArea>8.3 & training_data$SalePrice<12.5)
training_data = training_data[-which(training_data$GrLivArea>8.3 & training_data$SalePrice<12.5),]


# Removing Garage Area and using only Garage Cars
training_data$GarageArea = NULL
test_data$GarageArea = NULL

# Removing GarageYrBlt and YearRemodAdd and using only YearBlt
training_data$GarageYrBlt = NULL
training_data$YearRemodAdd = NULL
test_data$GarageYrBlt = NULL
test_data$YearRemodAdd = NULL

# Removing TotalRmsAbvGrd and using only GrLivArea
training_data$TotRmsAbvGrd = NULL
test_data$TotRmsAbvGrd = NULL





## Train, Validation Spliting
#We are going to split the annotated dataset in training and validation for the later evaluation of our regression models

# I found this function, that is worth to save for future ocasions.
splitdf <- function(dataframe, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)/1.5))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset=trainset,testset=testset)
}
splits <- splitdf(training_data, seed=1)
training <- splits$trainset
validation <- splits$testset



#If we inspect in detail the categorical variables of the dataset, we'll see that some are incomplete: 
#they only have a unique value for all the dataset.
#These features are not valuable. Remember the three aspects that a feature should have: 
#informative, <b>discriminative</b> and independent. Incomplete cases are not discriminative at all.
#In addition, this might create problems when fitting the regression model

#The following code show the incomplete cases

## remove incomplete cases
paste("Training set incomplete cases")
sapply(lapply(na.omit(training)[sapply(na.omit(training), is.factor)], droplevels), nlevels)
paste("Validation set incomplete cases")
sapply(lapply(na.omit(validation)[sapply(na.omit(validation), is.factor)], droplevels), nlevels)
paste("Test set incomplete cases")
sapply(lapply(na.omit(test_data)[sapply(na.omit(test_data), is.factor)], droplevels), nlevels)


#We remove the detected incomplete cases

# Remove the Utilities feature from the dataset (It only has one value)
training <- training[,-which(names(training) == "Utilities")]
validation <- validation[,-which(names(validation) == "Utilities")]
test_data <- test_data[,-which(names(test_data) == "Utilities")]




# Feature Engineering
# We here start the Feature Engineering.

## Filtering Methods
#We will rank the features according to their predictive power according to the methodologies seen in class: the Chi Squared Independence test and the Information Gain.

#### Full Model
#We first fit a lm model with all the features to have a baseline to evaluate the impact of the feature engineering.

set.seed(121)
train_control_config <- trainControl(method = "repeatedcv", 
                                     number = 5, 
                                     repeats = 1,
                                     returnResamp = "all")

full.lm.mod <- train(SalePrice ~ ., data = training, 
                     method = "lm", 
                     metric = "RMSE",
                     preProc = c("center", "scale"),
                     trControl=train_control_config)


for (x in names(validation)) {
  full.lm.mod$xlevels[[x]] <- union(full.lm.mod$xlevels[[x]], levels(validation[[x]]))
}
SalePriceIndex = which(colnames(validation) == "SalePrice")
full.lm.mod.pred <- predict(full.lm.mod, validation[,-SalePriceIndex])
full.lm.mod.pred[is.na(full.lm.mod.pred)] <- 0

my_data=as.data.frame(cbind(predicted=full.lm.mod.pred,observed=validation$SalePrice))

ggplot(my_data,aes(predicted,observed))+
  geom_point() + geom_smooth(method = "lm") +
  labs(x="Predicted") +
  ggtitle('Linear Model')

paste("Full Linear Regression RMSE = ", sqrt(mean((full.lm.mod.pred - validation$SalePrice)^2)))


### Chi-squared Selection
#Making use of the `FSelector` package <https://cran.r-project.org/web/packages/FSelector/FSelector.pdf>, rank the features according to the Chi Squared value. If you've problems with this package (some of us have problems with it), do some research to find another packages that will provide the Chi squared selection.

#Does it make sense to remove some features? Is so, do it! <b>(Tip: Sure it does)</b>

weights<- data.frame(chi.squared(SalePrice~., training_data))
weights$feature <- rownames(weights)
weights[order(weights$attr_importance, decreasing = TRUE),]
chi_squared_features <- weights$feature[weights$attr_importance >= 0.1]

#### Evaluation
#Evaluate the impact (in terms of RMSE) of the feature selection.
#To that end, execute the previous LM model taking as input the filtered training set

chi_squared.lm.mod <- train(SalePrice ~ ., data = training[append(chi_squared_features, "SalePrice")], 
                            method = "lm", 
                            metric = "RMSE",
                            preProc = c("center", "scale"),
                            trControl=train_control_config)

for (x in names(validation)) {
  chi_squared.lm.mod$xlevels[[x]] <- union(chi_squared.lm.mod$xlevels[[x]], levels(validation[[x]]))
}
chi_squared.lm.mod.pred <- predict(chi_squared.lm.mod, validation[,-SalePriceIndex])
chi_squared.lm.mod.pred[is.na(chi_squared.lm.mod.pred)] <- 0

my_data=as.data.frame(cbind(predicted=chi_squared.lm.mod.pred,observed=validation$SalePrice))

ggplot(my_data,aes(predicted,observed))+
  geom_point() + geom_smooth(method = "lm") +
  labs(x="Predicted") +
  ggtitle('Linear Model')

paste("Chi-Squared Filtered Linear Regression RMSE = ", sqrt(mean((chi_squared.lm.mod.pred - validation$SalePrice)^2)))


### Information Gain Selection
#Let's experiment now with Information Gain Selection.
#Making also use of the `FSelector` package <https://cran.r-project.org/web/packages/FSelector/FSelector.pdf>, rank the features according to their Information Gain and filter those which you consider, according to the IG value.

#Again, there're more alternatives to compute the IG.

weights<- data.frame(information.gain(SalePrice~., training_data))
weights$feature <- rownames(weights)
weights[order(weights$attr_importance, decreasing = TRUE),]
information_gain_features <- weights$feature[weights$attr_importance >= 0.05]


#### Evaluation
#Evaluate the impact of the IG selection in the model performance

ig.lm.mod <- train(SalePrice ~ ., data = training[append(information_gain_features, "SalePrice")], 
                   method = "lm", 
                   metric = "RMSE",
                   preProc = c("center", "scale"),
                   trControl=train_control_config)

for (x in names(validation)) {
  ig.lm.mod$xlevels[[x]] <- union(ig.lm.mod$xlevels[[x]], levels(validation[[x]]))
}
ig.lm.mod.pred <- predict(ig.lm.mod, validation[,-SalePriceIndex])
ig.lm.mod.pred[is.na(ig.lm.mod.pred)] <- 0

my_data=as.data.frame(cbind(predicted=ig.lm.mod.pred,observed=validation$SalePrice))

ggplot(my_data,aes(predicted,observed))+
  geom_point() + geom_smooth(method = "lm") +
  labs(x="Predicted") +
  ggtitle('Linear Model')

paste("IG Filtered Linear Regression RMSE = ", sqrt(mean((ig.lm.mod.pred - validation$SalePrice)^2)))


#Using the result of the evaluation, filter the dataset (according to the method and cutoff that you decide)

#Based on these results, we filter the training and validation set with the Information Gain features.
training <- training[append(information_gain_features, "SalePrice")]
validation <- validation[append(information_gain_features, "SalePrice")]


## Wrapper Methods
#Let us experiment now with Wrapper Methods. In particular, we are going to apply Forward Stepwise Selection Methods to find the best feature combination for this dataset.

### Stepwise


#### Backward Stepwise
#`caret` package provides a useful and easy way of experimenting with stepwise selection. Try it to know what a wrapper method suggests as the best possible subset of features and compare your results with the baseline.


 train_control_config_4_stepwise <- trainControl(method = "none")
# 
# backward.lm.mod <- train(SalePrice ~ ., data = training, 
#                          method = "glmStepAIC", 
#                          direction = "backward",
#                          trace = FALSE,
#                          metric = "RMSE",
#                          steps = 15,
#                          preProc = c("center", "scale"),
#                          trControl=train_control_config_4_stepwise)
# 
# #Printout only the selected features.
# 
# paste("Features Selected" ,backward.lm.mod$finalModel$formula[3])
# 
# 
# #Comput the RMSE of the selected model
# 
# for (x in names(validation)) {
#   backward.lm.mod$xlevels[[x]] <- union(backward.lm.mod$xlevels[[x]], levels(validation[[x]]))
# }
# backward.lm.mod.pred <- predict(backward.lm.mod, validation[,-ncol(validation)])
# backward.lm.mod.pred[is.na(backward.lm.mod.pred)] <- 0
# 
# 
# 
# paste("Forward Linear Regression RMSE = ", sqrt(mean((backward.lm.mod.pred - validation$SalePrice)^2)))
# 
# my_data=as.data.frame(cbind(predicted=backward.lm.mod.pred,observed=validation$SalePrice))
# ggplot(my_data,aes(predicted,observed))+
#   geom_point() + geom_smooth(method = "lm") +
#   labs(x="Predicted") +
#   ggtitle('Linear Model')


#### Forward Stepwise

#Try the same with forward stepwise.


# forward.lm.mod <- step(glm(training$SalePrice ~ 1, data = training[,-ncol(training)]), direction = "forward", scope=formula(glm(training$SalePrice ~ ., data = training[,-ncol(training)])))

forward.lm.mod <- train(x = training[-ncol(training)], y = training$SalePrice,
                        method = "glmStepAIC", 
                        direction = "forward",
                        steps = 10,
                        trace=FALSE,
                        metric = "RMSE",
                        preProc = c("center", "scale"),
                        trControl=train_control_config_4_stepwise)




#Printout only the selected features.

paste("Features Selected" ,forward.lm.mod$finalModel$formula[3])

#Compute the new RMSE

for (x in names(validation)) {
  forward.lm.mod$xlevels[[x]] <- union(forward.lm.mod$xlevels[[x]], levels(validation[[x]]))
}

forward.lm.mod.pred <- predict(forward.lm.mod, validation[,-which(names(validation) %in% c("SalePrice"))])
forward.lm.mod.pred[is.na(forward.lm.mod.pred)] <- 0



paste("Forward Linear Regression RMSE = ", sqrt(mean((forward.lm.mod.pred - validation$SalePrice)^2)))

my_data=as.data.frame(cbind(predicted=forward.lm.mod.pred,observed=validation$SalePrice))
ggplot(my_data,aes(predicted,observed))+
  geom_point() + geom_smooth(method = "lm") +
  labs(x="Predicted") +
  ggtitle('Linear Model')




forward_features <- c("OverallQual", "Neighborhood", "GrLivArea", "BsmtFinSF1" ,"MSSubClass", "OverallCond", "GarageCars", "YearBuilt", "LotArea", "MSZoning")



## Embedded
#Finally, we will experiment with embedded methods. 
#In particular we are going to focus on Ridge and  Lasso Regularization.

### Ridge Regression
#For this exercise, we are going to make use of the <a href="https://cran.r-project.org/web/packages/glmnet/index.html">`glmnet`</a> library. Take a look to the library and fit a glmnet model for Ridge Regression, using the grid of lambda values provided.

lambdas <- 10^seq(-2, 3, by = .1)
ridge.mod <- glmnet(x = data.matrix(training[,-ncol(training)]), y=training$SalePrice, alpha = 0, lambda = lambdas)


#### Evaluation
#Plotting the RMSE for the different lambda values, we can see the impact of this parameter in the model performance.
#Small values seem to work better for this dataset

RMSE = numeric(length(lambdas))
for (i in seq_along(lambdas)){
  ridge.pred=predict(ridge.mod, s=lambdas[i], data.matrix(validation[,-ncol(validation)]))
  RMSE[i] <- sqrt(mean((ridge.pred - validation$SalePrice)^2))
}
plot(lambdas, RMSE, main="Ridge", log="x", type = "b")




##### Cross Validation
#Making use of cv.glmnet <https://www.rdocumentation.org/packages/glmnet/versions/2.0-12/topics/cv.glmnet>, create a cross-validated Ridge Regression Model for the provided lambdas.

#Plotting again the error, CV give us a better understanding on the impact of lambda in the model performance

ridge.cv_fit <- cv.glmnet(x = data.matrix(training[,-ncol(training)]), y=training$SalePrice, alpha = 0, lambda = lambdas)
plot(ridge.cv_fit)


#<b>Interpretation:</b>

#  1. The plot shows the MSE (red dots) for the provided lambda values (included in the grid).
#2. The confidence intervals represent error estimates for the RSE, computed using CV. 
#3. The vertical lines show the locations of lambda.min (lambda that achives the best MSE) and lambda.1se (the largest lambda value within 1 standard error of lambda.min. Using lambda.1se hedges against overfitting by selecting a larger lambda value than the min).
#4. The numbers across the top are the number of nonzero coefficient estimates.

#Select the best lambda form the CV model, use it to predict the target value of the validation set and evaluate the results (in terms of RMSE)

bestlam <- ridge.cv_fit$lambda.min
paste("Best Lambda value from CV=", bestlam)
ridge.pred=predict(ridge.mod, s=bestlam, data.matrix(validation[,-ncol(validation)]))
paste("RMSE for lambda ", bestlam, " = ", sqrt(mean((ridge.pred - validation$SalePrice)^2)))



#Select the λ1se value from the CV model to predict on the validation set

lam1se <- ridge.cv_fit$lambda.1se
paste("Lambda 1se value from CV=", lam1se)
ridge.pred=predict(ridge.mod, s=lam1se, data.matrix(validation[,-ncol(validation)]))
paste("RMSE for lambda ", lam1se, " = ", sqrt(mean((ridge.pred - validation$SalePrice)^2)))

#As you can see, the result is almost the same, but the 1se value is less prone to overfitting

#Let's plot the predictions against the actual values to have an idea of the model performance

# Plot important coefficients
my_data=as.data.frame(cbind(predicted=ridge.pred,observed=validation$SalePrice))

ggplot(my_data,aes(my_data["1"],observed))+
  geom_point()+geom_smooth(method="lm")+
  scale_x_continuous(expand = c(0,0)) +
  labs(x="Predicted") +
  ggtitle('Ridge')


#Rank the variables according to the importance attributed by the model


# Print, plot variable importance
imp <- varImp(ridge.mod, lambda = bestlam)
names <- rownames(imp)[order(imp$Overall, decreasing=TRUE)]
importance <- imp[names,]

data.frame(row.names = names, importance)



### Lasso Regresion
#Using again the <a href="https://cran.r-project.org/web/packages/glmnet/index.html">`glmnet`</a> library, fit a Lasso Regression (take a look to the alpha parameter) using the grid of lambda values provided.

#### Evaluation
#Plot the RMSE for the different lambda values and Explain the results.

lambdas <- 10^seq(-3, 3, by = .1)

lasso.cv_fit <- cv.glmnet(x = data.matrix(training[,-ncol(training)]), y=training$SalePrice, alpha = 1, lambda = lambdas)
plot(lasso.cv_fit)



#<b>Interpretation:</b>
#As said in class, In contrast to Ridge Regression, Lasso Regression performs feature selection (it is forcing the coefficients to be 0), as you can see in the top numbers in the plot.


#Select the best lambda form the CV model, use it to predict the target value of the validation set and evaluate the results (in terms of RMSE)

bestlam <- lasso.cv_fit$lambda.min
paste("Best Lambda value from CV=", bestlam)
lasso.mod <- glmnet(x = data.matrix(training[,-ncol(training)]), y=training$SalePrice, alpha = 1, lambda = lambdas)
lasso.pred=predict(lasso.mod, s=bestlam, data.matrix(validation[,-ncol(validation)]))
paste("RMSE for lambda ", bestlam, " = ", sqrt(mean((lasso.pred - validation$SalePrice)^2)))


#Select the λ1se value from the CV model to predict on the validation set

lam1se <- lasso.cv_fit$lambda.1se
paste("Lambda 1se value from CV=", lam1se)
lasso.mod <- glmnet(x = data.matrix(training[,-ncol(training)]), y=training$SalePrice, alpha = 1, lambda = lambdas)
lasso.pred=predict(lasso.mod, s=lam1se, data.matrix(validation[,-ncol(validation)]))
paste("RMSE for lambda ", lam1se, " = ", sqrt(mean((lasso.pred - validation$SalePrice)^2)))


#Predictions against the actual values 

# Plot important coefficients
my_data=as.data.frame(cbind(predicted=lasso.pred,observed=validation$SalePrice))

ggplot(my_data,aes(my_data["1"],observed))+
  geom_point()+geom_smooth(method="lm")+
  scale_x_continuous(expand = c(0,0)) +
  labs(x="Predicted") +
  ggtitle('Lasso')


#Variable importance

# Print, plot variable importance
imp <- varImp(lasso.mod, lambda = bestlam)
names <- rownames(imp)[order(imp$Overall, decreasing=TRUE)]
importance <- imp[names,]

data.frame(row.names = names, importance)



#Variables selected by the lasso model (only those with importance larger than 0)

filtered_names <- rownames(imp)[order(imp$Overall, decreasing=TRUE)][1:28]
print(filtered_names)

#test_data[information_gain_features[23]]

# Prediction on the test data

log_prediction <- predict(lasso.cv_fit,  s=lasso.cv_fit$lambda.min, newx = data.matrix(test_data[information_gain_features]))
actual_pred <- exp(log_prediction)-1
hist(actual_pred)
submit <- data.frame(Id=test_data$Id,SalePrice=actual_pred)
colnames(submit) <-c("Id", "SalePrice")

submit$SalePrice[is.na(submit$SalePrice)] <- 0
replace_value_for_na <- sum(na.omit(submit$SalePrice))/(nrow(submit) - sum(submit$SalePrice == 0))
submit$SalePrice[submit$SalePrice == 0] <- replace_value_for_na

write.csv(submit,file="lasso_information_gain.csv",row.names=F)
