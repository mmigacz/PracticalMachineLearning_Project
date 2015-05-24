---
title: "Barbell Lifts Analysis"
output: html_document
---

#Synopsis
The goal of this project is to build a model with data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict the manner in which they did an exercise - barbell lifts.
The model should should predict "classe" variable with any other variables.

#Preparing data and dependencies
Initial preparations - loading required packages, loading data.


```r
suppressMessages(library(caret))

# Load data
srcTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if(!file.exists("pml-training.csv")){
  download.file(srcTrain, destfile = "pml-training.csv", method = "curl", mode="wb")
}

srcTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if(!file.exists("pml-testing.csv")){
  download.file(srcTest, destfile = "pml-testing.csv", method = "curl", mode="wb")
}
pml.training <- read.csv("pml-training.csv")
pml.testing <- read.csv("pml-testing.csv")
```

##Subseting and cleaning data
Some variables are for statistical purpose only. Because they are derived from the others (also correlated), than can be removed to simplify building prediction model.


```r
remove_cols_reqex <- "^(kurtosis|skewness|min|max|stddev|total|var|avg|ampl|num_window|cvtd_timestamp|X|new_window)"
pml.training <- pml.training[,grep(remove_cols_reqex, names(pml.training),invert=T)]
```

Split training data into training and validation subsets

```r
sub <- createDataPartition(y=pml.training$classe, p=.7, list=F) 
training <- pml.training[sub,]
validation <- pml.training[-sub,]
```

##Machine learning

Cross-validation with 3 fold, and allow running for paralell for shorten time

```r
ctrl <- trainControl(method="cv",number=3,allowParallel=T)
```

User random forest to build a model.

```r
modelRF <- train(classe~., data=training, method="rf", trControl=ctrl, allowParalell=T)
modelRF
```

```
## Random Forest 
## 
## 13737 samples
##    51 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (3 fold) 
## 
## Summary of sample sizes: 9158, 9158, 9158 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9908277  0.9883966  0.001705667  0.002157702
##   28    0.9957778  0.9946595  0.001533909  0.001940035
##   55    0.9912645  0.9889498  0.002431869  0.003074811
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 28.
```

Build confusion matrix for validation data

```r
confusionMatrix(predict(modelRF,validation),validation$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B    2 1136    1    0    0
##          C    0    1 1023    1    3
##          D    0    0    2  963    0
##          E    0    0    0    0 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.998           
##                  95% CI : (0.9964, 0.9989)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9974          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9988   0.9974   0.9971   0.9990   0.9972
## Specificity            0.9995   0.9994   0.9990   0.9996   1.0000
## Pos Pred Value         0.9988   0.9974   0.9951   0.9979   1.0000
## Neg Pred Value         0.9995   0.9994   0.9994   0.9998   0.9994
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2841   0.1930   0.1738   0.1636   0.1833
## Detection Prevalence   0.2845   0.1935   0.1747   0.1640   0.1833
## Balanced Accuracy      0.9992   0.9984   0.9980   0.9993   0.9986
```

The random forest accuracy on training set is 99.75% on the validation set. It suggests, that the out-of-sample error could be small.

##Test
Predict results on test data


```r
pml.testing.predicted = predict(modelRF,pml.testing)
pml.testing.predicted
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```





