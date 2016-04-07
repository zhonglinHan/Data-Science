rm(list = ls())
install.packages("method")
install.packages("mlbench")
install.packages('tree')
install.packages('rpart')
install.packages('randomForest')
install.packages('adabag')
install.packages('caret')    

install.packages("devtools")
library(devtools)
devtools::install_github('dmlc/xgboost',subdir='R-package')
## install_url("https://github.com/dmlc/xgboost.git")
### load libraries ###
library(xgboost)
library(methods)
library(adabag)
library(caret)
library(mlbench)
library (tree)
library(rpart)
library(randomForest)

################# BOOSTING TREE #######################
### read in data ####
train = read.csv("train_v1.csv", header = TRUE,stringsAsFactors = F) # without balanced
train.trans = read.csv("train_freq_v2.csv", header = TRUE,stringsAsFactors = F)
test.trans = read.csv("test_freq_v2.csv", header = TRUE,stringsAsFactors = F)

write.csv(train.trans, file = "train_freq_v2.csv", row.names = FALSE)
write.csv(test.trans, file = "test_freq_v2.csv", row.names = FALSE)


## now give the option of sqrt/log transformation
train.trans = train; 
for(ifeat in 1 : 93){
  i = ifeat + 1 
  train.trans[,i] = train.trans[,i]^(1/2) 
}

## now give the option to transfer each feature into its (relative) frequencies.
train.trans = train; 
for(ifeat in 1 : 93){
  cat("now processing feature ", ifeat,"\n")
  i = ifeat + 1 ;
  feat = train[,i]; 
  # get the frequency table

  train.freq = table(feat)
  train.freq = as.data.frame(train.freq) 
  names(train.freq) = c("value","freq")
  train.freq$value = as.integer(as.character(train.freq$value)) 
  # transform to relative frequency
  freq.sum = sum(train.freq$freq) ;
  for(j in 1:length(train.freq$value)){
    train.freq$freq[j] = train.freq$freq[j]/freq.sum ;
  }
  # replace feature values with their frequencies
  for(j in 1:length(train.freq$value)){
    train.trans[,i][train[,i] == train.freq$value[j]] = train.freq$freq[j] ;
  }
  rm(train.freq) ;
}

test <- read.csv("test.csv", header = TRUE,stringsAsFactors = F)# the official test set 
## now give the option to transfer each feature into its (relative) frequencies.
test.trans = test; 
for(ifeat in 1 : 93){
  cat("now processing feature ", ifeat,"\n")
  i = ifeat + 1 ;
  feat = test[,i]; 
  # get the frequency table
  
  test.freq = table(feat)
  test.freq = as.data.frame(test.freq) 
  names(test.freq) = c("value","freq")
  test.freq$value = as.integer(as.character(test.freq$value)) 
  # transform to relative frequency
  freq.sum = sum(test.freq$freq) ;
  for(j in 1:length(test.freq$value)){
    test.freq$freq[j] = test.freq$freq[j]/freq.sum ;
  }
  # replace feature values with their frequencies
  for(j in 1:length(test.freq$value)){
    test.trans[,i][test[,i] == test.freq$value[j]] = test.freq$freq[j] ;
  }
  rm(test.freq) ;
}










### process data ###
train = train.trans ;
train = subset(train, select = -c(id, setid))

y.train = train[,ncol(train)]
y.train = gsub('Class_','',y.train)
y.train = as.integer(y.train)-1 #xgboost take features in [0,numOfClass-1]

x.train = subset(train, select = -c(target))
x.train = as.matrix(x.train) 
x.train = matrix(as.numeric(x.train),nrow(x.train),ncol(x.train))

test = test.trans 
x.test = subset(test, select = -c(id))
#x.test = subset(test, select = -c(target))
x.test = as.matrix(x.test) 
x.test = matrix(as.numeric(x.test),nrow(x.test),ncol(x.test))

#### Cross Validation for Boosting Trees ####
# booster gbtree or gblinear
# cv.etas = c(0.2,0.25,0.3,0.35,0.4) ;
# cv.etas = c(0.2) ;
cv.etas = c(0.1) ;
# cv.gammas = c()
# cv.max_depths = c(6,7,8,9) ;
#cv.max_depths = c(8) ;
cv.max_depths = c(9) ;
# cv.min_child_weights = c(1,1.1,1.2,1.3) ;
# cv.min_child_weights = c(1.3) ;
cv.min_child_weights = c(1.3) ;
#cv.max_delta_steps = c(0,1,2,3) ;
cv.max_delta_steps = c(0) ;
# cv.subsamples = c(0.9) ;
cv.subsamples = c(0.8) ;
# cv.colsample_bytrees = c(0.6) ;
cv.colsample_bytrees = c(0.6) ;
#cv.subsamples = c(1,
#                  0.9,
#                  0.8)
#cv.subsamples = c(0.8)
#cv.colsample_bytrees = c(
 #                        0.8,
  #                       0.7,
  #                       0.6,
  #                       0.5,
   #                      0.4)
#cv.colsample_bytrees = c(0.8)
cv.record = NULL ;
cv.nround = 400 ;
cv.nfold = 10;


for(eta in cv.etas){
  for(max_depth in cv.max_depths){
    for(min_child_weight in cv.min_child_weights){
      for(max_delta_step in cv.max_delta_steps){
        for(subsample in cv.subsamples){
          for(colsample_bytree in cv.colsample_bytrees){
      cat("eta == ", eta, " max_depth == ", max_depth, " min_child_weight == ", min_child_weight) ;
      cat("\n")
      cat("max_delta_step == ", max_delta_step, " subsample == ", subsample, 
          " colsample_bytree == ", colsample_bytree) ;
      cat("\n")
      ptm <- proc.time()
      param <- list("objective" = "multi:softprob",
                    "eval_metric" = "mlogloss",
                    "eta" = eta,
                    "max_depth" = max_depth ,
                    "min_child_weight" = min_child_weight ,
                    "max_delta_step" = max_delta_step,
                    "subsample" = subsample ,
                    "colsample_bytree" = colsample_bytree,
                    "num_class" = 9,
                    "nthread" = 8)
      bst.cv = xgb.cv(param=param, data = x.train, label = y.train, 
                      nfold = cv.nfold, nrounds=cv.nround) ;
      cv.mlogloss = as.numeric(bst.cv$test.mlogloss.mean) + as.numeric(bst.cv$test.mlogloss.std) ;
      cv.eval = data.frame(iround = c(1:cv.nround), mlogloss = cv.mlogloss)
      cv.bestround = order(cv.eval$mlogloss) ;
      cv.bestround = cv.bestround[1] ;
      cv.bestmlogloss = cv.eval$mlogloss[cv.bestround] ;
      
      cv.record = rbind(cv.record, 
                        data.frame(eta = eta, max_depth = max_depth, min_child_weight = min_child_weight, 
                                   max_delta_step =max_delta_step, subsample = subsample, 
                                   colsample_bytree = colsample_bytree, 
                                   bestround = cv.bestround, bestloss = cv.bestmlogloss)) ;
      cat("running time is ",proc.time() - ptm)
          }
        }
      }
    }
  }
}

write.csv(cv.record, file = "boost_cv_8.csv", row.names = FALSE)
write.csv(train, file = "train_freq.csv", row.names = FALSE)
write.csv(test, file = "test_freq.csv", row.names = FALSE)


cv.record = read.csv("boost_cv_5.csv", header = TRUE,stringsAsFactors = F)

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "eta" = 0.1,
              "max_depth" = 9 ,
              "min_child_weight" = 1.3 ,
              "max_delta_step" = 0,
              "subsample" = 0.8 ,
              "colsample_bytree" = 0.6,
              "num_class" = 9,
              "nthread" = 8)
nround = 309
bst = xgboost(param=param, data = x.train, label = y.train, nrounds=nround)

pred = predict(bst,x.test)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
# pred.result <- data.frame(id = mytest$id, pred)
pred.result <- data.frame(id = test$id, pred)
names(pred.result)[2:10] = unique(train$target)

write.csv(pred.result, file = "prediction_v13.csv", row.names = FALSE)





