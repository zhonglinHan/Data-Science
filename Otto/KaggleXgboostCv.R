rm(list = ls())
# install packages
install.packages("method")
install.packages("mlbench")
install.packages("devtools")
library(devtools)
devtools::install_github('dmlc/xgboost',subdir='R-package')

library(xgboost)
library(methods)
library(mlbench)

train.trans = read.csv("train_freq_v2.csv", header = TRUE,stringsAsFactors = F)
test.trans = read.csv("test_freq_v2.csv", header = TRUE,stringsAsFactors = F)

### process data ###
train = train.trans ;
train = subset(train, select = -c(id, setid))

y.train = train[,ncol(train)]
#y.train = gsub('Class_','',y.train)
y.train = as.integer(y.train)-1 #xgboost take features in [0,numOfClass-1]

x.train = subset(train, select = -c(target))
x.train = as.matrix(x.train) 
x.train = matrix(as.numeric(x.train),nrow(x.train),ncol(x.train))

test = test.trans 
x.test = subset(test, select = -c(id))
#x.test = subset(test, select = -c(target))
x.test = as.matrix(x.test) 
x.test = matrix(as.numeric(x.test),nrow(x.test),ncol(x.test))


cv.etas = c(0.01) ;
cv.max_depths = c(5,7,9) ;
cv.min_child_weights = c(1.3) ;
cv.max_delta_steps = c(0) ;
cv.subsamples = c(0.8) ;
cv.colsample_bytrees = c(0.6) ;


cv.record = NULL ;
cv.nround = 10000 ;
cv.nfold = 5;

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

write.csv(cv.record, file = "boost_cv_12.csv", row.names = FALSE)
