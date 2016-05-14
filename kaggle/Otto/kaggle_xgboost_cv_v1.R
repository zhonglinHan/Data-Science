
rm(list = ls())
library(xgboost)
library(methods)

train = read.csv("train_v1.csv", header = TRUE,stringsAsFactors = F)

train = train ;
train = subset(train, select = -c(id, setid))

y.train = train[,ncol(train)]
y.train = gsub('Class_','',y.train)
y.train = as.integer(y.train)-1 #xgboost take features in [0,numOfClass-1]

x.train = subset(train, select = -c(target))
x.train = as.matrix(x.train) 
x.train = matrix(as.numeric(x.train),nrow(x.train),ncol(x.train))

# Set necessary parameter
#cv.etas = c(0.2,0.3,0.4,0.5) ;
cv.etas = c(0.2) ;
cv.max_depths = c(7) ;
cv.min_child_weights = c(1.2) ;
cv.record = NULL ;
cv.nround = 200 ;

for(eta in cv.etas){
  for(max_depth in cv.max_depths){
    for(min_child_weight in cv.min_child_weights){
      cat("eta == ", eta, " max_depth == ", max_depth, " min_cw == ", min_child_weight) ;
      cat("\n")
      ptm <- proc.time()
      param <- list("objective" = "multi:softprob",
                    "eval_metric" = "mlogloss",
                    "eta" = eta,
                    "max_depth" = max_depth ,
                    "min_child_weight" = min_child_weight ,
                    "num_class" = 9,
                    "nthread" = 8)
      bst.cv = xgb.cv(param=param, data = x.train, label = y.train, 
                      nfold = 10, nrounds=cv.nround) ;
      cv.mlogloss = as.numeric(bst.cv$test.mlogloss.mean) + as.numeric(bst.cv$test.mlogloss.std) ;
      cv.eval = data.frame(iround = c(1:cv.nround), mlogloss = cv.mlogloss)
      cv.bestround = order(cv.eval$mlogloss) ;
      cv.bestround = cv.bestround[1] ;
      cv.bestmlogloss = cv.eval$mlogloss[cv.bestround] ;
      
      cv.record = rbind(cv.record, 
                        data.frame(eta = eta, max_depth = max_depth, min_child_weight = min_child_weight, 
                                              bestround = cv.bestround, bestloss = cv.bestmlogloss)) ;
      cat("running time is ",proc.time() - ptm)
    }
  }
}
write.csv(cv.record, file = "boost_cv_1.csv", row.names = FALSE)



# Run Cross Valication








