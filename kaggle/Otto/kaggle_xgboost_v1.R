#require(xgboost)
#require(methods)
rm(list = ls())
library(xgboost)
library(methods)
# library(devtools)
# install_url("https://github.com/dmlc/xgboost.git")
# devtools::install_github('dmlc/xgboost',subdir='R-package')
train = read.csv("train_v1.csv", header = TRUE,stringsAsFactors = F)
mytrain_balance = read.csv("mytrain_balance_v1.csv", header = TRUE,stringsAsFactors = F)
mytest = read.csv("mytest_v1.csv", header = TRUE,stringsAsFactors = F)
mytrain = train[train$setid == 1 | train$setid == 2, ]
test <- read.csv("test.csv", header = TRUE,stringsAsFactors = F)# the official test set 
train_balance = read.csv("train_balance_v1.csv", header = TRUE, stringsAsFactors = F)

# train = read.csv('train.csv',header=TRUE,stringsAsFactors = F)
# test = read.csv('test.csv',header=TRUE,stringsAsFactors = F)
#train = mytrain_balance ;
#test = mytest ; 
train = train_balance ;
test = test ;
train = subset(train, select = -c(id, setid))
# test = subset(test, select = -c(id, setid))

#train = train[,-1]
#test = test[,-1]

y.train = train[,ncol(train)]
y.train = gsub('Class_','',y.train)
y.train = as.integer(y.train)-1 #xgboost take features in [0,numOfClass-1]

y.test = test[,ncol(test)]
y.test = gsub('Class_','',y.test)
y.test = as.integer(y.test)-1 #xgboost take features in [0,numOfClass-1]

x.train = subset(train, select = -c(target))
x.train = as.matrix(x.train) 
x.train = matrix(as.numeric(x.train),nrow(x.train),ncol(x.train))

x.test = subset(test, select = -c(id))
#x.test = subset(test, select = -c(target))
x.test = as.matrix(x.test) 
x.test = matrix(as.numeric(x.test),nrow(x.test),ncol(x.test))


#x = rbind(train[,-ncol(train)],test)
#x = as.matrix(x)
#x = matrix(as.numeric(x),nrow(x),ncol(x))
#trind = 1:length(y)
#teind = (nrow(train)+1):nrow(x)

# Set necessary parameter
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "eta" = 0.2,
              "max_depth" = 7 ,
              "min_child_weight" = 1.2 ,
              "num_class" = 9,
              "nthread" = 8)

# Run Cross Valication
cv.nround = 196
bst.cv = xgb.cv(param=param, data = x.train, label = y.train, 
                                nfold = 10, nrounds=cv.nround)
#bst.cv = xgb.cv(param=param, data = x[trind,], label = y, 
#                nfold = 3, nrounds=cv.nround)

# Train the model
nround = 196
bst = xgboost(param=param, data = x.train, label = y.train, nrounds=nround)
# bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
# save(bst, "bst.RData")

# Make prediction
pred = predict(bst,x.test)
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
# pred.result <- data.frame(id = mytest$id, pred)
pred.result <- data.frame(id = test$id, pred)
names(pred.result)[2:10] = unique(train$target)
prob = 0 ;
for(i in 1 : length(mytest$id)){
  icol = which(names(pred.result) == mytest$target[mytest$id == 
                                                       pred.result$id[i]]) ;
  p = pred.result[i, icol] ;
  p = max(min(p,1-10^{-15}),10^{-15}) ;
  prob = prob + log(p) ;             
}
testscore = -1/length(mytest$id) * prob ;


# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
write.csv(pred,file='prediction_v6.csv', quote=FALSE,row.names=FALSE)
write.csv(pred.result, file = "prediction_v8.csv", row.names = FALSE)
