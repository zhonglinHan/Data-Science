from nolearn.dbn import DBN
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from sklearn.metrics import zero_one_loss, classification_report, accuracy_score, log_loss

iris = load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
X_train_in, X_train_test, y_train_in, y_train_test = cross_validation.train_test_split(X_train, y_train, test_size=0.4, random_state=0)
clf = DBN([X_train.shape[1], 4, 3],learn_rates=0.05,epochs=200)

print 'Cross Validation'
clf.fit(X_train_in, y_train_in)
y_pred = clf.predict(X_train_test)
y_predprob = clf.predict_proba(X_train_test)
print classification_report(y_train_test,y_pred)
print 'The accuracy is: ', accuracy_score(y_train_test,y_pred)
print 'The log loss is:', log_loss(y_train_test, y_predprob)

print 'Train VS Test'
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_predprob = clf.predict_proba(X_test)
print classification_report(y_test,y_pred)
print 'The accuracy is: ', accuracy_score(y_test,y_pred)
print 'The log loss is:', log_loss(y_test, y_predprob)

