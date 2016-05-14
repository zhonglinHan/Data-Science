import numpy as np

def run_skfCV(X,y, clf, k):
    from sklearn.cross_validation import StratifiedKFold
    import numpy as np
    skf = StratifiedKFold(y, n_folds=k)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict_proba(X_test)[:,1]
    return y_pred