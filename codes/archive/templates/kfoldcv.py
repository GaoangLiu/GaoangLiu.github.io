from sklearn.model_selection import StratifiedKFold

def five_fold_cv(model, X, y, verbose=True):
    skf = StratifiedKFold(n_splits = 5)
    fold = 1
    scores = []
    print(X.shape, y.shape)
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)

        y_preds = model.predict(X_val)
#         y_preds = [x[1] for x in y_preds]

        score = roc_auc_score(y_val, y_preds)
        scores.append(score)
        if verbose:
            print('Fold', fold, '     ', score)
        fold += 1
    
    avg = np.mean(scores)
    if verbose:
        print('\nAverage:', avg)
    return avg
