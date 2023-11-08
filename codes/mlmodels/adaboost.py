class AdaBoost():
    def weak_learner(self, train, col_idx, thresh_val, inequal):
        '''Use weak classifier to make predictions 
        '''
        preds = np.ones((train.shape[0], 1))
        if inequal == 'lt':
            preds[train[:, col_idx] <= thresh_val] = -1
        else:
            preds[train[:, col_idx] > thresh_val] = -1
        return preds
            
    def build_stump(self, train, labels, weights):
        m, n = train.shape
        D = weights
        iter_count = 10
        best_stump = {}
        min_error = float('inf')
        best_preds = None
        
        for i in range(n):
            minv, maxv = train[:, i].min(), train[:, i].max()
            step = (maxv - minv) / iter_count 
            for thresh_val in np.arange(minv - step, maxv + step, step):
                for inequal in ['lt', 'gt']:
                    preds = self.weak_learner(train, i, thresh_val, inequal)
                    err = np.matrix(np.ones((m, 1)))
                    err[preds == labels] = 0
                    weighted_err = D.T * err
                    print(f"split dim {i}, thresh {thresh_val}, inequal {inequal}, weighted error {weighted_err}")
                    
                    if weighted_err < min_error:
                        min_error = weighted_err
                        best_preds = preds
                        best_stump = {'dim': i, 'thresh_val': thresh_val, 'inequal': inequal}

        return best_stump, min_error, best_preds
    
    def ada_boost_train(self, train, labels, steps=10):
        weak_learner = []
        m, n = train.shape()
#         weights = np.matrix(np.ones((m, 1))/m)
#         print(steps)
#         for i in range(steps):
#             continue
#             best_stump, error, preds = self.build_stump(train, labels, weights)
#             print("Sample weights ", weights)
            
#             alpha = log((1-error) / max(error, 1e-16)) / 2
#             expon = np.multiply(-1*alpha*labels.T, preds)
    

ab = AdaBoost()
ab.ada_boost_train(train, labels, 10)
# ab.weak_learner(train, 0, 1.5, 'lt')
