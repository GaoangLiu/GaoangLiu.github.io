import numpy as np
import math 

class NaiveBayes():
    def __init__(self):
        self.partial_vector_0 = []
        self.partial_vector_1 = []
        self.prob_class_1 = 0

    def train_naive_bayes(self, matrix, categories):
        '''Assume there are only two class.
        @param: categories is a list of 1's and 0's
        '''
        self.prob_class_1 = sum(categories) / len(categories)
        norm_features_0 = np.ones(len(matrix[0]))
        norm_features_1 = np.ones(len(matrix[0]))
        cnt0 = cnt1 = len(matrix[0])
        for i in range(len(categories)):
            if categories[i] == 1:
                norm_features_1 += matrix[i]
                cnt1 += sum(matrix[i])
            else:
                norm_features_0 += matrix[i]
                cnt0 += sum(matrix[i])
        self.partial_vector_0 = np.log(norm_features_0 / cnt0)
        self.partial_vector_1 = np.log(norm_features_1 / cnt1)
        

    def predict(self, test_matrix):
        '''@param: test_matrix: vector of vectors 
        '''
        return [1 if sum(vec * self.partial_vector_1) + math.log(self.prob_class_1) > \
            sum(vec * self.partial_vector_0) + math.log(1 - self.prob_class_1) else 0 \
                for vec in test_matrix]


if __name__ == "__main__":
    matrix = [[0, 1, 1, 0], [1, 0, 0, 0], [1, 1, 1, 0]]
    print(NaiveBayes().train_naive_bayes(matrix, [0, 0, 1]))
    # x = np.asarray([1,2,3,4])
    # print(np.log(x))
    # print(math.log(2))

    # y = np.asarray([1,2,3,4])
    # z = x + y
    # print(z)
    # print(np.asarray(x) / 3)
