from sklearn import preprocessing
import numpy as np 

le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 9])
print(le.classes_)

print(le.transform([1, 2, 2, 2, 9]))

print(le.fit_transform(['apple', 'pen', 'apple', 'applepen']))
print(le.classes_)


lb = preprocessing.LabelBinarizer()
print(lb.fit_transform([1, 3, 2]))
print(lb.fit_transform(['female', 'male', 'others', 'female']))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = np.array(['female', 'male', 'others']).reshape(-1, 1)
print(X)
print(enc.fit_transform(X).toarray())

