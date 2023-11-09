from sklearn.feature_extraction.text import CountVectorizer

cat_in_the_hat_docs=[
       "One Cent, Two Cents, Old Cent, New Cent: All About Money (Cat in the Hat's Learning Library",
       "Inside Your Outside: All About the Human Body (Cat in the Hat's Learning Library)",
       "Oh, The Things You Can Do That Are Good for You: All About Staying Healthy (Cat in the Hat's Learning Library)",
       "On Beyond Bugs: All About Insects (Cat in the Hat's Learning Library)",
       "There's No Place Like Space: All About Our Solar System (Cat in the Hat's Learning Library)" 
      ]

cv = CountVectorizer(cat_in_the_hat_docs)

count_vector = cv.fit_transform(cat_in_the_hat_docs)

print(count_vector[0]) 
print(cv.stop_words)

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np
corpus = ['this is the first document', 
    'this document is the second document',
    'and this is the third one',
    'is this the first document']
pipe = Pipeline([('count', CountVectorizer()),
                 ('tfid', TfidfTransformer())]).fit(corpus)    

print(pipe['count'].transform(corpus).toarray())