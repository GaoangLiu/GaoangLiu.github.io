from sklearn.metrics import jaccard_score
s1 = "我爱北京天安门"
s2 = "我爱北京天安门"
s3 = "我爱北京天安门，天安门上太阳升"
jaccard_score(list(s1), list(s2)) # 1.0
# jaccard_score(s1, s3) # 0.8