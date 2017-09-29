from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


corpus = [
        "我 来到 北京 清华大学",#第一类文本切词后的结果，词之间以空格隔开
        "他 来到 了 网易 杭研 大厦",#第二类文本的切词结果
        "小明 硕士 毕业 与 中国 科学院",#第三类文本的切词结果
        "我 爱 北京 天安门"
    ]
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
word = vectorizer.get_feature_names()
print(word)
weight = tfidf.toarray()
print(len(weight))
for i in range(len(weight)):
    print("-------这里输出第", i, "类文本的词语tf-idf权重------" )
    for j in range(len(word)):
        print (word[j], weight[i][j])


