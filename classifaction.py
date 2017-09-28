from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import lda
import numpy as np

categories = [
    'alt.atheism', 'soc.religion.christian',
    'comp.graphics', 'sci.med'
]

twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='classifaction', categories=categories, shuffle=True, random_state=42)
count_vect = CountVectorizer(stop_words='english')
X_train_counts = count_vect.fit_transform(twenty_train.data)
word = count_vect.get_feature_names()
print(X_train_counts.shape)

# tf_transformer = TfidfTransformer().fit(X_train_counts)
# X_train_tf = tf_transformer.fit_transform(X_train_counts)
# print(X_train_tf.shape)
#
# clf = MultinomialNB().fit(X_train_tf, twenty_train.target)
# # clf = GradientBoostingClassifier().fit(X_train_tf, twenty_train.target)
# # clf = SVC().fit(X_train_tf, twenty_train.target)
#
# predicted = clf.predict(tf_transformer.transform(count_vect.transform(twenty_test.data)))
#
# print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

model = lda.LDA(n_topics=150, n_iter=500, random_state=42)
model.fit(X_train_counts)
topic_word = model.topic_word_

# n = 20
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]
#     print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
# print(model.transform(count_vect.transform(twenty_test.data)))

clf = GradientBoostingClassifier().fit(model.doc_topic_, twenty_train.target)
predicted = clf.predict(model.transform(count_vect.transform(twenty_test.data)))

print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))