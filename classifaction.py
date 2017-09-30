from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import sklearn.svm as svm
from sklearn.svm import SVC
from sklearn.externals import joblib
import lda
import numpy as np

categories = [
    'alt.atheism', 'soc.religion.christian',
    'comp.graphics', 'sci.med'
]


twenty_train = fetch_20newsgroups(subset='train', categories=None)
twenty_test = fetch_20newsgroups(subset='test', categories=None)

# print(len(twent_train.data))
# print(len(twenty_test.data))
# labeled_length = int(len(twenty_train.data)/5)
# print(labeled_length)

count_vect = CountVectorizer(stop_words='english')
count_vect.fit(twenty_train.data)
X_train_counts = count_vect.transform(twenty_train.data)
# X_train_labeled_counts = count_vect.transform(twenty_train.data[:labeled_length])
# word = count_vect.get_feature_names()
# print(X_train_counts.shape)

tf_transformer = TfidfTransformer().fit(X_train_counts)
# X_train_tf = tf_transformer.transform(X_train_labeled_counts)
# print(X_train_tf.shape)

# twenty_labeled_data = twenty_train.data[:labeled_length]
# twenty_unlabeled_data = twenty_train.data[labeled_length:]
# twenty_train_labeled = twenty_train.target[:labeled_length]

balance = [100 for i in range(0, 100)]
i = 0
twenty_labeled_data = []
twenty_unlabeled_data = []
twenty_train_labeled = []
for data in twenty_train.data:
    if balance[twenty_train.target[i]] != 0:
        twenty_labeled_data.append(data)
        twenty_train_labeled.append(twenty_train.target[i])
        balance[twenty_train.target[i]] -= 1
    else:
        twenty_unlabeled_data.append(data)
    i += 1

count = [0 for i in range(0, 20)]
for i in range(0, len(twenty_train_labeled)):
    count[twenty_train_labeled[i]] += 1
print(count)
print(len(twenty_unlabeled_data))

#self-training
X_train_labeled_counts = count_vect.transform(twenty_labeled_data)
X_train_tf = tf_transformer.transform(X_train_labeled_counts)
for i in range(0, 200):
    clf = svm.SVC(decision_function_shape='ovo', kernel='linear', probability=True).fit(X_train_tf, twenty_train_labeled)
    predicted = clf.predict(tf_transformer.transform(count_vect.transform(twenty_unlabeled_data)))
    predicted_proba = clf.predict_proba(tf_transformer.transform(count_vect.transform(twenty_unlabeled_data)))

    score_dic = {}
    for j in range(0, len(predicted_proba)):
        sorted_proba = sorted(predicted_proba[j], reverse=True)
        score_dic[j] = sorted_proba[0] - sorted_proba[1]

    sorted(score_dic.items(), key=lambda d: d[1], reverse=True)
    j = 0
    balance = [1 for k in range(0, 20)]
    for key in score_dic.keys():
        if j == 20:
            break
        if balance[predicted[key]] == 0:
            continue
        balance[predicted[key]] -= 1
        # print(twenty_unlabeled_data[key])
        # print(score_dic[key])
        # print(predicted_proba[key])
        # print(predicted[key])
        twenty_labeled_data.append(twenty_unlabeled_data[key])
        twenty_train_labeled = np.append(twenty_train_labeled, predicted[key])
        # twenty_train_labeled.append(predicted[key])
        del twenty_unlabeled_data[key]
        j += 1

    X_train_labeled_counts = count_vect.transform(twenty_labeled_data)
    X_train_tf = tf_transformer.transform(X_train_labeled_counts)
    print(X_train_tf.shape)
    print(len(twenty_train_labeled))

    clf = svm.SVC(decision_function_shape='ovo', kernel='linear', probability=True).fit(X_train_tf, twenty_train_labeled)
    predicted = clf.predict(tf_transformer.transform(count_vect.transform(twenty_test.data)))
    predicted_proba = clf.predict_proba(tf_transformer.transform(count_vect.transform(twenty_test.data)))
    print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))



# clf = MultinomialNB().fit(X_train_tf, twenty_train.target[:labeled_length])
# clf = GradientBoostingClassifier().fit(X_train_tf, twenty_train.target)
# clf = svm.SVC(decision_function_shape='ovo', kernel='linear', probability=True).fit(X_train_tf, twenty_train.target[:labeled_length])
# predicted = clf.predict(tf_transformer.transform(count_vect.transform(twenty_test.data)))
# predicted_proba = clf.predict_proba(tf_transformer.transform(count_vect.transform(twenty_test.data)))
# print(predicted)
# print(predicted_proba)
# print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

# for i in range(50, 301, 50):
#     for j in range(500, 2001, 500):
#         model = lda.LDA(n_topics=i, n_iter=j)
#         model = model.fit(X_train_counts)
#         joblib.dump(model, 'lda'+str(i)+'-'+str(j)+'.pkl')

# model = lda.LDA(n_topics=200, n_iter=500)
# model = model.fit(X_train_counts)
# joblib.dump(model, 'lda200-500.pkl')

# model = joblib.load('lda50-500.pkl')

# topic_word = model.topic_word_
# n = 20
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(word)[np.argsort(topic_dist)][:-(n+1):-1]
#     print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))
# print(model.transform(count_vect.transform(twenty_test.data)))

# clf = GradientBoostingClassifier().fit(model.doc_topic_, twenty_train.target)
# predicted = clf.predict(model.transform(count_vect.transform(twenty_test.data)))
# print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
#
# clf = MultinomialNB().fit(model.doc_topic_, twenty_train.target)
# predicted = clf.predict(model.transform(count_vect.transform(twenty_test.data)))
# print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))
#
# clf = svm.SVC(decision_function_shape='ovo', kernel='linear').fit(model.doc_topic_, twenty_train.target)
# predicted = clf.predict(model.transform(count_vect.transform(twenty_test.data)))
# print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))