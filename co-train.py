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


class data_20news(object):
    def __init__(self):
        self.train_labeled = []
        self.train_labeled_target = []
        self.train_unlabeled = []
        self.train_unlabeled_target = []
        self.train = []
        self.test = []
        self.test_target = []
        self.target_names = []


def get_train_data(size):
    twenty_train = fetch_20newsgroups(subset='train', categories=None)
    twenty_test = fetch_20newsgroups(subset='test', categories=None)
    balance = [size for i in range(0, 20)]
    i = 0
    data = data_20news()
    twenty_labeled_data = []
    twenty_unlabeled_data = []
    twenty_train_labeled = []
    twenty_train_labeled_un = []
    for d in twenty_train.data:
        if balance[twenty_train.target[i]] != 0:
            twenty_labeled_data.append(d)
            twenty_train_labeled.append(twenty_train.target[i])
            balance[twenty_train.target[i]] -= 1
        else:
            twenty_unlabeled_data.append(d)
            twenty_train_labeled_un.append(twenty_train.target[i])
        i += 1
    data.train_labeled = twenty_labeled_data
    data.train_labeled_target = twenty_train_labeled
    data.train_unlabeled = twenty_unlabeled_data
    data.train_unlabeled_target = twenty_train_labeled_un
    data.train = twenty_train.data
    data.test = twenty_test.data
    data.test_target = twenty_test.target
    data.target_names = twenty_test.target_names
    count = [0 for i in range(0, 20)]
    for i in range(0, len(data.train_labeled)):
        count[data.train_labeled_target[i]] += 1
    print(count)
    return data


def co_training(data, n_iter, n_move, threshold):
    count_vect = CountVectorizer(stop_words='english')
    # count_vect.fit(data2.train)
    count_vect.fit(data.train)
    count_transformer = count_vect.transform(data.train)
    tfidf_transformer = TfidfTransformer().fit(count_transformer)
    lda_model = joblib.load('lda100-1000.pkl')
    X_train_labeled_counts = count_vect.transform(data.train_labeled)
    X_train_tfidf = tfidf_transformer.transform(X_train_labeled_counts)
    X_train_lda = lda_model.transform(X_train_labeled_counts)
    for i in range(0, n_iter):
        ##lda-part
        clf = svm.SVC(kernel='linear', probability=True).fit(X_train_lda, data.train_labeled_target)
        unlabeled_data_lda = lda_model.transform(count_vect.transform(data.train_unlabeled))
        predicted = clf.predict(unlabeled_data_lda)
        predicted_proba = clf.predict_proba(unlabeled_data_lda)

        score_dic = {}
        for j in range(0, len(predicted_proba)):
            sorted_proba = sorted(predicted_proba[j], reverse=True)
            score_dic[j] = sorted_proba[0] - sorted_proba[1]

        score_dic = sorted(score_dic.items(), key=lambda d: d[1], reverse=True)
        j = 0
        balance = [n_move for k in range(0, 20)]
        # keys = []
        # keys_predicted = []
        keys_dic = {}
        for key, score in score_dic:
            # print(predicted[key], score)
            if j == n_move*20:
                break
            if balance[predicted[key]] == 0:
                continue
            balance[predicted[key]] -= 1
            if score < threshold:
                print("lda no")
                j += 1
                continue

            # print(predicted[key], data.train_unlabeled_target[key])

            # data.train_labeled.append(data.train_unlabeled[key])
            # twenty_train_labeled = np.append(twenty_train_labeled, predicted[key])
            # data.train_labeled_target.append(predicted[key])
            # keys.append(key)
            # keys_predicted.append(predicted[key])
            keys_dic[key] = predicted[key]
            j += 1
        # print(keys_dic)
        # print(len(keys_dic))

        # keys = sorted(keys, reverse=True)
        # for key in keys:
        #     # print(key, len(twenty_unlabeled_data))
        #     del data.train_unlabeled[key]
        #     del data.train_unlabeled_target[key]

        ##tfidf-part
        clf = svm.SVC(kernel='linear', probability=True).fit(X_train_tfidf, data.train_labeled_target)
        unlabeled_data_tfidf = tfidf_transformer.transform(count_vect.transform(data.train_unlabeled))
        predicted = clf.predict(unlabeled_data_tfidf)
        predicted_proba = clf.predict_proba(unlabeled_data_tfidf)

        score_dic = {}
        for j in range(0, len(predicted_proba)):
            sorted_proba = sorted(predicted_proba[j], reverse=True)
            score_dic[j] = sorted_proba[0] - sorted_proba[1]

        score_dic = sorted(score_dic.items(), key=lambda d: d[1], reverse=True)
        j = 0
        balance = [n_move for k in range(0, 20)]
        for key, score in score_dic:
            # print(predicted[key], score)
            if key in keys_dic.keys():
                continue
            if j == n_move * 20:
                break
            if balance[predicted[key]] == 0:
                continue
            balance[predicted[key]] -= 1
            if score < threshold:
                print("tfidf no")
                j += 1
                continue
            # keys.append(key)
            # keys_predicted.append(predicted[key])
            keys_dic[key] = predicted[key]
            j += 1

        # keys = sorted(keys, reverse=True)
        keys_dic = sorted(keys_dic.items(), key=lambda d: d[0], reverse=True)
        # print(len(keys_dic))
        for key, predict in keys_dic:
            # print(key)
            # print(key, len(twenty_unlabeled_data))
            # print(data.train_unlabeled_target[key])
            data.train_labeled.append(data.train_unlabeled[key])
            data.train_labeled_target.append(predict)
            del data.train_unlabeled[key]
            del data.train_unlabeled_target[key]

        ##test-part
        X_train_labeled_counts = count_vect.transform(data.train_labeled)
        X_train_lda = lda_model.transform(X_train_labeled_counts)
        X_train_tfidf = tfidf_transformer.transform(X_train_labeled_counts)

        if i % 5 == 0:
            clf1 = svm.SVC(kernel='linear', probability=True).fit(X_train_lda, data.train_labeled_target)
            predicted1 = clf1.predict(lda_model.transform(count_vect.transform(data.test)))
            predicted_proba1 = clf1.predict_proba(lda_model.transform(count_vect.transform(data.test)))
            print("lda:")
            print(metrics.classification_report(data.test_target, predicted1, target_names=data.target_names))

            clf2 = svm.SVC(kernel='linear', probability=True).fit(X_train_tfidf, data.train_labeled_target)
            predicted2 = clf2.predict(tfidf_transformer.transform(count_vect.transform(data.test)))
            predicted_proba2 = clf2.predict_proba(tfidf_transformer.transform(count_vect.transform(data.test)))
            print("tfidf:")
            print(metrics.classification_report(data.test_target, predicted2, target_names=data.target_names))

            # print(predicted_proba1[5])
            # print(predicted_proba2[5])

            predicted = []
            for j in range(0, len(predicted_proba1)):
                predicted_proba = []
                for p in range(0, len(predicted_proba1[j])):
                    predicted_proba.append(predicted_proba1[j][p] + predicted_proba2[j][p])
                if j == 5:
                    print(predicted_proba)
                predicted.append(predicted_proba.index(max(predicted_proba)))
            # print(predicted)
            print("co-train:")
            print(metrics.classification_report(data.test_target, predicted, target_names=data.target_names))


if __name__ == '__main__':
    print(15, 201, 1, 0.05)
    data = get_train_data(15)
    # print(data.test_target[5])
    co_training(data, 201, 1, 0.05)