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


def co_training(data, data2, n_iter, n_move, threshold):
    count_vect = CountVectorizer(stop_words='english')
    # count_vect.fit(data2.train)
    count_vect.fit(data.train)
    count_transformer = count_vect.transform(data.train)
    tfidf_transformer = TfidfTransformer().fit(count_transformer)
    lda_model = joblib.load('lda100-1000.pkl')
    X_train_labeled_counts_lda = count_vect.transform(data.train_labeled)
    X_train_labeled_counts_tfidf = count_vect.transform(data2.train_labeled)
    X_train_lda = lda_model.transform(X_train_labeled_counts_lda)
    X_train_tfidf = tfidf_transformer.transform(X_train_labeled_counts_tfidf)

    # co-train prat
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
        # print(score_dic)
        j = 0
        balance = [n_move for k in range(0, 20)]
        # keys = []
        # keys_predicted = []
        content_dic_data2 = {}
        for key, score in score_dic:
            # print(predicted[key], score)
            if j == n_move*20:
                break
            if data.train_unlabeled[key] not in data2.train_unlabeled:
                continue
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
            content_dic_data2[data.train_unlabeled[key]] = predicted[key]
            j += 1
        # print(content_dic_data2)
        # print(len(keys_dic))

        # keys = sorted(keys, reverse=True)
        # for key in keys:
        #     # print(key, len(twenty_unlabeled_data))
        #     del data.train_unlabeled[key]
        #     del data.train_unlabeled_target[key]

        ##tfidf-part
        clf = svm.SVC(kernel='linear', probability=True).fit(X_train_tfidf, data2.train_labeled_target)
        unlabeled_data_tfidf = tfidf_transformer.transform(count_vect.transform(data2.train_unlabeled))
        predicted = clf.predict(unlabeled_data_tfidf)
        predicted_proba = clf.predict_proba(unlabeled_data_tfidf)

        score_dic = {}
        for j in range(0, len(predicted_proba)):
            sorted_proba = sorted(predicted_proba[j], reverse=True)
            score_dic[j] = sorted_proba[0] - sorted_proba[1]

        score_dic = sorted(score_dic.items(), key=lambda d: d[1], reverse=True)
        content_dic_data = {}
        j = 0
        balance = [n_move for k in range(0, 20)]
        for key, score in score_dic:
            # print(predicted[key], score)
            # if key in keys_dic.keys():
            #     continue
            if j == n_move * 20:
                break
            if data2.train_unlabeled[key] not in data.train_unlabeled:
                continue
            if balance[predicted[key]] == 0:
                continue
            balance[predicted[key]] -= 1
            if score < threshold:
                print("tfidf no")
                j += 1
                continue
            # keys.append(key)
            # keys_predicted.append(predicted[key])
            # keys_dic_data[key] = predicted[key]
            content_dic_data[data2.train_unlabeled[key]] = predicted[key]
            j += 1

        # change data part
        # keys = sorted(keys, reverse=True)
        # keys_dic_data = sorted(keys_dic_data.items(), key=lambda d: d[0], reverse=True)
        # keys_dic_data2 = sorted(keys_dic_data2.items(), key=lambda d: d[0], reverse=True)
        # print(len(keys_dic))
        for (content, predict) in content_dic_data2.items():
            # print(key)
            # print(key, len(twenty_unlabeled_data))
            # print(data.train_unlabeled_target[key])
            key = data2.train_unlabeled.index(content)
            data2.train_labeled.append(data2.train_unlabeled[key])
            data2.train_labeled_target.append(predict)
            del data2.train_unlabeled[key]
            del data2.train_unlabeled_target[key]

        for (content, predict) in content_dic_data.items():
            # print(key)
            # print(key, len(twenty_unlabeled_data))
            # print(data.train_unlabeled_target[key])
            key = data.train_unlabeled.index(content)
            data.train_labeled.append(data.train_unlabeled[key])
            data.train_labeled_target.append(predict)
            del data.train_unlabeled[key]
            del data.train_unlabeled_target[key]


        ##test-part
        X_train_labeled_counts_lda = count_vect.transform(data.train_labeled)
        X_train_labeled_counts_tfidf = count_vect.transform(data2.train_labeled)
        X_train_lda = lda_model.transform(X_train_labeled_counts_lda)
        X_train_tfidf = tfidf_transformer.transform(X_train_labeled_counts_tfidf)

        if i % 5 == 0:
            clf1 = svm.SVC(kernel='linear', probability=True).fit(X_train_lda, data.train_labeled_target)
            predicted1 = clf1.predict(lda_model.transform(count_vect.transform(data.test)))
            predicted_proba1 = clf1.predict_proba(lda_model.transform(count_vect.transform(data.test)))
            print("lda:")
            print(metrics.classification_report(data.test_target, predicted1, target_names=data.target_names))

            clf2 = svm.SVC(kernel='linear', probability=True).fit(X_train_tfidf, data2.train_labeled_target)
            predicted2 = clf2.predict(tfidf_transformer.transform(count_vect.transform(data2.test)))
            predicted_proba2 = clf2.predict_proba(tfidf_transformer.transform(count_vect.transform(data2.test)))
            print("tfidf:")
            print(metrics.classification_report(data2.test_target, predicted2, target_names=data2.target_names))

            # print(predicted_proba1[5])
            # print(predicted_proba2[5])

            predicted = []
            for j in range(0, len(predicted_proba1)):
                predicted_proba = []
                for p in range(0, len(predicted_proba1[j])):
                    predicted_proba.append(predicted_proba1[j][p] + predicted_proba2[j][p])
                # if j == 5:
                #     print(predicted_proba)
                predicted.append(predicted_proba.index(max(predicted_proba)))
            # print(predicted)
            print("co-train:")
            print(metrics.classification_report(data.test_target, predicted, target_names=data.target_names))

    clf1 = svm.SVC(kernel='linear', probability=True).fit(X_train_lda, data.train_labeled_target)
    predicted1 = clf1.predict(lda_model.transform(count_vect.transform(data.test)))
    predicted_proba1 = clf1.predict_proba(lda_model.transform(count_vect.transform(data.test)))
    print("lda:")
    print(metrics.classification_report(data.test_target, predicted1, target_names=data.target_names))

    clf2 = svm.SVC(kernel='linear', probability=True).fit(X_train_tfidf, data2.train_labeled_target)
    predicted2 = clf2.predict(tfidf_transformer.transform(count_vect.transform(data2.test)))
    predicted_proba2 = clf2.predict_proba(tfidf_transformer.transform(count_vect.transform(data2.test)))
    print("tfidf:")
    print(metrics.classification_report(data2.test_target, predicted2, target_names=data2.target_names))

    # print(predicted_proba1[5])
    # print(predicted_proba2[5])

    predicted = []
    for j in range(0, len(predicted_proba1)):
        predicted_proba = []
        for p in range(0, len(predicted_proba1[j])):
            predicted_proba.append(predicted_proba1[j][p] + predicted_proba2[j][p])
        # if j == 5:
        #     print(predicted_proba)
        predicted.append(predicted_proba.index(max(predicted_proba)))
    # print(predicted)
    print("co-train:")
    print(metrics.classification_report(data.test_target, predicted, target_names=data.target_names))


if __name__ == '__main__':
    print(50, 201, 1, 0.01)
    data = get_train_data(50)
    data2 = get_train_data(50)
    co_training(data, data2, 201, 1, 0.01)