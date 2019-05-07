## -*- coding: utf-8 -*-
# Author: Barbara McGillivray
# Date: 18/10/2017
# Python version: 3

# Import libraries:
from textblob.classifiers import NaiveBayesClassifier
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
import random
import codecs
import gensim
import csv
import os
import time

# Default parameters:
#istest_default = "yes"
istest_default = "no"
#combine_labels_default = "no"
combine_labels_default = "no"
coding_default = 1
#coding_default = 2
#stopwords_default = "no"
stopwords_default = "no"
#uniform_prior_default = "no"
uniform_prior_default = "yes"
#stemming_default = "no"
stemming_default = "yes"
#skip_model1_default = "yes"
skip_model1_default = "yes"
#skip_model2_default = "yes"
skip_model2_default = "no"
skip_model3_default = "no"
#skip_model3_default = "yes"
skip_model4_default = "no"
#skip_model4_default = "yes"
#skip_model5_default = "yes"
skip_model5_default = "no"
user_input_default = "yes"

# Best parameters:
combine_labels_default = "yes"
coding_default = 1
stopwords_default = "no"
stemming_default = "yes"
istest_default = "no"
skip_model1_default = "yes"
skip_model2_default = "yes"
skip_model3_default = "no"
skip_model4_default = "yes"
skip_model5_default = "yes"

number_test = 10
encoding="utf-8"

# User parameters:

istest = input('Is this a test? Please reply yes or not. Leave empty for default (' + str(istest_default) + ").")
user_input = input("Do you want to select the parameters manually or do you want to loop over all options? Select yes "
                   "for the former. Leave empty for default (" + str(user_input_default) + ").")

if user_input == "":
    user_input = user_input_default
if user_input == "yes":
    combine_labels_values = [input(
    'Do you want to combine the category labels (1 with 2, 4 with 5)? Please leave empty for default (' + str(
        combine_labels_default) + ").")]
    coding_values = [input('Which coding approach will you use? Please leave empty for default (' + str(coding_default) + ").")]
    stopwords_values = [input('Do you want to exclude stop words? Please leave empty for default (' + str(stopwords_default) + ").")]
    uniform_prior_values = [input(
        'Do you want to use a uniform prior for Multinomial Naive Bayes? Please leave empty for default '
        '(' + str(uniform_prior_default) + ").")]
    stemming_values = [input('Do you want to stem the words? Please leave empty for default '
                            '(' + str(stemming_default) + ").")]
else:
    combine_labels_values = ["yes", "no"]
    coding_values = [1] # [1,2]
    stopwords_values = ["yes", "no"]
    uniform_prior_values = ["yes", "no"]
    stemming_values = ["yes", "no"]

skip_model1 = input(
    'Do you want to skip the first model? Please leave empty for default (' + str(skip_model1_default) + ").")
skip_model2 = input(
    'Do you want to skip the second model? Please leave empty for default (' + str(skip_model2_default) + ").")
skip_model3 = input(
    'Do you want to skip the third model? Please leave empty for default (' + str(skip_model3_default) + ").")
skip_model4 = input(
    'Do you want to skip the fourth model? Please leave empty for default (' + str(skip_model4_default) + ").")
skip_model5 = input(
    'Do you want to skip the fifth model? Please leave empty for default (' + str(skip_model5_default) + ").")

# Set parameters:

if istest == "":
    istest = istest_default

if combine_labels_values == [""]:
    combine_labels_values = [combine_labels_default]

if coding_values == [""]:
    coding_values = [coding_default]

if stopwords_values == [""]:
    stopwords_values = [stopwords_default]

if stemming_values == [""]:
    stemming_values = [stemming_default]

if uniform_prior_values == [""]:
    uniform_prior_values = [uniform_prior_default]

if not skip_model1:
    skip_model1 = skip_model1_default

if not skip_model2:
    skip_model2 = skip_model2_default

if not skip_model3:
    skip_model3 = skip_model3_default

if not skip_model4:
    skip_model4 = skip_model4_default

if not skip_model5:
    skip_model5 = skip_model5_default

# Directory and file names:
dir_out = "output"
dir_in = "input"
annotated_file_name = "das_full_annotation.csv"  # Annotated data
input_file_name = "das_full.csv"
output_summary_file_all = "overview_models_parameters.csv"

annotated_number = 380  # Number of annotated statements
length_train = int(annotated_number*0.8)
print("length_train:", str(length_train))
length_test = annotated_number - length_train
print("length_test:", str(length_test))

# random indices between ... and ... which will be used to generate the training set and the test set:
random_indices = []

random_indices = list(range(annotated_number+1))
random.shuffle(random_indices)

random_indices_train = random_indices[:length_train]
print(str(len(random_indices_train)), "random_indices_train")
random_indices_test = random_indices[length_train:length_train+length_test]
print(str(len(random_indices_test)), "random_indices_test")

# create output directory if it doesn't exist:
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Today's date and time:
now = time.strftime("%c")

# ----------------------------
# Stemming:
# ----------------------------

stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super().build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

# --------------------------
# Training and test sets:
# --------------------------

train = list()
test = list()
to_classify = list()
top_frequency_das = list() # list of top 250 most frequent DAS from the annotation

print("Reading input data...")
input_file = open(os.path.join(dir_in, input_file_name), 'r')

if istest == "yes":
    max_number = number_test
else:
    max_number = sum(1 for row in input_file)

input_file.close()

print("max_number:" + str(max_number))

input_file = codecs.open(os.path.join(dir_in, input_file_name), 'r', encoding = 'UTF-8')
reader = csv.reader(input_file, delimiter=',', quotechar='"')
count = 0
das2freq = dict()
id2das = dict()
das2id = dict()

for row in reader:  # , max_col=5, max_row=max_number+1):
    count += 1
    if count < max_number:
        text = row[0]
        freq = row[1]
        if text != "":
            to_classify.append(text)
            das2freq[count] = freq
            id2das[count] = text
            das2id[text] = count

input_file.close()

to_classify = list(set(to_classify))
print(str(len(to_classify)) + " data points to classify")

def check_and_return(x):

    count = max(id2das.keys()) + 1
    if not x in das2id:
        das2freq[count] = 1
        id2das[count] = x
        das2id[x] = count
    return das2id[x]

# ------------------------------------
# Word embeddings:
# ------------------------------------

# We got ourselves a dictionary mapping word -> 100-dimensional vector. Now we can use it to build features.
# The simplest way to do that is by averaging word vectors for all words in a text.
# We will build a sklearn-compatible transformer that is initialised with a word -> vector dictionary.
class MeanEmbeddingVectorizer(object):

    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(glove_small))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

# This is a version that uses tf-idf weighting scheme for good measure:

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(glove_small))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

# ----------------------------
# Prepare overall summary:
# ----------------------------

with open(os.path.join(dir_out, output_summary_file_all), 'w', encoding='UTF-8') as outfile_summary_all:
    outwriter_all = csv.writer(outfile_summary_all, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
    outwriter_all.writerow(["Combine labels", "coding", "stopwords", "stemming", "uniform prior", "model", "accuracy",
                            "accuracy on top frequency DAS", "frequency-weighted accuracy",
                            "weighted precision", "weighted recall", "weighted F1 score",
                            "file with predictions on test set", "file with predictions"])

    # ----------------------------
    # Consider parameters:
    # ----------------------------

    for coding in coding_values:
        for combine_labels in combine_labels_values:
            for stopwords in stopwords_values:
                for stemming in stemming_values:

                    for uniform_prior in uniform_prior_values:

                        if (coding == 2 and combine_labels == "no") or coding != 2:  # exclude combined labels when approach is 2

                            print("Combine labels:", combine_labels, "coding:", coding, "stopwords:", stopwords,
                                  "stemming:", stemming, "uniform prior:", uniform_prior)

                            # NB files (first classifier):

                            output_file_nb = "Classified_NB_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                             "-stopwords-" + str(stopwords) + "-uniformprior_" + str(uniform_prior) + \
                                             "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            output_test_file_nb = "Classified_NB-test_" + "combined_labels_" + combine_labels + "-coding-approach" + \
                                                  str(coding) + "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            # NB with TF-IDF files (second classifier):

                            output_file_tfidf_nb = "Classified_TFIDF_NB_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            output_test_file_tfidf_nb = "Classified_TFIDF_NB-test_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                                        "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + \
                                                        "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            # SVM files (third classifier):

                            output_file_svm = "Classified_SVM_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                              "-stopwords-" + str(stopwords) + "-uniformprior_" + str(uniform_prior) + \
                                              "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            output_test_file_svm = "Classified_SVM-test_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                                   "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + \
                                                   "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            # word2vec files (fourth classifier):

                            output_file_w2v = "Classified_w2v_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                              "-stopwords-" + str(stopwords) + "-uniformprior_" + str(uniform_prior) + \
                                              "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            output_test_file_w2v = "Classified_w2v-test_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                                   "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + \
                                                   "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            # word2vec with TF-IDF files (fifth classifier):

                            output_file_w2v_tfidf = "Classified_w2v_tfidf_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                                    "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + \
                                                    "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            output_test_file_w2v_tfidf = "Classified_w2v_tfidf-test_" + "combined_labels_" + combine_labels + "-coding-approach" + str(
                                coding) + \
                                                         "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + \
                                                         "-stemming_" + str(stemming) + "-test_" + str(istest) + ".csv"

                            # summary file:

                            output_summary_file = "Classification_accuracy_" + "combined_labels_" + combine_labels + "-coding-approach" + \
                                                  str(coding) + "-stopwords-" + str(stopwords) + "-uniformprior_" + str(
                                uniform_prior) + "-stemming_" + str(stemming) + "-test_" + str(istest) + ".txt"

                            print("Reading annotated data...")
                            annotated_file = codecs.open(os.path.join(dir_in, annotated_file_name), 'r', encoding='UTF-8')
                            annotated_reader = csv.reader(annotated_file, delimiter='\t')  # , quotechar='|')

                            if istest == "yes":
                                max_number_ann = number_test
                            else:
                                max_number_ann = sum(1 for row in annotated_reader)

                            print("Max_number annotated:" + str(max_number_ann))
                            annotated_file.close()

                            annotated_file = codecs.open(
                                os.path.join(dir_in, annotated_file_name), 'r',
                                encoding='UTF-8')
                            annotated_reader = csv.reader(annotated_file, delimiter='\t')  # , quotechar='|')

                            train = list()
                            test = list()
                            top_frequency_das = list()  # list of top 250 most frequent DAS from the annotation

                            count = 0
                            count_labelled = 0
                            count_notlabelled = 0
                            for row_ann in annotated_reader:
                                count += 1

                                if count < max_number_ann+1 and count > 1:
                                    text = row_ann[0]
                                    if count <= 250:
                                        top_frequency_das.append(text)
                                    label = ""
                                    if coding == 1:
                                        label = str(row_ann[3])
                                        if combine_labels == "yes":
                                            if label == "3":
                                                label = "1"
                                            elif label == "5":
                                                label = "3"
                                            elif label == "4":
                                                label = "3"
                                    else:
                                        label = str(row_ann[4])

                                    if label != "?" and str(label) in ["0", "1", "2", "3", "4", "5"]:
                                        tuple = (text, label)

                                    if "FALSE" not in str(label) and str(label) in ["0", "1", "2", "3", "4", "5"]:
                                        if label != "" and label is not None:
                                            count_labelled += 1
                                            if count_labelled in random_indices_train:
                                                train.append(tuple)
                                            elif count_labelled in random_indices_test:
                                                test.append(tuple)
                                        else:
                                            count_notlabelled += 1

                            test_labels = np.asarray([x[1] for x in test])
                            print("Test labels:", str(set(test_labels)))
                            train_labels = np.asarray([x[1] for x in train])
                            print("Train labels:", str(set(train_labels)))
                            train_texts = [x[0] for x in train]
                            test_texts = [x[0] for x in test]
                            test_ids = [check_and_return(x) for x in test_texts]
                            test_texts_topfreq = [x for x in test_texts if x in top_frequency_das]
                            test_ids_topfreq = [check_and_return(x) for x in test_texts_topfreq]
                            test_labels_topfreq = np.asarray([x[1] for x in test if x[0] in top_frequency_das])
                            print("Training data points:")
                            print(str(len(train_texts)) + " training data points")
                            print("Test data points:")
                            print(str(len(test_texts)) + " test data points")
                            print("Top-frequency test data points:")
                            print(str(len(test_texts_topfreq)) + " top-frequency test data points")

                            annotated_file.close()

                            # ----------------------------
                            # word embeddings:
                            # ----------------------------

                            # http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
                            # https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb

                            # NB: This needs to be tested!!!
                            # download GloVe word vector representations
                            # bunch of small embeddings - trained on 6B tokens - 822 MB download, 2GB unzipped
                            # on a linux shell:
                            # wget http://nlp.stanford.edu/data/glove.6B.zip
                            # unzip glove.6B.zip
                            # move the files into dir_in

                            # Prepare word embeddings: the downloaded pretrained ones:
                            # with open(os.path.join(dir_annotated, "glove.6B.50d.txt"), "rb") as lines:
                            #    w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}

                            # reading glove files, this may take a while
                            # we're reading line by line and only saving vectors
                            # that correspond to words from our training set
                            # if you wan't to play around with the vectors and have
                            # enough RAM - remove the 'if' line and load everything

                            glove_small = {}
                            all_words = set(w for words in train_texts for w in words)
                            with open(os.path.join(dir_in, "glove.6B.50d.txt"), "rb") as infile:
                                for line in infile:
                                    parts = line.split()
                                    word = parts[0].decode(encoding)
                                    if (word in all_words):
                                        nums = np.array(parts[1:], dtype=np.float32)
                                        glove_small[word] = nums

                            # Train new word embeddings from scratch:

                            model = gensim.models.Word2Vec(train_texts, size=100)
                            w2v = dict(zip(model.wv.index2word, model.wv.syn0))

                            # define the actual models that will take tokenised text, vectorize and learn to classify the vectors
                            # with something fancy like Extra Trees:

                            etree_w2v = Pipeline([
                                ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                                ("extra trees", ExtraTreesClassifier(n_estimators=200))])

                            etree_w2v_tfidf = Pipeline([
                                ("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                ("extra trees", ExtraTreesClassifier(n_estimators=200))])

                            # ----------------------------
                            # Initialization of models:
                            # ----------------------------

                            if stopwords != "no":
                                if stemming != "no":
                                    count_vect = StemmedCountVectorizer(stop_words='english')
                                else:
                                    count_vect = CountVectorizer(stop_words='english')
                            else:
                                if stemming != "no":
                                    count_vect = StemmedCountVectorizer()
                                else:
                                    count_vect = CountVectorizer()

                            tfidf_transformer = TfidfTransformer()


                            if uniform_prior != "no":
                                mnnb = MultinomialNB(fit_prior=False)
                            else:
                                mnnb = MultinomialNB()


                            # accuracy values:
                            acc1 = 0
                            acc2 = 0
                            acc3 = 0
                            acc4 = 0
                            acc2_weighted = 0
                            acc2_topfreq = 0
                            acc2_gs = 0
                            acc3_weighted = 0
                            acc3_topfreq = 0
                            acc3_gs = 0
                            acc4_weighted = 0
                            acc4_topfreq = 0
                            acc4_cv = 0
                            acc5_weighted = 0
                            acc5_topfreq = 0
                            acc5_cv = 0
                            fea1 = ""
                            fea2 = ""
                            fea3 = ""
                            fea4 = ""

                            # ----------------------------
                            # Naive-Bayes classifier:
                            # http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
                            # ----------------------------

                            if skip_model1 != "yes":
                                print("Training Naive Bayes classifier...")
                                cl = NaiveBayesClassifier(train)

                                # testing:

                                predicted_test_cl = list()
                                for t in test_texts:
                                    predicted_test_cl.append(cl.classify(t))
                                print("There are", str(len(test_texts)), "test texts", "and ", str(len(predicted_test_cl)), "predicted test texts")

                                # accuracy:

                                print("Accuracy of Naive Bayes classifier:", cl.accuracy(test))
                                print("Most important features:", cl.show_informative_features(5))
                                acc1 = cl.accuracy(test)

                                feat1 = str(cl.show_informative_features(5))
                                #outfile_summary.write("Most important features: " + str(cl.show_informative_features(5)))

                                # Output:

                                print("Classifying statements...")
                                predicted_cl = list()
                                classified = 0
                                for t in to_classify:
                                    classified += 1
                                    print("Classifying "+str(classified))
                                    predicted_cl.append([t, cl.classify(t)])

                                print("Printing classified statements...")
                                with codecs.open(os.path.join(dir_out, output_file_nb), 'w', encoding = "UTF-8") as outfile_nb:
                                    outwriter = csv.writer(outfile_nb, delimiter='\t',
                                                           quoting=csv.QUOTE_MINIMAL)
                                    for [t, pred] in predicted_cl:
                                        outwriter.writerow([t, pred])

                                outwriter_all.writerow(
                                    [combine_labels, coding, stopwords, stemming, uniform_prior,
                                     "first classifier (Naive Bayes classifier)", acc1, ""])

                            # ------------------------------------
                            # Naive-Bayes classifier with TF-IDF:
                            # https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
                            # -----------------------------------

                            if skip_model2 != "yes":

                                print("Training TF-IDF Naive Bayes classifier...")

                                # Training:

                                # Learn the vocabulary dictionary and return a Document-Term matrix:
                                #X_train_counts = count_vect.fit_transform(train_texts)

                                # Apply TF-IDF and return a Document-Term matrix:
                                #X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

                                # Run Naive-Bayes classifier:
                                #clf = MultinomialNB().fit(X_train_tfidf, train_labels)

                                text_clf = Pipeline([('vect', count_vect), ('tfidf', tfidf_transformer), ('clf', mnnb)])
                                text_clf = text_clf.fit(train_texts, train_labels)

                                # testing:

                                print("Testing TF-IDF Naive Bayes classifier...")

                                predicted_test_tfidf_nb = text_clf.predict(test_texts)
                                predicted_test_tfidf_nb_topfreq = text_clf.predict(test_texts_topfreq)

                                print("Printing ", str(len(predicted_test_tfidf_nb)), "classified test set...")

                                predicted_test_cl = list()
                                for t in range(len(predicted_test_tfidf_nb)):
                                    predicted_test_cl.append([test_texts[t], predicted_test_tfidf_nb[t], test_labels[t]])
                                print("There are", str(len(test_texts)), "test texts", "and ",
                                      str(len(predicted_test_cl)), "predicted test texts")

                                outfile_test = open(os.path.join(dir_out, output_test_file_tfidf_nb), 'w', encoding='UTF-8')
                                testoutwriter = csv.writer(outfile_test, delimiter='\t',
                                                           # quotechar='|',
                                                           quoting=csv.QUOTE_MINIMAL)
                                for [t, pred, label] in predicted_test_cl:
                                    testoutwriter.writerow([t, pred, label])

                                outfile_test.close()

                                acc2 = accuracy_score(test_labels, predicted_test_tfidf_nb)

                                print("Accuracy:", str(acc2))

                                # weighted accuracy:
                                weights = np.array([das2freq[id] for id in test_ids], dtype=float)
                                y = np.array(predicted_test_tfidf_nb == test_labels, dtype=float)
                                print(weights.shape)
                                print(y.shape)
                                acc2_weighted = np.dot(y, weights.T) / np.sum(weights)
                                print("Weighted accuracy:", str(acc2_weighted))
                                acc2_topfreq = accuracy_score(test_labels_topfreq, predicted_test_tfidf_nb_topfreq)
                                print("Accuracy on top freq: ", str(acc2_topfreq))

                                # weighted average accuracy:
                                #acc_weighted2 = accuracy_score(test_labels, predicted_test_tfidf_nb)

                                # confusion matrix:
                                cm2 = (confusion_matrix(test_labels, predicted_test_tfidf_nb))
                                precision2 = precision_score(test_labels, predicted_test_tfidf_nb, average = None)
                                precision2_micro = precision_score(test_labels, predicted_test_tfidf_nb, average = 'micro')
                                precision2_macro = precision_score(test_labels, predicted_test_tfidf_nb, average = 'macro')
                                precision2_weighted = precision_score(test_labels, predicted_test_tfidf_nb, average = 'weighted')
                                recall2 = recall_score(test_labels, predicted_test_tfidf_nb, average = None)
                                recall2_micro = recall_score(test_labels, predicted_test_tfidf_nb, average = 'micro')
                                recall2_macro = recall_score(test_labels, predicted_test_tfidf_nb, average = 'macro')
                                recall2_weighted = recall_score(test_labels, predicted_test_tfidf_nb, average = 'weighted')

                                # Classification report:
                                cl_report2 = classification_report(test_labels, predicted_test_tfidf_nb)

                                f2 = f1_score(test_labels, predicted_test_tfidf_nb, average = None)
                                f_macro2 = f1_score(test_labels, predicted_test_tfidf_nb, average = 'macro')
                                f_micro2 = f1_score(test_labels, predicted_test_tfidf_nb, average = 'micro')
                                f_weighted2 = f1_score(test_labels, predicted_test_tfidf_nb, average = 'weighted')

                                # Grid Search: Almost all the classifiers will have various parameters which can be tuned to obtain optimal performance.

                                print("Grid Search for Naive Bayes classifier...")

                                # we are creating a list of parameters for which we would like to do performance tuning.
                                # All the parameters name start with the classifier name (remember the arbitrary name we gave).
                                # E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal
                                parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf__alpha': (1e-2, 1e-3)}

                                # We create an instance of the grid search by passing the classifier, parameters and n_jobs=-1
                                # which tells to use multiple cores from user machine:
                                gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5)
                                gs_clf = gs_clf.fit(train_texts, train_labels)

                                # Find the best mean score and the params:
                                acc2_gs = gs_clf.best_score_
                                params_gs_nb = gs_clf.best_params_

                                outwriter_all.writerow(
                                [combine_labels, coding, stopwords, stemming, uniform_prior,
                                 "second classifier (TF-IDF Naive Bayes classifier)", acc2, acc2_topfreq, acc2_weighted,
                                 precision2_weighted, recall2_weighted, f_weighted2,
                                 output_test_file_tfidf_nb, output_file_tfidf_nb])

                                # Output:

                                print("Classifying statements...")
                                predicted_cl = list()
                                classified = 0
                                predicted_tfidf_nb = text_clf.predict(to_classify)

                                for t in range(len(predicted_tfidf_nb)):
                                    classified += 1
                                    # print("Classifying "+str(classified))
                                    # print("Classifying " + to_classify[t])
                                    # print("t:" + str(to_classify[t]) + "label:" + str(predicted_tfidf_svm[t]))
                                    predicted_cl.append([to_classify[t], predicted_tfidf_nb[t]])

                                print("Printing classified statements...")
                                with codecs.open(os.path.join(dir_out, output_file_tfidf_nb), 'w',
                                                 encoding='UTF-8') as outfile_tfidf_nb:
                                    outwriter = csv.writer(outfile_tfidf_nb, delimiter='\t',
                                                           # quotechar='|',
                                                           quoting=csv.QUOTE_MINIMAL)
                                    for [t, pred] in predicted_cl:
                                        outwriter.writerow([t, pred])

                            # ----------------------------------------------
                            # Support Vector Machines (SVM) classifier with TF-IDF:
                            # https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
                            # ---------------------------------------------

                            if skip_model3 != "yes":

                                print("Training TF-IDF SVM classifier...")
                                text_clf_svm = Pipeline([('vect', count_vect), ('tfidf', tfidf_transformer),
                                                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                                         alpha = 1e-3, max_iter = 100, tol=1e-3, random_state = 42))])

                                # Training:

                                text_clf_svm = text_clf_svm.fit(train_texts, train_labels)

                                # Testing:

                                print("Testing TF-IDF SVM classifier...")
                                predicted_test_tfidf_svm = text_clf_svm.predict(test_texts)
                                acc3 = np.mean(predicted_test_tfidf_svm == test_labels)
                                print(str(acc3))
                                predicted_test_tfidf_svm_topfreq = text_clf_svm.predict(test_texts_topfreq)
                                acc3_topfreq = np.mean(predicted_test_tfidf_svm_topfreq == test_labels_topfreq)
                                print(str(acc3_topfreq))

                                weights = np.array([das2freq[id] for id in test_ids], dtype=float)
                                y = np.array(predicted_test_tfidf_svm == test_labels, dtype=float)
                                acc3_weighted = np.dot(y, weights.T) / np.sum(weights)
                                print("Weighted accuracy:", str(acc3_weighted))

                                precision3 = precision_score(test_labels, predicted_test_tfidf_svm, average=None)
                                precision3_micro = precision_score(test_labels, predicted_test_tfidf_svm,
                                                                   average='micro')
                                precision3_macro = precision_score(test_labels, predicted_test_tfidf_svm,
                                                                   average='macro')
                                precision3_weighted = precision_score(test_labels, predicted_test_tfidf_svm,
                                                                      average='weighted')
                                recall3 = recall_score(test_labels, predicted_test_tfidf_svm, average=None)
                                recall3_micro = recall_score(test_labels, predicted_test_tfidf_svm, average='micro')
                                recall3_macro = recall_score(test_labels, predicted_test_tfidf_svm, average='macro')
                                recall3_weighted = recall_score(test_labels, predicted_test_tfidf_svm,
                                                                average='weighted')

                                # Classification report:
                                cl_report3 = classification_report(test_labels, predicted_test_tfidf_svm)

                                # Confusion matrix:
                                cm3 = confusion_matrix(test_labels, predicted_test_tfidf_svm)

                                f3 = f1_score(test_labels, predicted_test_tfidf_svm, average=None)
                                f_macro3 = f1_score(test_labels, predicted_test_tfidf_svm, average='macro')
                                f_micro3 = f1_score(test_labels, predicted_test_tfidf_svm, average='micro')
                                f_weighted3 = f1_score(test_labels, predicted_test_tfidf_svm, average='weighted')

                                # Grid Search: Almost all the classifiers will have various parameters which can be tuned to obtain optimal performance.

                                print("Grid Search for SVM classifier...")

                                parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),
                                                  'clf-svm__alpha': (1e-2, 1e-3)}
                                gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1, cv=5)
                                gs_clf_svm = gs_clf_svm.fit(train_texts, train_labels)

                                acc3_gs = gs_clf_svm.best_score_
                                params_gs_svm = gs_clf_svm.best_params_

                                # Output:

                                print("Classifying statements...")
                                predicted_cl = list()
                                classified = 0
                                predicted_tfidf_svm = text_clf_svm.predict(to_classify)
                                for t in range(len(predicted_tfidf_svm)):
                                    classified += 1
                                    predicted_cl.append([to_classify[t], predicted_tfidf_svm[t]])

                                print("Printing classified statements...")
                                with codecs.open(os.path.join(dir_out, output_file_svm), 'w', encoding = 'UTF-8') as outfile_svm:
                                    outwriter = csv.writer(outfile_svm, delimiter='\t',
                                                           # quotechar='|',
                                                           quoting=csv.QUOTE_MINIMAL)
                                    for [t, pred] in predicted_cl:
                                        outwriter.writerow([t, pred])

                                print("Printing classified test set...")

                                predicted_test_cl = list()
                                for t in range(len(predicted_test_tfidf_svm)):
                                    predicted_test_cl.append([test_texts[t], predicted_test_tfidf_svm[t], test_labels[t]])
                                print("There are", str(len(test_texts)), "test texts", "and ",
                                      str(len(predicted_test_cl)), "predicted test texts")

                                outfile_test = open(os.path.join(dir_out, output_test_file_svm), 'w', encoding='UTF-8')
                                testoutwriter = csv.writer(outfile_test, delimiter='\t',
                                                               # quotechar='|',
                                                               quoting=csv.QUOTE_MINIMAL)
                                for [t, pred, label] in predicted_test_cl:
                                    testoutwriter.writerow([t, pred, label])
                                outfile_test.close()

                                outwriter_all.writerow(
                                [combine_labels, coding, stopwords, stemming, uniform_prior,
                                 "third classifier (SVM classifier)", acc3, acc3_topfreq, acc3_weighted,
                                 precision3_weighted, recall3_weighted, f_weighted3,
                                 output_test_file_svm, output_file_svm])

                            # ----------------------------------------------
                            # Word embeddings:
                            # https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb
                            # ---------------------------------------------

                            if skip_model4 != "yes":

                                # Training:

                                text_clf_w2v = etree_w2v.fit(train_texts, train_labels)

                                # Testing:
                                acc4_cv = cross_val_score(etree_w2v, test_texts, test_labels, cv=5).mean()

                                print("Testing word2vec classifier...")
                                predicted_test_w2v = text_clf_w2v.predict(test_texts)
                                acc4 = np.mean(predicted_test_w2v == test_labels)
                                print(str(acc4))
                                predicted_test_w2v_topfreq = text_clf_w2v.predict(test_texts_topfreq)
                                acc4_topfreq = np.mean(predicted_test_w2v_topfreq == test_labels_topfreq)
                                print(str(acc4_topfreq))

                                weights = np.array([das2freq[id] for id in test_ids], dtype=float)
                                print("weights:", str(weights))
                                y = np.array(predicted_test_w2v == test_labels, dtype=float)
                                print("y:", str(y))
                                acc4_weighted = np.dot(y, weights.T) / np.sum(weights)
                                print("Weighted accuracy:", str(acc4_weighted))

                                precision4 = precision_score(test_labels, predicted_test_w2v, average=None)
                                precision4_micro = precision_score(test_labels, predicted_test_w2v,
                                                                   average='micro')
                                precision4_macro = precision_score(test_labels, predicted_test_w2v,
                                                                   average='macro')
                                precision4_weighted = precision_score(test_labels, predicted_test_w2v,
                                                                      average='weighted')
                                recall4 = recall_score(test_labels, predicted_test_w2v, average=None)
                                recall4_micro = recall_score(test_labels, predicted_test_w2v, average='micro')
                                recall4_macro = recall_score(test_labels, predicted_test_w2v, average='macro')
                                recall4_weighted = recall_score(test_labels, predicted_test_w2v,
                                                                average='weighted')

                                # Classification report:
                                cl_report4 = classification_report(test_labels, predicted_test_w2v)

                                # Confusion matrix:
                                cm4 = confusion_matrix(test_labels, predicted_test_w2v)

                                f4 = f1_score(test_labels, predicted_test_w2v, average=None)
                                f_macro4 = f1_score(test_labels, predicted_test_w2v, average='macro')
                                f_micro4 = f1_score(test_labels, predicted_test_w2v, average='micro')
                                f_weighted4 = f1_score(test_labels, predicted_test_w2v, average='weighted')

                                # Output:

                                print("Classifying statements...")
                                predicted_cl = list()
                                classified = 0
                                predicted_w2v = text_clf_w2v.predict(to_classify)
                                for t in range(len(predicted_w2v)):
                                    classified += 1
                                    predicted_cl.append([to_classify[t], predicted_w2v[t]])

                                print("Printing classified statements...")
                                with codecs.open(os.path.join(dir_out, output_file_w2v), 'w', encoding='UTF-8') as outfile_w2v:
                                    outwriter = csv.writer(outfile_w2v, delimiter='\t',
                                                           # quotechar='|',
                                                           quoting=csv.QUOTE_MINIMAL)
                                    for [t, pred] in predicted_cl:
                                        outwriter.writerow([t, pred])

                                print("Printing classified test set...")

                                predicted_test_cl = list()
                                for t in range(len(predicted_test_w2v)):
                                    predicted_test_cl.append([test_texts[t], predicted_test_w2v[t], test_labels[t]])

                                outfile_test = open(os.path.join(dir_out, output_test_file_w2v), 'w', encoding='UTF-8')
                                testoutwriter = csv.writer(outfile_test, delimiter='\t',
                                                           # quotechar='|',
                                                           quoting=csv.QUOTE_MINIMAL)
                                for [t, pred, label] in predicted_test_cl:
                                    testoutwriter.writerow([t, pred, label])
                                outfile_test.close()

                                outwriter_all.writerow(
                                [combine_labels, coding, stopwords, stemming, uniform_prior,
                                 "fourth classifier (word2vec)", str(acc4), str(acc4_topfreq), acc4_weighted,
                                 precision4_weighted, recall4_weighted, f_weighted4,
                                 output_test_file_w2v, output_file_w2v])

                            # ----------------------------------------------
                            # Word embeddings using TF-IDF weights:
                            # https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking_python3.ipynb
                            # ---------------------------------------------

                            if skip_model5 != "yes":

                                # Training:

                                text_clf_w2v_tfidf = etree_w2v_tfidf.fit(train_texts, train_labels)

                                # Testing:
                                acc5_cv = cross_val_score(etree_w2v_tfidf, test_texts, test_labels, cv=5).mean()

                                print("Testing word2vec classifier...")
                                predicted_test_w2v_tfidf = text_clf_w2v_tfidf.predict(test_texts)
                                acc5 = np.mean(predicted_test_w2v_tfidf == test_labels)
                                print(str(acc5))
                                predicted_test_w2v_tfidf_topfreq = text_clf_w2v_tfidf.predict(test_texts_topfreq)
                                acc5_topfreq = np.mean(predicted_test_w2v_tfidf_topfreq == test_labels_topfreq)
                                print(str(acc5_topfreq))

                                weights = np.array([das2freq[id] for id in test_ids], dtype=float)
                                print("weights:", str(weights))
                                y = np.array(predicted_test_w2v_tfidf == test_labels, dtype=float)
                                print("y:", str(y))
                                acc5_weighted = np.dot(y, weights.T) / np.sum(weights)
                                print("Weighted accuracy:", str(acc5_weighted))

                                precision5 = precision_score(test_labels, predicted_test_w2v_tfidf, average=None)
                                precision5_micro = precision_score(test_labels, predicted_test_w2v_tfidf,
                                                                   average='micro')
                                precision5_macro = precision_score(test_labels, predicted_test_w2v_tfidf,
                                                                   average='macro')
                                precision5_weighted = precision_score(test_labels, predicted_test_w2v_tfidf,
                                                                      average='weighted')
                                recall5 = recall_score(test_labels, predicted_test_w2v_tfidf, average=None)
                                recall5_micro = recall_score(test_labels, predicted_test_w2v_tfidf, average='micro')
                                recall5_macro = recall_score(test_labels, predicted_test_w2v_tfidf, average='macro')
                                recall5_weighted = recall_score(test_labels, predicted_test_w2v_tfidf,
                                                                average='weighted')

                                # Classification report:
                                cl_report5 = classification_report(test_labels, predicted_test_w2v_tfidf)

                                # Confusion matrix:
                                cm5 = confusion_matrix(test_labels, predicted_test_w2v_tfidf)

                                f5 = f1_score(test_labels, predicted_test_w2v_tfidf, average=None)
                                f_macro5 = f1_score(test_labels, predicted_test_w2v_tfidf, average='macro')
                                f_micro5 = f1_score(test_labels, predicted_test_w2v_tfidf, average='micro')
                                f_weighted5 = f1_score(test_labels, predicted_test_w2v_tfidf, average='weighted')

                                # Output:

                                print("Classifying statements...")
                                predicted_cl = list()
                                classified = 0
                                predicted_w2v_tfidf = text_clf_w2v.predict(to_classify)
                                for t in range(len(predicted_w2v_tfidf)):
                                    classified += 1
                                    predicted_cl.append([to_classify[t], predicted_w2v_tfidf[t]])

                                print("Printing classified statements...")
                                with codecs.open(os.path.join(dir_out, output_file_w2v_tfidf), 'w', encoding='UTF-8') as outfile_w2v_tfidf:
                                    outwriter = csv.writer(outfile_w2v_tfidf, delimiter='\t',
                                                           # quotechar='|',
                                                           quoting=csv.QUOTE_MINIMAL)
                                    for [t, pred] in predicted_cl:
                                        outwriter.writerow([t, pred])

                                print("Printing classified test set...")

                                predicted_test_cl = list()
                                for t in range(len(predicted_test_w2v_tfidf)):
                                    predicted_test_cl.append([test_texts[t], predicted_test_w2v_tfidf[t], test_labels[t]])

                                outfile_test = open(os.path.join(dir_out, output_test_file_w2v_tfidf), 'w', encoding='UTF-8')
                                testoutwriter = csv.writer(outfile_test, delimiter='\t',
                                                           # quotechar='|',
                                                           quoting=csv.QUOTE_MINIMAL)
                                for [t, pred, label] in predicted_test_cl:
                                    testoutwriter.writerow([t, pred, label])
                                outfile_test.close()


                                outwriter_all.writerow(
                                [combine_labels, coding, stopwords, stemming, uniform_prior,
                                 "fifth classifier (word2vec with TF-IDF)", str(acc5), str(acc5_topfreq), acc5_weighted,
                                 precision5_weighted, recall5_weighted, f_weighted5,
                                 output_test_file_w2v_tfidf, output_file_w2v_tfidf])

                            # Word2Vec (https://github.com/adventuresinML/adventures-in-ml-code/blob/master/keras_word2vec.py)

                            with open(os.path.join(dir_out, output_summary_file), 'w', encoding='UTF-8') as outfile_summary:

                                if skip_model1 != "yes":
                                    outfile_summary.write("Accuracy of first classifier (Naive Bayes classifier):" + str(acc1)+"\n")
                                    outfile_summary.write("Most important features: " + str(fea1)+"\n")

                                if skip_model2 != "yes":
                                    outfile_summary.write("Accuracy of second classifier:" + str(acc2)+"\n")
                                    outfile_summary.write("Accuracy of second classifier on top-frequency DAS:" + str(acc2_topfreq) + "\n")
                                    outfile_summary.write("Weighted accuracy of second classifier:" + str(
                                        acc2_weighted) + "\n")
                                    outfile_summary.write("Precision of second classifier:" + str(
                                        precision2) + "\n")
                                    outfile_summary.write("Precision (micro) of second classifier:" + str(
                                        precision2_micro) + "\n")
                                    outfile_summary.write("Precision (macro) of second classifier:" + str(
                                        precision2_macro) + "\n")
                                    outfile_summary.write("Precision (weighted) of second classifier:" + str(
                                        precision2_weighted) + "\n")
                                    outfile_summary.write("Recall of second classifier:" + str(
                                        recall2) + "\n")
                                    outfile_summary.write("Recall (micro) of second classifier:" + str(
                                        recall2_micro) + "\n")
                                    outfile_summary.write("Recall (macro) of second classifier:" + str(
                                        recall2_macro) + "\n")
                                    outfile_summary.write("Recall (weighted) of second classifier:" + str(
                                        recall2_weighted) + "\n")

                                    outfile_summary.write("Classification report of second classifier:" + str(
                                        cl_report2) + "\n")
                                    outfile_summary.write("Confusion matrix report of second classifier:" + str(
                                        cm2) + "\n")
                                    outfile_summary.write("f1 of second classifier:" + str(
                                        f2) + "\n")
                                    outfile_summary.write("f1_macro of second classifier:" + str(
                                        f_macro2) + "\n")
                                    outfile_summary.write("f1_micro of second classifier:" + str(
                                        f_micro2) + "\n")
                                    outfile_summary.write("f1_weighted of second classifier:" + str(
                                        f_weighted2) + "\n")

                                    outfile_summary.write("With Grid Search:" + str(acc2_gs) + ", best parameters: " + str(params_gs_nb) + "\n")

                                if skip_model3 != "yes":
                                    outfile_summary.write("Accuracy of third classifier:" + str(acc3)+"\n")
                                    outfile_summary.write("Accuracy of third classifier on top-frequency DAS:" + str(
                                        acc3_topfreq) + "\n")
                                    outfile_summary.write("Weighted accuracy of third classifier:" + str(
                                        acc3_weighted) + "\n")
                                    outfile_summary.write("With cross validation:" + str(acc4_cv) + "\n")
                                    outfile_summary.write(
                                        "With Grid Search:" + str(acc3_gs) + ", best parameters: " + str(
                                            params_gs_svm) + "\n")

                                    outfile_summary.write("Precision of third classifier:" + str(
                                        precision3) + "\n")
                                    outfile_summary.write("Precision (micro) of third classifier:" + str(
                                        precision3_micro) + "\n")
                                    outfile_summary.write("Precision (macro) of third classifier:" + str(
                                        precision3_macro) + "\n")
                                    outfile_summary.write("Precision (weighted) of third classifier:" + str(
                                        precision3_weighted) + "\n")
                                    outfile_summary.write("Recall of third classifier:" + str(
                                        recall3) + "\n")
                                    outfile_summary.write("Recall (micro) of third classifier:" + str(
                                        recall3_micro) + "\n")
                                    outfile_summary.write("Recall (macro) of third classifier:" + str(
                                        recall3_macro) + "\n")
                                    outfile_summary.write("Recall (weighted) of third classifier:" + str(
                                        recall3_weighted) + "\n")

                                    outfile_summary.write("Classification report of third classifier:" + str(
                                        cl_report3) + "\n")
                                    outfile_summary.write("Confusion matrix report of third classifier:" + str(
                                        cm3) + "\n")
                                    outfile_summary.write("f1 of third classifier:" + str(
                                        f3) + "\n")
                                    outfile_summary.write("f1_macro of third classifier:" + str(
                                        f_macro3) + "\n")
                                    outfile_summary.write("f1_micro of third classifier:" + str(
                                        f_micro3) + "\n")
                                    outfile_summary.write("f1_weighted of third classifier:" + str(
                                        f_weighted3) + "\n")

                                if skip_model4 != "yes":
                                    outfile_summary.write("Accuracy of fourth classifier:" + str(acc4)+ "\n")
                                    outfile_summary.write("Accuracy of fourth classifier on top-frequency DAS:" + str(
                                        acc4_topfreq) + "\n")
                                    outfile_summary.write("Weighted accuracy of second classifier:" + str(
                                        acc4_topfreq) + "\n")

                                    outfile_summary.write("Precision of fourth classifier:" + str(
                                        precision4) + "\n")
                                    outfile_summary.write("Precision (micro) of fourth classifier:" + str(
                                        precision4_micro) + "\n")
                                    outfile_summary.write("Precision (macro) of fourth classifier:" + str(
                                        precision4_macro) + "\n")
                                    outfile_summary.write("Precision (weighted) of fourth classifier:" + str(
                                        precision4_weighted) + "\n")
                                    outfile_summary.write("Recall of fourth classifier:" + str(
                                        recall4) + "\n")
                                    outfile_summary.write("Recall (micro) of fourth classifier:" + str(
                                        recall4_micro) + "\n")
                                    outfile_summary.write("Recall (macro) of fourth classifier:" + str(
                                        recall4_macro) + "\n")
                                    outfile_summary.write("Recall (weighted) of fourth classifier:" + str(
                                        recall4_weighted) + "\n")

                                    outfile_summary.write("Classification report of fourth classifier:" + str(
                                        cl_report4) + "\n")
                                    outfile_summary.write("Confusion matrix report of fourth classifier:" + str(
                                        cm4) + "\n")
                                    outfile_summary.write("f1 of fourth classifier:" + str(
                                        f4) + "\n")
                                    outfile_summary.write("f1_macro of fourth classifier:" + str(
                                        f_macro4) + "\n")
                                    outfile_summary.write("f1_micro of fourth classifier:" + str(
                                        f_micro4) + "\n")
                                    outfile_summary.write("f1_weighted of fourth classifier:" + str(
                                        f_weighted4) + "\n")

                                if skip_model5 != "yes":

                                    outfile_summary.write("Accuracy of fifth classifier:" + str(acc5)+"\n")
                                    outfile_summary.write("Accuracy of fifth classifier on top-frequency DAS:" + str(
                                        acc5_topfreq) + "\n")
                                    outfile_summary.write("Accuracy of five classifier on top-frequency DAS:" + str(
                                        acc5_topfreq) + "\n")
                                    outfile_summary.write("With cross validation:" + str(acc5_cv) + "\n")

                                    outfile_summary.write("Precision of fifth classifier:" + str(
                                        precision5) + "\n")
                                    outfile_summary.write("Precision (micro) of fifth classifier:" + str(
                                        precision5_micro) + "\n")
                                    outfile_summary.write("Precision (macro) of fifth classifier:" + str(
                                        precision5_macro) + "\n")
                                    outfile_summary.write("Precision (weighted) of fifth classifier:" + str(
                                        precision5_weighted) + "\n")
                                    outfile_summary.write("Recall of fifth classifier:" + str(
                                        recall5) + "\n")
                                    outfile_summary.write("Recall (micro) of fifth classifier:" + str(
                                        recall5_micro) + "\n")
                                    outfile_summary.write("Recall (macro) of fifth classifier:" + str(
                                        recall5_macro) + "\n")
                                    outfile_summary.write("Recall (weighted) of fifth classifier:" + str(
                                        recall5_weighted) + "\n")

                                    outfile_summary.write("Classification report of fifth classifier:" + str(
                                        cl_report5) + "\n")
                                    outfile_summary.write("Confusion matrix report of fifth classifier:" + str(
                                        cm5) + "\n")
                                    outfile_summary.write("f1 of fifth classifier:" + str(
                                        f5) + "\n")
                                    outfile_summary.write("f1_macro of fifth classifier:" + str(
                                        f_macro5) + "\n")
                                    outfile_summary.write("f1_micro of fifth classifier:" + str(
                                        f_micro5) + "\n")
                                    outfile_summary.write("f1_weighted of fifth classifier:" + str(
                                        f_weighted5) + "\n")


