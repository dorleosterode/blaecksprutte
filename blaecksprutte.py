import argparse
import cPickle
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
import sys

import extract_mails

class StdLogger:
    def __init__(self):
        self.logger = None

    def verbose(self, level):
        self.logger = log

    def log_msg(self, level, msg):
        if self.logger is not None:
            self.logger.log(level, msg)

def validate(log):
    log.info("getting data")
    data, labels = extract_mails.get_training_data()
    log.info("splitting data")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)

    log.info("preprocess data")
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    X = vectorizer.transform(x_train)
    binarizer = MultiLabelBinarizer()
    binarizer.fit(labels)
    Y = binarizer.transform(y_train)

    log.info("train classifier")
    clf = OneVsRestClassifier(SGDClassifier())
    clf.fit(X, Y)

    log.info("evaluate classifier")
    Xt = vectorizer.transform(x_test)
    preds = clf.predict(Xt)
    real = binarizer.transform(y_test)

    print "average precision:"
    print label_ranking_average_precision_score(real, preds)
    print "ranking loss:"
    print label_ranking_loss(real, preds)

def train_from_bottom(log):
    log.info("extract all mails from database")
    train_data, train_labels = extract_mails.get_training_data()
    log.info("got {0} mails".format(len(train_data)))

    log.info("create the vocabulary")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_data)
    log.info("vocabulary size: {0}".format(len(vectorizer.vocabulary_)))
    binarizer = MultiLabelBinarizer()
    Y = binarizer.fit_transform(train_labels)

    log.info("train the classifier")
    clf = OneVsRestClassifier(SGDClassifier())
    clf.fit(X, Y)
    log.info("completed training")

    return vectorizer, binarizer, clf

def tag_new_mails(filename, log):
    log.info("get new mails")
    data, ids = extract_mails.get_new_mails()
    log.info("found {0} new mails".format(len(data)))
    if len(data) > 0:
        log.info("loading tagger")
        with open(filename, 'rb') as f:
            v, b, c = cPickle.load(f)
        log.info("predicting tags for new mails")
        X = v.transform(data)
        preds = c.predict(X)
        tags = b.inverse_transform(preds)
        log.info( "writing tags into database")
        extract_mails.write_tags(ids, tags)
        log.info("completed prediction")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the tagger from standard notmuch database", action="store_true")
    parser.add_argument("--tag", help="tag the mails with a new-tag", action="store_true")
    parser.add_argument("--verbose", help="print logging messages to stdout", action="store_true")
    args = parser.parse_args()

    filename = "tagger.pkl"

    level = logging.ERROR

    if args.verbose:
        level = logging.INFO

    log = logging.getLogger(__name__)
    out_hdlr = logging.StreamHandler(sys.stdout)
    out_hdlr.setFormatter(logging.Formatter('%(message)s'))
    out_hdlr.setLevel(level)
    log.addHandler(out_hdlr)
    log.setLevel(level)

    if args.tag:
        tag_new_mails(filename, log)
    elif args.train:
        v, b, c = train_from_bottom(log)
        with open(filename, 'wb') as f:
            cPickle.dump([v, b, c], f)

if __name__ == "__main__":
    main()
