import argparse
import cPickle
import logging
from notmuch import Database
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import sys
import warnings

import extract_mails

class StdLogger:
    def __init__(self):
        self.logger = None

    def verbose(self, level):
        self.logger = log

    def log_msg(self, level, msg):
        if self.logger is not None:
            self.logger.log(level, msg)

def atomic_pickle(o, filename):
    tmp = filename + '.tmp'
    with open(tmp, 'wb') as f:
        cPickle.dump(o, f, cPickle.HIGHEST_PROTOCOL)
    os.rename(tmp, filename)

def optimize(log, filename, progress=False):
    log.info("getting data")
    data, labels = extract_mails.get_training_data(progress)
    log.info("splitting data")
    x_train, x_test, y_train, y_test = train_test_split(data,
                                                        labels,
                                                        test_size=0.4,
                                                        random_state=0)

    log.info("preprocessing data")
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    X = vectorizer.transform(x_train)
    binarizer = MultiLabelBinarizer()
    binarizer.fit(labels)
    Y = binarizer.transform(y_train)

    # do a gridsearch for the best parameters
    log.info("doing gridsearch... this may take some time")
    pipe = Pipeline([
        ('feature_selection', SelectFromModel(
            RandomForestClassifier(n_estimators=10))),
        ('classification', SVC())
    ])
    clf = OneVsRestClassifier(pipe)
    parameters = {
        "estimator__feature_selection__threshold": ('mean', '0.5*mean', 0),
        "estimator__classification__kernel": ('linear', 'rbf'),
        "estimator__classification__C": (0.01, 0.1, 1, 10, 100)
    }

    grid_search = GridSearchCV(clf, parameters, n_jobs=-1, verbose=1,
                               scoring='f1_samples', error_score=0)
    grid_search.fit(X, Y)

    print grid_search.best_score_
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print "\t{0}: {1}".format(param_name, best_parameters[param_name])

    log.info("evaluating classifier")
    Xt = vectorizer.transform(x_test)
    preds = grid_search.best_estimator_.predict(Xt)
    real = binarizer.transform(y_test)

    print classification_report(real, preds, target_names = binarizer.classes_)

    # store the parameters from the best estimator and the pipeline,
    # so that the next time for training the best pipeline can be
    # used!
    clf.set_params(**best_parameters)
    atomic_pickle(clf, filename)

    return data, labels

def validate(log, filename, progress=False):
    log.info("getting data")
    data, labels = extract_mails.get_training_data(progress)
    log.info("splitting data")
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)

    log.info("preprocess data")
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    X = vectorizer.transform(x_train)
    binarizer = MultiLabelBinarizer()
    binarizer.fit(labels)
    Y = binarizer.transform(y_train)

    log.info("training best classifier")
    with open(filename, 'rb') as f:
        clf = cPickle.load(f)
    clf.fit(X, Y)

    log.info("evaluating classifier")
    Xt = vectorizer.transform(x_test)
    preds = clf.predict(Xt)
    real = binarizer.transform(y_test)

    print classification_report(real, preds, target_names = binarizer.classes_)

def train_from_bottom(log, filename, progress=False, data=None, labels=None):
    if data is None or labels is None:
        log.info("extract all mails from database")
        data, labels = \
            extract_mails.get_training_data(progress)
    log.info("got {0} mails".format(len(data)))

    log.info("create the vocabulary")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(data)
    log.info("vocabulary size: {0}".format(len(vectorizer.vocabulary_)))
    binarizer = MultiLabelBinarizer()
    Y = binarizer.fit_transform(labels)

    log.info("train the classifier")
    with open(filename, 'rb') as f:
        clf = cPickle.load(f)
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

def train(log, pipeline_filename, model_filename, progress):
    data, labels = None, None
    if not os.path.isfile(pipeline_filename):
        log.warn("no existing pipeline found: searching for best parameters. This may take some time!")
        data, labels = optimize(log, pipeline_filename, progress)
    v, b, c = train_from_bottom(log, pipeline_filename, progress, data, labels)
    atomic_pickle([v, b, c], model_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="print logging messages to stdout", action="store_true")
    parser.add_argument("--progress", help="print a progress bar",
                        action="store_true")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("train", help="train the tagger from standard notmuch database")
    subparsers.add_parser("tag", help="tag the mails with a new-tag")
    subparsers.add_parser("validate", help="show a classification report on stdout when trained on 0.6 of the maildir and tested on the other 0.4.")
    subparsers.add_parser("optimize", help="perform a grid search with 60 different possible hyperparameters to find the best ones")
    args = parser.parse_args()

    db = Database()
    path = db.get_path()
    db.close()

    model_filename = os.path.join(path, "blaecksprutte.db")
    pipeline_filename = os.path.join(path, "best_pipeline.db")

    warnings.simplefilter('ignore', UndefinedMetricWarning)
    warnings.simplefilter('ignore', FutureWarning)
    warnings.simplefilter('ignore', UserWarning)

    level = logging.ERROR

    if args.verbose:
        level = logging.INFO

    log = logging.getLogger(__name__)
    out_hdlr = logging.StreamHandler(sys.stdout)
    out_hdlr.setFormatter(logging.Formatter('%(message)s'))
    out_hdlr.setLevel(level)
    log.addHandler(out_hdlr)
    log.setLevel(level)

    if args.command == 'train':
        train(log, pipeline_filename, model_filename, args.progress)

    if args.command == 'tag':
        if not os.path.isfile(model_filename):
            log.warn("no existing model file found: training model. This may take some time!")
            train(log, pipeline_filename, model_filename, args.progress)
        tag_new_mails(model_filename, log)

    if args.command == 'validate':
        validate(log, pipeline_filename, args.progress)

    if args.command == 'optimize':
        optimize(log, pipeline_filename, args.progress)

if __name__ == "__main__":
    main()
