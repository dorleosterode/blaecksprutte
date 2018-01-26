import argparse
import cPickle
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

import extract_mails

def validate():
    print "getting data"
    data, labels = extract_mails.get_training_data()
    print "splitting data"
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=0)

    print "preprocess data"
    vectorizer = CountVectorizer()
    vectorizer.fit(data)
    X = vectorizer.transform(x_train)
    binarizer = MultiLabelBinarizer()
    binarizer.fit(labels)
    Y = binarizer.transform(y_train)

    print "train classifier"
    clf = OneVsRestClassifier(SGDClassifier())
    clf.fit(X, Y)

    print "evaluate classifier"
    Xt = vectorizer.transform(x_test)
    preds = clf.predict(Xt)
    real = binarizer.transform(y_test)

    print "average precision:"
    print label_ranking_average_precision_score(real, preds)
    print "ranking loss:"
    print label_ranking_loss(real, preds)

def train_from_bottom():
    train_data, train_labels = extract_mails.get_training_data()

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_data)
    binarizer = MultiLabelBinarizer()
    Y = binarizer.fit_transform(train_labels)

    clf = OneVsRestClassifier(SGDClassifier())
    clf.fit(X, Y)

    return vectorizer, binarizer, clf

def tag_new_mails(v, b, c):
    data, ids = extract_mails.get_new_mails()
    if len(data) > 0:
        X = v.transform(data)
        preds = c.predict(X)
        tags = b.inverse_transform(preds)
        extract_mails.write_tags(ids, tags)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", help="train the tagger from standard notmuch database", action="store_true")
    parser.add_argument("--tag", help="tag the mails with a new-tag", action="store_true")
    args = parser.parse_args()

    filename = "tagger.pkl"

    if args.tag:
        with open(filename, 'rb') as f:
            v, b, c = cPickle.load(f)
        tag_new_mails(v, b, c)
    elif args.train:
        v, b, c = train_from_bottom()
        with open(filename, 'wb') as f:
            cPickle.dump([v, b, c], f)

if __name__ == "__main__":
    main()
