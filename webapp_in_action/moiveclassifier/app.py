# -*- coding: utf-8 -*-

import pickle
import sqlite3
import os

import numpy as np
from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators

from moiveclassifier.vectorizer import vect
from moiveclassifier.update import update_model

cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, "pkl_objects/classifier.pkl"), "rb"))
db = os.path.join(cur_dir, "review.sqlite")


def classify(document):
    label = {0: "negative", 1: "positive"}
    X = vect.transform([document])
    print(clf.predict(X))
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba


def train(document, y):
    X = vect.transform([document])
    clf.partial_fit(X, [y])


def sqlite_entry(path, docuemt, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("insert into review_db (review, sentiment, date) values (?, ?, datetime(\"now\"))", (docuemt, y))
    conn.commit()
    conn.close()


# 听说 __name__ 的目的是让 Flask 找到同目录下的 templates 目录
app = Flask(__name__)


class ReviewForm(Form):
    movieview = TextAreaField("",
                              [validators.DataRequired(),
                               validators.length(min=15)])


@app.route("/")
def index():
    form = ReviewForm(request.form)
    return render_template("reviewform.html", form=form)


@app.route("/results", methods=["POST"])
def results():
    form = ReviewForm(request.form)
    if request.method == "POST" and form.validate():
        review = request.form["moviereview"]
        y, proba = classify(review)
        return render_template("results.html",
                               content=review,
                               prediction=y,
                               probability=round(proba * 100, 2))
    return render_template("reviewform.html", form=form)


@app.route("/thanks", methods=["POST"])
def feedback():
    feedback = request.form["feedback_button"]
    review = request.form["review"]
    prediction = request.form["prediction"]

    inv_label = {"negative": 0, "positive": 1}
    y = inv_label[prediction]
    if feedback == "Incorrect":
        y = int(not (y))
    train(review, y)
    sqlite_entry(db, review, y)
    return render_template("thanks.html")


if __name__ == '__main__':
    update_model(filepath=db, model=clf, batch=10000)
    app.run(debug=True)
