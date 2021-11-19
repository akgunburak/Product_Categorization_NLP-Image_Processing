# Libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import warnings
warnings.filterwarnings('ignore')


# Read the train set
train_set = pd.read_excel("data/train_preprocessed.xlsx")
train_set


# Read the test set
test_set = pd.read_excel("data/test_preprocessed.xlsx")
test_set


# Create tf-idf vectors
tfidf_vectorizer = TfidfVectorizer(max_features=200000)
train_text_tfidf = tfidf_vectorizer.fit_transform(train_set["text"]) 
test_text_tfidf = tfidf_vectorizer.transform(test_set["text"])


# Feature selection
ch2 = SelectKBest(chi2, k=50000)
train_text_tfidf = ch2.fit_transform(train_text_tfidf, train_set["CATEGORY_ID"])
test_text_tfidf = ch2.transform(test_text_tfidf)


# Train the ML models
# SVM
svm = svm.SVC(class_weight="balanced", C=1, kernel="linear")
svm.fit(train_text_tfidf, train_set["CATEGORY_ID"])  
svm_predict = svm.predict(test_text_tfidf)


# Random Forest
random_forest = RandomForestClassifier(class_weight="balanced", n_estimators=870,
                                       min_samples_split=4, min_samples_leaf=7,
                                       max_features="auto", max_depth=115)
random_forest.fit(train_text_tfidf, train_set["CATEGORY_ID"])
random_forest_predict = random_forest.predict(test_text_tfidf)


# Logistic Regression
log_reg = LogisticRegression(max_iter=len(train_set), solver="liblinear", penalty='l2', C=100)
log_reg.fit(train_text_tfidf, train_set["CATEGORY_ID"])
log_reg_predict = log_reg.predict(test_text_tfidf)


# XGBoost
xgb = xgboost.XGBClassifier(objective= 'binary:logistic', class_weight="balanced")
xgb.fit(train_text_tfidf, train_set["CATEGORY_ID"])
xgb_predict = xgb.predict(test_text_tfidf)


# Add the predictions into a dataframe
predictions = pd.DataFrame(svm_predict, columns=["svm"])
predictions["ran_for"] = random_forest_predict
predictions["log_reg"] = log_reg_predict
predictions["xgb"] = xgb_predict
predictions.to_csv("data/tfidf_ml.csv")