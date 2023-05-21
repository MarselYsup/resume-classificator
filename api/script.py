import joblib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

def prepare_data():
  df = pd.read_csv('./resources/resumesv1.4.csv', sep=',', encoding='utf-8')
  df.head()

  resumes = df.resume.tolist()
  mark = df.mark.tolist()

  marks = []

  for x in mark:
    if x < 15:
      marks.append(0)
    else:
      marks.append(1)

  return resumes, marks

def get_vectorizer_and_model():
  resumes, marks = prepare_data()
  vect_b = TfidfVectorizer(analyzer='word', stop_words='english')
  X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(resumes, marks)
  X_vect_train_b = vect_b.fit_transform(X_train_b)
  X_vect_test_b = vect_b.transform(X_test_b)

  bm = LogisticRegression()
  bm.fit(X_vect_train_b, y_train_b)
  y_predict_b = bm.predict(X_vect_test_b)

  cnf_matrix = metrics.confusion_matrix(y_test_b, y_predict_b)
  print(cnf_matrix)

  target_names = ['positive resume', 'negative resume']
  print(classification_report(y_test_b, y_predict_b, target_names=target_names))



  return vect_b, bm




vect, model = get_vectorizer_and_model()

joblib.dump(model, './model/model.pkl')
joblib.dump(vect, './model/vect.pkl')