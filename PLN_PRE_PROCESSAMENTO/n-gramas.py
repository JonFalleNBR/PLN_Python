"""
Vamos supor que queremos criar n-gramas para a frase abaixo:

"João quer aprender técnicas de PLN".

Após remover a stopword "de", teremos:

"João quer aprender técnicas PLN".

Se cada palavra fosse um token, teríamos o seguinte:

N-Grama    Tokens criados
1 - Unigram    ["João", "quer", "aprender", "técnicas", "PLN"]
2 - Bigram     [("João", "quer"), ("quer", "aprender"), ("aprender", "técnicas"), ("técnicas", "PLN")]
3 - Trigram    [("João", "quer", "aprender"), ("quer", "aprender", "técnicas"), ("aprender", "técnicas", "PLN")]
"""


%%time
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(2, 2))
bow_x_train = cv.fit_transform(X_train)
bow_x_test = cv.transform(X_test)
print('bow_x_train shape:', bow_x_train.shape)
print('bow_x_test shape:', bow_x_test.shape)

%%time
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
lr.fit(bow_x_train, ohe_y_train.ravel())
print(lr)

y_predict = lr.predict(bow_x_test)
print("LogReg Score :", accuracy_score(ohe_y_test, y_predict))
print(classification_report(ohe_y_test, y_predict, target_names=['Positive', 'Negative']))

%%time
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(loss='hinge', random_state=42)
svm.fit(bow_x_train, ohe_y_train.ravel())
print(svm)

y_predict = svm.predict(bow_x_test)
print("SVM Score :", accuracy_score(ohe_y_test, y_predict))
print(classification_report(ohe_y_test, y_predict, target_names=['Positive', 'Negative']))

%%time
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(3, 3))
bow_x_train = cv.fit_transform(X_train)
bow_x_test = cv.transform(X_test)
print('bow_x_train shape:', bow_x_train.shape)
print('bow_x_test shape:', bow_x_test.shape)

%%time
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
lr.fit(bow_x_train, ohe_y_train.ravel())
print(lr)

y_predict = lr.predict(bow_x_test)
print("LogReg Score :", accuracy_score(ohe_y_test, y_predict))
print(classification_report(ohe_y_test, y_predict, target_names=['Positive', 'Negative']))

%%time
from sklearn.linear_model import SGDClassifier
svm = SGDClassifier(loss='hinge', random_state=42)
svm.fit(bow_x_train, ohe_y_train.ravel())
print(svm)

y_predict = svm.predict(bow_x_test)
print("SVM Score :", accuracy_score(ohe_y_test, y_predict))
print(classification_report(ohe_y_test, y_predict, target_names=['Positive', 'Negative']))

# No arquivo do IMDB há os cenarios com TF-IDF