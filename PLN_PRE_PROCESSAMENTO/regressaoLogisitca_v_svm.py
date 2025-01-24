# Importação das bibliotecas necessárias
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt

# Configuração do modelo Logistic Regression
print("Treinando o modelo Logistic Regression...")
lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
lr.fit(bow_x_train, ohe_y_train.ravel())  # Treinamento do modelo
print("Modelo Logistic Regression treinado:", lr)

# Avaliação do modelo Logistic Regression
print("\nAvaliando o modelo Logistic Regression...")
y_predict_lr = lr.predict(bow_x_test)  # Previsões
print("LogReg Score:", accuracy_score(ohe_y_test, y_predict_lr))  # Score de acurácia
print("Classification Report Logistic Regression:")
print(classification_report(ohe_y_test, y_predict_lr, target_names=['Positive', 'Negative']))

# Matriz de Confusão para Logistic Regression
print("\nMatriz de Confusão - Logistic Regression:")
plot_confusion_matrix(lr, bow_x_test, ohe_y_test, values_format='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Configuração do modelo SVM
print("\nTreinando o modelo SVM...")
svm = SGDClassifier(loss='hinge', random_state=42)
svm.fit(bow_x_train, ohe_y_train.ravel())  # Treinamento do modelo
print("Modelo SVM treinado:", svm)

# Avaliação do modelo SVM
print("\nAvaliando o modelo SVM...")
y_predict_svm = svm.predict(bow_x_test)  # Previsões
print("SVM Score:", accuracy_score(ohe_y_test, y_predict_svm))  # Score de acurácia
print("Classification Report SVM:")
print(classification_report(ohe_y_test, y_predict_svm, target_names=['Positive', 'Negative']))

# Matriz de Confusão para SVM
print("\nMatriz de Confusão - SVM:")
plot_confusion_matrix(svm, bow_x_test, ohe_y_test, values_format='d')
plt.title("Confusion Matrix - SVM")
plt.show()
