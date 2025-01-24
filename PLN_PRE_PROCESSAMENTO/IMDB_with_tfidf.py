import pandas as pd
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix

# Carregar dataset
df = pd.read_csv('https://drive.google.com/u/0/uc?id=1ZlZsxrMHhZZb9ZTYABOiWw7bCPofY6cz&export=download', header=0)

# Pré-processamento do texto com NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Função de limpeza
stopwords = nltk.corpus.stopwords.words('english')
punctuations = list(string.punctuation)
stemmer = SnowballStemmer('english')
TAMANHO_MINIMO = 1
IGNORAR = ['...', 'br', '.so', '\'ll']

def prepara(texto):
    palavras = [i for i in word_tokenize(texto, language='english') if i not in punctuations]
    palavras = [i for i in palavras if i not in stopwords]
    palavras = [i for i in palavras if len(i) > TAMANHO_MINIMO]
    palavras = [i for i in palavras if i not in IGNORAR]
    palavras = [stemmer.stem(i) for i in palavras]
    return palavras

# Aplicando o pré-processamento
df['review2'] = df['review'].apply(prepara)
df['review2'] = df['review2'].apply(' '.join)

# Dividindo dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(df['review2'], df['sentiment'], test_size=0.2, random_state=0)

# Usando TfidfVectorizer
MAX_FEATURES = 100000
tfidf = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 3), max_features=MAX_FEATURES)
tfidf_X_train = tfidf.fit_transform(X_train)
tfidf_X_test = tfidf.transform(X_test)

# Binarizando os sentimentos
lb = LabelBinarizer()
ohe_y_train = lb.fit_transform(y_train)
ohe_y_test = lb.transform(y_test)

# Treinando o modelo com Regressão Logística
lr = LogisticRegression(penalty='l2', max_iter=50000, C=1, random_state=42)
lr.fit(tfidf_X_train, ohe_y_train.ravel())

# Avaliando o modelo
y_predict = lr.predict(tfidf_X_test)
print("LogReg Score:", accuracy_score(ohe_y_test, y_predict))
print(classification_report(ohe_y_test, y_predict, target_names=['Positive', 'Negative']))

# Plotando a matriz de confusão
plot_confusion_matrix(lr, tfidf_X_test, ohe_y_test, values_format='d')
