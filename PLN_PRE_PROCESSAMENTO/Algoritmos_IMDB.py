from IPython import get_ipython
from IPython.display import display, set_matplotlib_formats

# Importanto a bilbioteca pandas e conectando e lendo ao arquivo do IMDB

# %%
%matplotlib
inline
import pandas as pd

df = pd.read_csv('https://drive.google.com/u/0/uc?id=1ZlZsxrMHhZZb9ZTYABOiWw7bCPofY6cz&export=download', header=0)
df.head()

# Em seguida, vamos verificar se existem valores nulos nesse dataset:
df.isnull().sum()

# Assim, não existem valores nulos ou vazios para as colunas "review" e "sentiment". Em seguida, vamos verificar a quantidade de exemplos positivos e negativos:

df['sentiment'].value_counts()

# Como podemos ver acima, as classes estão balanceadas, com 25.000 exemplos para cada classe. Vamos plotar essa relação:

df["sentiment"].value_counts().plot.bar(title='Quantidade por Tipo', rot=90)

# A partir do nosso objeto dataframe "df", podemos chamar o método plot.bar() para plotar um gráfico de barras, que mostra a relação da quantidade de exemplos
# com valor "positive" e "negative". Veja ao executar o plot acima

## Com isso, foi possível conhecer um pouco o nosso dataset de filmes. A partir do próximo tópico, vamos começar a utilizar técnicas de PLN utilizando esse dataset.


# Pré-processamento do texto com NLTK
# A etapa de pré-processamento é muito importante para se criar modelos PLN adequados e eficazes.
# Dessa forma, nosso objetivo agora é iniciar esse pré-processamento.
# Agora que você já conhece um pouco seu dataset,


# importe a biblioteca NLTK do Python e realize o pré-processamento do texto utilizando as técnicas que já estudamos.

!pip
install
nltk
import nltk

nltk.download('stopwords')
nltk.download('punkt_tab')

exemplo = df['review'].values[0]
print(exemplo)  # Adicione esta linha para imprimir o conteúdo da variável "exemplo"
print(len(exemplo.split()))

# Vamos utilizar o código que já aprendemos para remover pontuações, stopwords e realizar a contagem de palavras:

from nltk.tokenize import word_tokenize
import string

stopwords = nltk.corpus.stopwords.words('english')
punctuations = list(string.punctuation)


def prepara(texto):
    palavras = [i for i in word_tokenize(texto, language='english') if i not in punctuations]
    palavras = [i for i in palavras if i not in stopwords]
    return palavras


exemplo_preparado = prepara(exemplo)
print(exemplo_preparado)
print(len(exemplo_preparado))

# Passamos de 307 para 195 tokens. Veja que existem algumas palavras muito pequenas (com apenas uma letra),
# além de algumas palavras que podemos ignorar. É possível criar um código adicional para removê-las.

# Além disso, vamos utilizar uma técnica chamada stemming. Stemming é a técnica de remover sufixos e prefixos
# de uma palavra, chamada stem. Por exemplo, o stem da palavra cooking é cook. Um bom algoritmo sabe que "ing"
# é um sufixo e pode ser removido. Stemming é muito usado em mecanismos de buscas para a indexação de palavras.
# Vamos utilizar stemming para simplificar ainda mais nosso problema, removendo variações de palavras.

# Vamos removê-las adicionando à nossa função o código abaixo:

# from nltk.stem import PorterStemmer
# ps = PorterStemmer()

# def preprocess_text(text):
#     words = text.split()
#     # Removendo palavras pequenas (menos que 2 caracteres)
#     words = [word for word in words if len(word) > 1]
#     # Aplicando stemming
#     stemmed_words = [ps.stem(word) for word in words]
#     return " ".join(stemmed_words)


from nltk.tokenize import word_tokenize
import string

stopwords = nltk.corpus.stopwords.words('english')
punctuations = list(string.punctuation)
TAMANHO_MIMINO = 1
IGNORAR = ['...', 'br', '.so', '\'ll']
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')


def prepara(texto):
    palavras = [i for i in word_tokenize(texto, language='english') if i not in punctuations]
    palavras = [i for i in palavras if i not in stopwords]
    palavras = [i for i in palavras if len(i) > TAMANHO_MIMINO]
    palavras = [i for i in palavras if i not in IGNORAR]
    palavras = [stemmer.stem(i) for i in palavras]
    return palavras


exemplo_preparado = prepara(exemplo)
print(exemplo_preparado)
print(len(exemplo_preparado))

# Assim, reduzimos de 195 para 175 tokens. Agora que já testamos a função,
# vamos executar essa função de preparação na coluna `review` do nosso dataset.
# Em seguida, juntaremos novamente o texto para ele voltar a ser uma sentença, e não uma lista de palavras.

# Como a função pode demorar, vamos utilizar a diretiva %%time para nos mostrar o tempo de processamento.


%%time
df['review2'] = df['review'].apply(prepara)
df['review2'] = df['review2'].apply(' '.join)

# Pronto! Podemos agora começar a utilizar um algoritmo de machine learning para tentar classificar os nossos exemplos.

# Vamos, primeiramente, separar os dados de treino e teste, sendo 20% para os dados de teste.
# Para isso, utilizaremos a função train_test_split.


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['review2'], df['sentiment'], test_size=0.2, random_state=0)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)

# Agora, criaremos o Bag-of-Words (BoW) para os dados de treino e teste
# utilizando a função CountVectorizer.
# Link de referência: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html

%%time
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 1))
bow_X_train = cv.fit_transform(X_train)
bow_X_test = cv.transform(X_test)
print('bow_X_train shape:', bow_x_train.shape)
print('bow_X_test shape:', bow_x_test.shape)

# Veja que temos um total de 32.283 palavras no nosso dicionário,
# gerando, assim, uma entrada de 32.283 colunas para cada exemplo!

# Como os valores de sentimento são "positive" e "negative",
# precisamos codificá-los em um formato que seja possível utilizar em modelos de machine learning.

# Vamos, então, criar o one-hot-encoding para esses valores, utilizando a função LabelBinarizer.
# Link de referência: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html

from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
ohe_y_train = lb.fit_transform(y_train)
ohe_y_test = lb.fit_transform(y_test)
print('ohe_y_train shape:', ohe_y_train.shape)
print('ohe_y_test shape:', ohe_y_test.shape)




