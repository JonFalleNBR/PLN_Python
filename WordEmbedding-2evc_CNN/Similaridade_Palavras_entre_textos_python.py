A simple real-world data for this demonstration is obtained from the movie review corpus provided by nltk (Pang & Lee, 2004). The first two reviews from the positive set and the negative set are selected. Then the first sentence of these four reviews is selected. We can first define 4 documents in Python as:

python
Copiar
Editar
d1 = "plot: two teen couples go to a church party, drink and then drive."
d2 = "films adapted from comic books have had plenty of success , whether they're about superheroes ( batman , superman , spawn ) , or geared toward kids ( casper ) or the arthouse crowd ( ghost world ) , but there's never really been a comic book like from hell before ."
d3 = "every now and then a movie comes along from a suspect studio , with every indication that it will be a stinker , and to everybody's surprise ( perhaps even the studio ) the film becomes a critical darling."
d4 = "damn that y2k bug."
documents = [d1, d2, d3, d4]
a. Preprocessing with nltk
The default functions of CountVectorizer and TfidfVectorizer in scikit-learn detect word boundary and remove punctuations automatically. However, if we want to do stemming or lemmatization, we need to customize certain parameters in CountVectorizer and TfidfVectorizer. Doing this overrides the default tokenization setting, which means that we have to customize tokenization, punctuation removal, and turning terms to lower case altogether.

Normalize by stemming:
python
Copiar
Editar
import nltk, string, numpy
nltk.download('punkt')  # first-time use only
stemmer = nltk.stem.porter.PorterStemmer()

def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def StemNormalize(text):
    return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
Normalize by lemmatization:
python
Copiar
Editar
nltk.download('wordnet')  # first-time use only
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
If we want more meaningful terms in their dictionary forms, lemmatization is preferred.

b. Turn text into vectors of term frequency:
python
Copiar
Editar
from sklearn.feature_extraction.text import CountVectorizer
LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
LemVectorizer.fit_transform(documents)
Normalized (after lemmatization) text in the four documents is tokenized, and each term is indexed:

python
Copiar
Editar
print(LemVectorizer.vocabulary_)
Output:

python
Copiar
Editar
{'spawn': 29, 'crowd': 11, 'casper': 5, 'church': 6, 'hell': 20, 'comic': 8, 'superheroes': 33, 'superman': 34, 'plot': 27, 'movie': 24, 'book': 3, 'suspect': 36, 'film': 17, 'party': 25, 'darling': 13, 'really': 28, 'teen': 37, 'everybodys': 16, 'damn': 12, 'batman': 2, 'couple': 9, 'drink': 14, 'like': 23, 'geared': 18, 'studio': 31, 'plenty': 26, 'surprise': 35, 'world': 39, 'come': 7, 'bug': 4, 'kid': 22, 'ghost': 19, 'arthouse': 1, 'y2k': 40, 'stinker': 30, 'success': 32, 'drive': 15, 'theyre': 38, 'indication': 21, 'critical': 10, 'adapted': 0}
And we have the tf matrix:

python
Copiar
Editar
tf_matrix = LemVectorizer.transform(documents).toarray()
print(tf_matrix)
Output:

python
Copiar
Editar
[[0 0 0 0 0 0 1 0 0 1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0]
 [1 1 1 2 0 1 0 0 2 0 0 1 0 0 0 0 0 1 1 1 1 0 1 1 0 0 1 0 1 1 0 0 1 1 1 0 0 0 1 1 0]
 [0 0 0 0 0 0 0 1 0 0 1 0 0 1 0 0 1 1 0 0 0 1 0 0 1 0 0 0 0 0 1 2 0 0 0 1 1 0 0 0 0]
 [0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1]]
Check the shape:

python
Copiar
Editar
tf_matrix.shape
Output:

python
Copiar
Editar
(4, 41)
c. Calculate idf and turn tf matrix to tf-idf matrix
Get idf:

python
Copiar
Editar
from sklearn.feature_extraction.text import TfidfTransformer
tfidfTran = TfidfTransformer(norm="l2")
tfidfTran.fit(tf_matrix)
print(tfidfTran.idf_)
Output:

python
Copiar
Editar
[1.91629073 1.91629073 1.91629073 1.91629073 1.91629073 ...]  # truncated for readability
Corroborate with manual calculation:

python
Copiar
Editar
import math
def idf(n, df):
    return math.log((n + 1.0) / (df + 1.0)) + 1

print("The idf for terms that appear in one document:", idf(4, 1))
print("The idf for terms that appear in two documents:", idf(4, 2))
Output:

python
Copiar
Editar
The idf for terms that appear in one document: 1.91629073187
The idf for terms that appear in two documents: 1.51082562377
d. Get the tf-idf matrix (4 by 41):
python
Copiar
Editar
tfidf_matrix = tfidfTran.transform(tf_matrix)
print(tfidf_matrix.toarray())
e. Get the pairwise similarity matrix (n by n):
python
Copiar
Editar
cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
print(cos_similarity_matrix)
Output:

python
Copiar
Editar
[[1.         0.         0.         0.        ]
 [0.         1.         0.03264186 0.        ]
 [0.         0.03264186 1.         0.        ]
 [0.         0.         0.         1.        ]]
f. Use TfidfVectorizer instead:
python
Copiar
Editar
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)