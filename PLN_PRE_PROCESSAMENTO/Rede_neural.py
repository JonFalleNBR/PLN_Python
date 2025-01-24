# A partir de agora, adicionaremos ao nosso exemplo redes neurais artificiais para tentar melhorar ainda mais os resultados.
# Para isso, vamos utilizar um framework de criação de redes neurais chamado TensorFlow (criado pela Google).
# Para seu uso, será usado outro framework, chamado Keras, o qual facilita o uso do TensorFlow.

# TensorFlow
# É um sistema para criação e treinamento de redes neurais baseado em gráficos de fluxo de dados.
# Os nós do gráfico representam operações matemáticas, enquanto as bordas do gráfico representam os arrays de dados multidimensionais (tensores) que fluem entre eles.
# Essa arquitetura flexível permite que você implante computação para uma ou mais CPUs ou GPUs em uma área de trabalho, servidor ou dispositivo móvel sem reescrever o código.
# No tutorial a seguir, vamos alterar nosso exemplo para utilizar o processamento de um modelo simples de rede neural.

# Tutorial
# Vamos conferir o tutorial com a utilização do TensorFlow https://www.tensorflow.org/ e do Keras https://keras.io/

# Antes disso, é preciso limitar a quantidade de palavras para 10.000, a fim de que nosso modelo não fique muito complexo,
# o que ocasionaria um uso de memória muito alto. Confira:

%%time
MAX_FEATURES = 10000  # Definindo o número máximo de features (palavras) que serão consideradas
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents='unicode', ngram_range=(1, 3), max_features=MAX_FEATURES)

# Transformando os dados de treino e teste com o TfidfVectorizer
tfidf_x_train = tfidf.fit_transform(X_train)
tfidf_x_test = tfidf.transform(X_test)

# Verificando o tamanho da entrada após o TF-IDF
print('tfidf_x_train shape:', tfidf_x_train.shape)
print('tfidf_x_test shape:', tfidf_x_test.shape)

# Veja que foram limitados a apenas 10.000 features. Ou seja, a camada de entrada da rede neural vai ter a dimensão 10.000.
# Realizando os imports necessários, teremos:

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.models import load_model

%load_ext tensorboard  # Carregando a extensão do TensorBoard para monitorar o treinamento

# Em seguida, defina os hiperparâmetros da rede:

NB_EPOCH = 5  # Número de épocas (quantidade de passagens sobre o conjunto de dados)
BATCH_SIZE = 128  # Tamanho do lote (quantidade de exemplos usados por vez para atualizar os pesos)
VERBOSE = 1  # Verbosidade do treinamento (mostra o progresso)
OPTIMIZER = Adam()  # Usando o otimizador Adam para ajustar os pesos da rede

# Com isso, estamos definindo:
# - número de épocas: quantidade de vezes que os dados de treino serão submetidos à rede;
# - batch size: quantidade de exemplos que serão submetidos à rede a cada passo de treinamento;
# - verbose: indica que queremos acompanhar os resultados de execução a cada época;
# - otimizador: escolhemos o otimizador Adam().

# Os hiperparâmetros acima podem ser variados durante o desenvolvimento do modelo para tentar obter melhores resultados.
# Recomenda-se que, a cada treinamento, os valores de hiperparâmetros utilizados e os resultados sejam salvos para que se possa manter um histórico das tentativas.

# Em seguida, vamos definir a arquitetura da rede neural:

model = Sequential()

# Camada de entrada: 16 neurônios e a forma de entrada será igual ao número de características no conjunto de dados
model.add(Dense(16, input_shape=(tfidf_x_train.shape[1],)))

# Função de ativação 'relu' para a camada escondida
model.add(Activation('relu'))

# Camada de dropout com 50% de neurônios desativados para evitar overfitting
model.add(Dropout(0.5))

# Camada de saída com um neurônio e função de ativação 'sigmoid' para problemas binários (classificação)
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Exibindo um resumo do modelo para visualizar a arquitetura da rede neural
model.summary()

# Observe que criamos uma rede neural com 160 mil parâmetros.

# Em seguida, compilaremos e treinaremos o modelo:

model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

# Callback para monitorar o treinamento no TensorBoard
tbCallBack = TensorBoard(log_dir='logs_pln_nn', histogram_freq=0, write_graph=True)

# Treinando o modelo com os dados de treino
modelo_v1 = model.fit(tfidf_x_train.toarray(), ohe_y_train,
                      batch_size=BATCH_SIZE,
                      epochs=NB_EPOCH,
                      verbose=VERBOSE,
                      validation_split=0.2,  # Usando 20% dos dados de treino para validação
                      callbacks=[tbCallBack])

# Vamos verificar o resultado no TensorBoard:
%tensorboard --logdir logs_pln_nn

# Na próxima figura, vemos o resultado no TensorBoard, uma ferramenta do framework TensorFlow (que é utilizado pelo Keras) para exibir os gráficos de resultado da acurácia e função de perda para os datasets de treino e teste.
# É importante ressaltar que nos interessa mais o resultado do teste, pois é baseado em dados inéditos e que o modelo ainda não conhece.

# Gráficos
# Em seguida, imprimiremos o score dos dados de teste:

loss, accuracy = model.evaluate(tfidf_x_test.toarray(), ohe_y_test, verbose=False)
print("Acurácia do Teste:  {:.4f}".format(accuracy))

# Perceba que conseguimos um resultado um pouco melhor que antes: 89,8%.

# Para finalizar, vamos plotar os gráficos de acurácia e perda da rede:

import matplotlib.pyplot as plt

# Plotando a acurácia durante o treinamento
plt.plot(modelo_v1.history['accuracy'])
plt.plot(modelo_v1.history['val_accuracy'])
plt.title('Acurácia do Modelo')
plt.ylabel('Acurácia')
plt.xlabel('Epoch')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()

# Plotando a perda durante o treinamento
plt.plot(modelo_v1.history['loss'])
plt.plot(modelo_v1.history['val_loss'])
plt.title('Perda do Modelo')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Treino', 'Teste'], loc='upper left')
plt.show()
