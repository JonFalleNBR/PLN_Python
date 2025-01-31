from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['review2'], df['sentiment'], test_size=0.2, random_state=0)
print('X_train shape:',X_train.shape)
print('y_train shape:',y_train.shape)
print('X_test shape:',X_test.shape)
print('y_test shape:',y_test.shape)

#saida
##X_train shape: (40000,)
##y_train shape: (40000,)
##X_test shape: (10000,)
##y_test shape: (10000,)

%%time
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
tk_X_train = tokenizer.texts_to_sequences(X_train)
tk_X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1
print(X_train[0])
print(tk_X_train[0])

#saida
# one review mention watch oz episod hook they right exact happen me.
# the first thing struck oz brutal unflinch scene violenc set right word go trust
# show faint heart timid this show pull punch regard drug sex violenc it hardcor classic use word.
# it call oz nicknam given oswald maximum secur state penitentari it focus main emerald citi
# experiment section prison cell glass front face inward privaci high agenda em citi home many..aryan
# muslim gangsta latino christian italian irish scuffl death stare dodgi deal shadi agreement never far away.
# would say main appeal show due fact goe show would n't dare forget pretti pictur paint mainstream audienc
# forget charm forget romanc oz n't mess around the first episod ever saw struck nasti surreal
# could n't say readi watch develop tast oz got accustom high level graphic violenc not violenc injustic
# crook guard sold nickel inmat kill order get away well manner middl class inmat turn prison bitch
# due lack street skill prison experi watch oz may becom comfort uncomfort view .that get touch darker side

# [136, 1, 1230, 419, 2, 889, 1267, 304, 1, 698, 790, 951, 10, 49, 161, 899, 85, 1862, 64, 118, 4548, 333, 644]
# CPU times: user 6.25 s, sys: 4.98 ms, total: 6.25 s
# Wall time: 6.26 s

for word in ['movi','film','the','one']:
     print('{}: {}'.format(word, tokenizer.word_index[word]))

     #saida
     # movi: 2
     # film: 3
     # the: 4
     # one: 7

print('Max size:',max([len(i) for i in tk_X_train]))

#saida
#Max size: 1374
from keras.preprocessing.sequence import pad_sequences
maxlen = max([len(i) for i in tk_X_train])
tk_pad_X_train = pad_sequences(tk_X_train, padding='post', maxlen=maxlen)
tk_pad_X_test = pad_sequences(tk_X_test, padding='post', maxlen=maxlen)
print(tk_pad_X_train[0, :])

#saida
#[136    1 1230 ...    0    0    0]

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
ohe_y_train = lb.fit_transform(y_train)
ohe_y_test = lb.fit_transform(y_test)
print('ohe_y_train shape:',ohe_y_train.shape)
print('ohe_y_test shape:',ohe_y_test.shape)

#saida
#ohe_y_train shape: (40000, 1)
# ohe_y_test shape: (10000, 1)

Podemos, agora, utilizar a classe Embedding do Keras para incorporar essa nova codificação.
Essa camada Embedding pega os inteiros calculados anteriormente e os mapeia para um vetor denso de
incorporação. Mas, antes, vamos fazer as importações necessárias, conforme a seguir.


fro m keras.models
import Sequential
fro m keras.layers
import Embedding, Flatten, Dense, GlobalMaxPool1D, Dropout, Conv1D, GlobalMaxPooling1D
fro m keras.optimizers import Adam


E definir os hiperparâmetros do treinamento:


NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()

#Agora, sim, vamos definir uma arquitetura de rede neural:

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#saida

# Model: "sequential_1"
# __________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, 1374, 50)          3695700
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 68700)             0
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                687010
# _________________________________________________________________
# dense_3 (Dense)              (None, 1)                 11
# =================================================================
# Total params: 4,382,721
# Trainable params: 4,382,721
# Non-trainable params: 0


	%%time
history = model.fit(tk_pad_X_train, ohe_y_train,
                    epochs=NB_EPOCH,
                    verbose=1,
                    validation_data=(tk_pad_X_test, ohe_y_test),
                    batch_size=BATCH_SIZE)


# Epoch 1/20
# 313/313 [==============================] - 16s 51ms/step - loss: 0.6966 - accuracy: 0.4974 - val_loss: 0.6931 - val_accuracy: 0.5035
# Epoch 2/20
# 313/313 [==============================] - 16s 51ms/step - loss: 0.6931 - accuracy: 0.4997 - val_loss: 0.6931 - val_accuracy: 0.4965
# Epoch 3/20
# 313/313 [==============================] - 16s 50ms/step - loss: 0.6931 - accuracy: 0.4997 - val_loss: 0.6931 - val_accuracy: 0.5036
# Epoch 4/20
# 313/313 [==============================] - 16s 51ms/step - loss: 0.6930 - accuracy: 0.4990 - val_loss: 0.6931 - val_accuracy: 0.4965
# Epoch 5/20
# 313/313 [==============================] - 16s 50ms/step - loss: 0.6929 - accuracy: 0.4989 - val_loss: 0.6931 - val_accuracy: 0.4965
# Epoch 6/20
# 313/313 [==============================] - 16s 50ms/step - loss: 0.6929 - accuracy: 0.5009 - val_loss: 0.6931 - val_accuracy: 0.4965
# Epoch 7/20
# 313/313 [==============================] - 15s 49ms/step - loss: 0.5487 - accuracy: 0.6553 - val_loss: 0.3094 - val_accuracy: 0.8735
# Epoch 8/20
# 313/313 [==============================] - 16s 50ms/step - loss: 0.2216 - accuracy: 0.9151 - val_loss: 0.2807 - val_accuracy: 0.8847
# Epoch 9/20
# 313/313 [==============================] - 16s 50ms/step - loss: 0.1095 - accuracy: 0.9659 - val_loss: 0.3252 - val_accuracy: 0.8799
# Epoch 10/20
# 313/313 [==============================] - 15s 49ms/step - loss: 0.0341 - accuracy: 0.9929 - val_loss: 0.4032 - val_accuracy: 0.8763
# Epoch 11/20
# 313/313 [==============================] - 15s 49ms/step - loss: 0.0075 - accuracy: 0.9991 - val_loss: 0.4808 - val_accuracy: 0.8752
# Epoch 12/20
# 313/313 [==============================] - 16s 50ms/step - loss: 0.0023 - accuracy: 0.9999 - val_loss: 0.5383 - val_accuracy: 0.8762
# Epoch 13/20
# 313/313 [==============================] - 16s 50ms/step - loss: 0.0010 - accuracy: 1.0000 - val_loss: 0.5811 - val_accuracy: 0.8757
# Epoch 14/20
# 313/313 [==============================] - 15s 49ms/step - loss: 6.0870e-04 - accuracy: 1.0000 - val_loss: 0.6149 - val_accuracy: 0.8743
# Epoch 15/20
# 313/313 [==============================] - 15s 49ms/step - loss: 3.9885e-04 - accuracy: 1.0000 - val_loss: 0.6414 - val_accuracy: 0.8748
# Epoch 16/20
# 313/313 [==============================] - 15s 49ms/step - loss: 2.7970e-04 - accuracy: 1.0000 - val_loss: 0.6655 - val_accuracy: 0.8751
# Epoch 17/20
# 313/313 [==============================] - 16s 50ms/step - loss: 2.0181e-04 - accuracy: 1.0000 - val_loss: 0.6877 - val_accuracy: 0.8754
# Epoch 18/20
# 313/313 [==============================] - 15s 49ms/step - loss: 1.5003e-04 - accuracy: 1.0000 - val_loss: 0.7090 - val_accuracy: 0.8748
# Epoch 19/20
# 313/313 [==============================] - 15s 49ms/step - loss: 1.1429e-04 - accuracy: 1.0000 - val_loss: 0.7290 - val_accuracy: 0.8750
# Epoch 20/20
# 313/313 [==============================] - 15s 49ms/step - loss: 8.8089e-05 - accuracy: 1.0000 - val_loss: 0.7458 - val_accuracy: 0.8746


#Vamos calcular a acurácia:

loss, accuracy = model.evaluate(tk_pad_X_test, ohe_y_test, verbose=False)
print("Acurácia do Teste:  {:.4f}".format(accuracy))

#saida
#Acurácia do Teste:  0.8746

import matplotlib.pyplot as plt
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training accuracy')
    plt.plot(x, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
plot_history(history)


# Conforme visto acima, o treinamento chegou próximo de 100% para os dados de treino,
# mas não para os dados de teste. Isso pode ser pelo fato de não termos adicionado uma camada Dropout.

# Veja que o resultado não alterou após uma quantidade de épocas, logo, vamos limitar a execução para 10 épocas.

# Além disso, vamos adicionar à nossa rede uma camada Max Pooling após o Embedding,
# como uma forma de reduzir o tamanho dos vetores de entrada.

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=max_len))
model.add(GlobalMaxPool1D())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=OPTIMIZER, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Model: "sequential_4"
# __________________________________________________________________
# Layer (type)                 Output Shape              Param #
# ==================================================================
# embedding_4 (Embedding)      (None, 1374, 50)          3695700
# __________________________________________________________________
# global_max_pooling1d_2 (Glob (None, 50)                0
# __________________________________________________________________
# dense_7 (Dense)              (None, 16)                816
# __________________________________________________________________
# dropout (Dropout)            (None, 16)                0
# __________________________________________________________________
# dense_8 (Dense)              (None, 1)                 17
# ==================================================================
# Total params: 3,696,533
# Trainable params: 3,696,533
# Non-trainable params: 0

#Vamos treinar e ver o resultado.

%%time
history = model.fit(tk_pad_X_train, ohe_y_train,
                    epochs=NB_EPOCH,
                    verbose=1,
                    validation_data=(tk_pad_X_test, ohe_y_test),
                    batch_size=BATCH_SIZE)

# Início do treinamento, 1ª época
Epoch 1/10
313/313 [==============================] - 16s 51ms/step - loss: 0.5674 - accuracy: 0.7181 - val_loss: 0.3804 - val_accuracy: 0.8452
# O modelo começa o treinamento. Durante a 1ª época:
# - A perda (loss) é 0.5674, indicando que o modelo está cometendo erros.
# - A acurácia nos dados de treino (accuracy) é de 71.81%.
# - Nos dados de validação, a perda (val_loss) é 0.3804 e a acurácia (val_accuracy) é 84.52%.
# Isso mostra que o modelo está se saindo bem nos dados de validação também.

# 2ª época
Epoch 2/10
313/313 [==============================] - 16s 52ms/step - loss: 0.3586 - accuracy: 0.8548 - val_loss: 0.3159 - val_accuracy: 0.8669
# Na 2ª época, o modelo melhora:
# - A perda diminui para 0.3586, o que indica que está cometendo menos erros.
# - A acurácia nos dados de treino sobe para 85.48%.
# - Nos dados de validação, a perda diminui para 0.3159 e a acurácia aumenta para 86.69%.
# Isso indica que o modelo está aprendendo e melhorando.

# 3ª época
Epoch 3/10
313/313 [==============================] - 16s 52ms/step - loss: 0.2883 - accuracy: 0.8917 - val_loss: 0.2983 - val_accuracy: 0.8738
# Na 3ª época, o modelo continua a melhorar:
# - A perda nos dados de treino diminui ainda mais para 0.2883.
# - A acurácia nos dados de treino chega a 89.17%.
# - Nos dados de validação, a perda é 0.2983 e a acurácia sobe para 87.38%, o que é muito bom.

# 4ª época
Epoch 4/10
313/313 [==============================] - 16s 52ms/step - loss: 0.2427 - accuracy: 0.9157 - val_loss: 0.2947 - val_accuracy: 0.8784
# Na 4ª época, o modelo tem mais progresso:
# - A perda de treino diminui para 0.2427 e a acurácia sobe para 91.57%.
# - Nos dados de validação, a perda é 0.2947 e a acurácia é de 87.84%.
# O modelo está ficando cada vez mais preciso.

# 5ª época
Epoch 5/10
313/313 [==============================] - 17s 53ms/step - loss: 0.2009 - accuracy: 0.9341 - val_loss: 0.3017 - val_accuracy: 0.8773
# Na 5ª época, o modelo continua a melhorar, mas com uma leve diminuição na acurácia de validação:
# - A perda de treino é 0.2009 e a acurácia de treino é 93.41%.
# - Nos dados de validação, a perda sobe ligeiramente para 0.3017 e a acurácia diminui para 87.73%.

# 6ª época
Epoch 6/10
313/313 [==============================] - 17s 55ms/step - loss: 0.1652 - accuracy: 0.9491 - val_loss: 0.3171 - val_accuracy: 0.8768
# Na 6ª época, o modelo tem uma boa performance nos dados de treino:
# - A perda diminui para 0.1652 e a acurácia de treino sobe para 94.91%.
# - Nos dados de validação, a perda aumenta para 0.3171 e a acurácia diminui para 87.68%.
# O modelo ainda está melhorando, mas começa a mostrar sinais de overfitting (quando a acurácia de validação não melhora tanto).

# 7ª época
Epoch 7/10
313/313 [==============================] - 17s 53ms/step - loss: 0.1370 - accuracy: 0.9599 - val_loss: 0.3386 - val_accuracy: 0.8755
# A 7ª época mostra mais melhorias nos dados de treino, mas a validação tem uma leve queda:
# - A perda de treino diminui para 0.1370 e a acurácia chega a 95.99%.
# - Nos dados de validação, a perda sobe para 0.3386 e a acurácia diminui para 87.55%.
# A perda de validação começa a aumentar, o que pode ser um sinal de que o modelo está começando a se ajustar demais aos dados de treino.

# 8ª época
Epoch 8/10
313/313 [==============================] - 17s 53ms/step - loss: 0.1124 - accuracy: 0.9703 - val_loss: 0.3649 - val_accuracy: 0.8736
# Na 8ª época, a acurácia de treino continua subindo:
# - A perda diminui ainda mais para 0.1124 e a acurácia de treino atinge 97.03%.
# - Nos dados de validação, a perda sobe para 0.3649 e a acurácia diminui para 87.36%.
# A validação está mostrando uma queda na acurácia, o que indica que o modelo está começando a se ajustar em excesso aos dados de treino.

# 9ª época
Epoch 9/10
313/313 [==============================] - 16s 52ms/step - loss: 0.0921 - accuracy: 0.9768 - val_loss: 0.3953 - val_accuracy: 0.8739
# Na 9ª época, o modelo continua muito bem nos dados de treino:
# - A perda diminui para 0.0921 e a acurácia de treino é de 97.68%.
# - Nos dados de validação, a perda aumenta para 0.3953 e a acurácia é de 87.39%.
# O modelo ainda está melhorando nos dados de treino, mas a validação continua a ter uma leve queda.

# 10ª época
Epoch 10/10
313/313 [==============================] - 16s 52ms/step - loss: 0.0729 - accuracy: 0.9832 - val_loss: 0.4180 - val_accuracy: 0.8723
# No final do treinamento, o modelo atinge:
# - Perda de treino de 0.0729 e acurácia de treino de 98.32%.
# - Nos dados de validação, a perda chega a 0.4180 e a acurácia é de 87.23%.
# Embora a acurácia de treino tenha subido, a acurácia de validação tem uma leve diminuição, indicando que o modelo pode ter começado a overfitar.


#Vamos calcular a acurácia novamente:

loss, accuracy = model.evaluate(tk_pad_X_test, ohe_y_test, verbose=False)
print("Acurácia do Teste:  {:.4f}".format(accuracy))

#saida
#Acurácia do Teste:  0.8723

#plotando o resultado
plot_history(history)



