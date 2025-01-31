Vamos experimentar o uso de uma CNN para o nosso problema:

Código	NB_EPOCH = 3
BATCH_SIZE = 128
VERBOSE = 1
OPTIMIZER = Adam()
Após definir os hiperparâmetros, vamos definir a rede neural CNN:

Código	model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_len))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dropout(rate=0.4))
model.add(Dense(32, activation='relu'))
model.add(Dropout(rate=0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
Resultado	Model: "sequential_13"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_13 (Embedding)     (None, 1374, 128)         9460992
_________________________________________________________________
conv1d_11 (Conv1D)           (None, 1370, 64)          41024
_________________________________________________________________
max_pooling1d_5 (MaxPooling1 (None, 685, 64)           0
_________________________________________________________________
flatten_5 (Flatten)          (None, 43840)             0
_________________________________________________________________
dropout_16 (Dropout)         (None, 43840)             0
_________________________________________________________________
dense_24 (Dense)             (None, 32)                1402912
_________________________________________________________________
dropout_17 (Dropout)         (None, 32)                0
_________________________________________________________________
dense_25 (Dense)             (None, 1)                 33
=================================================================
Total params: 10,904,961
Trainable params: 10,904,961
Non-trainable params: 0
Agora, o treinamento:

Código	%%time
history = model.fit(tk_pad_X_train, ohe_y_train,
                    epochs=NB_EPOCH,
                    verbose=1,
                    validation_data=(tk_pad_X_test, ohe_y_test),
                    batch_size=BATCH_SIZE)
Resultado	Epoch 1/3
313/313 [==============================] - 47s 152ms/step - loss: 0.4381 - accuracy: 0.7850 - val_loss: 0.2709 - val_accuracy: 0.8882
Epoch 2/3
313/313 [==============================] - 47s 150ms/step - loss: 0.2601 - accuracy: 0.9154 - val_loss: 0.2674 - val_accuracy: 0.8907
Epoch 3/3
313/313 [==============================] - 48s 153ms/step - loss: 0.1906 - accuracy: 0.9366 - val_loss: 0.2868 - val_accuracy: 0.8909
CPU times: user 3min, sys: 5.4 s, total: 3min 5s
Wall time: 2min 23s
Vejamos o resultado:

Código	loss, accuracy = model.evaluate(tk_pad_X_test, ohe_y_test, verbose=False)
print("Acurácia do Teste:  {:.4f}".format(accuracy))
Resultado	Acurácia do Teste: 0.8909
Confira, agora, o gráfico do treinamento a seguir.