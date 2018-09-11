#################text classification

import tensorflow as tf
from tensorflow import keras

data = keras.datasets.imdb 

(tr1,tr2), (te1,te2) = data.load_data(num_words=10000)

tr1.shape,te1.shape


print("tr1:{},tr2:{}".format(len(tr1),len(tr2)))

print(tr1[0])

#####convert integers to text back
w_i = data.get_word_index()

x=w_i.items()
print(x)




w_i = {k:(v+3) for k,v in w_i.items()} 
w_i["<PAD>"] = 0
w_i["<START>"] = 1
w_i["<UNK>"] = 2  # unknown
w_i["<UNUSED>"] = 3

rev_w_i = dict([(value, key) for (key, value) in w_i.items()])

def decode_review(text):
    return ' '.join([rev_w_i.get(i, '?') for i in text])


decode_review(tr1[0])


#### convert integers to tensors using padding method
tr1= keras.preprocessing.sequence.pad_sequences(tr1,value=w_i["<PAD>"],padding='post',maxlen =256)
te1= keras.preprocessing.sequence.pad_sequences(te1,value=w_i["<PAD>"],padding='post',maxlen =256)


len(tr1[0])
print(tr1[0])

############build a model

voc_size = 10000

model =keras.Sequential()
model.add(keras.layers.Embedding(voc_size,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation = "relu"))
model.add(keras.layers.Dense(1,activation="sigmoid"))
model.summary()

#########compile the model
model.compile(optimizer=tf.train.AdamOptimizer(),loss ='binary_crossentropy',metrics= ['accuracy'])

x_val = tr1[:10000]
partial_x_train = tr1[10000:]

y_val = tr2[:10000]
partial_y_train = tr2[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

#############evaluate ther model

results = model.evaluate(te1,te2)
print(results)

################graph of accuarcy and loss overtime


hist_dic =history.history
hist_dic.keys()

############################plot the graph

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure
acc_values = hist_dict['acc']
val_acc_values = hist_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
