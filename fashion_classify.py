import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.keras.__version__)


fs = keras.datasets.fashion_mnist
(tr1,tr2), (te1,te2) = fs.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

tr1.shape
len(tr2)
tr2
te1.shape
len(te1)


#plt.figure()
#plt.imshow(tr1[6])
#plt.colorbar()
#plt.grid(False)

tr1 = tr1/255
te1 = te1/255


plt.figure(figsize=(5,5))
for i in range(25):
     plt.subplot(5,5,i+1)
     plt.xticks([])
     plt.yticks([])
     plt.grid(False)
     plt.imshow(tr1[i],cmap = plt.cm.binary)
     plt.xlabel (class_names[tr2[i]]) 
      
#########model building

model = keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(128,activation ='relu'),
        keras.layers.Dense(10,activation='softmax')])

 

###############model compile
model.compile(optimizer=tf.train.AdamOptimizer(),loss ='sparse_categorical_crossentropy',metrics= ['accuracy'])


#######train the model

model.fit(tr1,tr2, epochs=5)



#####evaluate on test data

te_l,te_ac= model.evaluate(te1,te2)

print(te_ac)


##########model prediction using trained data

pred = model.predict(te1)

pred[0]


#########highest confidence value

np.argmax(pred[0])
te2[0]
