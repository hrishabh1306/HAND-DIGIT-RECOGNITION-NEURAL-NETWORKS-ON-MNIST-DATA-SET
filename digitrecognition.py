import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# mnist=tf.keras.datasets.mnist
# (x_train,y_train),(test_x,test_y)=mnist.load_data()

# # normalising
# x_train=tf.keras.utils.normalize(x_train,axis=1)
# test_x=tf.keras.utils.normalize(test_x,axis=1)

# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(10,activation='softmax'))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# model.fit(x_train,y_train,epochs=1)

# model.save('handwritten.model')

model=tf.keras.models.load_model('handwritten.model')

# loss , accuracy= model.evaluate(test_x , test_y)

# print(loss)
# print(accuracy)

image_number=1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img=cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"THIS DIGIT IS PROBABLY A {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print("error")
    finally:
        image_number=image_number+1
