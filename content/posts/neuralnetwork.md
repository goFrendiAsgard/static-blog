---
title: "Neural Network From Scratch"
date: 2019-07-31T07:48:41+07:00
categories:
- Machine Learning
tags:
- Macine Learning
---
# Implementation with Tensorflow

TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.

Unlike `sklearn.neural_network`, tensorflow give us more freedom to set up our neural-network.

The following implementation was taken directly from https://www.tensorflow.org/tutorials/keras/basic_classification

First of all, let's try to import tensorflow and keras (which is now also part of tensorflow)

## Importing Tensorflow and Keras


```python
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
```

## Explore the dataset

We will try to perform classification task on mnist's dataset (http://yann.lecun.com/exdb/mnist/). The dataset contains of `70000` gray-scale images. Each image has `28 x 28` dimension and belong to one (and only one) of the following 10 classes:


```python
class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
```

Now, let's download `fashion_mnist` dataset from `keras.datasets` and split them into `train` and `test` set. By default, the dataset contains of `60000` training set and `10000` test set.


```python
mnist = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

```

Let's explore the data a little bit

Here is a bit information about our `train_labels`. It is a one-dimension array with 60000 elements


```python
train_labels
```




    array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)




```python
train_labels.shape
```




    (60000,)



Now, let's explore our `train_images`


```python
train_images
```




    array([[[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           ...,
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]]], dtype=uint8)




```python
train_images.shape
```




    (60000, 28, 28)



Just to make sure, let's see our first image and label in detail


```python
index = 0
label = train_labels[index]
image = train_images[index]

print("label: ", label) # this is the first train_labels
print("which is: ", class_names[label]) # use our pre-defined class_names to get textual representation of the label
plt.figure()
plt.imshow(image) # if you just want to see the matrix representation of the image, use `image` instead
plt.show()

```

    label:  5
    which is:  five



![png](neuralnetwork_files/neuralnetwork_14_1.png)


## Configuring the neural network model

Finally, let's build our neural network.

First of all, we define 3 layers here:

* flatten layer with input_shape = 28x28: This one will transform our 2 dimensional matrix into 1 dimensional matrix (or a vector). The output of this layer will be an array with 784 elements
* dense layer containing 128 neuron with relu activation: This one will create a layer containing 128 neuron. Each of them is connected to the output of our previous layer (an array containing 784 elements). Each neuron activation is depending on `relu` function (https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
* dense layer containing 10 neuron with softmax activation: Finally, since we have 10 classes, it is natural to have 10 neuron in our output layer. Each neuron should show us how probable is an image belong to a particular class. Finally, we will use softmax to return the prediction result (https://en.wikipedia.org/wiki/Softmax_function)

After defining the layers, we need to define our optimizer, loss function, and metrics:

* optimizer: How to optimize
* loss function: How to calculate error
* metrics: How to measure the quality of the network


```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


```

## Train the neural network model


```python
model.fit(train_images, train_labels, epochs=100)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```

    Train on 60000 samples
    Epoch 1/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 9.4768 - accuracy: 0.4109
    Epoch 2/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 8.0383 - accuracy: 0.5005
    Epoch 3/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 7.2688 - accuracy: 0.5484
    Epoch 4/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 7.1353 - accuracy: 0.5569
    Epoch 5/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 6.9730 - accuracy: 0.5669
    Epoch 6/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 6.9570 - accuracy: 0.5681
    Epoch 7/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 6.6057 - accuracy: 0.5896
    Epoch 8/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 6.1260 - accuracy: 0.6193
    Epoch 9/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 5.8815 - accuracy: 0.6345
    Epoch 10/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 5.6206 - accuracy: 0.6507
    Epoch 11/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 4.6773 - accuracy: 0.7093
    Epoch 12/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 4.5473 - accuracy: 0.7174
    Epoch 13/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 4.5345 - accuracy: 0.7182
    Epoch 14/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 4.4753 - accuracy: 0.7219
    Epoch 15/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.5269 - accuracy: 0.7186
    Epoch 16/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.4178 - accuracy: 0.7256
    Epoch 17/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 4.4035 - accuracy: 0.7265
    Epoch 18/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.3175 - accuracy: 0.7318
    Epoch 19/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 4.3240 - accuracy: 0.7314
    Epoch 20/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.2989 - accuracy: 0.7329
    Epoch 21/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 4.1754 - accuracy: 0.7407
    Epoch 22/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 4.3550 - accuracy: 0.7295
    Epoch 23/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 4.2358 - accuracy: 0.7369
    Epoch 24/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.2346 - accuracy: 0.7370
    Epoch 25/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 4.1078 - accuracy: 0.7448
    Epoch 26/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.1685 - accuracy: 0.7412
    Epoch 27/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.2378 - accuracy: 0.7368
    Epoch 28/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.2262 - accuracy: 0.7375
    Epoch 29/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.1886 - accuracy: 0.7399
    Epoch 30/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.1263 - accuracy: 0.7438
    Epoch 31/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.2394 - accuracy: 0.7368
    Epoch 32/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.0048 - accuracy: 0.7514
    Epoch 33/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.1238 - accuracy: 0.7440
    Epoch 34/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.1525 - accuracy: 0.7421
    Epoch 35/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.1032 - accuracy: 0.7453
    Epoch 36/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.1175 - accuracy: 0.7443
    Epoch 37/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.3530 - accuracy: 0.7297
    Epoch 38/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.1691 - accuracy: 0.7412
    Epoch 39/100
    60000/60000 [==============================] - 5s 86us/sample - loss: 4.0590 - accuracy: 0.7480
    Epoch 40/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 4.0694 - accuracy: 0.7473
    Epoch 41/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 4.0510 - accuracy: 0.7485
    Epoch 42/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 4.0803 - accuracy: 0.7467
    Epoch 43/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 4.2356 - accuracy: 0.7370
    Epoch 44/100
    60000/60000 [==============================] - 5s 88us/sample - loss: 4.1049 - accuracy: 0.7452
    Epoch 45/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 4.0528 - accuracy: 0.7484
    Epoch 46/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 4.0291 - accuracy: 0.7498
    Epoch 47/100
    60000/60000 [==============================] - 5s 88us/sample - loss: 3.9522 - accuracy: 0.7547
    Epoch 48/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 3.9992 - accuracy: 0.7517
    Epoch 49/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.0291 - accuracy: 0.7499
    Epoch 50/100
    60000/60000 [==============================] - 6s 96us/sample - loss: 3.9588 - accuracy: 0.7542
    Epoch 51/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 3.9611 - accuracy: 0.7541
    Epoch 52/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 4.0488 - accuracy: 0.7487
    Epoch 53/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 3.9929 - accuracy: 0.7522
    Epoch 54/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.9734 - accuracy: 0.7533
    Epoch 55/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 4.0326 - accuracy: 0.7496
    Epoch 56/100
    60000/60000 [==============================] - 6s 95us/sample - loss: 4.0214 - accuracy: 0.7504
    Epoch 57/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.0237 - accuracy: 0.7502
    Epoch 58/100
    60000/60000 [==============================] - 6s 95us/sample - loss: 4.0089 - accuracy: 0.7511
    Epoch 59/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.8984 - accuracy: 0.7580
    Epoch 60/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.9781 - accuracy: 0.7530
    Epoch 61/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 3.9318 - accuracy: 0.7560
    Epoch 62/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.1118 - accuracy: 0.7448
    Epoch 63/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.0048 - accuracy: 0.7514
    Epoch 64/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 4.0615 - accuracy: 0.7479
    Epoch 65/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 3.9387 - accuracy: 0.7555
    Epoch 66/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.9485 - accuracy: 0.7548
    Epoch 67/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 3.9242 - accuracy: 0.7564
    Epoch 68/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 3.8543 - accuracy: 0.7608
    Epoch 69/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.9845 - accuracy: 0.7527
    Epoch 70/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.0102 - accuracy: 0.7510
    Epoch 71/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.0308 - accuracy: 0.7498
    Epoch 72/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.8557 - accuracy: 0.7606
    Epoch 73/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.8924 - accuracy: 0.7584
    Epoch 74/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 4.0663 - accuracy: 0.7476
    Epoch 75/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.8677 - accuracy: 0.7599
    Epoch 76/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 4.0528 - accuracy: 0.7485
    Epoch 77/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 3.9462 - accuracy: 0.7551
    Epoch 78/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 3.8967 - accuracy: 0.7581
    Epoch 79/100
    60000/60000 [==============================] - 5s 90us/sample - loss: 4.0119 - accuracy: 0.7510
    Epoch 80/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 3.8886 - accuracy: 0.7587
    Epoch 81/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 3.8775 - accuracy: 0.7594
    Epoch 82/100
    60000/60000 [==============================] - 5s 86us/sample - loss: 3.7995 - accuracy: 0.7642
    Epoch 83/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 3.7972 - accuracy: 0.7643
    Epoch 84/100
    60000/60000 [==============================] - 5s 86us/sample - loss: 3.9107 - accuracy: 0.7572
    Epoch 85/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 3.8611 - accuracy: 0.7604
    Epoch 86/100
    60000/60000 [==============================] - 5s 86us/sample - loss: 3.9305 - accuracy: 0.7560
    Epoch 87/100
    60000/60000 [==============================] - 5s 86us/sample - loss: 3.8945 - accuracy: 0.7583
    Epoch 88/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 3.8754 - accuracy: 0.7595
    Epoch 89/100
    60000/60000 [==============================] - 5s 87us/sample - loss: 3.8740 - accuracy: 0.7596
    Epoch 90/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.9132 - accuracy: 0.7571
    Epoch 91/100
    60000/60000 [==============================] - 6s 93us/sample - loss: 3.8803 - accuracy: 0.7591
    Epoch 92/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 3.9693 - accuracy: 0.7537
    Epoch 93/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 4.0703 - accuracy: 0.7474
    Epoch 94/100
    60000/60000 [==============================] - 6s 94us/sample - loss: 3.8868 - accuracy: 0.7588
    Epoch 95/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 3.8406 - accuracy: 0.7616
    Epoch 96/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 3.7705 - accuracy: 0.7660
    Epoch 97/100
    60000/60000 [==============================] - 6s 92us/sample - loss: 3.8150 - accuracy: 0.7632
    Epoch 98/100
    60000/60000 [==============================] - 5s 92us/sample - loss: 3.8774 - accuracy: 0.7593
    Epoch 99/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 3.9108 - accuracy: 0.7573
    Epoch 100/100
    60000/60000 [==============================] - 5s 91us/sample - loss: 3.9316 - accuracy: 0.7560
    10000/10000 [==============================] - 1s 55us/sample - loss: 3.8109 - accuracy: 0.7635
    Test accuracy: 0.7635


## Prediction


```python
predictions = model.predict(test_images)
np.argmax(predictions[0])

index = 0
prediction_label = np.argmax(predictions[0])
target_label = test_labels[index]
image = test_images[index]

print("target label: ", target_label, class_names[target_label])
print("prediction label: ", prediction_label, class_names[prediction_label])
plt.figure()
plt.imshow(image) # if you just want to see the matrix representation of the image, use `image` instead
plt.show()

```

    target label:  7 seven
    prediction label:  7 seven



![png](neuralnetwork_files/neuralnetwork_20_1.png)

