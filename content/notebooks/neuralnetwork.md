---
title: "Neural Network From Scratch"
date: 2019-07-31T07:48:41+07:00
categories:
- Machine Learning
tags:
- Macine Learning
---
# The Problem

Artificial neural network is no magic. The main purpose of neural network (and any classical machine learning algorithm) is to find out `magic numbers` in order to find out the best `classifier` or `regressor`.

Suppose we have the following data:

```
age height weight
50  170    75
40  180    ?
```

Now, we want to know the weight of someone whose age is `40`, and height is `180`.

This is a regression problem. Mathematically you can formulate the problem as follow:

$$weight = w1.age + w2.height + w3$$

Now our goal is to find out 3 magic numbers, `w1`, `w2`, and `w3`.

# A simple approach (Non-neural-network)

You might be tempted to solve the equation using pure `algebra`. Since we have single data with three variables (aka: the magic numbers), I think we can't solve the equation with pure algebra.

So, let's solve the using brute-force experiments!!!

## Finding the regressor

First of all, we have to define our `loss function` (how far we are from the target). In our case, the loss function is as follow:

$$error = |actual\_weight - predicted\_weight|$$
$$error = |actual\_weight - (w1.age + w2.height + w3)|$$

Let's implement the loss function in Python:


```python
def loss_function(actual_weight, w1, age, w2, height, w3):
    error = abs(actual_weight - (w1*age + w2*height + w3))
    return error
```

Now, let's try to assign `1`-`100` to `w1`, `w2`, and `w3` in order to get the best numbers for each of them


```python
age = 50.0
height = 170.0
actual_weight = 75.0
best_error = 0 # error threshold, we are only interested for error that is less or equal to 0

for w1 in range(-101, 101):
    for w2 in range(-100, 101):
        for w3 in range(-100, 101):
            error = loss_function(actual_weight, w1, age, w2, height, w3)
            if error <= best_error:
                best_error = error
                print("error:", error, "w1:", w1, "w2:", w2, "w3:", w3)

```

    error: 0.0 w1: -101 w2: 30 w3: 25
    error: 0.0 w1: -100 w2: 30 w3: -25
    error: 0.0 w1: -99 w2: 29 w3: 95
    error: 0.0 w1: -99 w2: 30 w3: -75
    error: 0.0 w1: -98 w2: 29 w3: 45
    error: 0.0 w1: -97 w2: 29 w3: -5
    error: 0.0 w1: -96 w2: 29 w3: -55
    error: 0.0 w1: -95 w2: 28 w3: 65
    error: 0.0 w1: -94 w2: 28 w3: 15
    error: 0.0 w1: -93 w2: 28 w3: -35
    error: 0.0 w1: -92 w2: 27 w3: 85
    error: 0.0 w1: -92 w2: 28 w3: -85
    error: 0.0 w1: -91 w2: 27 w3: 35
    error: 0.0 w1: -90 w2: 27 w3: -15
    error: 0.0 w1: -89 w2: 27 w3: -65
    error: 0.0 w1: -88 w2: 26 w3: 55
    error: 0.0 w1: -87 w2: 26 w3: 5
    error: 0.0 w1: -86 w2: 26 w3: -45
    error: 0.0 w1: -85 w2: 25 w3: 75
    error: 0.0 w1: -85 w2: 26 w3: -95
    error: 0.0 w1: -84 w2: 25 w3: 25
    error: 0.0 w1: -83 w2: 25 w3: -25
    error: 0.0 w1: -82 w2: 24 w3: 95
    error: 0.0 w1: -82 w2: 25 w3: -75
    error: 0.0 w1: -81 w2: 24 w3: 45
    error: 0.0 w1: -80 w2: 24 w3: -5
    error: 0.0 w1: -79 w2: 24 w3: -55
    error: 0.0 w1: -78 w2: 23 w3: 65
    error: 0.0 w1: -77 w2: 23 w3: 15
    error: 0.0 w1: -76 w2: 23 w3: -35
    error: 0.0 w1: -75 w2: 22 w3: 85
    error: 0.0 w1: -75 w2: 23 w3: -85
    error: 0.0 w1: -74 w2: 22 w3: 35
    error: 0.0 w1: -73 w2: 22 w3: -15
    error: 0.0 w1: -72 w2: 22 w3: -65
    error: 0.0 w1: -71 w2: 21 w3: 55
    error: 0.0 w1: -70 w2: 21 w3: 5
    error: 0.0 w1: -69 w2: 21 w3: -45
    error: 0.0 w1: -68 w2: 20 w3: 75
    error: 0.0 w1: -68 w2: 21 w3: -95
    error: 0.0 w1: -67 w2: 20 w3: 25
    error: 0.0 w1: -66 w2: 20 w3: -25
    error: 0.0 w1: -65 w2: 19 w3: 95
    error: 0.0 w1: -65 w2: 20 w3: -75
    error: 0.0 w1: -64 w2: 19 w3: 45
    error: 0.0 w1: -63 w2: 19 w3: -5
    error: 0.0 w1: -62 w2: 19 w3: -55
    error: 0.0 w1: -61 w2: 18 w3: 65
    error: 0.0 w1: -60 w2: 18 w3: 15
    error: 0.0 w1: -59 w2: 18 w3: -35
    error: 0.0 w1: -58 w2: 17 w3: 85
    error: 0.0 w1: -58 w2: 18 w3: -85
    error: 0.0 w1: -57 w2: 17 w3: 35
    error: 0.0 w1: -56 w2: 17 w3: -15
    error: 0.0 w1: -55 w2: 17 w3: -65
    error: 0.0 w1: -54 w2: 16 w3: 55
    error: 0.0 w1: -53 w2: 16 w3: 5
    error: 0.0 w1: -52 w2: 16 w3: -45
    error: 0.0 w1: -51 w2: 15 w3: 75
    error: 0.0 w1: -51 w2: 16 w3: -95
    error: 0.0 w1: -50 w2: 15 w3: 25
    error: 0.0 w1: -49 w2: 15 w3: -25
    error: 0.0 w1: -48 w2: 14 w3: 95
    error: 0.0 w1: -48 w2: 15 w3: -75
    error: 0.0 w1: -47 w2: 14 w3: 45
    error: 0.0 w1: -46 w2: 14 w3: -5
    error: 0.0 w1: -45 w2: 14 w3: -55
    error: 0.0 w1: -44 w2: 13 w3: 65
    error: 0.0 w1: -43 w2: 13 w3: 15
    error: 0.0 w1: -42 w2: 13 w3: -35
    error: 0.0 w1: -41 w2: 12 w3: 85
    error: 0.0 w1: -41 w2: 13 w3: -85
    error: 0.0 w1: -40 w2: 12 w3: 35
    error: 0.0 w1: -39 w2: 12 w3: -15
    error: 0.0 w1: -38 w2: 12 w3: -65
    error: 0.0 w1: -37 w2: 11 w3: 55
    error: 0.0 w1: -36 w2: 11 w3: 5
    error: 0.0 w1: -35 w2: 11 w3: -45
    error: 0.0 w1: -34 w2: 10 w3: 75
    error: 0.0 w1: -34 w2: 11 w3: -95
    error: 0.0 w1: -33 w2: 10 w3: 25
    error: 0.0 w1: -32 w2: 10 w3: -25
    error: 0.0 w1: -31 w2: 9 w3: 95
    error: 0.0 w1: -31 w2: 10 w3: -75
    error: 0.0 w1: -30 w2: 9 w3: 45
    error: 0.0 w1: -29 w2: 9 w3: -5
    error: 0.0 w1: -28 w2: 9 w3: -55
    error: 0.0 w1: -27 w2: 8 w3: 65
    error: 0.0 w1: -26 w2: 8 w3: 15
    error: 0.0 w1: -25 w2: 8 w3: -35
    error: 0.0 w1: -24 w2: 7 w3: 85
    error: 0.0 w1: -24 w2: 8 w3: -85
    error: 0.0 w1: -23 w2: 7 w3: 35
    error: 0.0 w1: -22 w2: 7 w3: -15
    error: 0.0 w1: -21 w2: 7 w3: -65
    error: 0.0 w1: -20 w2: 6 w3: 55
    error: 0.0 w1: -19 w2: 6 w3: 5
    error: 0.0 w1: -18 w2: 6 w3: -45
    error: 0.0 w1: -17 w2: 5 w3: 75
    error: 0.0 w1: -17 w2: 6 w3: -95
    error: 0.0 w1: -16 w2: 5 w3: 25
    error: 0.0 w1: -15 w2: 5 w3: -25
    error: 0.0 w1: -14 w2: 4 w3: 95
    error: 0.0 w1: -14 w2: 5 w3: -75
    error: 0.0 w1: -13 w2: 4 w3: 45
    error: 0.0 w1: -12 w2: 4 w3: -5
    error: 0.0 w1: -11 w2: 4 w3: -55
    error: 0.0 w1: -10 w2: 3 w3: 65
    error: 0.0 w1: -9 w2: 3 w3: 15
    error: 0.0 w1: -8 w2: 3 w3: -35
    error: 0.0 w1: -7 w2: 2 w3: 85
    error: 0.0 w1: -7 w2: 3 w3: -85
    error: 0.0 w1: -6 w2: 2 w3: 35
    error: 0.0 w1: -5 w2: 2 w3: -15
    error: 0.0 w1: -4 w2: 2 w3: -65
    error: 0.0 w1: -3 w2: 1 w3: 55
    error: 0.0 w1: -2 w2: 1 w3: 5
    error: 0.0 w1: -1 w2: 1 w3: -45
    error: 0.0 w1: 0 w2: 0 w3: 75
    error: 0.0 w1: 0 w2: 1 w3: -95
    error: 0.0 w1: 1 w2: 0 w3: 25
    error: 0.0 w1: 2 w2: 0 w3: -25
    error: 0.0 w1: 3 w2: -1 w3: 95
    error: 0.0 w1: 3 w2: 0 w3: -75
    error: 0.0 w1: 4 w2: -1 w3: 45
    error: 0.0 w1: 5 w2: -1 w3: -5
    error: 0.0 w1: 6 w2: -1 w3: -55
    error: 0.0 w1: 7 w2: -2 w3: 65
    error: 0.0 w1: 8 w2: -2 w3: 15
    error: 0.0 w1: 9 w2: -2 w3: -35
    error: 0.0 w1: 10 w2: -3 w3: 85
    error: 0.0 w1: 10 w2: -2 w3: -85
    error: 0.0 w1: 11 w2: -3 w3: 35
    error: 0.0 w1: 12 w2: -3 w3: -15
    error: 0.0 w1: 13 w2: -3 w3: -65
    error: 0.0 w1: 14 w2: -4 w3: 55
    error: 0.0 w1: 15 w2: -4 w3: 5
    error: 0.0 w1: 16 w2: -4 w3: -45
    error: 0.0 w1: 17 w2: -5 w3: 75
    error: 0.0 w1: 17 w2: -4 w3: -95
    error: 0.0 w1: 18 w2: -5 w3: 25
    error: 0.0 w1: 19 w2: -5 w3: -25
    error: 0.0 w1: 20 w2: -6 w3: 95
    error: 0.0 w1: 20 w2: -5 w3: -75
    error: 0.0 w1: 21 w2: -6 w3: 45
    error: 0.0 w1: 22 w2: -6 w3: -5
    error: 0.0 w1: 23 w2: -6 w3: -55
    error: 0.0 w1: 24 w2: -7 w3: 65
    error: 0.0 w1: 25 w2: -7 w3: 15
    error: 0.0 w1: 26 w2: -7 w3: -35
    error: 0.0 w1: 27 w2: -8 w3: 85
    error: 0.0 w1: 27 w2: -7 w3: -85
    error: 0.0 w1: 28 w2: -8 w3: 35
    error: 0.0 w1: 29 w2: -8 w3: -15
    error: 0.0 w1: 30 w2: -8 w3: -65
    error: 0.0 w1: 31 w2: -9 w3: 55
    error: 0.0 w1: 32 w2: -9 w3: 5
    error: 0.0 w1: 33 w2: -9 w3: -45
    error: 0.0 w1: 34 w2: -10 w3: 75
    error: 0.0 w1: 34 w2: -9 w3: -95
    error: 0.0 w1: 35 w2: -10 w3: 25
    error: 0.0 w1: 36 w2: -10 w3: -25
    error: 0.0 w1: 37 w2: -11 w3: 95
    error: 0.0 w1: 37 w2: -10 w3: -75
    error: 0.0 w1: 38 w2: -11 w3: 45
    error: 0.0 w1: 39 w2: -11 w3: -5
    error: 0.0 w1: 40 w2: -11 w3: -55
    error: 0.0 w1: 41 w2: -12 w3: 65
    error: 0.0 w1: 42 w2: -12 w3: 15
    error: 0.0 w1: 43 w2: -12 w3: -35
    error: 0.0 w1: 44 w2: -13 w3: 85
    error: 0.0 w1: 44 w2: -12 w3: -85
    error: 0.0 w1: 45 w2: -13 w3: 35
    error: 0.0 w1: 46 w2: -13 w3: -15
    error: 0.0 w1: 47 w2: -13 w3: -65
    error: 0.0 w1: 48 w2: -14 w3: 55
    error: 0.0 w1: 49 w2: -14 w3: 5
    error: 0.0 w1: 50 w2: -14 w3: -45
    error: 0.0 w1: 51 w2: -15 w3: 75
    error: 0.0 w1: 51 w2: -14 w3: -95
    error: 0.0 w1: 52 w2: -15 w3: 25
    error: 0.0 w1: 53 w2: -15 w3: -25
    error: 0.0 w1: 54 w2: -16 w3: 95
    error: 0.0 w1: 54 w2: -15 w3: -75
    error: 0.0 w1: 55 w2: -16 w3: 45
    error: 0.0 w1: 56 w2: -16 w3: -5
    error: 0.0 w1: 57 w2: -16 w3: -55
    error: 0.0 w1: 58 w2: -17 w3: 65
    error: 0.0 w1: 59 w2: -17 w3: 15
    error: 0.0 w1: 60 w2: -17 w3: -35
    error: 0.0 w1: 61 w2: -18 w3: 85
    error: 0.0 w1: 61 w2: -17 w3: -85
    error: 0.0 w1: 62 w2: -18 w3: 35
    error: 0.0 w1: 63 w2: -18 w3: -15
    error: 0.0 w1: 64 w2: -18 w3: -65
    error: 0.0 w1: 65 w2: -19 w3: 55
    error: 0.0 w1: 66 w2: -19 w3: 5
    error: 0.0 w1: 67 w2: -19 w3: -45
    error: 0.0 w1: 68 w2: -20 w3: 75
    error: 0.0 w1: 68 w2: -19 w3: -95
    error: 0.0 w1: 69 w2: -20 w3: 25
    error: 0.0 w1: 70 w2: -20 w3: -25
    error: 0.0 w1: 71 w2: -21 w3: 95
    error: 0.0 w1: 71 w2: -20 w3: -75
    error: 0.0 w1: 72 w2: -21 w3: 45
    error: 0.0 w1: 73 w2: -21 w3: -5
    error: 0.0 w1: 74 w2: -21 w3: -55
    error: 0.0 w1: 75 w2: -22 w3: 65
    error: 0.0 w1: 76 w2: -22 w3: 15
    error: 0.0 w1: 77 w2: -22 w3: -35
    error: 0.0 w1: 78 w2: -23 w3: 85
    error: 0.0 w1: 78 w2: -22 w3: -85
    error: 0.0 w1: 79 w2: -23 w3: 35
    error: 0.0 w1: 80 w2: -23 w3: -15
    error: 0.0 w1: 81 w2: -23 w3: -65
    error: 0.0 w1: 82 w2: -24 w3: 55
    error: 0.0 w1: 83 w2: -24 w3: 5
    error: 0.0 w1: 84 w2: -24 w3: -45
    error: 0.0 w1: 85 w2: -25 w3: 75
    error: 0.0 w1: 85 w2: -24 w3: -95
    error: 0.0 w1: 86 w2: -25 w3: 25
    error: 0.0 w1: 87 w2: -25 w3: -25
    error: 0.0 w1: 88 w2: -26 w3: 95
    error: 0.0 w1: 88 w2: -25 w3: -75
    error: 0.0 w1: 89 w2: -26 w3: 45
    error: 0.0 w1: 90 w2: -26 w3: -5
    error: 0.0 w1: 91 w2: -26 w3: -55
    error: 0.0 w1: 92 w2: -27 w3: 65
    error: 0.0 w1: 93 w2: -27 w3: 15
    error: 0.0 w1: 94 w2: -27 w3: -35
    error: 0.0 w1: 95 w2: -28 w3: 85
    error: 0.0 w1: 95 w2: -27 w3: -85
    error: 0.0 w1: 96 w2: -28 w3: 35
    error: 0.0 w1: 97 w2: -28 w3: -15
    error: 0.0 w1: 98 w2: -28 w3: -65
    error: 0.0 w1: 99 w2: -29 w3: 55
    error: 0.0 w1: 100 w2: -29 w3: 5


Perfect, we have several predictors!!! 

Let's take the first one: `w1=-100`, `w2=30`, and `w3=-25`

For `age=50`, `height=170`, we get

$$prediction\_weight = -100 . 50 + 30 . 170 - 25 . 1$$
$$prediction\_weight = -5000 + 5100 - 25$$
$$prediction\_weight = 75$$

The `prediction_weight` is equal to `actual_weight`!!!

## Using the regressor

Finally, for `age=40` and `weight=180`, we get this:

$$prediction\_weight = -100.40 + 30.180 - 25.1$$
$$prediction\_weight = -4000 + 5400 - 25$$
$$prediction\_weight = 1375$$

Well, not so make sense, probably we need to use another available predictor.

__Note:__ If your problem can be perfectly solved with pure-algebra/brute-force. Just use them for your own good. Neural network or any other machine learning algorithm, should only be used if the solution is not obvious.

## What just happened?

We have just jump into a problem named `overfitting`. Overfitting is a problem where our predictor/regressor is correct for the training data, but incorrect for the testing data.

Another problems with our approach are:

* we naively believe that w1, w2, and w3 are integer
* we naively believe that the predictor is a straight linear line. Probably we need some logarithm, power, and other eccentric operations in order to get the correct predictor/classifier

Neural networks can definitely solve our two last problems. Overfitting is still a common problem in machine learning.

# Neural Network approach

As already stated, neural network is no magic.

In our previous approach, the regressor is formulated as follow: 

$$weight = w1.age + w2.height + w3$$

Suppose we have a function `f` to process the result, the regressor should looks like this:

$$weight = f(w1.age + w2.height + w3)$$

This is what a single neuron in neural network do !!!

![]()



# Neural Network implementation using Tensorflow

TensorFlow is an open-source software library for dataflow programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks.

Unlike `sklearn.neural_network`, tensorflow give us more freedom to set up our neural-network.

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



![png](neuralnetwork_files/neuralnetwork_23_1.png)


## Configuring the neural network model

Finally, let's build our neural network.

First of all, we define 3 layers here:

* flatten layer with input_shape = 28x28: This one will transform our 2 dimensional matrix into 1 dimensional matrix (or a vector). The output of this layer will be an array with 784 elements
* dense layer containing 1024 neuron with sigmoid activation: This one will create a layer containing 128 neuron. Each of them is connected to the output of our previous layer (an array containing 784 elements). Each neuron activation is depending on `sigmoid` function (https://en.wikipedia.org/wiki/Sigmoid_function)
* dense layer containing 10 neuron with softmax activation: Finally, since we have 10 classes, it is natural to have 10 neuron in our output layer. Each neuron should show us how probable is an image belong to a particular class. Finally, we will use softmax to return the prediction result (https://en.wikipedia.org/wiki/Softmax_function)

After defining the layers, we need to define our optimizer, loss function, and metrics:

* optimizer: How to optimize. We use adam optimization (https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
* loss function: How to calculate error
* metrics: How to measure the quality of the network


```python
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(1024, activation=tf.nn.sigmoid),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


```

## Train the neural network model


```python
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```

    Train on 60000 samples
    Epoch 1/5
    60000/60000 [==============================] - 8s 133us/sample - loss: 0.3687 - accuracy: 0.8954
    Epoch 2/5
    60000/60000 [==============================] - 8s 129us/sample - loss: 0.2767 - accuracy: 0.9184
    Epoch 3/5
    60000/60000 [==============================] - 8s 131us/sample - loss: 0.2467 - accuracy: 0.9273
    Epoch 4/5
    60000/60000 [==============================] - 8s 132us/sample - loss: 0.2285 - accuracy: 0.9304
    Epoch 5/5
    60000/60000 [==============================] - 8s 132us/sample - loss: 0.2158 - accuracy: 0.9337
    10000/10000 [==============================] - 1s 71us/sample - loss: 0.1969 - accuracy: 0.9367
    Test accuracy: 0.9367


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



![png](neuralnetwork_files/neuralnetwork_29_1.png)


# Further discussion

For classical machine learning (as this one), data preprocessing is quite important. Please look at this: https://github.com/shayan09/MNIST-Handwriting-Recognition-using-Keras/blob/master/Basic%20Keras%20NN.ipynb for comparison.
