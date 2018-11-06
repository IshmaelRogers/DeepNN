# Deep Neural Networks

# Ishmael Rogers
# Robotics Engineer, Infinitely Deep Research Group
# www.idrg.io
# 2018

Take the classifier and turn it into a deep network

Let's focus on how the optimizer computes the graidents for arbitrary functions. Introduce the concept of regulaization which helps train larger models.


Review questions

consider our simple model from the deep learning repository, it takes in a 28 x 28 image the output was 10 classes

How many trained parameters did the model have?

The image was 28 x 28, therefore matrix W takes in 28 x 28 pixels (rows)

If the output is 10 classes, that means that b has to be a 10 x 1

28 x 28 x 10 + 10 = 7850


 NOTE: In general if you have N, inputs and K, outputs, then the number of parameters equal
 
 (N + 1) x K

# Limitations of linear models

interaction are limited
Cannot handle products inputs

Linear models are efficient and stable 
small cahnges in input can never yield big changes in output

derivative of linear function is constant 

Ideally we want to keep parameters inside big linear functions while also making the model non-linear.

## Introduce linearities

RELU simpliest non-linear function with a nice derivative that happens to be a step function. 

linear if x > 0  and a zero everywhere else


# Network of ReLUs

Take a logistic classifier and do the minimal aount of change to make it nonliner

instead of single matrix multiply as the classifier, we insert a RELU in the middle to get a

# Two layer Neural Networks

The hidden layer to a network allows it to model more complex functions. Using a non linear activation function on the hudden layer lets it model non-linear functions. 

1. from inputs to RElU

The first layer  consists of the set of weights and biases applied to X and passed through ReLUs. The output of this layer is fed to the next one, but is not observable outside the network, hence it is known as a hidden layer.

2. From Relu to classifier 

The second layer consists of the weights and biases applied to these intermediate outputs, followed by the softmax function to generate probabilities

Parameter H the number of RELU units in the classifier ***

Can be as big as we want. 





# The chain rule 

# Backpropagation 

# Deep Neural Networks in Tensorflow 

# Training a Deep Learning Network 

# Save and Restore TensorFlow Models 

# Finetuning

# Regularization 

# Dropout 

# Deep Neural Network Lab

