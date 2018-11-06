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

Parameter H, the number of RELU units in the classifier ***

Can be as big as we want. 

## TensorFlow ReLU

We use the TensorFlow ReLU function 

tf.nn.relu(
    features,
    name=None
)

The code applies the tf.nn.relu() function to the hidden_layer, effectively turning off any negative weights and acting like an on/off switch. Adding additional layers, like the output layer, after an activation function turns the model into a nonlinear function. This nonlinearity allows the network to solve more complex problems.

# The chain rule 

The math 

# Backpropagation 

Makes computing derivatives of complex functions very effcient as long as the function is made up of simple blocks with simple derivatives.

Running the model towards the prediction is forward prop

The model that goes backwards is back prop

To run SDG for every simngle batch of data in the training set run the forward prop then back prop that will provide the graident for each of the weights in the model. Then apply gradients with learning rate to orignial weights and update them. Repeat many times to optimize model!

Each block of backprop takes twice the memory need for the forward prop and twice to compute important for sizing modeling and fitting it in memory.

# Deep Neural Networks in Tensorflow 

Expanding on the idea of a logistic classifier to build a deep neural network.



# Training a Deep Learning Network 

# Save and Restore TensorFlow Models 

# Finetuning

# Regularization 

# Dropout 

# Deep Neural Network Lab

