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

Please use the multi-layer perceptron folder to follow along. 


1. You'll use the MNIST dataset provided by TensorFlow, which batches and One-Hot encodes the data for you.

2. The focus here is on the architecture of multilayer neural networks, not parameter tuning, so here we'll just give you the learning parameters.

3. The variable n_hidden_layer determines the size of the hidden layer in the neural network. This is also known as the width of a layer.

4. Deep neural networks use multiple layers with each layer requiring it's own weight and bias. 
- The 'hidden_layer' weight and bias is for the hidden layer. 
- The 'out' weight and bias is for the output layer. 

If the neural network were deeper, there would be weights and biases for each additional layer.

5. The MNIST data is made up of 28px by 28px images with a single channel. The 

tf.reshape() 

function above reshapes the 28px by 28px matrices in x into row vectors of 784px.

6. Combining linear functions together using a ReLU will give you a two layer network.

7. This is the same optimization technique used in the Intro to TensorFLow lab

8. The MNIST library in TensorFlow provides the ability to receive the dataset in batches. Calling the mnist.train.next_batch() function returns a subset of the training data


# Training a Deep Learning Network 


# Save and Restore TensorFlow Models 

# Finetuning

# Regularization 

The network at just the right size for our data is hard to optimize. 

In practice we train networks that are way too big for our data then prevent them from overfitting

# Early termination 

Look at performance of validation set. 

Stop and train as soon as we stop inmproving 

Prevents over optimization

## Regularizing 

- applying artificial constraints on the network that implicitly reduce the number of free parameters.

# L2 Regularization

Adds another term to the loss which penalizes large weights.

Add L2 Norm of the weights to the loss and multiply by a small constant

NOTE: L2 norm stands for the sum of the squares of the individual elements in a vector 
 
 Structure of network doesnt change because the L2 Norm is being added
 
 Derivative is X
 

Question


# Dropout 

Dropout is a regularization technique for reducing overfitting. The technique temporarily drops units (artificial neurons) from the network, along with all of those units' incoming and outgoing connections

imagine 1 layer connected to another layer the values that go from 1 layer to the next are activations. Take the activations and randomly, for every example you train the network on, set half of them to zero. 

Completely and randomly and take half the data and destroy it 

Network can never rely on any given activation to be present because it may be destroyed

Forced to learn a redundant representation for everything, to make sure some of the info remains

Prevents overfitting

Takes a concensus 

Evaluating the network trained with dropout no longer want randomness

something determoinistic is desired 

Take concensuses over the 

average the activation 

y e = average of all yt obtained during training 

during =training

0 out activation that i drop out 
scale remaining activations by factor of 2 

to remove these dropouts and scaling operations from the nueral network

The result is an average of these activations that is properly scaled


TensorFlow provides the 
tf.nn.dropout(
    x,
    keep_prob,
    noise_shape=None,
    seed=None,
    name=None
)

function to implement the dropout

See droput.py

Result of code is 

[[ 9.559999, 16.], [ 0. , 0.], [ 4.72, 28.32])




# Deep Neural Network Lab


Refer to Deep NN Lab folder
