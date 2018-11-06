# Deep Neural Networks

# Ishmael Rogers
# Robotics Engineer, Infinitely Deep Research Group
# www.idrg.io
# 2018

In the Deep Learning repository we developed a simple classifier to perform classification tasks. In this repository we will go a step further and turn the single classifier into a Deep Neural Network. 

This README document focuses on how the optimizer computes the graidents for arbitrary functions and introduces the concept of regularization which helps train larger models.

## Review questions

Consider the simple classifier from the deep learning repository

if the classifier takes in a 28 x 28 image and the output is 10 classes, 
then, how many trained parameters did the model have?

The image was 28 x 28, therefore matrix W takes in 28 x 28 pixels (rows)

If the output is 10 classes, that means that b has to be a 10 x 1

28 x 28 x 10 + 10 = 7850 trained parameters!
NOTE: In general if you have N, inputs and K, outputs, then the number of parameters equals:
 
 (N + 1) x K

# Limitations of linear models

The interactions that are possible with linear models are limited. More specifically, they tend to handle inputs that need to be added together but cannot handle products of inputs.

There are some positive aspects of linear models that force us to use them despite their limitations.

0. Mathematically inear models are efficient and stable therefore, small cahnges in input can never yield big changes in output.
1. The derivative of linear function is constant 

NOTE: Ideally we want to keep parameters inside big linear functions while also making the model non-linear.

## Introduce non-linearities

The ReLU is the simpliest non-linear function (with a nice derivative that happens to be a step function.) 

The ReLU is linear if x > 0  and a zero everywhere else

# Network of ReLUs

Constructing a network of ReLUs can be extremely useful.

0. Take a logistic classifier and make the minimal amount of changes to make it non-linear

NOTE: instead of a single matrix multiply as the classifier, we insert a RELU in the middle to get a

# Two layer Neural Networks

The hidden layer in a network allows it to model more complex functions. 
NOTE: Using a non linear activation function on the hidden layer lets it model non-linear functions very well. 
NOTE: Functions can become non-linear when the inputs are multiplied together as briefly discussed previously.

0. Inputs to ReLU

The first layer consists of the set of weights and biases applied to X and is passed through ReLUs. The output of this layer is fed to the next one, but is not observable outside the network, hence it is known as a hidden layer.

1. From Relu to classifier 

The second layer consists of the weights and biases applied to these intermediate outputs, followed by the softmax function to generate probabilities
NOTE: The parameter H is the number of RELU units in the classifier
NOTE: H Can be as big as we want. 

## TensorFlow ReLU

To take adavantage of the power of ReLUs, we use the TensorFlow ReLU function 

tf.nn.relu(
    features,
    name=None
)

The code applies the tf.nn.relu() function to the hidden_layer:

0. Turning off any negative weights and acting like an on/off switch. 

Adding additional layers, like the output layer, after an activation function turns the model into a nonlinear function. 

This nonlinearity allows the network to solve more complex problems.

# The chain rule 

Insert chain rules here

# Backpropagation 

Makes computing derivatives of complex functions very effcient as long as the function is made up of simple blocks with simple derivatives.

NOTE: Running the model towards the prediction is called forward propagation. 
NOTE: The model that goes backward is back propagation.

To run SDG for every single batch of data in the training set 

0. Run the forward prop 
1. Run the back prop 
NOTE: that will provide the graident for each of the weights in the model. 
2. Apply obtained gradients with learning rate to orignial weights and update them. 
3. Repeat many times to optimize model!

NOTE: Each block of back prop takes twice the memory needed for the forward prop and twice to compute 
NOTE: This idea is important for sizing models and fitting it them into memory.

# Deep Neural Networks in Tensorflow 

Finally we will expand on the idea of a logistic classifier to build a deep neural network. 

Please use the code in the multi-layer perceptron folder to follow along. 


0. You'll use the MNIST dataset provided by TensorFlow, which batches and One-Hot encodes the data for you.

1. The focus here is on the architecture of multilayer neural networks, not parameter tuning, so the learning parameters are provided.

2. The variable "n_hidden_layer" determines the size of the hidden layer in the neural network. 
NOTE: This is also known as the width of a layer.

3. Deep neural networks use multiple layers with each layer requiring it's own weight and bias. 
   - The 'hidden_layer' weight and bias is for the hidden layer. 
   - The 'out' weight and bias is for the output layer. 
NOTE: If the neural network were deeper, there would be weights and biases for each additional layer.

4. The MNIST data is made up of 28px by 28px images with a single channel. The 

tf.reshape() 

function above reshapes the 28px by 28px matrices in x into row vectors of 784px.

5. Combine linear functions together using a ReLU will give you a two layer network.

6. This is the same optimization technique used in the Intro to TensorFlow lab

8. The MNIST library in TensorFlow provides the ability to receive the dataset in batches. 
NOTE: Calling the "mnist.train.next_batch()" function returns a subset of the training data


# Training a Deep Learning Network 


# Save and Restore TensorFlow Models 

See Save and Restore folder in this repository for code designed to save and load TensorFlow Models

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
