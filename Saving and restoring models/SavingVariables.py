import tensorflow as tf 

#the file path to save the data 

save_file = './model.ckpt'

#two Tesnsor Variable: weights and bias

weights = tf.Variable(tf.truncated_normal([2, 3]))
bias = tf.Variable(tf.truncated_normal([3]))

#class used to save and/or restore Tensor Variables 

saver = tf. train.Saver()

with tf.Session() as sess:
	#inintialize all the variables
	sess.run(tf.globale_variables_initializer())

	#show the values of the weights and bias
	print('Weights: ')
	print(sess.run(weights))
	print('Bias: ')
	print(sess.run(bias))

	#Save the model
	saver.save(sess, save_file)




