#Copyright: Febin John Sam (fjohnsam96@gmail.com)

#Code written based on tensorflow 1
#Tensorflow 2 package used and version 2 behaviour disabled

import numpy as np
import tensorflow.compat.v1 as tf
print(tf.__version__)
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import math
import os
import input_data
import mnist_reader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

print("*************************************************************")

# argparse used to take command line arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train the model",action="store_true")
parser.add_argument("--test", help="test the model",action="store_true")
parser.add_argument("--layer",help="use logistic regression on output from specified layer",choices=['1','2','3'])
args = parser.parse_args()

# Function to train the neural network
def train(X_train, y_train, X_val, y_val):

	sess=tf.Session()

	n_traindata=1000
	inputsize=784
	hiddensize=150
	classes=10
	hiddenlayers=3
	param = {}
	param['w1'] = tf.Variable(tf.random_uniform((inputsize,hiddensize),-0.05,0.05,seed=420))
	param['b1'] = tf.Variable(tf.zeros(hiddensize,1))
	param['w2'] = tf.Variable(tf.random_uniform((hiddensize,hiddensize),-0.05,0.05,seed=420))
	param['b2'] = tf.Variable(tf.zeros(hiddensize,1))
	param['w3'] = tf.Variable(tf.random_uniform((hiddensize,classes),-0.05,0.05,seed=420))
	param['b3'] = tf.Variable(tf.zeros(classes,1))

	x=tf.placeholder(tf.float32,(None,inputsize),name="x")
	y=tf.placeholder(tf.float32,(None,classes),name="y")

	#parametres
	w1,b1 = param['w1'],param['b1']
	w2,b2 = param['w2'],param['b2']
	w3,b3 = param['w3'],param['b3']

	#Network
	z1=tf.add(tf.matmul(x,w1),b1,name="z1")
	a1=tf.maximum(0.0,z1)
	z2=tf.add(tf.matmul(a1,w2),b2,name="z2")
	a2=tf.maximum(0.0,z2)
	scores=tf.add(tf.matmul(a2,w3),b3,name="scores")

	loss=None

	softmax_matrix = tf.exp(scores) / tf.reshape(tf.reduce_sum(tf.exp(scores), axis=1),(-1,1))

	loss_matrix=tf.multiply(tf.log(softmax_matrix),y)
	data_loss = - tf.math.reduce_sum(loss_matrix) 
	reg_loss =  tf.math.reduce_sum(tf.multiply(w1,w1))+tf.math.reduce_sum(tf.multiply(w2,w2))+tf.math.reduce_sum(tf.multiply(w3,w3))
	loss = data_loss + reg_loss * 0.5

	trainer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00005).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(softmax_matrix, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")

	# To store network parameters
	saver = tf.train.Saver()

	num_iters=500*60
	batch_size=500

	num_train = X_train.shape[0]
	iterations_per_epoch = max(num_train / batch_size, 1)

	count = 0
	best_val_acc = 0

	sess.run(tf.global_variables_initializer())

	for it in range(num_iters):
		X_batch = None
		y_batch = None

		# create batch using np.random.choice function
		batch_indices = np.random.choice(num_train, batch_size)
		batch_indices = batch_indices.tolist()
		X_batch = X_train[batch_indices]
		y_batch = y_train[batch_indices]


		sess.run(trainer,feed_dict={x:X_batch,y:y_batch})

		if it % 500 ==0 :

			a = sess.run(accuracy,feed_dict={x:X_train,y:y_train})
			print("Training accuracy %f" %a,end="   ")

			b = sess.run(accuracy,feed_dict={x:X_val,y:y_val})
			print("Validation accuracy %f" %b )	
			print("\n")

			# if validation accuracy improved then store the current parameters
			if b > best_val_acc :
				best_val_acc = b
				count = 0
				saver.save(sess, 'weights/my_model')

			# if validation acc doesnt improves for simultaneous 5 iterations then early stoping
			elif count == 4 :
				print("Early stopping...")
				a*=100
				b*=100
				print("Final training accuracy %f"%a)
				print("Final validation accuracy %f"%b)
				break
			else:
				count+=1
	
# Function to test the trained network parameters
def test(X_test,y_test):

	sess=tf.Session() 

	saver = tf.train.import_meta_graph('weights/my_model.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./weights/'))
	
	# To retrieve the placeholders for feeding test data,
	# because the tf variables and placeholders were previously declared locally in train function
	graph = tf.get_default_graph()

	x = graph.get_tensor_by_name("x:0")
	y = graph.get_tensor_by_name("y:0")

	accuracy = graph.get_tensor_by_name("accuracy:0")

	print("Test accuracy %f" % sess.run(accuracy,feed_dict={x:X_test,y:y_test}))


# Function to run logistic regression on data representation at layer 1, 2 and 3
def logistic(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot):

	sess=tf.Session() 

	saver = tf.train.import_meta_graph('weights/my_model.meta')
	saver.restore(sess,tf.train.latest_checkpoint('./weights/'))
	
	# To retrieve the placeholders for feeding test data,
	# because the tf variables and placeholders were previously declared locally in train function
	graph = tf.get_default_graph()
	x = graph.get_tensor_by_name("x:0")
	y = graph.get_tensor_by_name("y:0")

	logreg = LogisticRegression(solver='saga',random_state=0,max_iter=200)

	#layer1
	if args.layer=='1':

		z1 = graph.get_tensor_by_name("z1:0")

		X_train_layer1 = sess.run(z1,feed_dict={x:X_train,y:y_train_onehot})
		X_test_layer1 = sess.run(z1,feed_dict={x:X_test,y:y_test_onehot})

		logreg.fit(X_train_layer1,y_train)
		score1 = logreg.score(X_test_layer1,y_test)
		print("Layer 1 Logistic Regression accuracy ",score1)

	#layer2
	if args.layer=='2':

		z2 = graph.get_tensor_by_name("z2:0")

		X_train_layer2 = sess.run(z2,feed_dict={x:X_train,y:y_train_onehot})
		X_test_layer2 = sess.run(z2,feed_dict={x:X_test,y:y_test_onehot})

		logreg.fit(X_train_layer2,y_train)
		score2 = logreg.score(X_test_layer2,y_test)
		print("Layer 2 Logistic Regression accuracy ",score2)

	#layer2
	if args.layer=='3':

		scores = graph.get_tensor_by_name("scores:0")

		X_train_layer3 = sess.run(scores,feed_dict={x:X_train,y:y_train_onehot})
		X_test_layer3 = sess.run(scores,feed_dict={x:X_test,y:y_test_onehot})

		logreg.fit(X_train_layer3,y_train)
		score3 = logreg.score(X_test_layer3,y_test)
		print("Layer 3 Logistic Regression accuracy ",score3)



def main():

	# Reading the dataset
	X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
	X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

	# Splitting into train and validation set
	X_train2, X_val, y_train2, y_val = train_test_split(X_train,y_train, test_size=1/6, random_state=42)

	# Converting the image lables(numbers) into one hot vectors
	y_train_onehot = np.eye(10)[y_train]
	y_train2_onehot = np.eye(10)[y_train2]
	y_val_onehot = np.eye(10)[y_val]
	y_test_onehot = np.eye(10)[y_test]

	# Scaling the pixel values to prevent overflow
	X_train=X_train/255
	X_train2=X_train2/255
	X_val=X_val/255
	X_test=X_test/255	


	if args.train:
   		train(X_train2,y_train2_onehot,X_val,y_val_onehot)
	if args.test:
	    test(X_test,y_test_onehot)
	if args.layer=='1':
		logistic(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)
	if args.layer=='2':
		logistic(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)
	if args.layer=='3':
		logistic(X_train,y_train,y_train_onehot,X_test,y_test,y_test_onehot)


if __name__=='__main__':
	main()
