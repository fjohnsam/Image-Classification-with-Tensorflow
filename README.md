# Image-Classification-with-Tensorflow

HyperParameters:

Number of layers = 3
Hidden layer size = 150 Learning rate = 0.00005 Regularization parameter = 0.5 Batch size = 500
Number of epochs = 50 Patience = 5
Activation function = ReLU
Optimiser = Gradient descent
Batch generator = np.random.choice() Train - Validation split = 5 : 1
Logistic regression solver = ‘saga’

Results:
1. Test accuracy obtained = 87.22%
2. Logistic Regression accuracy at layer 1 = 84.3 % 
3. Logistic Regression accuracy at layer 2 = 87.3 % 
4. Logistic Regression accuracy at layer 3 = 87.54 %

Inferences:   
1. The increasing accuracy of Logistic regression from layer 1 to Layer 3 indicates that deeper the network, better learning is done. At deeper layers, more detailed features are learned.
2. When the learning rate was set too low, there was hardly any visible learning done. When the learning rate was set too high, then the descent path diverged from the desired path and got stuck in local minima.
3. Softmax prevents the overflow (blowing up) of the loss values.
4. The pixel values also were scaled to values between 0-1, to prevent
overflow.
5. The logistic regression Saga solver is suitable for large datasets and
hence gave good accuracy.
6. The final test accuracy of >87% indicates that the 3 layers neural
network is very well capable of learning the given dataset.
