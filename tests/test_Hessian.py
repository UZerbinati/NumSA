#We import all the library we are gona need
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numsa.TFHessian import *

#Defining the Loss Function
def Loss1(X):
    return 2*X[0]**3*X[1]**3;



def test_Loss1():
    #Defining the Hessian class for the above loss function in x
    x0 = tf.Variable([1.0,1.0])
    H =  Hessian(Loss1,x0)
    Error = [];
    ErrorH = [];
    N = 10
    for n in range(N):
        h = 1/(2**n);
        ErrorH = ErrorH + [h];
        x = tf.Variable([1.0+h,1.0+h]);
        v = tf.Variable([h,h]);
        Grad, Hv = H.action(v,True)
        err = abs(Loss1(x)-Loss1(x0)-tf.tensordot(Grad,v,1)-0.5*tf.tensordot(v,Hv,1));
        Error = Error + [err];
    plt.loglog(ErrorH,Error,"*-")
    plt.loglog(ErrorH,[10*h**3 for h in ErrorH],"--")
    plt.legend(["2nd Order Taylor Error","3rd Order Convergence"])
    order = (tf.math.log(Error[7])-tf.math.log(Error[9]))/(np.log(ErrorH[7])-np.log(ErrorH[9]))
    assert 2.9 < order;
def test_eig1():
    #Defining the Hessian class for the above loss function in x
    x0 = tf.Variable([1.0,1.0])
    H =  Hessian(Loss1,x0)
    #Using Power iteration method to compute the maximum eigenvalue
    assert abs(H.eig("pi-max")-30) < 1e-4;



def test_sym_Hessian():
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    # Inputs
    training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
    # Outputs

    training_label = np.array([[0],[1],[1],[0]], "float32")
    training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_label))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2,input_dim=2, activation='sigmoid',bias_initializer=init),#2 nodes hidden layer
        tf.keras.layers.Dense(1)
    ])
    def loss_fn(y,x):
        return tf.math.reduce_mean(tf.math.squared_difference(y, x));
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(training_label, predictions);
        return loss_value;
    for epoch in range(1):
        for step, (x,y) in enumerate(training_dataset):
            # Compute Hessian and Gradients
            H = Hessian(Loss,model.trainable_weights)
            fullH, grad = H.mat("KERAS",grad=True);
            #Reshaping the Hessians
            grads = [tf.Variable(grad[0:4].reshape(2,2),dtype=np.float32),
                     tf.Variable(grad[4:6].reshape(2,),dtype=np.float32),
                     tf.Variable(grad[6:8].reshape(2,1),dtype=np.float32),
                     tf.Variable(grad[8].reshape(1,),dtype=np.float32),]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(training_label, predictions);
        return loss_value;
    H = Hessian(Loss,model.trainable_weights)
    fullH = H.mat("KERAS");
    assert np.linalg.norm(fullH-fullH.T) < 1e-4;



def test_prod_Hessian():
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    # Inputs
    training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
    # Outputs

    training_label = np.array([[0],[1],[1],[0]], "float32")
    training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_label))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2,input_dim=2, activation='sigmoid',bias_initializer=init),#2 nodes hidden layer
        tf.keras.layers.Dense(1)
    ])
    def loss_fn(y,x):
        return tf.math.reduce_mean(tf.math.squared_difference(y, x));
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(training_label, predictions);
        return loss_value;
    for epoch in range(1):
        for step, (x,y) in enumerate(training_dataset):
            # Compute Hessian and Gradients
            H = Hessian(Loss,model.trainable_weights)
            fullH, grad = H.mat("KERAS",grad=True);
            #Reshaping the Hessians
            grads = [tf.Variable(grad[0:4].reshape(2,2),dtype=np.float32),
                     tf.Variable(grad[4:6].reshape(2,),dtype=np.float32),
                     tf.Variable(grad[6:8].reshape(2,1),dtype=np.float32),
                     tf.Variable(grad[8].reshape(1,),dtype=np.float32),]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(training_label, predictions);
        return loss_value;
    H = Hessian(Loss,model.trainable_weights)
    fullH = H.mat("KERAS");
    w = np.random.rand(9,1)
    assert (abs(fullH@w-(H.vecprod(w,"KERAS").reshape(9,1)))<1e-6).all()



def test_constant_weights():
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    # Inputs
    training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
    # Outputs

    training_label = np.array([[0],[1],[1],[0]], "float32")
    training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_label))
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(2,input_dim=2, activation='sigmoid',bias_initializer=init),#2 nodes hidden layer
        tf.keras.layers.Dense(1)
    ])
    def loss_fn(y,x):
        return tf.math.reduce_mean(tf.math.squared_difference(y, x));
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(training_label, predictions);
        return loss_value;
    for epoch in range(1):
        for step, (x,y) in enumerate(training_dataset):
            # Compute Hessian and Gradients
            H = Hessian(Loss,model.trainable_weights)
            fullH, grad = H.mat("KERAS",grad=True);
            #Reshaping the Hessians
            grads = [tf.Variable(grad[0:4].reshape(2,2),dtype=np.float32),
                     tf.Variable(grad[4:6].reshape(2,),dtype=np.float32),
                     tf.Variable(grad[6:8].reshape(2,1),dtype=np.float32),
                     tf.Variable(grad[8].reshape(1,),dtype=np.float32),]
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        #return tf.math.reduce_mean(tf.math.squared_difference(training_label, predictions));
        return tf.math.reduce_mean(predictions);
    H = Hessian(Loss,model.trainable_weights)
    fullH = H.mat("KERAS");
    print("Shape {}".format(fullH.shape));
    assert  (np.sum(fullH[:,8])+np.sum(fullH[8,:])+fullH[8,8]+fullH[7,7]+fullH[6,6]+fullH[6,7]+fullH[7,6])==0;
