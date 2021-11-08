#We import all the library we are gona need
import pytest
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numsa.TFHessian import *
import dsdl

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
    H =  Hessian(Loss1,x0,"KERAS")
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
            H = Hessian(Loss,model.trainable_weights,"KERAS")
            fullH = H.mat();
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                labels = model(training_data, training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(training_label, labels)
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(training_label, predictions);
        return loss_value;
    H = Hessian(Loss,model.trainable_weights,"KERAS")
    fullH = H.mat();
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
            H = Hessian(Loss,model.trainable_weights,"KERAS")
            fullH = H.mat();
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                labels = model(training_data, training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(training_label, labels)
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        # Compute the loss value for this minibatch.
        loss_value = loss_fn(training_label, predictions);
        return loss_value;
    H = Hessian(Loss,model.trainable_weights,"KERAS")
    fullH = H.mat();
    w = np.random.rand(9,1)
    assert (abs(fullH@w-(H.vecprod(w).reshape(9,1)))<1e-6).all()



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
            H = Hessian(Loss,model.trainable_weights,"KERAS")
            fullH = H.mat();
            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                labels = model(training_data, training=True)  # Logits for this minibatch
                # Compute the loss value for this minibatch.
                loss_value = loss_fn(training_label, labels)
                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
                grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
    def Loss(weights):
        predictions = model(training_data, training=True) #Logits for this minibatch
        #return tf.math.reduce_mean(tf.math.squared_difference(training_label, predictions));
        return tf.math.reduce_mean(predictions);
    H = Hessian(Loss,model.trainable_weights,"KERAS")
    fullH = H.mat();
    print("Shape {}".format(fullH.shape));
    assert  (np.sum(fullH[:,8])+np.sum(fullH[8,:])+fullH[8,8]+fullH[7,7]+fullH[6,6]+fullH[6,7]+fullH[7,6])==0;

def test_Newton():
    itmax = 100; # Number of epoch.
    tol = 1e-8
    step_size = 1; #Learning rate
    def Loss(x):
        return (x[0]**2)*(x[1]**2)+x[0]*x[1];
    #Defining the Hessian class for the above loss function in x0
    x = tf.Variable(0.1*np.ones((2,1),dtype=np.float32))
    H =  Hessian(Loss,x)
    grad = H.grad().numpy();
    print("Computed the  first gradient ...")
    q = H.pCG(grad,1,1,tol=tol,itmax=100);
    print("Computed search search diratcion ...")
    print("Entering the Netwton optimization loop")
    for it in tqdm(range(itmax)):
        x = x - tf.constant(step_size,dtype=np.float32)*tf.Variable(q,dtype=np.float32);
        x =  tf.Variable(x)
        if it%50 == 0:
            print("Lost funciton at this iteration {}  and gradient norm {}".format(Loss(x),np.linalg.norm(grad)));
        if np.linalg.norm(grad)<tol:
            print("Lost funciton at this iteration {}, gradient norm {} and is achived at point {}"
          .format(Loss(x),np.linalg.norm(grad),x));
            break
        H =  Hessian(Loss,x)
        grad = H.grad().numpy();
        q = H.pCG(grad,1,1,tol=tol,itmax=100);
    assert Loss(x)<tol;
@pytest.mark.mpi
def test_FEDNL_Newton_Vanilla():
    comm = MPI.COMM_WORLD

    ds = dsdl.load("a1a")

    X, Y = ds.get_train()
    indx = np.array_split(range(X.shape[0]),int(comm.Get_size()));
    tfX = []
    tfY = []
    for k in range(len(indx)):
        tfX = tfX + [tf.sparse.from_dense(np.array(X[indx[comm.Get_rank()]].todense(), dtype=np.float32))]
        tfY = tfY + [tf.convert_to_tensor(np.array(Y[indx[comm.Get_rank()]], dtype=np.float32).reshape(X[indx[comm.Get_rank()]].shape[0], 1))]

    tfXs = tf.sparse.from_dense(np.array(X.todense(), dtype=np.float32))
    tfYs = tf.convert_to_tensor(np.array(Y, dtype=np.float32).reshape(X.shape[0], 1))
    #Defining the Loss Function
    def LossSerial(x):
        lam = 1e-3; #Regularisation
        x = tf.reshape(x, (119, 1))
        Z = tf.sparse.sparse_dense_matmul(tfXs, x, adjoint_a=False)
        Z = tf.math.multiply(tfYs, Z)
        S = tf.reduce_sum(tf.math.log(1 + tf.math.exp(-Z)) / tfXs.shape[0]) + lam*tf.norm(x)**2

        return S
    #Defining the Loss Function
    def Loss(x,comm):
        lam = 1e-3; #Regularisation
        x = tf.reshape(x, (119, 1))
        Z = tf.sparse.sparse_dense_matmul(tfX[comm.Get_rank()], x, adjoint_a=False)
        Z = tf.math.multiply(tfY[comm.Get_rank()], Z)
        S = tf.reduce_sum(tf.math.log(1 + tf.math.exp(-Z)) / tfX[comm.Get_rank()].shape[0]) + lam*tf.norm(x)**2
        return S
    ################! Setting Of The Solver!##################
    itmax = 50
    tol = 1e-4;
    step_size=1;
    ###########################################################
    x = tf.Variable(0.1*np.ones((119,1),dtype=np.float32))

    H = Hessian(Loss,x);
    H.shift(x)#,start=0*np.identity(x.numpy().shape[0])) #We initialize the shifter
    #We now collect and average the loc Hessians in the master node (rk 0)
    Hs = H.comm.gather(H.memH, root=0);
    if H.comm.Get_rank()==0:
        Hm = (1/len(Hs))*np.sum(Hs,0);
    else:
        Hm = None
    print("The master Hessian has been initialised")
    for it in tqdm(range(itmax)):
        # Obtaining the compression of the difference between local mat
        # and next local mat.
        U,sigma,Vt,ell = H.shift(x,{"comp":MatSVDCompDiag,"rk":119,"type":"mat"});
        shift = Vt.transpose()@np.diag(sigma)@U.transpose();
        #print("Updating local Hessian")
        H.memH = H.memH+step_size*shift;
        grad = H.grad().numpy();
        #Now we update the master Hessian and perform the Newton method step
        Shifts = H.comm.gather(shift, root=0);
        Grads = H.comm.gather(grad, root=0);
        Ells = H.comm.gather(ell, root=0);
        if H.comm.Get_rank() == 0:
            #print("Computing the avarage of the local shifts and grad ...")
            Shift = (1/len(Shifts))*np.sum(Shifts,0);
            Grad = (1/len(Grads))*np.sum(Grads,0);
            Ell = (1/len(Ells))*np.sum(Ells,0);
            res = np.linalg.norm(Grad);
            #print("Computing the master Hessian ...")
            Hm = Hm + step_size*Shift;
            #print("Searching new search direction ...")
            A = Hm #A = Hm + Ell*np.identity(Hm.shape[0]);
            q = np.linalg.solve(A,Grad);
            #print("Found search dir, ",q);
            if it%25 == 0:
                print("(FedNL) [Iteration. {}] Lost funciton at this iteration {}  and gradient norm {}".format(it,LossSerial(x),np.linalg.norm(Grad)));
            x = x - tf.Variable(q,dtype=np.float32);
            x =  tf.Variable(x)
        else:
            res = None
        #Distributing the search direction
        x = H.comm.bcast(x,root=0)
        res = H.comm.bcast(res,root=0)
        if res<tol:
                break
    assert(it < 5);
