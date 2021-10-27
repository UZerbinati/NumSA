#We import all the library we are gona need
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numsa.TFHessian import *
from mpi4py import MPI
comm = MPI.COMM_WORLD

import dsdl

ds = dsdl.load("a1a")

X, Y = ds.get_train()
print(X.shape, Y.shape)
print("World Size, ", comm.Get_size());
indx = np.array_split(range(X.shape[0]),int(comm.Get_size()));

#Setting the parameter of this run, we will use optimization nomeclature not ML one.
itmax = 100; # Number of epoch.
tol = 1e-4
step_size = 0.2; #Learning rate
Err = [];
#Defining the Loss Function
def Loss(x,comm):
    rank = comm.Get_rank();
    S = tf.Variable(0.0);
    pX =X[indx[rank]]
    pY =Y[indx[rank]]
    for j in range(pX.shape[0]):
        a = tf.constant((pX[j,:].todense().reshape(119,1)),dtype=np.float32);
        b = tf.constant(pY[j],dtype=np.float32)
        a = tf.reshape(a,(119,1));
        x = tf.reshape(x,(119,1));
        dot = tf.matmul(tf.transpose(a),x);
        S = S+tf.math.log(1+tf.math.exp(-b*dot))
    S = (1/pX.shape[0])*S;
    return S;
#Defining the Hessian class for the above loss function in x0
x = tf.Variable(0.1*np.ones((119,1),dtype=np.float32))
H =  Hessian(Loss,x)
grad = H.grad().numpy();
print("Computed the  first gradient ...")
q = grad #H.pCG(grad,10,2,tol=1e-3,itmax=10);
print("Computed search search diratcion ...")
for it in tqdm(range(itmax)):
    x = x - tf.constant(step_size,dtype=np.float32)*tf.Variable(q,dtype=np.float32);
    x =  tf.Variable(x)
    if it%50 == 0:
        print("Lost funciton at this iteration {}  and gradient norm {}".format(Loss(x,H.comm),np.linalg.norm(grad)));
    if np.linalg.norm(grad)<tol:
        break
    H =  Hessian(Loss,x)
    grad = H.grad().numpy();
    q = grad #H.pCG(grad,10,2,tol=1e-3,itmax=10);
itmax = 200; # Number of epoch.
for it in tqdm(range(itmax)):
    x = x - tf.constant(step_size,dtype=np.float32)*tf.Variable(q,dtype=np.float32);
    x =  tf.Variable(x)
    Err = Err + [np.linalg.norm(grad)];
    if it%5 == 0:
        print("Lost funciton at this iteration {}  and gradient norm {}".format(Loss(x,H.comm),np.linalg.norm(grad)));
    if np.linalg.norm(grad)<tol:
        break
    H =  Hessian(Loss,x)
    grad = H.grad().numpy();
    U, s, Vt = H.RandMatSVD(50,10)
    q = (Vt.transpose()@np.linalg.inv(np.diag(s))@U.transpose())@grad;
print("Lost funciton at this iteration {}  and gradient norm {}".format(Loss(x,H.comm),np.linalg.norm(grad)));
