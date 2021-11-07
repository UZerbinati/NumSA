import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numsa.TFHessian import *
import dsdl

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
H.loc = True;
grad = H.grad().numpy();
memH = H.mat()
#memH = np.identity(x.shape[0])#H.mat()
for it in tqdm(range(itmax)):
    #Redefine the Hessian
    H = Hessian(Loss,x);
    H.loc = True;
    #Computing Matrix to be compressed
    tbcomp = H.mat()-memH;
    print("Norm of TBC",np.linalg.norm(tbcomp));
    #Computing diagonal app
    ell = np.linalg.norm(tbcomp,ord="fro");
    l = 1 #Rank Compression
    #Compressing
    U, sigma, Vt = la.svd(tbcomp, full_matrices=False);
    sigma = sigma[0:l];
    U = U[:,0:l];
    Vt = Vt[0:l,:];
    #Updating the Hessian
    shift = Vt.transpose()@np.diag(sigma)@U.transpose(); 
    memH = memH+step_size*shift;

    grad = H.grad().numpy();
    #Comunicating Hessian to other core
    Hs = H.comm.gather(memH, root=0);
    ells = H.comm.gather(ell, root=0);
    Grads = H.comm.gather(grad, root=0);
    if H.comm.Get_rank() == 0:
        #Summing Hessian and Gradient up
        Hm = (1/len(Hs))*np.sum(Hs,0);
        Grad = (1/len(Grads))*np.sum(Grads,0);
        Ell = (1/len(ells))*np.sum(ells,0);
        q = np.linalg.solve(Hm,Grad);
        if it%1 == 0:
            print("(FedNL) [Iteration. {}] Lost funciton at this iteration {}  and gradient norm {}".format(it,LossSerial(x),np.linalg.norm(Grad)));
        if np.linalg.norm(Grad)<tol:
            break
        x = x - tf.Variable(q,dtype=np.float32);
        x =  tf.Variable(x)
    #Distributing the search direction
    x = H.comm.bcast(x,root=0)
