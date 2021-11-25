import tensorflow as tf
import numpy as np
import scipy.linalg as la
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
atol = 1e-5;
rtol = 1e-6
step_size=1;
###########################################################
x = tf.Variable(0.1*np.ones((119,1),dtype=np.float32))

Res = [];
TBCHistory = [];
Err = [];
LossStar =  0.33691510558128357;

H = Hessian(Loss,x);
H.loc = True;
#We now collect and average the loc Hessians in the master node (rk 0)

print("Compressing Initial Hessian ")
cU, cS, cVt = MatSVDComp(H.mat()-np.diag(np.diag(H.mat())),40);
print("Assembling the the comp ...")
C =  cU@np.diag(cS)@cVt;
print("Built the compression ...")
A = np.diag(np.diag(H.mat()))+C;
H.shift(x,start=A) #We initialize the shifter
Hs = H.comm.gather(H.memH, root=0);
if H.comm.Get_rank()==0:
    Hm = (1/len(Hs))*np.sum(Hs,0);
else:
    Hm = None
print("The master Hessian has been initialised")
for it in tqdm(range(itmax)):
    # Obtaining the compression of the difference between local mat
    # and next local mat.
    U,sigma,Vt,ell = H.shift(x,{"comp":MatSVDCompDiag,"rk":1,"type":"mat"});
    shift = Vt.transpose()@np.diag(sigma)@U.transpose();
    TBCHistory = TBCHistory + [sigma[0]];
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
        if it == 0:
            normalgrad = np.linalg.norm(grad);
        Res = Res + [res];
        #print("Computing the master Hessian ...")
        Hm = Hm + step_size*Shift;
        #print("Searching new search direction ...")
        A = Hm; #A = Hm + Ell*np.identity(Hm.shape[0]);
        q = np.linalg.solve(A,Grad);
        #print("Found search dir, ",q);
        if it%25 == 0:
            print("(FedNL) [Iteration. {}] Lost funciton at this iteration {}  and gradient norm {}".format(it,LossSerial(x),np.linalg.norm(Grad)));
        Err = Err + [np.abs(LossSerial(x)-LossStar)];
        x = x - tf.Variable(q,dtype=np.float32);
        x =  tf.Variable(x)
    else:
        res = None
        normalgrad = None
    #Distributing the search direction
    x = H.comm.bcast(x,root=0)
    res = H.comm.bcast(res,root=0)
    normalgrad = H.comm.bcast(normalgrad,root=0)
    if res<atol or res<rtol*normalgrad:
        break
    if np.abs(LossStar-LossSerial(x))<1e-7:
        break
print("Lost funciton at this iteration {}, gradient norm {} and error {}.".format(LossSerial(x),np.linalg.norm(grad),abs(LossSerial(x)-LossStar)))
print("Error History: {}".format(Err))
