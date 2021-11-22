import tensorflow as tf
import numpy as np
import scipy.linalg as la
import pandas as pd
import matplotlib.pyplot as plt
from numsa.TFHessian import *
import dsdl
from copy import copy, deepcopy

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
itmax = 100
tol = 1e-4;
N = 119;
step_size = 2.0
avg = False;

bkitmax = 1;
tau = 0.5
c = 0.5;
###########################################################
Residuals = [];
TBCHistory = [];
Err = [];
x = tf.Variable(0.1*np.ones((119,1),dtype=np.float32))
LossStar =  0.33691510558128357;
H = Hessian(Loss,x);
Hm = Hessian(Loss,x);
M = H.comm.Get_size();
H.shift(x, opt={"type":"act"},start=H.vecprod)
#We now collect and average the loc Hessians in the master node (rk 0)
#initU, initS, initVt = H.RandMatSVD(40,10);
#init = initU@np.diag(initS)@initVt;
init = H.mat()
Inits = H.comm.gather(init, root=0);

if H.comm.Get_rank()==0:
    QInv = np.linalg.inv((1/len(Inits))*np.sum(Inits,0));
    #QInv = 1e3*np.identity(N)
else:
    QInv = None;

print("The master Hessian has been initialised")
for it in tqdm(range(itmax)):
    # Obtaining the compression of the difference between local mat
    # and next local mat.
    U,sigma,Vt = H.shift(x,{"comp":ActHalko,"rk":1,"type":"act"});
    #print("TBC: {}".format(sigma[0]));
    #print("Updating local Hessian")
    Hm = Hessian(Loss,x);
    H.memH =copy(Hm.vecprod);
    grad = H.grad().numpy();
    #Now we update the master Hessian and perform the Newton method step
    ShiftUs = H.comm.gather(U, root=0);
    ShiftVs = H.comm.gather(sigma[0]*Vt, root=0);
    Grads = H.comm.gather(grad, root=0);
    Umat = np.zeros((N,H.comm.Get_size()));
    Vmat = np.zeros((H.comm.Get_size(),N));
    if H.comm.Get_rank() == 0:
        #print("Computing the avarage of the local shifts and grad ...")
        if not avg:
            for j in range(H.comm.Get_size()):
                Umat[:,j] = ShiftUs[j].reshape(N,);
                Vmat[j,:] = ShiftVs[j];
            Uu, Su, Vut = np.linalg.svd(Umat);
            Uv, Sv, Vvt = np.linalg.svd(Vmat);
            #Building the Rank 1 approximation
            u = Uu[:,0].reshape((N,1));
            v = (Su[0]*Sv[0])*Vut[0,:].reshape(1,M)@Uv[:,0].reshape(M,1)@Vvt[0,:].reshape(1,N)
        else:
            u = (1/len(ShiftUs))*np.sum(ShiftUs,0);
            v = (1/len(ShiftVs))*np.sum(ShiftVs,0);
        Grad = (1/len(Grads))*np.sum(Grads,0);
        res = np.linalg.norm(Grad);
        Residuals = Residuals + [res];
        TBCHistory = TBCHistory + [sigma[0]];
        #print("Computing the master Hessian ...")
        #SHERMAN-MORRISON
        normal = (1+v@QInv@u);
        #print("Normalisation: ",normal);
        A = QInv@u@v@QInv;
        #print("Searching new search direction ...")
        QInv = QInv - (1/(1+normal))*A;
        #Back traking
        step = step_size;
        q =  QInv@Grad;
        for bkit in range(bkitmax):
                m = Grad.T@q
                t = -c*m;
                if LossSerial(x)-LossSerial(x - tf.Variable(step*q,dtype=np.float32))>step*tau:
                    break
                else:
                    step = tau*step;
        #print("Found search dir, ",q.shape);
        if it%1 == 0:
            Err = Err + [np.abs(LossStar-LossSerial(x))];
            print("(FedNL) [Iteration. {}] Lost funciton at this iteration {}  and gradient norm {}, back tracing it {} and step {}.".format(it,LossSerial(x),np.linalg.norm(Grad),bkit,step));
        x = x - tf.Variable(step*q,dtype=np.float32);
        x =  tf.Variable(x)
    else:
        res = None
    #Distributing the search direction
    x = H.comm.bcast(x,root=0)
    res = H.comm.bcast(res,root=0)
    if res<tol:
            break
if H.comm.Get_rank() == 0:   
    print("[Newton] Lost funciton at this iteration {}  and gradient norm {}, lost err {}.".format(LossSerial(x),np.linalg.norm(grad),np.abs(LossStar-LossSerial(x))));
    print("[Error History]: {}.".format(Err))
