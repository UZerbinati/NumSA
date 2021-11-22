#We import all the library we are gona need
import tensorflow as tf
import numpy as np
import pandas as pd
#%matplotlib inline
import matplotlib.pyplot as plt
from numsa.TFHessian import *
import dsdl

ds = dsdl.load("a1a")

X, Y = ds.get_train()

#Setting the parameter of this run, we will use optimization nomeclature not ML one.
itmax = 100; # Number of epoch.
atol = 1e-5
rtol = 1e-6
step_size = 1; #Learning rate
Err = [];
tfX = tf.sparse.from_dense(np.array(X.todense(), dtype=np.float32))
tfY = tf.convert_to_tensor(np.array(Y, dtype=np.float32).reshape(X.shape[0], 1))


#Defining the Loss Function
def Loss(x):
    lam = 1e-3;
    x = tf.reshape(x, (119, 1))
    Z = tf.sparse.sparse_dense_matmul(tfX, x, adjoint_a=False)
    Z = tf.math.multiply(tfY, Z)
    S = tf.reduce_sum(tf.math.log(1 + tf.math.exp(-Z)) / tfX.shape[0])+lam*tf.norm(x)**2
    return S

#Defining the Hessian class for the above loss function in x0
x = tf.Variable(0.1*np.ones((119,1),dtype=np.float32))
LossStar =  0.33691510558128357;
H =  Hessian(Loss,x)
H.verbose = 1; 
grad = H.grad().numpy();
normalgrad = np.linalg.norm(grad);
for it in tqdm(range(itmax)):
    Err = Err + [np.linalg.norm(grad)];
    if it%1 == 0:
        if H.comm.Get_rank() == 0:
            print("[Newton] Lost funciton at this iteration {}  and gradient norm {}, lost err {}.".format(Loss(x),np.linalg.norm(grad),np.abs(LossStar-Loss(x))));
    if np.linalg.norm(grad)<atol or np.linalg.norm(grad)<rtol*normalgrad:
        break
    if np.abs(LossStar-Loss(x))<1e-7:
        break
    H =  Hessian(Loss,x)
    H.verbose = 1; 
    grad = H.grad().numpy();
    q = H.CG(grad,rtol=1e-2,atol=1e-7,itmax=1000);
    x = x - tf.constant(step_size,dtype=np.float32)*tf.Variable(q,dtype=np.float32);
    x =  tf.Variable(x)
print("[Newton] Lost funciton at this iteration {}  and gradient norm {}, lost err {}.".format(Loss(x),np.linalg.norm(grad),np.abs(LossStar-Loss(x))));
