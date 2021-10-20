#!/usr/bin/env python
# coding: utf-8

# ### Newton Method

# Our objective is to minimize the function:
# \begin{equation}
#     f(\vec{x}) = \frac{1}{m} \sum_{i=1}^m \log\Bigg(1+\exp\Big(-b_j \vec{a_j}^T\vec{x}\Big)\Bigg)\qquad for \; x \in \mathbb{R}^d
# \end{equation}
# where $d$ is the feature number and $\vec{a}_j$ are the data while $b_j$ are the labels.
# Now we would like to this applying the newton method to find a point that minimize such a function. This is possible because since $f$ is convex, all stationary points are minimizers and we search for the "roots" of the equation $\nabla f=0$.
# The newton method we implement is of the form,
# \begin{equation}
#     \vec{x}_{n+1} = \vec{x}_n -\gamma Hf(\vec{x}_n)^{-1}\nabla f(\vec{x}_n)
# \end{equation}
# where $\gamma$ is the step size.
# We solve the system $Hf(\vec{x}_n)q=\nabla f(\vec{x}_n)$ using the CG method where as a preconditioned we have taken a the inverse of $Hf(\vec{x}_n)$ computed using the random SVD presented in [1].

# In[5]:


#We import all the library we are gona need
import tensorflow as tf
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from numsa.TFHessian import *
import dsdl


# In[11]:


ds = dsdl.load("a1a")

X, Y = ds.get_train()
X = X[1:100];
Y = Y[1:100];
print(X.shape, Y.shape)


# In[ ]:


#Setting the parameter of this run, we will use optimization nomeclature not ML one.
itmax = 500; # Number of epoch.
tol = 1e-2
step_size = 0.2; #Learning rate
#Defining the Loss Function
def Loss(x):
    S = tf.Variable(0.0);
    for j in range(X.shape[0]):
        a = tf.constant((X[j,:].todense().reshape(119,1)),dtype=np.float32);
        b = tf.constant(Y[j],dtype=np.float32)
        a = tf.reshape(a,(119,1));
        x = tf.reshape(x,(119,1));
        dot = tf.matmul(tf.transpose(a),x);
        S = S+tf.math.log(1+tf.math.exp(-b*dot))
    S = (1/X.shape[0])*S;
    return S;
#Defining the Hessian class for the above loss function in x0
x = tf.Variable(0.1*np.ones((119,1),dtype=np.float32))
H =  Hessian(Loss,x)
grad = H.grad().numpy();
print("Computed the  first gradient ...")
q = H.pCG(grad,10,2,tol=1e-3,itmax=1000);
print("Computed search search diratcion ...")
print("Entering the Netwton optimization loop")
for it in tqdm(range(itmax)):
    x = x - tf.constant(step_size,dtype=np.float32)*tf.Variable(q,dtype=np.float32);
    x =  tf.Variable(x)
    if it%50 == 0:
        print("Lost funciton at this iteration {}  and gradient norm {}".format(Loss(x),np.linalg.norm(grad)));
    if np.linalg.norm(grad)<tol:
        break
    H =  Hessian(Loss,x)
    grad = H.grad().numpy();
    q = H.pCG(grad,10,2,tol=1e-3,itmax=1000);


# In[ ]:


print("Lost funciton at this iteration {}  and gradient norm {}".format(Loss(x),np.linalg.norm(grad)));

