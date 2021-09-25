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
