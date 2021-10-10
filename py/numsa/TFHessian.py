import tensorflow as tf
import numpy as np
import scipy.linalg as la
import sys
from mpi4py import MPI
from tqdm import tqdm

def util_shape_product(Layers):
    N = 0;
    for layer in Layers:
        if len(layer)==2:
            N = N+layer[0]*layer[1];
        elif len(layer)==1:
            N = N+layer[0];
    return N;
class Hessian:
    """
    This class has been created to work with Hessians in TensorFlow.
    In particular the Hessian class is initialised as follow.
    H = Hessian(LossFunction,x0)
    One can compute the action of the Hessian as follow,
    H.action(v) = [d^2(Loss)/d(x^2) (x0)]v
    """
    def __init__(self,loss,where,flag="None"):
        self.flag = flag;
        self.LossFunction = loss;
        self.x0 = where;
        self.SwitchVerbose(False);
        self.comm = MPI.COMM_WORLD;
    def SwitchVerbose(self,state):
        self.verbose = state;
        if not(state):
        	self.new_tqdm = lambda x : x  	
        else:
        	self.new_tqdm = tqdm
    def action(self,v,grad=False):
        """
        If the grad option is True, also the gradient is returned.
        """
        comm = self.comm;
        nprs = comm.Get_size()
        #Check if the lost funciton have been writen from MPI support
        if "comm" in self.LossFunction.__code__.co_varnames:
            #Farward diff. tape for the second derivative
            with tf.autodiff.ForwardAccumulator(self.x0,v) as acc:
                #Backward diff. tape for the first derivative
                with tf.GradientTape() as tape:
                    # It is important to evaluate the lost function inside
                    # the two tape in order to use automatic diff. 
                    LossEvaluation = self.LossFunction(self.x0,comm)
                    backward = tape.gradient(LossEvaluation, self.x0);
                    #print("gradient: ",backward)
            Hvs = comm.gather(acc.jvp(backward), root=0);
            Grads = comm.gather(backward,root=0); 
            if grad:
                Hv = [];
                Grad = [];
                for i in range(len(self.x0)):
                    Hv = Hv+[sum([Hvs[k][i] for k in range(len(Hvs))])];
                    Grad = Grad+[sum([Grads[k][i] for k in range(len(Grads))])];
                return Hv, Grad;
            else:
                Hv = [];
                for i in range(len(self.x0)):
                    Hv = Hv+[sum([Hvs[k][i] for k in range(len(Hvs))])];
                return Hv;
        else:
            #Farward diff. tape for the second derivative
            with tf.autodiff.ForwardAccumulator(self.x0,v) as acc:
                #Backward diff. tape for the first derivative
                with tf.GradientTape() as tape:
                    # It is important to evaluate the lost function inside
                    # the two tape in order to use automatic diff. 
                    LossEvaluation = self.LossFunction(self.x0)
                    backward = tape.gradient(LossEvaluation, self.x0);
                    #print("gradient: ",backward)
            if grad:
                return backward, acc.jvp(backward);
            else:
                return acc.jvp(backward);
    def vecprod(self,w):
        if self.flag=="KERAS":
            model_weights = self.x0;
            u = np.zeros(w.shape[0],)
            v = [];
            #Cyling over the number of layer in the NN to build the vector of the conical
            #base.
            for j in range(len(model_weights)):
                starter = util_shape_product([model_weights[r].shape for r in range(j)])
                vj = np.zeros((util_shape_product([model_weights[j].shape]),));
                vj = w[starter:starter+vj.shape[0]].reshape(vj.shape[0],); 
                vj = tf.Variable(vj, dtype=np.float32);
                vj = tf.reshape(vj, model_weights[j].shape);
                v = v + [vj];
            #Filling the Hessian Matrix
            bindex = 0 #row where we start the filling
            tindex = util_shape_product([model_weights[0].shape]) #row where we end the filling
            #Column where we start the filling;
            for s in range(len(model_weights)-1):
                layerH = self.action(v);
                #removeing none in the layerH
                layerH = [ tf.Variable([0]) if l==None else l for l in layerH];
                u[bindex:tindex] = layerH[s].numpy().reshape(util_shape_product([model_weights[s].shape]));
                bindex = tindex;
                tindex = tindex+util_shape_product([model_weights[s+1].shape])
            u[bindex:tindex] = layerH[-1].numpy().reshape(util_shape_product([model_weights[-1].shape]));
            return u;
    def mat(self,grad=False):
        """
        Setting up the MPI support
        --------------------------
        My idea for parallelizing the code is very stupid, I devide the construction of the 
        canonical base among the processes and the evaluation of the Hessian vector product
        among the same process. Then we use one last process to sum the result Hessian into
        the final Hessian.
        UZ
        """
        comm = self.comm;
        nprs = comm.Get_size()
        nsect = nprs;
        model_weights = self.x0;
        if self.verbose:
            print("MPI the world is {} process big !".format(nprs));
        rank = comm.Get_rank();

        N = util_shape_product([layer.shape for layer in model_weights]);
        matH = np.zeros((N,N));
        Grad = np.zeros((N,));

        if self.flag == "KERAS":
            if (grad):
                #Cycling over the number of layer in the NN
                for k in range(len(model_weights)):
                    #Cycling over the possible combination of the canonical base with 1.0 in the 
                    #layer k.
                    NBase = np.array_split(range(util_shape_product([model_weights[k].shape])),nsect);
                    for i in self.new_tqdm(NBase[rank]): #range(util_shape_product([model_weights[k].shape])):
                        v = [];
                        #Cyling over the number of layer in the NN to build the vector of the conical
                        #base.
                        for j in range(len(model_weights)):
                            vj = np.zeros((util_shape_product([model_weights[j].shape]),));
                            if j == k:
                                vj[i] = 1.0;
                            vj = tf.Variable(vj, dtype=np.float32);
                            vj = tf.reshape(vj, model_weights[j].shape);
                            v = v + [vj];
                        #Filling the Hessian Matrix
                        bindex = 0 #row where we start the filling
                        tindex = util_shape_product([model_weights[0].shape]) #row where we end the filling
                        #Column where we start the filling;
                        starti = util_shape_product([model_weights[r].shape for r in range(k)])
                        for s in range(len(model_weights)-1):
                            layerGrad, layerH =self.action(v, grad=True)
                            Grad[bindex:tindex] = layerGrad[s].numpy().reshape(util_shape_product([model_weights[s].shape]),);
                            matH[bindex:tindex,starti+i] =layerH[s].numpy().reshape(util_shape_product([model_weights[s].shape]));
                            bindex = tindex;
                            tindex = tindex+util_shape_product([model_weights[s+1].shape])
                        layerGrad, layerH =self.action(v, grad=True)
                        Grad[bindex:tindex] = layerGrad[-1].numpy().reshape(util_shape_product([model_weights[-1].shape]),);
                        matH[bindex:tindex,starti+i] =layerH[-1].numpy().reshape(util_shape_product([model_weights[-1].shape]));
                matH = comm.gather(matH,root=0);
            else:
                #Cycling over the number of layer in the NN
                for k in range(len(model_weights)):
                    #Cycling over the possible combination of the canonical base with 1.0 in the 
                    #layer k.
                    Base = util_shape_product([model_weights[k].shape]);
                    NBase = np.array_split(range(Base),nsect);
                    for i in self.new_tqdm(NBase[rank]):
                        v = [];
                        #Cyling over the number of layer in the NN to build the vector of the conical
                        #base.
                        for j in range(len(model_weights)):
                            vj = np.zeros((util_shape_product([model_weights[j].shape]),));
                            if j == k:
                                vj[i] = 1.0;
                            vj = tf.Variable(vj, dtype=np.float32);
                            vj = tf.reshape(vj, model_weights[j].shape);
                            v = v + [vj];
                        #Filling the Hessian Matrix
                        bindex = 0 #row where we start the filling
                        tindex = util_shape_product([model_weights[0].shape]) #row where we end the filling
                        #Column where we start the filling;
                        starti = util_shape_product([model_weights[r].shape for r in range(k)])
                        for s in range(len(model_weights)-1):
                            layerH = self.action(v);
                            #removeing none in the layerH
                            layerH = [ tf.Variable([0]) if l==None else l for l in layerH];
                            matH[bindex:tindex,starti+i] = layerH[s].numpy().reshape(util_shape_product([model_weights[s].shape]));
                            bindex = tindex;
                            tindex = tindex+util_shape_product([model_weights[s+1].shape])
                        matH[bindex:tindex,starti+i] = layerH[-1].numpy().reshape(util_shape_product([model_weights[-1].shape]));
                matH = comm.gather(matH,root=0);
        if (grad):
            if rank == 0:
                return sum(matH), sum(Grad);
            else:
                return 1;
        else:
            if rank == 0:
                return sum(matH);
            else:
                return 1;
    def RandMatSVD(self,k,p):
        model_weights = self.x0;
        if self.flag == "KERAS":
            N = util_shape_product([layer.shape for layer in model_weights]);
            l = k+p;
            mu, sigma = 0, 1 # mean and standard deviation
            omega = np.random.normal(mu, sigma, (N,l))
            Y = np.zeros((N,l));
            Bt = np.zeros((N,l));
            for i in range(l):
                Y[:,i] = self.vecprod(omega[:,i].reshape(N,1))
            Q,R = la.qr(Y);
            #HALKO 4.1
            #
            #(N,l) Y = A * Omega (N,N)(N,l)
            #(N,l)(l,l) QR = Y (N,l)
            #(l,N) B = Q^t A (l,N)(N,N)
            #Bt = A^t Q = AQ
            #USV^t = B
            Q,R = la.qr(Y);
            for i in range(l):
                Bt[:,i] = self.vecprod(Q[:,i])
            B = Bt.T
            U, sigma, Vt = la.svd(B); 
            return U, sigma, Vt;

    def eig(self,flag,itmax=10):
        if flag == "pi-max":
            v = self.x0;
            v = (1/tf.norm(v))*v;
            for i in range(itmax):
                v = self.action(v);
                v = (1/tf.norm(v))*v;
            return tf.tensordot(v,self.action(v),1);
