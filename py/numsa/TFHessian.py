import tensorflow as tf
import numpy as np
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
    def __init__(self,loss,where):
        self.LossFunction = loss;
        self.x0 = where;
        self.verbose = False;
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
    def mat(self,model_weights,flag,grad=False):
        """
        Setting up the MPI support
        --------------------------
        My idea for parallelizing the code is very stupid, I devide the construction of the 
        canonical base among the processes and the evaluation of the Hessian vector product
        among the same process. Then we use one last process to sum the result Hessian into
        the final Hessian.
        UZ
        """
        comm = MPI.COMM_WORLD;
        nprs = comm.Get_size()
        if nprs == 1:
            nsect = 1;
        else:
            nsect = nprs - 1;
        if self.verbose:
            print("MPI the world is {} process big !".format(nprs));
        rank = comm.Get_rank();

        N = util_shape_product([layer.shape for layer in model_weights]);
        matH = np.zeros((N,N));
        Grad = np.zeros((N,));
        if flag == "KERAS":
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
                        if rank == 0:
                            for l in range(1,nprs):
                                matH = matH + com.recv(source=l);
                        else:
                            comm.send(matH, dest=0);
            else:
                #Cycling over the number of layer in the NN
                for k in range(len(model_weights)):
                    #Cycling over the possible combination of the canonical base with 1.0 in the 
                    #layer k.
                    NBase = np.array_split(range(util_shape_product([model_weights[k].shape])),nsect);
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
                        if rank == 0:
                            for l in range(1,nprs):
                                matH = matH + com.recv(source=l);
                        else:
                            comm.send(matH, dest=0);

        if (grad):
            return matH, Grad;
        else:
            return matH;
    def eig(self,flag,itmax=10):
        if flag == "pi-max":
            v = self.x0;
            v = (1/tf.norm(v))*v;
            for i in range(itmax):
                v = self.action(v);
                v = (1/tf.norm(v))*v;
            return tf.tensordot(v,self.action(v),1);
