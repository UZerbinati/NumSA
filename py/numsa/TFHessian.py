import tensorflow as tf

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
        if grad:
            return backward, acc.jvp(backward);
        else:
            return acc.jvp(backward);
