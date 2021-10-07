import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numsa.TFHessian import *
from tqdm import tqdm
import mpi4py as MPI

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
# Inputs
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
# Outputs

training_label = np.array([[0],[1],[1],[0]], "float32")
training_dataset = tf.data.Dataset.from_tensor_slices((training_data, training_label))
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2,input_dim=2, activation='sigmoid',bias_initializer=init),#2 nodes hidden layer
    tf.keras.layers.Dense(1)
])
def loss_fn(y,x):
    return tf.math.reduce_mean(tf.math.squared_difference(y, x));
def Loss(weights):
    predictions = model(training_data, training=True) #Logits for this minibatch
    # Compute the loss value for this minibatch.
    loss_value = loss_fn(training_label, predictions);
    return loss_value;
for epoch in range(100):
    for step, (x,y) in enumerate(training_dataset):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            labels = model(training_data, training=True)  # Logits for this minibatch
            # Compute the loss value for this minibatch.
            loss_value = loss_fn(training_label, labels)
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
def Loss(weights):
    predictions = model(training_data, training=True) #Logits for this minibatch
    loss_value = loss_fn(training_label, predictions);
    return loss_value;
H = Hessian(Loss,model.trainable_weights)
H.SwitchVerbose(True);
print("\nRank: ",H.comm.Get_rank())
fullH = np.zeros((9,9));
fullH = H.mat(model.trainable_weights,"KERAS");
if H.comm.Get_rank() == 0:
    print("Inside H, \n{}".format(fullH))
    plt.imshow(np.log10(abs(fullH)+1e-16));
    plt.colorbar();
    plt.show()
