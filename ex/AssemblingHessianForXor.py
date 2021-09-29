"""
Example on how to assemble an Hessian from a Neural Network
"""


fullH = np.zeros((9,9));
for i in range(4):
    v1 = np.zeros((4,),dtype=np.float32);
    v1[i] = 1.0; 
    v1 = tf.Variable(v1.reshape(2,2),dtype=np.float32)
    v2 = np.zeros((2,),dtype=np.float32);
    v2 = tf.Variable(v2,dtype=np.float32)
    v3 = tf.Variable(np.zeros((2,1),dtype=np.float32));
    v4 = tf.Variable(np.zeros((1,),dtype=np.float32));
    v = [v1,v2,v3,v4]
    fullH[0:4,i] = H.action(v)[0].numpy().reshape(4,);
    fullH[4:6,i] = H.action(v)[1].numpy().reshape(2,);
    fullH[6:8,i] = H.action(v)[2].numpy().reshape(2,);
    fullH[8,i] = H.action(v)[3].numpy().reshape(1,);
for i in range(2):
    v1 = np.zeros((4),dtype=np.float32);
    v1 = tf.Variable(v1.reshape(2,2),dtype=np.float32)
    v2 = np.zeros((2,),dtype=np.float32);
    v2[i] = 1.0;
    v2 = tf.Variable(v2,dtype=np.float32)
    v3 = tf.Variable(np.zeros((2,1),dtype=np.float32));
    v4 = tf.Variable(np.zeros((1,),dtype=np.float32));
    v = [v1,v2,v3,v4]
    fullH[0:4,4+i] = H.action(v)[0].numpy().reshape(4,);
    fullH[4:6,4+i] = H.action(v)[1].numpy().reshape(2,);
    fullH[6:8,4+i] = H.action(v)[2].numpy().reshape(2,);
    fullH[8,4+i] = H.action(v)[3].numpy().reshape(1,);
for i in range(2):
    v1 = np.zeros((4),dtype=np.float32);
    v1 = tf.Variable(v1.reshape(2,2),dtype=np.float32)
    v2 = np.zeros((2,),dtype=np.float32);
    v2 = tf.Variable(v2,dtype=np.float32)
    v3 = np.zeros((2,),dtype=np.float32);
    v3[i] = 1;
    v3 = tf.Variable(v3.reshape(2,1),dtype=np.float32);
    v4 = tf.Variable(np.zeros((1,),dtype=np.float32));
    v = [v1,v2,v3,v4]
    fullH[0:4,6+i] = H.action(v)[0].numpy().reshape(4,);
    fullH[4:6,6+i] = H.action(v)[1].numpy().reshape(2,);
    fullH[6:8,6+i] = H.action(v)[2].numpy().reshape(2,);
    fullH[8,6+i] = H.action(v)[3].numpy().reshape(1,);
for i in range(1):
    v1 = np.zeros((4),dtype=np.float32);
    v1 = tf.Variable(v1.reshape(2,2),dtype=np.float32)
    v2 = np.zeros((2,),dtype=np.float32);
    v2 = tf.Variable(v2,dtype=np.float32)
    v3 = np.zeros((2,),dtype=np.float32);
    v3 = tf.Variable(v3.reshape(2,1),dtype=np.float32);
    v4 = np.zeros(1,);
    v4[i] = 1.0;
    v4 = tf.Variable(v4.reshape(1,),dtype=np.float32);
    v = [v1,v2,v3,v4]
    fullH[0:4,8+i] = H.action(v)[0].numpy().reshape(4,);
    fullH[4:6,8+i] = H.action(v)[1].numpy().reshape(2,);
    fullH[6:8,8+i] = H.action(v)[2].numpy().reshape(2,);
    fullH[8,8+i] = H.action(v)[3].numpy().reshape(1,);
plt.imshow((np.log10(abs(fullH)+1e-16)))
plt.colorbar()
print(np.min(abs(fullH)))
