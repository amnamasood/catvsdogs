import os
from PIL import Image
import numpy as np
import tensorflow as tf

def  load_data(): 
    data = np.empty((402,7500),dtype="float32")
    label = np.empty((402,2),dtype ="float32")
    imgs = os.listdir("./train")
    num = len(imgs)
    j=0
    for i in range(num):
        if not imgs[i].startswith('.'):
            label[j,0]=1 if imgs[i].split('.')[0]=='cat' else 0
            label[j,1]=0 if imgs[i].split('.')[0]=='cat' else 1
            img = Image.open("./train/"+imgs[i])
            img = img.resize((50,50), Image.ANTIALIAS)
            arr = np.asarray (img, dtype ="float32")
            arr=arr.flatten() 
            data[j,:]=arr
            j=j+1
    return data, label

def  load_test(): 
    data = np.empty((100,7500),dtype="float32")
    label = np.empty((100,2),dtype ="float32")
    imgs = os.listdir("./test")
    num = len(imgs)
    j=0
    for i in range(num):
        if not imgs[i].startswith('.'):
            label[j,0]=1 if imgs[i].split('.')[0]=='cat' else 0
            label[j,1]=0 if imgs[i].split('.')[0]=='cat' else 1
            img = Image.open("./test/"+imgs[i])
            img = img.resize((50,50), Image.ANTIALIAS)
            arr = np.asarray (img, dtype ="float32")
            arr=arr.flatten()
            #data [j,:,:,:] = [arr[:,:,0],arr[:,:, 1],arr[:,:, 2]] 
            data[j,:]=arr
            j=j+1
    return data, label

data,label = load_data()
test_data, test_label = load_test()
x = tf.placeholder(tf.float32, [None, 7500])
W = tf.Variable(tf.truncated_normal([7500, 2],stddev=1))
b = tf.Variable(tf.constant(0.1,shape=[2]))
#y = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.nn.log_softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None,2])
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * y, reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
j=0
k=19
for i in range(20):
  batch_xs = data[j:k, :]
  batch_ys = label[j:k,:]
  j=j+20
  k=k+20
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  #print(sess.run(b))
  #print(sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys}))
#ytest = tf.round(y)
#check1 = tf.cast(check1,tf.float32)
#correct_prediction = tf.equal(y_,ytest)
#print(sess.run(W))
ytest = tf.nn.softmax(tf.matmul(x,W)+b)
correct_prediction = tf.equal(tf.argmax(ytest,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_data, y_: test_label}))