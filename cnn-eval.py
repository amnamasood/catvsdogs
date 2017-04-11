import os
from PIL import Image
import numpy as np
import tensorflow as tf
import glob
import math

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')

def make_lab(filenames):
    n=len(filenames)
    label=np.zeros((n,2),np.float32)
    for i,files in enumerate(filenames):
     if 'cat' in str(files):
       label[i,0]=1.0
     else:
       label[i,1]=1.0
    return label

def input_pipeline(batch_size):
    files=glob.glob('test/*.jpg')
    label=make_lab(files)
    files=tf.convert_to_tensor(files,dtype=tf.string)
    label=tf.convert_to_tensor(label,dtype=tf.float32)
    
    filename_queue = tf.train.slice_input_producer([files,label],shuffle=True)

    labels=filename_queue[1]
    image = tf.image.decode_jpeg(tf.read_file(filename_queue[0]), channels=3)
    image = tf.image.resize_images(image, 28, 28)
    
    min_after_dequeue = 5
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.batch( [image, labels], batch_size=batch_size, capacity=capacity)
    image_batch=tf.reshape(image_batch,[batch_size,28,28,3])
    label_batch=tf.reshape(label_batch,[batch_size,2])
    return image_batch, label_batch


x,y_ = input_pipeline(100)

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([4 * 4 * 64, 100])
b_fc1 = bias_variable([100])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([100, 64])
b_fc2 = bias_variable([64])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2) + b_fc2)

W_fc3 = weight_variable([64, 2])
b_fc3 = bias_variable([2])
h_fc3 = tf.matmul(h_fc2,W_fc3) + b_fc3
y_conv = tf.nn.softmax(h_fc3)

correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver({'W_conv1':W_conv1,'b_conv1':b_conv1,'W_conv2':W_conv2,'b_conv2':b_conv2,'W_fc1':W_fc1,'b_fc1':b_fc1,'W_fc2':W_fc2,'b_fc2':b_fc2,'W_fc3':W_fc3,'b_fc3':b_fc3})

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
saver.restore(sess, '/tmp/model-cnn3.ckpt')


# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print(sess.run(y_conv))
print(accuracy.eval())
print(sess.run(b_fc2))
coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
