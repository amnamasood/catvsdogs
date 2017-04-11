import os
from PIL import Image
import numpy as np
import tensorflow as tf
import glob

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
    image = tf.image.resize_images(image, 50, 50)
    image = tf.reshape(image, [1,7500])
    
    min_after_dequeue = 5
    capacity = min_after_dequeue + 3 * batch_size
    image_batch, label_batch = tf.train.batch( [image, labels], batch_size=batch_size, capacity=capacity)
    image_batch=tf.reshape(image_batch,[batch_size,7500])
    label_batch=tf.reshape(label_batch,[batch_size,2])
    return image_batch, label_batch


x,y_ = input_pipeline(100)

W1 = tf.Variable(tf.truncated_normal([7500, 6],stddev=1.0/7500))
b1 = tf.Variable(tf.zeros([6]))

y1 = tf.matmul(x,W1) + b1
a1 = tf.nn.relu(y1)
#a1 = tf.sigmoid(y1)

W2 = tf.Variable(tf.truncated_normal([6, 6],stddev=1.0/6))
b2 = tf.Variable(tf.zeros([6]))

y2 = tf.matmul(a1,W2) + b2
a2 = tf.nn.relu(y2)
#a2 = tf.sigmoid(y2)

W3 = tf.Variable(tf.truncated_normal([6, 2],stddev=1.0/6))
b3 = tf.Variable(tf.zeros([2]))

y3 = tf.matmul(a2,W3) + b3
#y = tf.nn.relu(y3)
y = tf.nn.softmax(y3)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver({'W1':W1,'b1':b1,'W2':W2,'b2':b2,'W3':W3,'b3':b3})
# Create the graph, etc.
init_op = tf.initialize_all_variables()

# Create a session for running operations in the Graph.
sess = tf.Session()

# Initialize the variables (like the epoch counter).
sess.run(init_op)

saver.restore(sess, '/tmp/model-mlp2.ckpt')
# Start input enqueue threads.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

print(sess.run(y))
print(sess.run(accuracy))

coord.request_stop()

# Wait for threads to finish.
coord.join(threads)
sess.close()
