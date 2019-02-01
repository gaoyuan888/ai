from PIL import Image
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# 定义图
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv1 = tf.constant(0.1, shape=[32])

h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv2 = tf.constant(0.1, shape=[64])

h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W_fc1 = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc1 = tf.constant(0.1, shape=[1024])

h_pool2 = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc2 = tf.constant(0.1, shape=[10])

y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
prediction = tf.argmax(y_conv, 1)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()  # defaults to saving all variables
process_train = False

with tf.Session() as sess:
    if process_train:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(100)
            _, train_accuracy = sess.run([train_step, accuracy],
                feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            if i % 100 == 0:
                print("step %d, training accuracy %g" % (i, train_accuracy))

            # 保存模型参数，注意把这里改为自己的路径
            saver.save(sess, './mnist_model/model.ckpt')
            print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images,
                y_: mnist.test.labels, keep_prob: 1.0}))
    else:
        saver.restore(sess, "./mnist_model/model.ckpt")

    pred_file = "./3.png"
    img_content = Image.open(pred_file)
    img_content = img_content.resize([28, 28])

    pred_content = img_content.convert("1")
    pred_pixel = list(pred_content.getdata())  # get pixel values
    pred_pixel = [(255 - x) * 1.0 / 255.0 for x in pred_pixel]

    pred_num = sess.run(prediction, feed_dict={x: [pred_pixel], keep_prob: 1.0})

    print('recognize result:')
    print(pred_num)
