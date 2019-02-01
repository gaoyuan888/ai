import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

train_data = input_data.read_data_sets("/tmp/data/", one_hot=True)

# 参数
learning_rate = 0.01
training_epochs = 20
batch_size = 120

# 输入数据
x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])  # 模型参数
W = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1), name="W")
b = tf.Variable(tf.constant(0.1, shape=[10]), name="b")
# 构造模型
y_pred = tf.nn.softmax(tf.matmul(x_data, W) + b)
# 损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y_pred), reduction_indices=1))
# 优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    # 开始执行程序，每次选择batch的样本
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(train_data.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = train_data.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x_data: batch_xs, y_data: batch_ys})
            avg_cost += c / total_batch
        if (epoch + 1) % 10 == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)

            # 模型测试
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print "Accuracy:", accuracy.eval({x_data: train_data.test.images, y_data: train_data.test.labels})

