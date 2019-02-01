import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow as tf

embeddings_file = './w2v_data/GoogleNews-vectors-negative2000.bin'
log_path = './w2v_model'

demo_vocab_size = 1000

# 建立一个 list of vectors 和 a list of words
with open(embeddings_file, "rb") as f:
    header = f.readline()
    vocab_size, vector_size = map(int, header.split())
    binary_len = np.dtype('float32').itemsize * vector_size

    # 每一行是一个单词和一个词向量，使用空格' '分开
    words, embeddings = [], []
    for line in range(demo_vocab_size):
        word = []
        while True:
            ch = f.read(1)
            if ch == b' ':
                word = b''.join(word).decode('utf-8').encode("gbk", "ignore").decode("utf-8")
                break
            word.append(ch)
        words.append(word)

        # 读入空格右边的词向量, 存入embeddings
        vector = np.fromstring(f.read(binary_len), dtype='float32')
        embeddings.append(vector)


with tf.Session() as sess:
    X = tf.Variable([0.0], name='embedding')
    place = tf.placeholder(tf.float32, shape=[len(words), vector_size])
    set_x = tf.assign(X, place, validate_shape=False)

    sess.run(tf.global_variables_initializer())
    sess.run(set_x, feed_dict={place: embeddings})

    # 需要保存一个metadata文件,给词典里每一个词分配一个身份
    with open(log_path + '/metadata.tsv', 'w') as f:
        for word in words:
            f.write(word + "\n")

    # 写 TensorFlow summary
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = 'embedding:0'
    embedding_conf.metadata_path = os.path.join('metadata.tsv')
    projector.visualize_embeddings(summary_writer, config)

    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_path, "w2v.ckpt"))


#   `cd ./data && tensorboard --logdir .` 打开 tensorboard
