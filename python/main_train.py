# coding: utf-8
import cv2
import numpy as np
import random
import os
import tensorflow as tf
import tempfile
import queue
from tensorflow.python.framework.graph_util import convert_variables_to_constants

NUM_CLASSES = 5
IMG_SIZE = 50


class DataSetQueue:
    def __init__(self):
        self.dataSetArray = []
        self.train_img_dirs = ['chu', 'left', 'normal', 'right', 'smile']
        self.dataSetQueue = queue.Queue()
        self.__train_data_prepare()

    def __train_data_prepare(self):
        # 学習用画像データ
        # train_image = []
        # 学習データのラベル
        # train_label = []

        for i, d in enumerate(self.train_img_dirs):
            files = os.listdir('./FaceLearning/' + d)
            for f in files:
                # 画像読み取り
                img = cv2.imread('./FaceLearning/' + d + '/' + f)
                img = cv2.resize(img, (28, 28))
                img = img.flatten().astype(np.float32) / 255.0
                # train_image.append(img)
                img = np.asanyarray(img)

                tmp = np.zeros(NUM_CLASSES)
                tmp[i] = 1
                # train_label.append(tmp)
                data_set = DataSet(img, tmp)
                self.dataSetArray.append(data_set)

        random.shuffle(self.dataSetArray)
        for i in range(len(self.dataSetArray)):
            self.dataSetQueue.put(self.dataSetArray[i])

    def batch(self, num):
        image_array = []
        label_array = []
        for i in range(num):
            tmp = self.dataSetQueue.get()
            image_array.append(tmp.image)
            label_array.append(tmp.label)
        return image_array, label_array

    def finish(self):
        self.dataSetQueue.task_done()


class DataSet:
    def __init__(self, arg_image, arg_label):
        self.image = np.asanyarray(arg_image)
        self.label = np.asanyarray(arg_label)


def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, w, name):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', name=name)


def max_pool_2x2(x, name):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)


def deepnn(x):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, 28, 28, 3], name='input')

    with tf.name_scope('conv1'):
        w_conv1 = weight_variable([5, 5, 3, 32], name='w_conv1')
        b_conv1 = bias_variable([32], name='b_conv1')
        h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, name='inconvd1') + b_conv1, name='h_conv1')

    with tf.name_scope('pool1'):
        h_pool1 = max_pool_2x2(h_conv1, name='h_pool1')

    with tf.name_scope('conv2'):
        w_conv2 = weight_variable([5, 5, 32, 64], name='w_conv2')
        b_conv2 = bias_variable([64], name='b_con1')
        h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, name='inconvd2') + b_conv2, name='h_conv2')

    with tf.name_scope('pool2'):
        h_pool2 = max_pool_2x2(h_conv2, name='h_pool2')

    with tf.name_scope('fc1'):
        w_fc1 = weight_variable([7 * 7 * 64, 1024], name='w_fc1')
        b_fc1 = bias_variable([1024], name='b_fc1')

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1, name='h_fc1')

    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

    with tf.name_scope('fc2'):
        w_fc2 = weight_variable([1024, 5], name='w_fc2')
        b_fc2 = bias_variable([5], name='b_fc2')
        mul_fc2 = tf.matmul(h_fc1_drop, w_fc2)
        y_conv = tf.add(mul_fc2, b_fc2,name='result')
    return y_conv, keep_prob


def main():
    dataSetQueue = DataSetQueue()
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder(tf.float32, [None, 28 * 28 * 3], name='x')
        y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES], name='y_')

        y_conv, keep_prob = deepnn(x)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_, name='softmax')
            cross_entropy = tf.reduce_mean(cross_entropy, name='reduce_mean')

        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy, name='train_step')

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1), name='equal')
            correct_prediction = tf.cast(correct_prediction, tf.float32, name='cast')

        accuracy = tf.reduce_mean(correct_prediction, name='accuracy_1')

        # graph_location = tempfile.mkdtemp()
        # print('Saving graph to : %s' % graph_location)

        # train_witer = tf.summary.FileWriter(graph_location)
        # rain_witer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(90):
                image, label = dataSetQueue.batch(50)
                if i % 5 == 0:
                    train_accuracy = accuracy.eval(feed_dict={
                        x: image, y_: label, keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))

                train_step.run(feed_dict={x: image, y_: label, keep_prob: 0.5})
            [print(n.name) for n in sess.graph.as_graph_def().node]
            minimal_graph = convert_variables_to_constants(sess, sess.graph_def, ['accuracy_1'])
            tf.train.write_graph(minimal_graph, './', 'trained_graph.pb', as_text=False)
            tf.train.write_graph(minimal_graph, './', 'train_graph.txt', as_text=True)


if __name__ == '__main__':
    main()
