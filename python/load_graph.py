from main_train import DataSetQueue
import tensorflow as tf


def import_graph_def():
    with open('trained_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def test():
    with tf.Session() as sess:
        dataSetQueue = DataSetQueue()
        image, label = dataSetQueue.batch(50)
        print('accuracy = ',
              sess.run('accuracy_1:0', feed_dict={'x:0': image, 'y_:0': label, 'dropout/keep_prob:0': 1.0}))


def main():
    graph = tf.Graph()
    with graph.as_default():
        import_graph_def()
        test()


if __name__ == '__main__':
    main()
