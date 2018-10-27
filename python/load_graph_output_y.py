from main_train import DataSetQueue
import tensorflow as tf
import time
import numpy as np
import cv2

class GraphAndSession():
    def __init__(self,argGraph,session):
        self.sess = session
        self.graph = argGraph
    
    def getSession(self):
        return self.sess

def import_graph_def():
    f = open('trained_graph.pb', 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    f.close()
    

def getSession():
    graph = tf.Graph()
    # if(graph is None):
    #     return 10
    graph.as_default()
    #import_graph_def()
    # sess = tf.Session()
    # if(sess is None):
    #     return 100
    # return 100
    
    try:
        f = open('trained_graph.pb', 'rb')
    except :
        return 100

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    f.close()
    sess = tf.Session()
    #graphAndSession = GraphAndSession(graph,sess)
    return sess

def hoge(i):
    return 100*i

def createImgArray():
    img = np.ones((28*28*3),dtype=np.float32)
    img[0] = 1.5
    img[1] = 2.88
    img[2] = 3.94
    return img

def estimate(dataPtr,sess):
    img = []

    try:
        #flatImg = np.frombuffer(dataPtr,dtype=np.float32).flatten()
        #showImg = np.reshape(flatImg,(28,28,3))
        #cv2.imshow('hoge',showImg)
        #cv2.waitkey(1)
        # showImg = np.reshape(dataPtr,(28,28,3))
        # cv2.imshow('hoge',showImg)
        # cv2.waitKey(0)
        #dataPtr = dataPtr/255.0
        img.append(dataPtr)
        ans =  sess.run('fc2/result:0', feed_dict={'x:0': img, 'dropout/keep_prob:0': 1.0})
        #print(ans)
        return np.argmax(ans)
    except:
        return 1000


def estimate2(argImg,sess):
    img = []
    dataSetQueue = DataSetQueue()
    image, label = dataSetQueue.batch(1)
    img.append(argImg)

    #img.append(image)
    #img.append(np.frombuffer(dataPtr,dtype=np.float).flatten())
    ans =  sess.run('fc2/result:0', feed_dict={'x:0': image, 'dropout/keep_prob:0': 1.0})
    #print(ans)
    return np.argmax(ans)

def close_sess(session):
    session.close()

def test():
    with tf.Session() as sess:
        for i in range(10):
            dataSetQueue = DataSetQueue()
            image, label = dataSetQueue.batch(1)
            print(label)
            t =  time.time()
            result = sess.run('fc2/result:0', feed_dict={'x:0': image, 'dropout/keep_prob:0': 1.0})
            print('result = ',result)
            print(np.argmax(result))
            print(time.time() - t)

def main2():
    graph = tf.Graph()
    with graph.as_default():
        import_graph_def()
        test()

class SampleClass(object):
    def __init__(self):
        self.x = 1
        self.y = 1

    def getX(self):
        return self.x


def getSampleClassInstance():
    sampleClass = SampleClass()
    return sampleClass

def getSampleClassInstanceMenberX(instance):
    if(instance is None):
        return 100
    return instance.getX()
    

def imgFunction(imgPtr):
    try:
        img=np.frombuffer(imgPtr, dtype=np.uint8).reshape((28, 28,3))
    except :
        return 10        
    cv2.imshow('img',img)
    cv2.waitKey(1)
    cv2.imwrite("LennaG.png",img)
    return 100

def main():
    dataSetQueue = DataSetQueue()
    img, label = dataSetQueue.batch(1)
    sess = getSession()
    print(estimate2(img,sess))
    close_sess(sess)

if __name__ == '__main__':
    main()
