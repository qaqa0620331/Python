# TensorFlow深度學習
```
Deep Learning with TensorFlow - Second Edition
Giancarlo Zaccone, Md. Rezaul Karim
March 2018 2018年3月

https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-tensorflow-second-edition

```


```
第 1 章 深度學習入門 1
1．1 機器學習簡介 1
1．1．1 監督學習 2
1．1．2 無監督學習 2
1．1．3 強化學習 3
1．2 深度學習定義 3
1．2．1 人腦的工作機制 3
1．2．2 深度學習歷史 4
1．2．3 應用領域 5
1．3 神經網路 5
1．3．1 生物神經元 5
1．3．2 人工神經元 6
1．4 人工神經網路的學習方式 8
1．4．1 反向傳播演算法 8
1．4．2 權重優化 8
1．4．3 隨機梯度下降法 9
1．5 神經網路架構 10
1．5．1 多層感知器 10
1．5．2 DNN架構 11
1．5．3 卷積神經網路 12
1．5．4 受限玻爾茲曼機 12
1．6 自編碼器 13
1．7 迴圈神經網路 14
1．8 幾種深度學習框架對比 14
1．9 小結 16

2章 TensorFlow初探 17
2．1 總覽 17
2．1．1 TensorFlow 1．x版本特性 18
2．1．2 使用上的改進 18
2．1．3 TensorFlow安裝與入門 19
2．2 在Linux上安裝TensorFlow 19
2．3 為TensorFlow啟用NVIDIA GPU 20
2．3．1 * 1步：安裝NVIDIA CUDA 20
2．3．2 * 2步：安裝NVIDIA cuDNN v5．1+ 21
2．3．3 第3步：確定GPU卡的CUDA計算能力為3．0+ 22
2．3．4 第4步：安裝libcupti-dev庫 22
2．3．5 第5步：安裝Python（或Python3） 22
2．3．6 第6步：安裝並升級PIP（或PIP3） 22
2．3．7 第7步：安裝TensorFlow 23
2．4 如何安裝TensorFlow 23
2．4．1 直接使用pip安裝 23
2．4．2 使用virtualenv安裝 24
2．4．3 從原始程式碼安裝 26
2．5 在Windows上安裝TensorFlow 27
2．5．1 在虛擬機器上安裝TensorFlow 27
2．5．2 直接安裝到Windows 27
2．6 測試安裝是否成功 28
```
```
python3
Python 3.6.0 |Anaconda 4.3.0 (64-bit)| (default, Dec 23 2016, 12:22:00) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant("Hello Tensorflow!")
>>> sess = tf.Session()
2018-06-16 23:57:27.696815: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-16 23:57:27.696860: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-16 23:57:27.696873: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
>>> print sess.run(hello)
  File "<stdin>", line 1
    print sess.run(hello)
             ^
SyntaxError: invalid syntax
>>> print(sess.run(hello))
b'Hello Tensorflow!'
```
```
ksu@ksu:~$ python2
Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant("Hello Tensorflow!")
>>> sess = tf.Session()
2018-06-17 00:01:00.510786: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-17 00:01:00.510826: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2018-06-17 00:01:00.510834: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
>>> print sess.run(hello)
Hello Tensorflow!
```

Session類別的run()函式:run (fetches, feed_dict=None, options=None, run_metadata)

placeholder(參數1,參數2,參數3)
>* 參數1==>data type資料類型 ==>  tf.float32
>* 參數2==>shape of a placeholder==>形狀(shape):the number of rows and columns it has
>* 參數3==>變數名稱name, very useful for debugging and code analysis purposes.

#### 範例1:兩數相乘
```
import tensorflow as tf
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

a = tf.placeholder("float")
b = tf.placeholder("float")

y = tf.multiply(a, b)
sess = tf.Session()
print(sess.run(y, feed_dict={a: 3, b: 5}))
```
```
import tensorflow as tf
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

x = tf.constant(8)
y = tf.constant(9)
z = tf.multiply(x, y)

sess = tf.Session()
out_z = sess.run(z)

print('The multiplication of x and y: %d' % out_z)
```

#### 範例2:兩向量相加(第二版)

```
#定義計算圖Computational Graph==>劃出此範例的Computational Graph
v_1 = tf.constant([1,2,3,4])
v_2 = tf.constant([2,1,5,3])
v_add = tf.add(v_1,v_2) # You can also write v_1 + v_2 instead

#執行計算圖Computational Graph
with tf.Session() as sess:
   print(sess.run(v_add))
```
```
print(tf.Session().run(tf.add(tf.constant([1,2,3,4]),tf.constant([2,1,5,3]))))
```


2．7 計算圖Computational Graph

https://bobondemon.github.io/2017/11/29/TF-Notes-Computational-Graph-in-Tensorflow/

deferred execution

Neural networks as computational graphs


2．8 為何採用計算圖 29

2．9 程式設計模型The programming model 
```
A TensorFlow program is generally divided into three phases:
Construction of the computational graph
Running a session, which is performed for the operations defined in the graph
Resulting data collection and analysis
```
```
import tensorflow as tf

# 建立computational graph計算圖
with tf.Session() as session:
    x = tf.placeholder(tf.float32,[1],name="x") #定義變數
    y = tf.placeholder(tf.float32,[1],name="y") #定義變數
    z = tf.constant(2.0)   #定義常數
    y = x * z
    
# 定義computational model
x_in = [100]
y_output = session.run(y,{x:x_in}) #執行計算圖
print(y_output)
```


2．10 資料模型Data model==>階(Rank)+形狀(shape)+資料類型(data type)
```
import tensorflow as tf

scalar = tf.constant(100)
vector = tf.constant([1,2,3,4,5])
matrix = tf.constant([[1,2,3],[4,5,6]])
cube_matrix = tf.constant([[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]]])

# 使用get_shape()去取得形狀(shape)
print(scalar.get_shape())
print(vector.get_shape())
print(matrix.get_shape())
print(cube_matrix.get_shape())
```
2．10．2 形狀(shape)

scalar1.get_shape()

vector1.get_shape()

matrix1.get_shape()

cube1.get_shape()

2．10．3 資料類型(data type)

To build a tensor with a constant value, pass a NumPy array to the tf.constant()
operator, and the result will be a TensorFlow tensor with that value
```
import tensorflow as tf
import numpy as np

tensor_1d = np.array([1,2,3,4,5,6,7,8,9,10])
tensor_1d = tf.constant(tensor_1d)

with tf.Session() as sess:
    print (tensor_1d.get_shape())
    print (sess.run(tensor_1d))
```   

To build a tensor, with variable values, use a NumPy array and pass it to the
tf.Variable constructor, the result will be a TensorFlow variable tensor with that initial
value:
``` 
import tensorflow as tf
import numpy as np

#tensore 2d con valori variabili
tensor_2d = np.array([(1, 2, 3), (4, 5, 6), (7, 8, 9)])
tensor_2d = tf.Variable(tensor_2d)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tensor_2d.get_shape())
    print(sess.run(tensor_2d))


tensor_3d = np.array([[[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8]],
                      [[ 9, 10, 11],[12, 13, 14],[15, 16, 17]],
                      [[18, 19, 20],[21, 22, 23],[24, 25, 26]]])

tensor_3d = tf.convert_to_tensor(tensor_3d, dtype=tf.float64)
with tf.Session() as sess:
    print(tensor_3d.get_shape())
    print(sess.run(tensor_3d))


interactive_session = tf.InteractiveSession()
tensor = np.array([1, 2, 3, 4, 5])
tensor = tf.constant(tensor)
print(tensor.eval())
interactive_session.close()

"""
Python 2.7.10 (default, Oct 14 2015, 16:09:02) 
[GCC 5.2.1 20151010] on linux2
Type "copyright", "credits" or "license()" for more information.
>>> ================================ RESTART ================================
>>> 
(10,)
[ 1  2  3  4  5  6  7  8  9 10]
(3, 3)
[[1 2 3]
 [4 5 6]
 [7 8 9]]
(3, 3, 3)
[[[  0.   1.   2.]
  [  3.   4.   5.]
  [  6.   7.   8.]]

 [[  9.  10.  11.]
  [ 12.  13.  14.]
  [ 15.  16.  17.]]

 [[ 18.  19.  20.]
  [ 21.  22.  23.]
  [ 24.  25.  26.]]]
[1 2 3 4 5]
>>> 
"""

``` 

2．10．4 變數Variables

Variables are TensorFlow objects that hold and update parameters. A variable must be
initialized; also you can save and restore it to analyze your code.
``` 
import tensorflow as tf
import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

#value = tf.Variable(0, name="value")

value = tf.get_variable("value", shape=[], dtype=tf.int32, initializer=None, regularizer=None, trainable=True, collections=None)

one = tf.constant(1)
update_value = tf.assign_add(value, one)
initialize_var = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(initialize_var)
    print(sess.run(value))
    for _ in range(5):
        sess.run(update_value)
        print(sess.run(value))
``` 
2．10．5 取回Fetches
``` 
import tensorflow as tf

constant_A = tf.constant([100.0])
constant_B = tf.constant([300.0])
constant_C = tf.constant([3.0])

sum_ = tf.add(constant_A, constant_B)
mul_ = tf.multiply(constant_A, constant_C)

with tf.Session() as sess:
    result = sess.run([sum_, mul_])
    print(result)
``` 

2．10．6 注入 38
```
import tensorflow as tf
import numpy as np

a = 3
b = 2

x = tf.placeholder(tf.float32, shape=(a, b))
y = tf.add(x, x)

data = np.random.rand(a, b)

sess = tf.Session()

print(sess.run(y,feed_dict={x: data}))
```
```
2．11 TensorBoard 38
2．12 實現一個單輸入神經元 39
2．13 單輸入神經元原始程式碼 43


2．14 遷移到TensorFlow 1．x版本 43
2．14．1 如何用腳本升級 44
2．14．2 局限 47
2．14．3 手動升級代碼 47
2．14．4 變數 47
2．14．5 匯總函數 47
2．14．6 簡化的數學操作 48
2．14．7 其他事項 49
2．15 小結 49
```

第二版新增:Linear regression and beyond

```
# Import libraries (Numpy, Tensorflow, matplotlib)
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf

import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

# Create 1000 points following a function y=0.1 * x + 0.4 (i.e. y = W * x + b) with some normal random distribution
num_points = 1000
vectors_set = []
for i in range(num_points):
    W = 0.1  # W
    b = 0.4  # b
    x1 = np.random.normal(0.0, 1.0)
    nd = np.random.normal(0.0, 0.05)
    y1 = W * x1 + b
    # Add some impurity with the some normal distribution -i.e. nd
    y1 = y1 + nd
    # Append them and create a combined vector set
    vectors_set.append([x1, y1])

# Seperate the data point across axixes
x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# Plot and show the data points on a 2D space
plot.plot(x_data, y_data, 'ro', label='Original data')
plot.legend()
plot.show()

#tf.name_scope organize things on the tensorboard graph view
with tf.name_scope("LinearRegression") as scope:
	W = tf.Variable(tf.zeros([1]))
	b = tf.Variable(tf.zeros([1]))
	y = W * x_data + b

# Define a loss function that take into account the distance between the prediction and our dataset
with tf.name_scope("LossFunction") as scope:
	loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.6)
train = optimizer.minimize(loss)

# Annotate loss, weights and bias (Needed for tensorboard)
loss_summary = tf.summary.scalar("loss", loss)
w_ = tf.summary.histogram("W", W)
b_ = tf.summary.histogram("b", b)

# Merge all the summaries
merged_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Writer for tensorboard (Directory)
writer_tensorboard = tf.summary.FileWriter('logs/', tf.get_default_graph())

for i in range(6):
	sess.run(train)
	print(i, sess.run(W), sess.run(b), sess.run(loss))
	plot.plot(x_data, y_data, 'ro', label='Original data')
	plot.plot(x_data, sess.run(W)*x_data + sess.run(b))
	plot.xlabel('X')
	plot.xlim(-2, 2)
	plot.ylim(0.1, 0.6)
	plot.ylabel('Y')
	plot.legend()
	plot.show()

```
```
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from numpy import genfromtxt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import os
from tensorflow.python.framework import ops
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()

def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    return features, labels

def normalizer(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return(dataset - mu)/sigma

def bias_vector(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l

features,labels = read_boston_data()
normalized_features = normalizer(features)
data, label = bias_vector(normalized_features,labels)
n_dim = data.shape[1]

# Train-test split
train_x, test_x, train_y, test_y = train_test_split(data,label,test_size = 0.25,random_state = 100)

learning_rate = 0.0001
training_epochs = 100000
log_loss = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,n_dim]) #takes any number of rows but n_dim columns
Y = tf.placeholder(tf.float32,[None,1]) # #takes any number of rows but only 1 continuous column
W = tf.Variable(tf.ones([n_dim,1])) # W weight vector 

init_op = tf.global_variables_initializer()

# LInear regression operation: First line will multiply features matrix to weights matrix and can be used for prediction. 
#The second line is cost or loss function (squared error of regression line). 
#Finally, the third line perform one step of gradient descent optimization to minimize the cost function. 
 
y_ = tf.matmul(X, W)
cost_op = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_op)

sess = tf.Session()
sess.run(init_op)

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:train_x,Y:train_y})
    log_loss = np.append(log_loss,sess.run(cost_op,feed_dict={X: train_x,Y: train_y}))

plt.plot(range(len(log_loss)),log_loss)
plt.axis([0,training_epochs,0,np.max(log_loss)])
plt.show()

pred_y = sess.run(y_, feed_dict={X: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse)) 

fig, ax = plt.subplots()
ax.scatter(test_y, pred_y)
ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

sess.close()
```

```
第3章 用TensorFlow構建前饋神經網路 51
3．1 前饋神經網路介紹 51
3．1．1 前饋和反向傳播 52
3．1．2 權重和偏差 53
3．1．3 傳遞函數 53
3．2 手寫數字分類 54
3．3 探究MNIST資料集 55
3．4 Softmax分類器 57
3．5 TensorFlow模型的保存和還原 63
3．5．1 保存模型 63
3．5．2 還原模型 63
3．5．3 Softmax原始程式碼 65
3．5．4 Softmax啟動器原始程式碼 66
3．6 實現一個五層神經網路 67
3．6．1 視覺化 69
3．6．2 五層神經網路原始程式碼 70
3．7 ReLU分類器 72
3．8 視覺化 73
3．9 Dropout優化 76
3．10 視覺化 78
3．11 小結 80
```
第4章 TensorFlow與卷積神經網路 82

4．1 CNN簡介 82

4．2 CNN架構 84

4．3 構建你的* 一個CNN(要跑一段時間)

Python2.7
```
import tensorflow as tf
import numpy as np
#import mnist_data 

batch_size = 128
test_size = 256
img_size = 28
num_classes = 10

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden):

    conv1 = tf.nn.conv2d(X, w,\
                         strides=[1, 1, 1, 1],\
                         padding='SAME')

    conv1_a = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1_a, ksize=[1, 2, 2, 1]\
                        ,strides=[1, 2, 2, 1],\
                        padding='SAME')
    conv1 = tf.nn.dropout(conv1, p_keep_conv)

    conv2 = tf.nn.conv2d(conv1, w2,\
                         strides=[1, 1, 1, 1],\
                         padding='SAME')
    conv2_a = tf.nn.relu(conv2)
    conv2 = tf.nn.max_pool(conv2_a, ksize=[1, 2, 2, 1],\
                        strides=[1, 2, 2, 1],\
                        padding='SAME')
    conv2 = tf.nn.dropout(conv2, p_keep_conv)

    conv3=tf.nn.conv2d(conv2, w3,\
                       strides=[1, 1, 1, 1]\
                       ,padding='SAME')

    conv3 = tf.nn.relu(conv3)


    FC_layer = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1],\
                        strides=[1, 2, 2, 1],\
                        padding='SAME')
    
    FC_layer = tf.reshape(FC_layer, [-1, w4.get_shape().as_list()[0]])    
    FC_layer = tf.nn.dropout(FC_layer, p_keep_conv)


    output_layer = tf.nn.relu(tf.matmul(FC_layer, w4))
    output_layer = tf.nn.dropout(output_layer, p_keep_hidden)

    result = tf.matmul(output_layer, w_o)
    return result


#mnist = mnist_data.read_data_sets("ata/")
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

trX, trY, teX, teY = mnist.train.images,\
                     mnist.train.labels, \
                     mnist.test.images, \
                     mnist.test.labels

trX = trX.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img
teX = teX.reshape(-1, img_size, img_size, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, img_size, img_size, 1])
Y = tf.placeholder("float", [None, num_classes])

w = init_weights([3, 3, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([3, 3, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
w4 = init_weights([128 * 4 * 4, 625]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([625, num_classes])         # FC 625 inputs, 10 outputs (labels)

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_o, p_keep_conv, p_keep_hidden)

Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)
cost = tf.reduce_mean(Y_)
optimizer  = tf.train.\
           RMSPropOptimizer(0.001, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

with tf.Session() as sess:
    #tf.initialize_all_variables().run()
    tf.global_variables_initializer().run()
    for i in range(100):
        training_batch = \
                       zip(range(0, len(trX), \
                                 batch_size),
                             range(batch_size, \
                                   len(trX)+1, \
                                   batch_size))
        for start, end in training_batch:
            sess.run(optimizer , feed_dict={X: trX[start:end],\
                                          Y: trY[start:end],\
                                          p_keep_conv: 0.8,\
                                          p_keep_hidden: 0.5})

        test_indices = np.arange(len(teX)) # Get A Test Batch
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]

        print(i, np.mean(np.argmax(teY[test_indices], axis=1) ==\
                         sess.run\
                         (predict_op,\
                          feed_dict={X: teX[test_indices],\
                                     Y: teY[test_indices], \
                                     p_keep_conv: 1.0,\
                                     p_keep_hidden: 1.0})))

"""
Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.
Successfully extracted to train-images-idx3-ubyte.mnist 9912422 bytes.
Loading ata/train-images-idx3-ubyte.mnist
Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.
Successfully extracted to train-labels-idx1-ubyte.mnist 28881 bytes.
Loading ata/train-labels-idx1-ubyte.mnist
Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.
Successfully extracted to t10k-images-idx3-ubyte.mnist 1648877 bytes.
Loading ata/t10k-images-idx3-ubyte.mnist
Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.
Successfully extracted to t10k-labels-idx1-ubyte.mnist 4542 bytes.
Loading ata/t10k-labels-idx1-ubyte.mnist
(0, 0.95703125)
(1, 0.98046875)
(2, 0.9921875)
(3, 0.99609375)
(4, 0.99609375)
(5, 0.98828125)
(6, 0.99609375)
(7, 0.99609375)
(8, 0.98828125)
(9, 0.98046875)
(10, 0.99609375)
(11, 1.0)
(12, 0.9921875)
(13, 0.98046875)
(14, 0.98828125)
(15, 0.9921875)
(16, 0.9921875)
(17, 0.9921875)
(18, 0.9921875)
(19, 1.0)
(20, 0.98828125)
(21, 0.99609375)
(22, 0.98828125)
(23, 1.0)
(24, 0.9921875)
(25, 0.99609375)
(26, 0.99609375)
(27, 0.98828125)
(28, 0.98828125)
(29, 0.9921875)
(30, 0.99609375)
(31, 0.9921875)
(32, 0.99609375)
(33, 1.0)
(34, 0.99609375)
(35, 1.0)
(36, 0.9921875)
(37, 1.0)
(38, 0.99609375)
(39, 0.99609375)
(40, 0.99609375)
(41, 0.9921875)
(42, 0.98828125)
(43, 0.9921875)
(44, 0.9921875)
(45, 0.9921875)
(46, 0.9921875)
(47, 0.98828125)
(48, 0.99609375)
(49, 0.99609375)
(50, 1.0)
(51, 0.98046875)
(52, 0.99609375)
(53, 0.98828125)
(54, 0.99609375)
(55, 0.9921875)
(56, 0.99609375)
(57, 0.9921875)
(58, 0.98828125)
(59, 0.99609375)
(60, 0.99609375)
(61, 0.98828125)
(62, 1.0)
(63, 0.98828125)
(64, 0.98828125)
(65, 0.98828125)
(66, 1.0)
(67, 0.99609375)
(68, 1.0)
(69, 1.0)
(70, 0.9921875)
(71, 0.99609375)
(72, 0.984375)
(73, 0.9921875)
(74, 0.98828125)
(75, 0.99609375)
(76, 1.0)
(77, 0.9921875)
(78, 0.984375)
(79, 1.0)
(80, 0.9921875)
(81, 0.9921875)
(82, 0.99609375)
(83, 1.0)
(84, 0.98828125)
(85, 0.98828125)
(86, 0.99609375)
(87, 1.0)
(88, 0.99609375)
"""

```

4．4 CNN表情識別 95
4．4．1 表情分類器原始程式碼 104
4．4．2 使用自己的圖像測試模型 107
```
__author__ = 'Charlie'
import pandas as pd
import numpy as np
import os, sys, inspect
from six.moves import cPickle as pickle
import scipy.misc as misc

IMAGE_SIZE = 48
NUM_LABELS = 7
VALIDATION_PERCENT = 0.1  # use 10 percent of training images for validation

IMAGE_LOCATION_NORM = IMAGE_SIZE / 2

np.random.seed(0)

emotion = {0:'anger', 1:'disgust',\
           2:'fear',3:'happy',\
           4:'sad',5:'surprise',6:'neutral'}

class testResult:

    def __init__(self):
        self.anger = 0
        self.disgust = 0
        self.fear = 0
        self.happy = 0
        self.sad = 0
        self.surprise = 0
        self.neutral = 0
        
    def evaluate(self,label):
        
        if (0 == label):
            self.anger = self.anger+1
        if (1 == label):
            self.disgust = self.disgust+1
        if (2 == label):
            self.fear = self.fear+1
        if (3 == label):
            self.happy = self.happy+1
        if (4 == label):
            self.sad = self.sad+1
        if (5 == label):
            self.surprise = self.surprise+1
        if (6 == label):
            self.neutral = self.neutral+1

    def display_result(self,evaluations):
        print("anger = "    + str((self.anger/float(evaluations))*100)    + "%")
        print("disgust = "  + str((self.disgust/float(evaluations))*100)  + "%")
        print("fear = "     + str((self.fear/float(evaluations))*100)     + "%")
        print("happy = "    + str((self.happy/float(evaluations))*100)    + "%")
        print("sad = "      + str((self.sad/float(evaluations))*100)      + "%")
        print("surprise = " + str((self.surprise/float(evaluations))*100) + "%")
        print("neutral = "  + str((self.neutral/float(evaluations))*100)  + "%")
            

def read_data(data_dir, force=False):
    def create_onehot_label(x):
        label = np.zeros((1, NUM_LABELS), dtype=np.float32)
        label[:, int(x)] = 1
        return label

    pickle_file = os.path.join(data_dir, "EmotionDetectorData.pickle")
    if force or not os.path.exists(pickle_file):
        train_filename = os.path.join(data_dir, "train.csv")
        data_frame = pd.read_csv(train_filename)
        data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()
        print "Reading train.csv ..."

        train_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
        print train_images.shape
        train_labels = np.array([map(create_onehot_label, data_frame['Emotion'].values)]).reshape(-1, NUM_LABELS)
        print train_labels.shape

        permutations = np.random.permutation(train_images.shape[0])
        train_images = train_images[permutations]
        train_labels = train_labels[permutations]
        validation_percent = int(train_images.shape[0] * VALIDATION_PERCENT)
        validation_images = train_images[:validation_percent]
        validation_labels = train_labels[:validation_percent]
        train_images = train_images[validation_percent:]
        train_labels = train_labels[validation_percent:]

        print "Reading test.csv ..."
        test_filename = os.path.join(data_dir, "test.csv")
        data_frame = pd.read_csv(test_filename)
        data_frame['Pixels'] = data_frame['Pixels'].apply(lambda x: np.fromstring(x, sep=" ") / 255.0)
        data_frame = data_frame.dropna()
        test_images = np.vstack(data_frame['Pixels']).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)

        with open(pickle_file, "wb") as file:
            try:
                print 'Picking ...'
                save = {
                    "train_images": train_images,
                    "train_labels": train_labels,
                    "validation_images": validation_images,
                    "validation_labels": validation_labels,
                    "test_images": test_images,
                }
                pickle.dump(save, file, pickle.HIGHEST_PROTOCOL)

            except:
                print("Unable to pickle file :/")

    with open(pickle_file, "rb") as file:
        save = pickle.load(file)
        train_images = save["train_images"]
        train_labels = save["train_labels"]
        validation_images = save["validation_images"]
        validation_labels = save["validation_labels"]
        test_images = save["test_images"]

    return train_images, train_labels, validation_images, validation_labels, test_images
```

```
from scipy import misc
import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
import os, sys, inspect
from datetime import datetime
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import EmotionDetectorUtils

emotion = {0:'anger', 1:'disgust',\
           2:'fear',3:'happy',\
           4:'sad',5:'surprise',6:'neutral'}


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('author_img.jpg')     
gray = rgb2gray(img)
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()


""""
lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
"""



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "EmotionDetector/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/EmotionDetector_logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")




train_images, train_labels, valid_images, valid_labels, test_images = \
                  EmotionDetectorUtils.read_data(FLAGS.data_dir)


sess = tf.InteractiveSession()

new_saver = tf.train.import_meta_graph('logs/model.ckpt-1000.meta')
new_saver.restore(sess, 'logs/model.ckpt-1000')
tf.get_default_graph().as_graph_def()

x = sess.graph.get_tensor_by_name("input:0")
y_conv = sess.graph.get_tensor_by_name("output:0")

image_0 = np.resize(gray,(1,48,48,1))

result = sess.run(y_conv, feed_dict={x:image_0})
label = sess.run(tf.argmax(result, 1))
print(emotion[label[0]])

```
```
import tensorflow as tf
import numpy as np
#import os, sys, inspect
from datetime import datetime
import EmotionDetectorUtils

"""
lib_path = os.path.realpath(
    os.path.abspath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
"""


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("data_dir", "EmotionDetector/", "Path to data files")
tf.flags.DEFINE_string("logs_dir", "logs/EmotionDetector_logs/", "Path to where log files are to be saved")
tf.flags.DEFINE_string("mode", "train", "mode: train (Default)/ test")

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_ITERATIONS = 1001
REGULARIZATION = 1e-2
IMAGE_SIZE = 48
NUM_LABELS = 7
VALIDATION_PERCENT = 0.1


def add_to_regularization_loss(W, b):
    tf.add_to_collection("losses", tf.nn.l2_loss(W))
    tf.add_to_collection("losses", tf.nn.l2_loss(b))

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def conv2d_basic(x, W, bias):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.bias_add(conv, bias)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], \
                          strides=[1, 2, 2, 1], padding="SAME")


def emotion_cnn(dataset):
    with tf.name_scope("conv1") as scope:
        #W_conv1 = weight_variable([5, 5, 1, 32])
        #b_conv1 = bias_variable([32])
        tf.summary.histogram("W_conv1", weights['wc1'])
        tf.summary.histogram("b_conv1", biases['bc1'])
        conv_1 = tf.nn.conv2d(dataset, weights['wc1'],\
                              strides=[1, 1, 1, 1], padding="SAME")
        h_conv1 = tf.nn.bias_add(conv_1, biases['bc1'])
        #h_conv1 = conv2d_basic(dataset, W_conv1, b_conv1)
        h_1 = tf.nn.relu(h_conv1)
        h_pool1 = max_pool_2x2(h_1)
        add_to_regularization_loss(weights['wc1'], biases['bc1'])

    with tf.name_scope("conv2") as scope:
        #W_conv2 = weight_variable([3, 3, 32, 64])
        #b_conv2 = bias_variable([64])
        tf.summary.histogram("W_conv2", weights['wc2'])
        tf.summary.histogram("b_conv2", biases['bc2'])
        conv_2 = tf.nn.conv2d(h_pool1, weights['wc2'],\
                              strides=[1, 1, 1, 1], padding="SAME")
        h_conv2 = tf.nn.bias_add(conv_2, biases['bc2'])
        #h_conv2 = conv2d_basic(h_pool1, weights['wc2'], biases['bc2'])
        h_2 = tf.nn.relu(h_conv2)
        h_pool2 = max_pool_2x2(h_2)
        add_to_regularization_loss(weights['wc2'], biases['bc2'])

    with tf.name_scope("fc_1") as scope:
        prob=0.5
        image_size = IMAGE_SIZE / 4
        h_flat = tf.reshape(h_pool2, [-1, image_size * image_size * 64])
        #W_fc1 = weight_variable([image_size * image_size * 64, 256])
        #b_fc1 = bias_variable([256])
        tf.summary.histogram("W_fc1", weights['wf1'])
        tf.summary.histogram("b_fc1", biases['bf1'])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, weights['wf1']) + biases['bf1'])
        h_fc1_dropout = tf.nn.dropout(h_fc1, prob)
        
    with tf.name_scope("fc_2") as scope:
        #W_fc2 = weight_variable([256, NUM_LABELS])
        #b_fc2 = bias_variable([NUM_LABELS])
        tf.summary.histogram("W_fc2", weights['wf2'])
        tf.summary.histogram("b_fc2", biases['bf2'])
        #pred = tf.matmul(h_fc1, weights['wf2']) + biases['bf2']
        pred = tf.matmul(h_fc1_dropout, weights['wf2']) + biases['bf2']

    return pred

weights = {
    'wc1': weight_variable([5, 5, 1, 32], name="W_conv1"),
    'wc2': weight_variable([3, 3, 32, 64],name="W_conv2"),
    'wf1': weight_variable([(IMAGE_SIZE / 4) * (IMAGE_SIZE / 4) * 64, 256],name="W_fc1"),
    'wf2': weight_variable([256, NUM_LABELS], name="W_fc2")
}

biases = {
    'bc1': bias_variable([32], name="b_conv1"),
    'bc2': bias_variable([64], name="b_conv2"),
    'bf1': bias_variable([256], name="b_fc1"),
    'bf2': bias_variable([NUM_LABELS], name="b_fc2")
}

def loss(pred, label):
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label))
    tf.summary.scalar('Entropy', cross_entropy_loss)
    reg_losses = tf.add_n(tf.get_collection("losses"))
    tf.summary.scalar('Reg_loss', reg_losses)
    return cross_entropy_loss + REGULARIZATION * reg_losses


def train(loss, step):
    return tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=step)


def get_next_batch(images, labels, step):
    offset = (step * BATCH_SIZE) % (images.shape[0] - BATCH_SIZE)
    batch_images = images[offset: offset + BATCH_SIZE]
    batch_labels = labels[offset:offset + BATCH_SIZE]
    return batch_images, batch_labels


def main(argv=None):
    train_images, train_labels, valid_images, valid_labels, test_images = \
                  EmotionDetectorUtils.read_data(FLAGS.data_dir)
    print "Train size: %s" % train_images.shape[0]
    print 'Validation size: %s' % valid_images.shape[0]
    print "Test size: %s" % test_images.shape[0]

    global_step = tf.Variable(0, trainable=False)
    dropout_prob = tf.placeholder(tf.float32)
    input_dataset = tf.placeholder(tf.float32, \
                                   [None, IMAGE_SIZE, IMAGE_SIZE, 1],name="input")
    input_labels = tf.placeholder(tf.float32, [None, NUM_LABELS])

    pred = emotion_cnn(input_dataset)
    output_pred = tf.nn.softmax(pred,name="output")
    loss_val = loss(pred, input_labels)
    train_op = train(loss_val, global_step)

    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph_def)
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print "Model Restored!"

        for step in xrange(MAX_ITERATIONS):
            batch_image, batch_label = get_next_batch(train_images,\
                                                      train_labels, step)
            feed_dict = {input_dataset: batch_image, \
                         input_labels: batch_label}

            sess.run(train_op, feed_dict=feed_dict)
            if step % 10 == 0:
                train_loss, summary_str = sess.run([loss_val, summary_op],\
                                                   feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step=step)
                print "Training Loss: %f" % train_loss

            if step % 100 == 0:
                valid_loss = sess.run(loss_val, \
                                      feed_dict={input_dataset: valid_images,\
                                                 input_labels: valid_labels})
                print "%s Validation Loss: %f" % (datetime.now(), valid_loss)
                saver.save(sess, FLAGS.logs_dir + 'model.ckpt', global_step=step)


if __name__ == "__main__":
    tf.app.run()



"""
>>> 
Train size: 3761
Validation size: 417
Test size: 1312
WARNING:tensorflow:Passing a `GraphDef` to the SummaryWriter is deprecated. Pass a `Graph` object instead, such as `sess.graph`.
Training Loss: 1.962236
2016-11-05 22:39:36.645682 Validation Loss: 1.962719
Training Loss: 1.907290
Training Loss: 1.849100
Training Loss: 1.871116
Training Loss: 1.798998
Training Loss: 1.885601
Training Loss: 1.849380
Training Loss: 1.843139
Training Loss: 1.933691
Training Loss: 1.829839
Training Loss: 1.839772
2016-11-05 22:42:58.951699 Validation Loss: 1.822431
Training Loss: 1.772197
Training Loss: 1.666473
Training Loss: 1.620869
Training Loss: 1.592660
Training Loss: 1.422701
Training Loss: 1.436721
Training Loss: 1.348217
Training Loss: 1.432023
Training Loss: 1.347753
Training Loss: 1.299889
2016-11-05 22:46:55.144483 Validation Loss: 1.335237
Training Loss: 1.108747
Training Loss: 1.197601
Training Loss: 1.245860
Training Loss: 1.164120
Training Loss: 0.994351
Training Loss: 1.072356
Training Loss: 1.193485
Training Loss: 1.118093
Training Loss: 1.021220
Training Loss: 1.069752
2016-11-05 22:50:17.677074 Validation Loss: 1.111559
Training Loss: 1.099430
Training Loss: 0.966327
Training Loss: 0.960916
Training Loss: 0.844742
Training Loss: 0.979741
Training Loss: 0.891897
Training Loss: 1.013132
Training Loss: 0.936738
Training Loss: 0.911577
Training Loss: 0.862605
2016-11-05 22:53:30.999141 Validation Loss: 0.999061
Training Loss: 0.800337
Training Loss: 0.776097
Training Loss: 0.799260
Training Loss: 0.919926
Training Loss: 0.758807
Training Loss: 0.807968
Training Loss: 0.856378
Training Loss: 0.867762
Training Loss: 0.656170
Training Loss: 0.688761
2016-11-05 22:56:53.256991 Validation Loss: 0.931223
Training Loss: 0.696454
Training Loss: 0.725157
Training Loss: 0.674037
Training Loss: 0.719200
Training Loss: 0.749460
Training Loss: 0.741768
Training Loss: 0.702719
Training Loss: 0.734194
Training Loss: 0.669155
Training Loss: 0.641528
2016-11-05 23:00:06.530139 Validation Loss: 0.911489
Training Loss: 0.764550
Training Loss: 0.646964
Training Loss: 0.724712
Training Loss: 0.726692
Training Loss: 0.656019
Training Loss: 0.690552
Training Loss: 0.537638
Training Loss: 0.680097
Training Loss: 0.554115
Training Loss: 0.590837
2016-11-05 23:03:15.351156 Validation Loss: 0.818303
Training Loss: 0.656608
Training Loss: 0.567394
Training Loss: 0.545324
Training Loss: 0.611726
Training Loss: 0.600910
Training Loss: 0.526467
Training Loss: 0.584986
Training Loss: 0.567015
Training Loss: 0.555465
Training Loss: 0.630097
2016-11-05 23:06:26.575298 Validation Loss: 0.824178
Training Loss: 0.662920
Training Loss: 0.512493
Training Loss: 0.475912
Training Loss: 0.455112
Training Loss: 0.567875
Training Loss: 0.582927
Training Loss: 0.509225
Training Loss: 0.602916
Training Loss: 0.521976
Training Loss: 0.445122
2016-11-05 23:09:40.136353 Validation Loss: 0.803449
Training Loss: 0.435535
Training Loss: 0.459343
Training Loss: 0.481706
Training Loss: 0.460640
Training Loss: 0.554570
Training Loss: 0.427962
Training Loss: 0.512764
Training Loss: 0.531128
Training Loss: 0.364465
Training Loss: 0.432366
2016-11-05 23:12:50.769527 Validation Loss: 0.851074
>>> 
"""


```

```
第5章 優化TensorFlow自編碼器 112
5．1 自編碼器簡介 112
5．2 實現一個自編碼器 113
5．3 增強自編碼器的魯棒性 119
5．4 構建去噪自編碼器 120
5．5 卷積自編碼器 127
5．5．1 編碼器 127
5．5．2 解碼器 128
5．5．3 卷積自編碼器原始程式碼 134
5．6 小結 138

第6章 迴圈神經網路 139
6．1 RNN的基本概念 139
6．2 RNN的工作機制 140
6．3 RNN的展開 140
6．4 梯度消失問題 141
6．5 LSTM網路 142
6．6 RNN圖像分類器 143
6．7 雙向RNN 149
6．8 文本預測 155
6．8．1 資料集 156
6．8．2 困惑度 156
6．8．3 PTB模型 156
6．8．4 運行常式 157
6．9 小結 158

第7章 GPU計算 160
7．1 GPGPU計算 160
7．2 GPGPU的歷史 161
7．3 CUDA架構 161
7．4 GPU程式設計模型 162
7．5 TensorFlow中GPU的設置 163
7．6 TensorFlow的GPU管理 165
7．7 GPU記憶體管理 168
7．8 在多GPU系統上分配單個GPU 168
7．9 使用多個GPU 170
7．10 小結 171

第8章 TensorFlow高 級程式設計 172
8．1 Keras簡介 172
8．2 構建深度學習模型 174
8．3 影評的情感分類 175
8．4 添加一個卷積層 179
8．5 Pretty Tensor 181
8．6 數字分類器 182
8．7 TFLearn 187
8．8 泰坦尼克號倖存者預測器 188
8．9 小結 191

第9章 TensorFlow高級多媒體程式設計 193
9．1 多媒體分析簡介 193
9．2 基於深度學習的大型對象檢測 193
9．2．1 瓶頸層 195
9．2．2 使用重訓練的模型 195
9．3 加速線性代數 197
9．3．1 TensorFlow的核心優勢 197
9．3．2 加速線性代數的準時編譯 197
9．4 TensorFlow和Keras 202
9．4．1 Keras簡介 202
9．4．2 擁有Keras的好處 203
9．4．3 視頻問答系統 203
9．5 Android上的深度學習 209
9．5．1 TensorFlow演示程式 209
9．5．2 Android入門 211
9．6 小結 214

10章 強化學習 215
10．1 強化學習基本概念 216
10．2 Q-learning演算法 217
10．3 OpenAI Gym框架簡介 218
10．4 FrozenLake-v0實現問題 220
10．5 使用TensorFlow實現Q-learning 223
10．6 小結 227
```
