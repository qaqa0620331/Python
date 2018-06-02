# 教科書
>* Python Machine Learning Cookbook: Prateek Joshi  June 2016
>* https://detail.tmall.com/item.htm?id=558284517556&ns=1&abbucket=19
>* https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-cookbook
>* https://github.com/PacktPublishing/Python-Machine-Learning-Cookbook

# packages and modules



# 監督學習: Supervised Learning 

### 第1章 監督學習: Supervised Learning 

1.2 資料預處理技術==>preprocessing.py
```
import numpy as np
from sklearn import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])

# mean removal
data_standardized = preprocessing.scale(data)
print "\nMean =", data_standardized.mean(axis=0)
print "Std deviation =", data_standardized.std(axis=0)

# min max scaling
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print "\nMin max scaled data:\n", data_scaled

# normalization
data_normalized = preprocessing.normalize(data, norm='l1')
print "\nL1 normalized data:\n", data_normalized

# binarization
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print "\nBinarized data:\n", data_binarized

# one hot encoding
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoded_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print "\nEncoded vector:\n", encoded_vector

```

### 1.3 標記編碼方法==>label_encoder.py
```

import numpy as np
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)

# print classes
print "\nClass mapping:"
for i, item in enumerate(label_encoder.classes_):
    print item, '-->', i

# transform a set of classes
labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print "\nLabels =", labels 
print "Encoded labels =", list(encoded_labels)

# inverse transform
encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print "\nEncoded labels =", encoded_labels
print "Decoded labels =", list(decoded_labels)
```
### 1.4 創建線性回歸器==>linear_regression_singlevar.py

##### 線性回歸linear regression

##### 資料集:data_singlevar.txt
```
4.94,4.37
-1.58,1.7
-4.45,1.88
-6.06,0.56
-1.22,2.23
...............
```

##### 程式:linear_regression_singlevar.py
```
import sys

import numpy as np

filename = sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
y_train = np.array(y[:num_training])

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
y_test = np.array(y[num_training:])

# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(X_test)

# Plot outputs
import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Measure performance ### 1.5 計算回歸準確性
import sklearn.metrics as sm

print "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) 
print "Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2) 
print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2) 
print "Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2) 
print "R2 score =", round(sm.r2_score(y_test, y_test_pred), 2)

# Model persistence 1.6 保存模型資料
import cPickle as pickle

output_model_file = '3_model_linear_regr.pkl'

with open(output_model_file, 'w') as f:
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'r') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print "\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2) 
```
##### 執行程式
```

python linear_regression_singlevar.py data_singlevar.txt
　
```

## 多元回歸

>* 1.7 創建嶺回歸器Building a ridge regressor==>
>* 1.8 創建多項式回歸器Building a polynomial regressor ==>

##### 資料集:data_multivar.txt
```
0.39,2.78,7.11,-8.07
1.65,6.7,2.42,12.24
5.67,6.38,3.79,23.96
2.31,6.27,4.8,4.29
3.67,6.67,2.38,16.37
3.64,3.14,2.38,12.44
7.0,3.85,8.39,13.45
```
##### 程式:regression_multivar.py
```
import sys

import numpy as np

filename = sys.argv[1]
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = [float(i) for i in line.split(',')]
        xt, yt = data[:-1], data[-1]
        X.append(xt)
        y.append(yt)

# Train/test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Training data
#X_train = np.array(X[:num_training]).reshape((num_training,1))
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])

# Test data
#X_test = np.array(X[num_training:]).reshape((num_test,1))
X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

# Create linear regression object
from sklearn import linear_model

linear_regressor = linear_model.LinearRegression()
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

# Train the model using the training sets
linear_regressor.fit(X_train, y_train)
ridge_regressor.fit(X_train, y_train)

# Predict the output
y_test_pred = linear_regressor.predict(X_test)
y_test_pred_ridge = ridge_regressor.predict(X_test)

# Measure performance
import sklearn.metrics as sm

print "LINEAR:"
print "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) 
print "Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2) 
print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2) 
print "Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2) 
print "R2 score =", round(sm.r2_score(y_test, y_test_pred), 2)

print "\nRIDGE:"
print "Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2) 
print "Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2) 
print "Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2) 
print "Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2) 
print "R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2)

# Polynomial regression
from sklearn.preprocessing import PolynomialFeatures

polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [0.39,2.78,7.11]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print "\nLinear regression:\n", linear_regressor.predict(datapoint)
print "\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint)

# Stochastic Gradient Descent regressor
sgd_regressor = linear_model.SGDRegressor(loss='huber', n_iter=50)
sgd_regressor.fit(X_train, y_train)
print "\nSGD regressor:\n", sgd_regressor.predict(datapoint)
```

### 房屋價格的估算==>Estimating housing prices==>housing.py

>* 1.9 估算房屋價格Estimating housing prices 
>* 1.10 計算特徵的相對重要性Computing the relative importance of features
```
import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

def plot_feature_importances(feature_importances, title, feature_names):
    # Normalize the importance values 
    feature_importances = 100.0 * (feature_importances / max(feature_importances))

    # Sort the values and flip them
    index_sorted = np.flipud(np.argsort(feature_importances))

    # Arrange the X ticks
    pos = np.arange(index_sorted.shape[0]) + 0.5

    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()

if __name__=='__main__':
    # Load housing data
    housing_data = datasets.load_boston() 

    # Shuffle the data
    X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.8 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit decision tree regression model
    dt_regressor = DecisionTreeRegressor(max_depth=4)
    dt_regressor.fit(X_train, y_train)

    # Fit decision tree regression model with AdaBoost
    ab_regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=400, random_state=7)
    ab_regressor.fit(X_train, y_train)

    # Evaluate performance of Decision Tree regressor
    y_pred_dt = dt_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_dt)
    evs = explained_variance_score(y_test, y_pred_dt) 
    print "\n#### Decision Tree performance ####"
    print "Mean squared error =", round(mse, 2)
    print "Explained variance score =", round(evs, 2)

    # Evaluate performance of AdaBoost
    y_pred_ab = ab_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_ab)
    evs = explained_variance_score(y_test, y_pred_ab) 
    print "\n#### AdaBoost performance ####"
    print "Mean squared error =", round(mse, 2)
    print "Explained variance score =", round(evs, 2)

    # Plot relative feature importances 
    plot_feature_importances(dt_regressor.feature_importances_, 
            'Decision Tree regressor', housing_data.feature_names)
    plot_feature_importances(ab_regressor.feature_importances_, 
            'AdaBoost regressor', housing_data.feature_names)
```



### 1.11 評估共用單車的需求分佈Estimating bicycle demand distribution==>bike_sharing.py

>* 問題:共享單車的需求分佈
>* 演算法:random forest regressor
```
import sys
import csv

import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from housing import plot_feature_importances

def load_dataset(filename):
    file_reader = csv.reader(open(filename, 'rb'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[2:13])
        y.append(row[-1])

    # Extract feature names
    feature_names = np.array(X[0])

    # Remove the first row because they are feature names
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

if __name__=='__main__':
    # Load the dataset from the input file
    X, y, feature_names = load_dataset(sys.argv[1])
    X, y = shuffle(X, y, random_state=7) 

    # Split the data 80/20 (80% for training, 20% for testing)
    num_training = int(0.9 * len(X))
    X_train, y_train = X[:num_training], y[:num_training]
    X_test, y_test = X[num_training:], y[num_training:]

    # Fit Random Forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_samples_split=1)
    rf_regressor.fit(X_train, y_train)

    # Evaluate performance of Random Forest regressor
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred) 
    print "\n#### Random Forest regressor performance ####"
    print "Mean squared error =", round(mse, 2)
    print "Explained variance score =", round(evs, 2)

    # Plot relative feature importances 
    plot_feature_importances(rf_regressor.feature_importances_, 'Random Forest regressor', feature_names)

```


# 第2章 創建分類器CONSTRUCTING A CLASSIFIER


## 2.2 建立簡單分類器Building a simple classifier==>simple_classifier.py
```
import numpy as np
import matplotlib.pyplot as plt

# input data
X = np.array([[3,1], [2,5], [1,8], [6,4], [5,2], [3,5], [4,7], [4,-1]])

# labels
y = [0, 1, 1, 0, 0, 1, 1, 0]

# separate the data into classes based on 'y'
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

# plot input data
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')

# draw the separator line
line_x = range(10)
line_y = line_x

# plot labeled data and separator line 
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], color='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], color='black', marker='x')
plt.plot(line_x, line_y, color='black', linewidth=3)

plt.show()
```

### 2.3 建立邏輯回歸分類器Building a logistic regression classifier
```

```




2.4 建立樸素貝葉斯分類器Building a Naive Bayes classifier





2.5 將資料集分割成訓練集和測試集Splitting the dataset for training and testing




2.6 用交叉驗證檢驗模型準確性Evaluating the accuracy using cross-validation





2.7 混淆矩陣視覺化Visualizing the confusion matrix




2.8 提取性能報告Extracting the performance report





2.9 根據汽車特徵評估品質Evaluating cars based on their characteristics



2.10 生成驗證曲線Extracting validation curves





2.11 生成學習曲線Extracting learning curves



2.12 估算收入階層Estimating the income bracket


### 第3章 預測建模PREDICTIVE MODELING

3.2 用SVM建立線性分類器Building a linear classifier using Support Vector Machine (SVMs)
```
import numpy as np
import matplotlib.pyplot as plt

import utilities 

# Load input data
input_file = 'data_multivar.txt'
X, y = utilities.load_data(input_file)

###############################################
# Separate the data into classes based on 'y'
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])

# Plot the input data
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='black', edgecolors='black', marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='None', edgecolors='black', marker='s')
plt.title('Input data')

###############################################
# Train test split and SVM training
from sklearn import cross_validation
from sklearn.svm import SVC

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=5)

params = {'kernel': 'linear'}
#params = {'kernel': 'poly', 'degree': 3}
#params = {'kernel': 'rbf'}
classifier = SVC(**params)
classifier.fit(X_train, y_train)
utilities.plot_classifier(classifier, X_train, y_train, 'Training dataset')

y_test_pred = classifier.predict(X_test)
utilities.plot_classifier(classifier, X_test, y_test, 'Test dataset')

###############################################
# Evaluate classifier performance

from sklearn.metrics import classification_report

target_names = ['Class-' + str(int(i)) for i in set(y)]
print "\n" + "#"*30
print "\nClassifier performance on training dataset\n"
print classification_report(y_train, classifier.predict(X_train), target_names=target_names)
print "#"*30 + "\n"

print "#"*30
print "\nClassification report on test dataset\n"
print classification_report(y_test, y_test_pred, target_names=target_names)
print "#"*30 + "\n"

plt.show()
```

3.3 用SVM建立非線性分類器Building a nonlinear classifier using SVMs




3.4 解決類型數量不平衡問題Tackling class imbalance



3.5 提取置信度Extracting confidence measurements



3.6 尋找最優超參數Finding optimal hyperparameters



3.7 建立事件預測器Building an event predictor



3.8 估算交通流量Estimating traffic
```
```
# 無監督學習 Unsupervised Learning

## 第4章 無監督學習——聚類CLUSTERING WITH UNSUPERVISED LEARNING

### 4.2 用k-means演算法聚類資料Clustering data using the k-means algorithm ==>kmeans.py

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

import utilities

# Load data
data = utilities.load_data('data_multivar.txt')
num_clusters = 4

# Plot data
plt.figure()
plt.scatter(data[:,0], data[:,1], marker='o', 
        facecolors='none', edgecolors='k', s=30)
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Train the model
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
kmeans.fit(data)

# Step size of the mesh
step_size = 0.01

# Plot the boundaries
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# Predict labels for all points in the mesh
predicted_labels = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()])

# Plot the results
predicted_labels = predicted_labels.reshape(x_values.shape)
plt.figure()
plt.clf()
plt.imshow(predicted_labels, interpolation='nearest',
           extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.scatter(data[:,0], data[:,1], marker='o', 
        facecolors='none', edgecolors='k', s=30)

centroids = kmeans.cluster_centers_
plt.scatter(centroids[:,0], centroids[:,1], marker='o', s=200, linewidths=3,
        color='k', zorder=10, facecolors='black')
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Centoids and boundaries obtained using KMeans')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
```


### 4.3 用向量量化壓縮圖片Compressing an image using vector quantization ==> vector_quantization.py

vector quantization is the N-dimensional version of "rounding off".

http://www.datacompression.com/vq.shtml

Vector quantization is popularly used in image compression
where we store each pixel using fewer bits than the original image to achieve compression

```
import argparse

import numpy as np
from scipy import misc 
from sklearn import cluster
import matplotlib.pyplot as plt

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compress the input image \
            using clustering')
    parser.add_argument("--input-file", dest="input_file", required=True,
            help="Input image")
    parser.add_argument("--num-bits", dest="num_bits", required=False,
            type=int, help="Number of bits used to represent each pixel")
    return parser

def compress_image(img, num_clusters):
    # Convert input image into (num_samples, num_features) 
    # array to run kmeans clustering algorithm 
    X = img.reshape((-1, 1))  

    # Run kmeans on input data
    kmeans = cluster.KMeans(n_clusters=num_clusters, n_init=4, random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_

    # Assign each value to the nearest centroid and 
    # reshape it to the original image shape
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)

    return input_image_compressed

def plot_image(img, title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_file = args.input_file
    num_bits = args.num_bits

    if not 1 <= num_bits <= 8:
        raise TypeError('Number of bits should be between 1 and 8')

    num_clusters = np.power(2, num_bits)

    # Print compression rate
    compression_rate = round(100 * (8.0 - args.num_bits) / 8.0, 2)
    print "\nThe size of the image will be reduced by a factor of", 8.0/args.num_bits
    print "\nCompression rate = " + str(compression_rate) + "%"

    # Load input image
    input_image = misc.imread(input_file, True).astype(np.uint8)

    # original image 
    plot_image(input_image, 'Original image')

    # compressed image 
    input_image_compressed = compress_image(input_image, num_clusters)
    plot_image(input_image_compressed, 'Compressed image; compression rate = ' 
            + str(compression_rate) + '%')

    plt.show()

```

python vector_quantization.py --input-file flower_image.jpg --num-bits 4

python vector_quantization.py --input-file flower_image.jpg --num-bits 2

python vector_quantization.py --input-file flower_image.jpg --num-bits 1

### 4.4 建立均值漂移聚類模型Building a Mean Shift clustering model ==> mean_shift.py

>* http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/TUZEL1/MeanShift.pdf

```
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

import utilities

# Load data from input file
X = utilities.load_data('data_multivar.txt')

# Estimating the bandwidth 
bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Compute clustering with MeanShift
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_estimator.fit(X)
labels = meanshift_estimator.labels_
centroids = meanshift_estimator.cluster_centers_
num_clusters = len(np.unique(labels))

print "Number of clusters in input data =", num_clusters

###########################################################
# Plot the points and centroids 

import matplotlib.pyplot as plt
from itertools import cycle

plt.figure()

# specify marker shapes for different clusters
markers = '.*xv'

for i, marker in zip(range(num_clusters), markers):
    # plot the points belong to the current cluster
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='k')

    # plot the centroid of the current cluster
    centroid = centroids[i]
    plt.plot(centroid[0], centroid[1], marker='o', markerfacecolor='k',
             markeredgecolor='k', markersize=15)

plt.title('Clusters and their centroids')
plt.show()

```

### 4.5 用凝聚層次聚類進行資料分組Grouping data using agglomerative clustering ==> agglomerative.py

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

def add_noise(x, y, amplitude):
    X = np.concatenate((x, y))
    X += amplitude * np.random.randn(2, X.shape[1])
    return X.T

def get_spiral(t, noise_amplitude=0.5):
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)

    return add_noise(x, y, noise_amplitude)

def get_rose(t, noise_amplitude=0.02):
    # Equation for "rose" (or rhodonea curve); if k is odd, then
    # the curve will have k petals, else it will have 2k petals
    k = 5       
    r = np.cos(k*t) + 0.25 
    x = r * np.cos(t)
    y = r * np.sin(t)

    return add_noise(x, y, noise_amplitude)

def get_hypotrochoid(t, noise_amplitude=0):
    a, b, h = 10.0, 2.0, 4.0
    x = (a - b) * np.cos(t) + h * np.cos((a - b) / b * t) 
    y = (a - b) * np.sin(t) - h * np.sin((a - b) / b * t) 

    return add_noise(x, y, 0)

def perform_clustering(X, connectivity, title, num_clusters=3, linkage='ward'):
    plt.figure()
    model = AgglomerativeClustering(linkage=linkage, 
                    connectivity=connectivity, n_clusters=num_clusters)
    model.fit(X)

    # extract labels
    labels = model.labels_

    # specify marker shapes for different clusters
    markers = '.vx'

    for i, marker in zip(range(num_clusters), markers):
        # plot the points belong to the current cluster
        plt.scatter(X[labels==i, 0], X[labels==i, 1], s=50, 
                    marker=marker, color='k', facecolors='none')

    plt.title(title)

if __name__=='__main__':
    # Generate sample data
    n_samples = 500 
    np.random.seed(2)
    t = 2.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
    X = get_spiral(t)

    # No connectivity
    connectivity = None 
    perform_clustering(X, connectivity, 'No connectivity')

    # Create K-Neighbors graph 
    connectivity = kneighbors_graph(X, 10, include_self=False)
    perform_clustering(X, connectivity, 'K-Neighbors connectivity')

    plt.show()

```


### 4.6 評價聚類演算法的聚類效果Evaluating the performance of clustering algorithms  ==> performance.py

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

import utilities

# Load data
data = utilities.load_data('data_perf.txt')

scores = []
range_values = np.arange(2, 10)

for i in range_values:
    # Train the model
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(data)
    score = metrics.silhouette_score(data, kmeans.labels_, 
                metric='euclidean', sample_size=len(data))

    print "\nNumber of clusters =", i
    print "Silhouette score =", score
                    
    scores.append(score)

# Plot scores
plt.figure()
plt.bar(range_values, scores, width=0.6, color='k', align='center')
plt.title('Silhouette score vs number of clusters')

# Plot data
plt.figure()
plt.scatter(data[:,0], data[:,1], color='k', s=30, marker='o', facecolors='none')
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()

```

### 4.7 用DBSCAN演算法自動估算集群數量Automatically estimating the number of clusters using DBSCAN algorithm   ==> estimate_clusters.py

```
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

import utilities

# Load data
data = utilities.load_data('data_perf.txt')

scores = []
range_values = np.arange(2, 10)

for i in range_values:
    # Train the model
    kmeans = KMeans(init='k-means++', n_clusters=i, n_init=10)
    kmeans.fit(data)
    score = metrics.silhouette_score(data, kmeans.labels_, 
                metric='euclidean', sample_size=len(data))

    print "\nNumber of clusters =", i
    print "Silhouette score =", score
                    
    scores.append(score)

# Plot scores
plt.figure()
plt.bar(range_values, scores, width=0.6, color='k', align='center')
plt.title('Silhouette score vs number of clusters')

# Plot data
plt.figure()
plt.scatter(data[:,0], data[:,1], color='k', s=30, marker='o', facecolors='none')
x_min, x_max = min(data[:, 0]) - 1, max(data[:, 0]) + 1
y_min, y_max = min(data[:, 1]) - 1, max(data[:, 1]) + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

plt.show()

```

### 4.8 探索股票資料的模式Finding patterns in stock market data ==> stock_market.py

```
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
from matplotlib.finance import quotes_historical_yahoo_ochl as quotes_yahoo

# Input symbol file
symbol_file = 'symbol_map.json'

# Choose a time period
start_date = datetime.datetime(2004, 4, 5)
end_date = datetime.datetime(2007, 6, 2)

# Load the symbol map
with open(symbol_file, 'r') as f:
    symbol_dict = json.loads(f.read())

symbols, names = np.array(list(symbol_dict.items())).T

quotes = [quotes_yahoo(symbol, start_date, end_date, asobject=True) 
                for symbol in symbols]

# Extract opening and closing quotes
opening_quotes = np.array([quote.open for quote in quotes]).astype(np.float)
closing_quotes = np.array([quote.close for quote in quotes]).astype(np.float)

# The daily fluctuations of the quotes 
delta_quotes = closing_quotes - opening_quotes

# Build a graph model from the correlations
edge_model = covariance.GraphLassoCV()

# Standardize the data 
X = delta_quotes.copy().T
X /= X.std(axis=0)

# Train the model
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Build clustering model using affinity propagation
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Print the results of clustering
for i in range(num_labels + 1):
    print "Cluster", i+1, "-->", ', '.join(names[labels == i])


```

### 4.9 建立客戶細分模型Building a customer segmentation model ==> customer_segmentation.py

```
import csv

import numpy as np
from sklearn import cluster, covariance, manifold
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Load data from input file
input_file = 'wholesale.csv'
file_reader = csv.reader(open(input_file, 'rb'), delimiter=',')
X = []
for count, row in enumerate(file_reader):
    if not count:
        names = row[2:]
        continue

    X.append([float(x) for x in row[2:]])

# Input data as numpy array
X = np.array(X)

# Estimating the bandwidth 
bandwidth = estimate_bandwidth(X, quantile=0.8, n_samples=len(X))

# Compute clustering with MeanShift
meanshift_estimator = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_estimator.fit(X)
labels = meanshift_estimator.labels_
centroids = meanshift_estimator.cluster_centers_
num_clusters = len(np.unique(labels))

print "\nNumber of clusters in input data =", num_clusters

print "\nCentroids of clusters:"
print '\t'.join([name[:3] for name in names])
for centroid in centroids:
    print '\t'.join([str(int(x)) for x in centroid])

################
# Visualizing data

centroids_milk_groceries = centroids[:, 1:3]

# Plot the nodes using the coordinates of our centroids_milk_groceries
plt.figure()
plt.scatter(centroids_milk_groceries[:,0], centroids_milk_groceries[:,1], 
        s=100, edgecolors='k', facecolors='none')

offset = 0.2
plt.xlim(centroids_milk_groceries[:,0].min() - offset * centroids_milk_groceries[:,0].ptp(),
        centroids_milk_groceries[:,0].max() + offset * centroids_milk_groceries[:,0].ptp(),)
plt.ylim(centroids_milk_groceries[:,1].min() - offset * centroids_milk_groceries[:,1].ptp(),
        centroids_milk_groceries[:,1].max() + offset * centroids_milk_groceries[:,1].ptp())

plt.title('Centroids of clusters for milk and groceries')
plt.show()

```



