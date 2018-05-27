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

### 1.7 創建嶺回歸器Building a ridge regressor==>

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
### 1.8 創建多項式回歸器Building a polynomial regressor ==>
```
```
### 1.9 估算房屋價格Estimating housing prices ==>housing.py
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
### 1.10 計算特徵的相對重要性Computing the relative importance of features ==>
```
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



# 監督學習

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


第3章 預測建模PREDICTIVE MODELING



3.2 用SVM建立線性分類器Building a linear classifier using Support Vector Machine (SVMs)




3.3 用SVM建立非線性分類器Building a nonlinear classifier using SVMs




3.4 解決類型數量不平衡問題Tackling class imbalance



3.5 提取置信度Extracting confidence measurements



3.6 尋找最優超參數Finding optimal hyperparameters



3.7 建立事件預測器Building an event predictor



3.8 估算交通流量Estimating traffic
```
# 無監督學習 Unsupervised Learning

## 第4章 無監督學習——聚類CLUSTERING WITH UNSUPERVISED LEARNING

4.2 用k-means演算法聚類資料Clustering data using the k-means algorithm




4.3 用向量量化壓縮圖片Compressing an image using vector quantization




4.4 建立均值漂移聚類模型Building a Mean Shift clustering model



4.5 用凝聚層次聚類進行資料分組Grouping data using agglomerative clustering



4.6 評價聚類演算法的聚類效果Evaluating the performance of clustering algorithms



4.7 用DBSCAN演算法自動估算集群數量Automatically estimating the number of clusters using DBSCAN algorithm



4.8 探索股票資料的模式Finding patterns in stock market data



4.9 建立客戶細分模型Building a customer segmentation model





```
第5章 構建推薦引擎
5.1 簡介
5.2 為資料處理構建函數組合
5.3 構建機器學習流水線
5.4 尋找最近鄰
5.5 構建一個KNN分類器
5.6 構建一個KNN回歸器
5.7 計算歐氏距離分數
5.8 計算皮爾遜相關係數
5.9 尋找資料集中的相似使用者
5.10 生成電影推薦

第6章 分析文本資料
6.1 簡介
6.2 用標記解析的方法預處理資料
6.3 提取文本資料的詞幹
6.4 用詞形還原的方法還原文本的基本形式
6.5 用分塊的方法劃分文本
6.6 創建詞袋模型
6.7 創建文本分類器
6.8 識別性別
6.9 分析句子的情感
6.10 用主題建模識別文本的模式

第7章 語音辨識
7.1 簡介
7.2 讀取和繪製音訊資料
7.3 將音訊信號轉換為頻域
7.4 自訂參數生成音訊信號
7.5 合成音樂
7.6 提取頻域特徵
7.7 創建隱瑪律科夫模型
7.8 創建一個語音辨識器

第8章 解剖時間序列和時序資料
8.1 簡介
8.2 將資料轉換為時間序列格式
8.3 切分時間序列資料
8.4 操作時間序列資料
8.5 從時間序列資料中提取統計數字
8.6 針對序列資料創建隱瑪律科夫模型
8.7 針對序列文本資料創建條件隨機場
8.8 用隱瑪律科夫模型分析股票市場資料

第9章 圖像內容分析 IMAGE CONTENT ANALYSIS
9.2 用OpenCV-Pyhon操作圖像 Operating on images using OpenCV-Python==>operating_on_images.py

9.3 檢測邊 Detecting edges==>edge_detector.py

9.4 長條圖均衡化 Histogram equalization

histogram_equalizer.py

9.5 檢測棱角 Detecting corners

corner_detector.py

9.6 檢測SIFT特徵點 Detecting SIFT feature points==>feature_detector.py

9.7 創建Star特徵檢測器 Building a Star feature detector==>star_detector.py

9.8 利用視覺碼本和向量量化創建特徵 Creating features using visual codebook and vector quantization

build_features.py

9.9 用極端隨機森林訓練圖像分類器 Training an image classifier using Extremely Random Forests

trainer.py

9.10 創建一個物件識別器 Building an object recognizer==>object_recognizer.py


第10章 人臉識別 BIOMETRIC FACE RECOGNITION

10.2 從網路攝像頭採集和處理視頻資訊Capturing and processing video from a webcam

video_capture.py

10.3 用Haar級聯創建一個人臉識別器Building a face detector using Haar cascades

face_recognizer.py

10.4 創建一個眼睛和鼻子檢測器Building eye and nose detectors

eye_nose_detector.py

10.5 做主成分分析Performing Principal Components Analysis

pca.py

10.6 做核主成分分析Performing Kernel Principal Components Analysis

kpca.py

10.7 做盲源分離Performing blind source separation

blind_source_separation.py

10.8 用局部二值模式長條圖創建一個人臉識別器Building a face recognizer using Local Binary Patterns Histogram

face_recognizer.py



第11章 深度神經網路DEEP NEURAL NETWORKS

11.2 創建一個感知器Building a perceptron

perceptron.py

11.3 創建一個單層神經網路Building a single layer neural network

single_layer.py

11.4 創建一個深度神經網路Building a deep neural network

deep_neural_network.py


11.5 創建一個向量量化器Creating a vector quantizer

vector_quantization.py

11.6 為序列資料分析創建一個遞迴神經網路Building a recurrent neural network for sequential data analysis

recurrent_network.py

11.7 在光學字元辨識資料庫中將字元視覺化Visualizing the characters in an optical character recognition database

visualize_characters.py


11.8 用神經網路創建一個光學字元辨識器Building an optical character recognizer using neural networks


ocr.py




第12章 視覺化數據VISUALIZING DATA


12.2 畫3D散點圖Plotting 3D scatter plots



12.3 畫氣泡圖Plotting bubble plots



12.4 畫動態氣泡圖Animating bubble plots



12.5 畫圓形圖Drawing pie charts



12.6 畫日期格式的時間序列資料Plotting date-formatted time series data




12.7 畫長條圖Plotting histograms




12.8 視覺化熱力圖Visualizing heat maps




12.9 動態信號的視覺化類比Animating dynamic signals


```
