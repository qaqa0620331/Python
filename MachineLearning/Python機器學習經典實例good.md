https://detail.tmall.com/item.htm?id=558284517556&ns=1&abbucket=19


https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-cookbook

https://github.com/PacktPublishing/Python-Machine-Learning-Cookbook

Python Machine Learning Cookbook

Prateek Joshi

June 2016

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
```
1.3 標記編碼方法
1.4 創建線性回歸器
1.5 計算回歸準確性
1.6 保存模型資料
1.7 創建嶺回歸器
1.8 創建多項式回歸器
1.9 估算房屋價格
1.10 計算特徵的相對重要性
1.11 評估共用單車的需求分佈

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
