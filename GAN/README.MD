Computer evolves to generate baroque music!

https://www.youtube.com/watch?v=SacogDL_4JU

https://blog.csdn.net/qq_31780525/article/details/71932020

https://tw.wxwenku.com/d/100505574


https://blog.csdn.net/qq_25737169/article/details/78857788

[系列活動] 一日搞懂生成式對抗網路
https://www.slideshare.net/tw_dsconf/ss-78795326


生成對抗模式 GAN 的介紹

https://www.slideshare.net/yenlung/gan-90396897


與高中生談人工智慧與深度學習'
https://www.slideshare.net/yenlung/ss-82198270


強化學習 Reinforcement Learning

https://www.slideshare.net/yenlung/reinforcement-learning-90737484

Recurrent Neural Network 遞迴式神經網路

https://www.slideshare.net/yenlung/recurrent-neural-network-89538572


https://github.com/tkarras/progressive_growing_of_gans


吳志忠 - 利用 Python 與人工智慧快速打造人性化聊天機器人 - PyConTW2017

https://www.youtube.com/watch?v=zL2v0czpsyk


https://wiseodd.github.io/

https://github.com/wiseodd/generative-models



Awesome-GANs with Tensorflow
https://github.com/kozistr/Awesome-GANs


gans-awesome-applications
https://github.com/nashory/gans-awesome-applications




https://github.com/tkarras/progressive_growing_of_gans

# Mini-batch

http://hp.stuhome.net/index.php/2016/09/20/tensorflow_batch_minibatch/


https://zhuanlan.zhihu.com/p/25601482


深度學習的優化演算法，說白了就是梯度下降。每次的參數更新有兩種方式。

Batch gradient descent
第一種，遍歷全部資料集算一次損失函數，然後算函數對各個參數的梯度，更新梯度。
這種方法每更新一次參數都要把資料集裡的所有樣本都看一遍，計算量開銷大，計算速度慢，不支援線上學習，這稱為Batch gradient descent，批梯度下降。

stochastic gradient descent
另一種，每看一個資料就算一下損失函數，然後求梯度更新參數，這個稱為隨機梯度下降，stochastic gradient descent。
這個方法速度比較快，但是收斂性能不太好，可能在最優點附近晃來晃去，hit不到最優點。
兩次參數的更新也有可能互相抵消掉，造成目標函數震盪的比較劇烈。

mini-batch gradient decent
為了克服兩種方法的缺點，現在一般採用的是一種折中手段，mini-batch gradient decent，小批的梯度下降，
這種方法把資料分為若干個批，按批來更新參數，這樣，一個批中的一組資料共同決定了本次梯度的方向，下降起來就不容易跑偏，減少了隨機性。
另一方面因為批的樣本數與整個資料集相比小了很多，計算量也不是很大。


https://zh-tw.coursera.org/learn/machine-learning

### 案例1:vanilla_gan

https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py


sudo python3 test.py

要裝套件
sudo apt-get install python3-tk

執行==> sudo python3 test.py

成果==>程式碼改成for it in range(10000):

每一個iter會產生結果圖

Iter: 0
D loss: 1.724
G_loss: 1.553

Iter: 1000
D loss: 0.002649
G_loss: 9.038

Iter: 2000
D loss: 0.00579
G_loss: 8.99

Iter: 3000
D loss: 0.02634
G_loss: 5.401

Iter: 4000
D loss: 0.02967
G_loss: 6.889

Iter: 5000
D loss: 0.1539
G_loss: 5.191

Iter: 6000
D loss: 0.355
G_loss: 4.346

Iter: 7000
D loss: 0.3148
G_loss: 3.902

Iter: 8000
D loss: 0.4361
G_loss: 3.636

Iter: 9000
D loss: 0.4466
G_loss: 3.099



### 案例2:CGAN==>把label加進來

https://github.com/wiseodd/generative-models/blob/master/GAN/conditional_gan/cgan_tensorflow.py


if not os.path.exists('out_cgan/'):
    os.makedirs('out_cgan/')
    
    for it in range(100000):
    if it % 1000 == 0:
        n_sample = 16

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, 7] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        fig = plot(samples)
        plt.savefig('out_cgan/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
    
### 案例3:infoGAN


https://blog.csdn.net/u011699990/article/details/71599067

https://blog.csdn.net/hjimce/article/details/55657325


### REGULAR VONVOLUTION VS TRANSPOSE CONVOLUTION

STRIDE

PADDING

3*3 ----> 6*6

 6*6 ---->3*3 


### 案例4:DCGAN

https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0

### 案例5:ACGAN ==> 多種class

##### Tensorflow實作

##### PyTorch實作

gitlimlab/ACGAN-PyTorch

https://github.com/gitlimlab/ACGAN-PyTorch

### 6:EBGAN == energy-based GAN

Energy-based Generative Adversarial Network
Junbo Zhao, Michael Mathieu, Yann LeCun
(Submitted on 11 Sep 2016 (v1), last revised 6 Mar 2017 (this version, v4))

We introduce the "Energy-based Generative Adversarial Network" model (EBGAN) which views the discriminator as an energy function that attributes low energies to the regions near the data manifold and higher energies to other regions. 

Similar to the probabilistic GANs, a generator is seen as being trained to produce contrastive samples with minimal energies, while the discriminator is trained to assign high energies to these generated samples. 
Viewing the discriminator as an energy function allows to use a wide variety of architectures and loss functionals in addition to the usual binary classifier with logistic output. Among them, we show one instantiation of EBGAN framework as using an auto-encoder architecture, with the energy being the reconstruction error, in place of the discriminator. We show that this form of EBGAN exhibits more stable behavior than regular GANs during training. We also show that a single-scale architecture can be trained to generate high-resolution images.

D==>使用autoencode

Mean-squRED rECONSTRUCTION ERROR AS D(X)
##### Tensorflow實作

https://github.com/wiseodd/generative-models/tree/master/GAN/ebgan

##### PyTorch實作

https://github.com/wiseodd/generative-models/tree/master/GAN/ebgan

### WGAN 

GAN==>training instability
WGAN==>generate only low-quality samples or fail to converge
==>due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior


News text generation with adversarial deep learning
http://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8925528&fileOId=8925531


##### Lipschitz限制

##### Improved WGAN gradient penalty (WGAN-GP)

Improved Training of Wasserstein GANs
Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron Courville

https://arxiv.org/abs/1704.00028



##### Tensorflow實作

https://github.com/igul222/improved_wgan_training

##### PyTorch實作
