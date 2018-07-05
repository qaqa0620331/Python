Computer evolves to generate baroque music!

https://www.youtube.com/watch?v=SacogDL_4JU

https://tw.wxwenku.com/d/100505574


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


