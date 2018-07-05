https://github.com/wiseodd/generative-models


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
    
### 案例3:DCGAN

### 案例4:infoGAN
