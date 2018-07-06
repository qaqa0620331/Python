
# WGAN==>PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION

We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024². We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.


Progressive Growing of GANs for Improved Quality, Stability, and Variation

https://www.youtube.com/watch?v=G06dEcZ-QTg&feature=youtu.be

https://github.com/tkarras/progressive_growing_of_gans

https://drive.google.com/drive/folders/0B4qLcYyJmiz0NHFULTdYc05lX0U

INCREASING VARIATION USING MINIBATCH STANDARD DEVIATION

minibatch discrimination

variation problem

first compute
the standard deviation for each feature in each spatial location over the minibatch. 
We then average these estimates over all features and spatial locations to arrive at a single value. 
We replicate the
value and concatenate it to all spatial locations and over the minibatch, yielding one additional (constant)
feature map.



unrolling the discriminator (Metz et al., 2016) to regularize its updates

a “repelling regularizer” (Zhao et al., 2017) that adds a new loss term
to the generator, trying to encourage it to orthogonalize the feature vectors in a minibatch. 

The multiple generators of Ghosh et al. (2017) also serve a similar goal.

# CGAN ==> Image-to-Image Translation with Conditional Adversarial Networks

We investigate conditional adversarial networks as a
general-purpose solution to image-to-image translation
problems. These networks not only learn the mapping from
input image to output image, but also learn a loss function
to train this mapping. This makes it possible to apply
the same generic approach to problems that traditionally
would require very different loss formulations. We demonstrate
that this approach is effective at synthesizing photos
from label maps, reconstructing objects from edge maps,
and colorizing images, among other tasks. As a community,
we no longer hand-engineer our mapping functions,
and this work suggests we can achieve reasonable results
without hand-engineering our loss functions either.

https://github.com/phillipi/

https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix





人工智慧試衣間
