# What:

# 多有用==>應用領域

# 工作原理How do GANs work?


This method of training a GAN is taken from game theory called the minimax game.

# 兩階段訓練模式

Pass 1: Train discriminator and freeze generator (freezing means setting training as false. 
The network does only forward pass and no backpropagation is applied)

Pass 2: Train generator and freeze discriminator


# Steps to train a GAN

Step 1: Define the problem. Do you want to generate fake images or fake text. 
Here you should completely define the problem and collect data for it.

Step 2: Define architecture of GAN. Define how your GAN should look like. 
Should both your generator and discriminator be multi layer perceptrons, or convolutional neural networks? 
This step will depend on what problem you are trying to solve.

Step 3: Train Discriminator on real data for n epochs. 
Get the data you want to generate fake on and train the discriminator to correctly predict them as real. 
Here value n can be any natural number between 1 and infinity.

Step 4: Generate fake inputs for generator and train discriminator on fake data. 
Get generated data and let the discriminator correctly predict them as fake.

Step 5: Train generator with the output of discriminator. 
Now when the discriminator is trained, you can get its predictions and use it as an objective for training the generator. 
Train the generator to fool the discriminator.

Step 6: Repeat step 3 to step 5 for a few epochs.

Step 7: Check if the fake data manually if it seems legit. 
If it seems appropriate, stop training, else go to step 3. 
This is a bit of a manual task, as hand evaluating the data is the best way to check the fakeness. 
When this step is over, you can evaluate whether the GAN is performing well enough.

Now just take a breath and look at what kind of implications this technique could have. If hypothetically you had a fully functional generator, you can duplicate almost anything. To give you examples, you can generate fake news; create books and novels with unimaginable stories; on call support and much more. You can have artificial intelligence as close to reality; a true artificial  intelligence! That’s the dream!!
# reference:
>* https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/
