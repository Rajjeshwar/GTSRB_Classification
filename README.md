# GTSRB_Classification
Multiclass-classification on the GTSRB dataset
   
To beginwith, here is a brief explanation of Convolutional Neural Networks: 

![image](https://user-images.githubusercontent.com/80246631/142187970-d2078d47-fae0-40ee-89c4-3aeccf0d6033.png)

In this project, we performed a multiclass classification on the GTSRB dataset using traditional CNNs and have then compared it with a VGG19 transfer learning model. To maximize accuracy average and soft voting ensembling were performed. 

## Design decisions:

### CNN

![image](https://user-images.githubusercontent.com/80246631/142197356-47049e7e-e504-427a-9acd-e5ad96d6f468.png)


1. The input images were resized to 32 x 32px each. On manually verifying I decided to use the rgb format for the images as color stood out as being one of the most important features used to define a sign.
2. 30% of the training images were used to cross validate the model while training.
3. For the output activation we use a softmax activation to obtain the probabilities of predictions for each label per value of X_train. 
4. Sparse-categorical cross entropy is used as our loss function since we have 43 output classes. Using categorical cross entropy would entail one hot encoded output vectors of length 43 each which can be avoided with this choice. 
5. For optimization, we used a standard ADAM optimizer with default values for learning rate(0.001), beta1(0.9) and beta2(0.99).
6. Weights have been initialized with the Glorot uniform distribution. I tried using random weights but the loss function failed to decrease. 
7. To reduce overfitting a dropout layer was added in the penultimate dense layer with a rate of 0.5. 
8. Since traffic signs have a lot of edges and other such low level sharp features we use max pooling layers to extract these during training for the model to better understand the image.

### Using VGG19 architecture 

![image](https://user-images.githubusercontent.com/80246631/142196981-e8bb0f1f-b28a-44d8-9581-02ade4f91d6c.png)


1. To compare how the CNN model performs against VGG19 for this dataset we use transfer learning to import the architecture of the model. 
2. The weights that were pre-trained on the imagenet dataset were however not used. The reasoning for this is that the imagenet dataset has a significant difference in feature distribution for the data. Furthermore, on trying to fit the model with pre-trained weights on this data I observed significant underfitting. 
3. Therefore only the model architecture was used, the weights were re-trained. The output layer was replaced with a BatchNorm layer followed by a dense layer with a sigmoid activation to bind the output values between 0 and 1.
4. For the final layer we again use a softmax activation. 

### Model averaging 

We create an ensemble that averages over the outputs of the two models. `tf.keras.Average(inputs=model_input, outputs=ensemble_output)`

### Soft voting 

Finally, to verify how the two models perform when using soft voting on the outputs of the CNN and the VGG19 model we add the predictions of the two models and calculate the log loss. 

## Evaluation and Metrics: 

We calculate the log loss(cross entropy loss) for each of the models to analyze their performance. Log-loss is indicative of how close the prediction probability is to the corresponding actual/true value (0 or 1 in case of binary classification). The more the predicted probability diverges from the actual value, the higher is the log-loss value. 

First let's see how well the CNN model performed using `log_loss_score = metrics.log_loss(Y_test, pred_Cnn)` and use `np.argmax(pred_Cnn, axis=1)` to find the mean precision, recall and f1 scores.

 ```
 Log loss score: 0.242151899069741
 
 Train accuracy = 97.60
 Test accuracy = 94.65
 
 weighted average 
 precision = 0.95
 recall = 0.95
 f1 score = 0.95
 
 ```
As we can see the CNN is really good at classifying on both the train, val and test sets. There is some overfit observed but overall the accuracy is still good.

For transfer learning using VGG19:

 ```
 Log loss score: 0.3427089769729845
 
 Train accuracy = 93.28
 Test accuracy = 92.55
 
 weighted average 
 precision = 0.92
 recall = 0.93
 f1 score = 0.92
 
 ```
 
 For model average ensemble:
 
 ```
 Log loss score: 0.22207731239115464
 
 Train accuracy = 97.76
 Test accuracy = 95.19
 ```
 
 Soft voting:
 
 ```
 Log loss score: 0.19205694852732097
 ```

 ## Install Requirements: 
 
The following were used for making this program-

1. Tensorflow
2. sklearn
3. numpy
4. pandas
5. os module
6. unittest
 
 ```
 pip install -r requirements.txt
 ```
 
 The following link provides a good walkthrough to setup tensorflow:
 
  ```
https://towardsdatascience.com/tensorflow-gpu-installation-made-easy-use-conda-instead-of-pip-52e5249374bc
 ```
 
 
 ## Format code to PEP-8 standards (Important for contributing to the repo): 
 
 This repository is strictly based on *PEP-8* standards. To assert PEP-8 standards after editing your own code, use the following: 
 
 ```
 black GTSRB_Classification_setup.py
 black GTSRB_model_training_evaluation.py
 ```
 
If you wish to use change the dataset used here change the following to correctly reflect the directory in `CatsVSDogs-Dataload.py`:

`data_dir_train = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\data\Roadsigns\Train"`

`data_dir_test = r"C:\Users\Desktop\Desktop\JuPyter Notebooks\data\Roadsigns\Test"`



NOTE: This was trained on a 2080Super using tensorflow GPU, images were resized to fit vram constraints. Training will take longer on GPUs not running CUDA, on CPUs and if larger datasets are used.

### Reference: 

1. https://cs231n.github.io/convolutional-networks/
2. https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
3. https://towardsdatascience.com/what-is-stratified-cross-validation-in-machine-learning-8844f3e7ae8e
4. https://www.superdatascience.com/blogs/convolutional-neural-networks-cnn-softmax-crossentropy
