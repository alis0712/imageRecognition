# Image Recognition

The goal of this project is to use a neural architecture such as CNN or a variation of CNN and detect a set of 100 images using the CIFAR dataset. To do this I first loaded in the data, and started off with MNIST dataset and then added in CIFAR-10 data. Fashion MNIST data was used as benchmark. Fashion MNIST data is a 28x28 grayscale data it includes labels such as t-shirts, shoes, pants and so forth. CIFAR-10 dataset is a 32x32x3 color images. It includes dataset such as automobile, frog, horse, and so forth. Figure below shows what CIFAR-10 and MNIST data pertains: 

![image](https://user-images.githubusercontent.com/62857780/209239438-ed94b707-3955-4961-83e3-d52f92c125ed.png)
CIFAR-10 and Fashion MNIST Datasets



When the python code for CNN is implemented, it takes in a kernel of the input image then implements a pooling layer where it applies the ReLU activation function, it then flattens the layer and implements either forward or backward propagation to the image and then spits out the results. The whole process of CNN is identified in the figure below: 

![image](https://user-images.githubusercontent.com/62857780/209239528-8b4bffe5-0124-4fc5-93f5-c719b59f4993.png)

CNN code execution process

So, CNN is just an ANN with convolution layers. Utilizing Functional API is a better way of creating models. The process of CNN is summarized below
	Train the model
	Evaluate the model
	Make predictions
	Further refine the model using data augmentation by using techniques such as batch normalization
Figures below show how the data Fashion MNIST data and CIFAR data is taken in python: 

![image](https://user-images.githubusercontent.com/62857780/209239746-87fdeaa1-057d-424c-9142-6220e5ebd20c.png)

![image](https://user-images.githubusercontent.com/62857780/209239766-9f27f2bd-5637-42e3-99c2-7971a66550f2.png)

Figures representing the input data in Python using Jupyter Notebook IDE


## Fashion MNIST Dataset 

The process of implementing Fashion MNIST dataset in python is as follows: 
	Drop-in replacement for MNIST, exact same format. 
	X.shape = N X 28 X 28 (grayscale)
	Pixel values 0…255
	Not the right shape for CNN. CNN expects to N x H x W x Color
	We must reshape to N x 28 x 28 x 1
Fashion CIFAR-10 Dataset 
The process of implementing CIFAR-10 dataset in python is as follows: 
	Data is N x 32 x 32 x 3-pixel values from 0….255
	Slight inconvenience: labels are N x 1
	Just call flatten () to fix it. 
Figure below shows how convolve function is implemented in Python: 

![image](https://user-images.githubusercontent.com/62857780/209239890-f2b10fac-c3f4-4ed0-bd95-6494e3a79ee3.png)

Shows how the Convolve function is implemented in Python

## Data Augmentation 

Data augmentation is a technique where developers significantly increase the diversity of data available for training models without collecting new data. This process is typically done using techniques such as padding, cropping, and horizontal flipping which are used to train large neural networks. Images can be seen by the human eye, so it allows users to invent new data. However, we need to keep in mind that coming with newer forms of data results in more space taken up by memory. There are also endless number of ways I can invent new data. We shift data by 1, 2, 3, 4, 5,…n pixels. We also shift the data by moving it clockwise or counterclockwise, up, down, left, right and so forth. And we can do this using TensorFlow’s Keras API, we just need to be aware of generators and iterators which are like for-loops. 

Example of data augmentation program implemented in Python 

![image](https://user-images.githubusercontent.com/62857780/209239969-6bf9ae5a-f5c0-4392-8b7f-edcf001684b3.png)


## Batch Normalization
Is a method which is used to make CNN faster and more efficient through normalization of layer’s inputs by either recentering and rescaling. The recentering and rescaling process is done through a process called internal covariate shift. The \beta\ and\ \gamma are learned through gradient descent. Batch normalization is calculated the following way:  
z=\frac{x\ -\ \mu_b}{\sigma_B} , y=z\gamma+\ \beta
The process of batch normalization is as follows: 
Feedforward Network: BatchNorm --> Dense --> BathNorm --> Dense -->  BatchNorm --> Dense
Batch normalization acts as a regularization. The regularization process helps with overfitting. Since every batch is slightly different, slightly different values of mean and standard deviation are obtained. They are often the true mean and standard deviation of the whole dataset. And this is essentially noise and using noise during the training makes the neural network impermeable to noise. In convolution layers Batch Norms are applied the following way: 
Conv --> BN --> Conv --> BN --> Conv --> BN --> Flatten --> Dense --> Dense



## Results and Conclusion
After implementing the code, we get the following results: 
Fashion MNIST Data – Training Dataset Results 
![image](https://user-images.githubusercontent.com/62857780/209240311-619a92be-27a6-48cd-8270-15e813cd4b3e.png)
Training data set accuracy and Test data set accuracy
With Fashion MNIST data we notice a training data accuracy of 0.9572 with a test accuracy of 0.8983

Fashion MNIST Data – Loss Per Iteration Results  
I then plotted the loss per iteration graph and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240195-b3834fde-b1e3-4001-bba9-a219d6e31cc9.png)

Loss per iteration test and training dataset

Fashion MNIST Data – Accuracy Per Iteration
I then plotted the accuracy per iteration graph and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240245-eec87ba6-0a0a-43d5-be5b-5bb7ea4e3503.png)

Accuracy per iteration on test and training dataset

Fashion MNIST Data – Confusion Matrix
I then plotted fashion MNIST datas confusion matrix and obtained the following results:

![image](https://user-images.githubusercontent.com/62857780/209240357-4caaf637-b8cb-495f-be24-b2210d3bbbce.png)

Confusion Matrix

Fashion MNIST Data – Misclassified Examples

I then plotted the misclassified examples and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240514-5fb10e01-2ae7-4d31-ba49-02a1091e272d.png)

Misclassified Examples – Predicted Dress but gave it coat

CIFAR-10 Training Dataset Results
![image](https://user-images.githubusercontent.com/62857780/209240590-41ed3b96-4679-4716-9c02-2efdb22f56d9.png)

Training data set accuracy and Test data set accuracy
With CIFAR-10 data we notice a training data accuracy of 0.9184 with a test accuracy of 0.6916. To improve test accuracy, we then performed data augmentation with batch normalization to improve the accuracy of the CIFAR-10 dataset. 


CIFAR-10-Loss Per Iteration
I then plotted loss per iteration on the test and training dataset and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240627-46eaad85-91a0-4bfc-ae82-288f0b659e4a.png)

Loss per iteration on test and training dataset

CIFAR-10-Accuracy Per Iteration
I then plotted accuracy per iteration on the test and training dataset and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240671-debec67e-3341-47ee-bccc-8b05739a80d2.png)

Accuracy per iteration on test and training dataset

CIFAR-10 Data – Confusion Matrix
I then plotted the confusion matrix and obtained the following results: 
![image](https://user-images.githubusercontent.com/62857780/209240711-d43420a8-cd0a-4a07-a660-0c8553ace522.png)
Confusion Matrix

CIFAR-10 Data – Misclassified Data
I then went ahead and plotted the CIFAR-10 misclassified data and obtained the following results:

![image](https://user-images.githubusercontent.com/62857780/209240756-89ede0e0-8fbb-4f8f-b631-40ef8ade0c45.png)

Misclassified Examples – Predicted ship but gave it automobile

CIFAR-10 Augmented Training Dataset Results: 
After augmenting CIFAR-10 dataset we obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240803-cab8a69d-2f7d-45bc-8dfa-6c0e2f8b9425.png)

Results obtained after augmenting CIFAR-10 dataset – Training dataset accuracy and Test dataset accuracy

CIFAR-10 Data – Loss Per Iteration – Augmented Data: 
I then plotted the loss per iteration graph and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240889-456be146-9176-4c6d-9932-146ef05e9426.png)

Loss per iteration on test and training dataset

CIFAR-10 Data – Accuracy Per Iteration – Augmented Data: 
I then plotted the accuracy per iteration graph and obtained the following results:

![image](https://user-images.githubusercontent.com/62857780/209240917-cfb0ad7c-cacf-47e4-a3c8-bc615ebf79be.png)

Accuracy per iteration on test and training dataset

CIFAR-10 Data – Confusion Matrix – Augmented Data: 
I then plotted the CIFAR-10 Confusion Matrix data and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240945-421519c6-40c7-4b4d-be81-94611fe0f20c.png)

Confusion Matrix of augmented CIFAR-10 data

CIFAR-10 Data – Misclassified Data – Augmented Data: 
I then plotted the misclassified data and obtained the following results: 

![image](https://user-images.githubusercontent.com/62857780/209240997-49ced3d9-d441-4b8b-af7f-fc9561c4db03.png)

Misclassified Examples – Predicted frog but gave bird

## Conclusion
Convolution Neural Networks (CNN) is a powerful algorithm that can be used to solve various problems related to image recognition. One of the ways we can improve the accuracy of CNN as shown in this project is through batch normalization, other ways we can improve the results are as follows: 
	Using bigger pre-trained models
	Using K-fold cross organization
	Using MixUp to augment images
	Using CutMix technique to augment images
	Using Ensemble learning


## References
https://www.ibm.com/cloud/learn/neural-networks
https://www.cs.toronto.edu/~urtasun/courses/CSC411
https://machinelearningmastery.com/batch-normalization-for-training-of-deep-neural-networks/
http://www.cs.cornell.edu/courses/cs5670/2021sp/lectures/lec21_cnns_for_web.pdf
https://github.com/s9k96/Image-Classification-on-CIFAR10-using-CNN/blob/master/main.ipynb

