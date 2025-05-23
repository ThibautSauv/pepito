# Introduction

This is a small project using Pépito the Cat data [Twitter account](https://x.com/PepitoTheCat). This Twitter bot is linked to a Raspberry Pi and a camera. The camera takes a picture of a cat trapdoor every time Pépito the Cat goes in or out. The picture is then posted on Twitter with a message (in our out).

There are two types of data:
- The pictures for classification
- The timeseries data of the cat going in and out

The goal of this project is to classify the pictures of the cat trapdoor to know if Pépito the Cat is going in or out. This will be done using a Convolutional Neural Network (CNN).
The timeseries data could be used to predict when Pépito the Cat will go in or out, or to analyze his behavior.

# Data

The image dataset is available [here](https://huggingface.co/datasets/PepitoTheCat2007/pepito-images). It is presplitted into a training and a testing set.

![99/1 train_test split](./imgs/train_test.png)

The classes are:
- 0: In
- 1: Out

The repartition of the classes is supposedly balanced. The real repartition is the following:
![61/39](./imgs/in_out.png)

The images are of different sizes and shapes. They are in color. The dataset is composed of 20078 images.
```python	
{
    (480, 640, 3): 15730,
    (240, 320, 3): 1700,
    (288, 384, 3): 709,
    (855, 1200, 3): 1419,
    (900, 1200, 3): 482,
    (450, 600, 3): 36,
    (901, 1053, 3): 1,
    (853, 1200, 3): 1
}
```
The minimum size is (240, 320, 3) so the images can be reshaped to a common 224x224 size (usually used in CNNs).

# Data Preprocessing

The images sometimes do not feature any cat in them. Thus, they should be removed from the dataset. After processing the images through YOLO, we have the following repartition:

![YOLO repartition: 75% of cat detected](./imgs/yolo.png)

YOLO sometimes detects other objects in the image, whether it is hallucinated or not. The cat is detected in 75% of the images. 

With this new data set, we have the following repartition:
![55/45 repartition for filtered dataset](./imgs/filtered_in_out.png)

# Performance
With only 5 epochs, the model is able to reach 97.56% accuracy on the test set. The misclassification is mostly due to the cat not being in the image, and the trapdoor being closed. Thus the image should be removed from the dataset.

Removing the images where the cat is not present, the model reaches 99.8% accuracy on the test set, with only 5 epochs.

Applying PCA to the feature map of the model, the model doesn't learn anymore. This was the case for the minimal dimension possible for PCA (512x49 -> 32x49).

TODO: Let's try with higher dimensions (maximal dimension possible for PCA being 512x49 -> 49x49).

# Conclusion
The model used is a VGG11 pretrained, with the last layers replaced by two fully connected layers. The first layer has 512 neurons and the second one has 2 neurons. Seeing the performance of the model, it could be interesting to use a smaller model to reduce the size of the model and the computation time (even though it is already quite fast).



# TODO:
- [ ] Use smaller models/preprocessing to reduce the size of the model (VGG11 seems to have figured out to look at the trapdoor)
- [x] Understand misclassification
- [x] Filter out no-cat images from the dataset ~~or relabel them to a new class~~
- [ ] Use **better** techniques to reduce features but keep the information
- [ ] Compute and Analyze metrics