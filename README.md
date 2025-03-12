# Introduction

This is a small project using Pépito the Cat data [Twitter account](https://x.com/PepitoTheCat). This Twitter bot is linked to a Raspberry Pi and a camera. The camera takes a picture of a cat trapdoor every time Pépito the Cat goes in or out. The picture is then posted on Twitter with a message (in our out).

There are two types of data:
- The pictures for classification
- The timeseries data of the cat going in and out

The goal of this project is to classify the pictures of the cat trapdoor to know if Pépito the Cat is going in or out. This will be done using a Convolutional Neural Network (CNN).
The timeseries data could be used to predict when Pépito the Cat will go in or out, or to analyze his behavior.