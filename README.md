# CNN for Multi-Digit Classification

This project explores how Convolutional Neural Networks (CNNs) can be used to effectively identify a series of digits from real-world images that are obtained from “The Street View House Numbers (SVHN) Dataset”.  CNNs have evolved dramatically every year since the inception of the ImageNet Challenge in 2010. 

## Problem Statement
I am attempting to predict a series of numbers given an image of house numbers from the SVHN dataset. An important thing to take note is that instead of the standard identification of numbers, as with the MNIST dataset, I now need to correctly detect the numbers and the sequence of numbers. 

## Programming Language
I used Python and Tensorflow to build the model. This implementation also uses TensorBoard extensively for visualizations.

## Problems running Tensorflow? Use TFAMI.
I recommend starting a GPU instance using Amazon's AWS. I have created an image and replicated it across all regions. You can easily run this set of code on the GPU instance within a few minutes. Simply search for `TFAMI` under `community AMIs` when you are launching your instance. More information on the specific IDs can be obtained from the following [Github repository](https://github.com/ritchieng/tensorflow-aws-ami).

## How to use this code base
1. Create the relevant folders with the commands 
	- ```
	mkdir log_trial_1
	mkdir log_trial_2
	```
2. You can load the data and pre-process all the images with one single command ```python load_data.py```
3. Load the first model using the command ```python model_trial_1.py```
    - The output should resemble something similar to this [output](https://github.com/ritchieng/NumNum/blob/master/NumNum/model_trial_1_command_results.txt).
4. You can view Tensorboard's visualizations using the command ```tensorboard --logdir=log_trial_1```
5. Load the second model using the commmand ```python model_trial_2.py```
    - The output should resemble something similar to this [output](https://github.com/ritchieng/NumNum/blob/master/NumNum/model_trial_1_command_results.txt).
6. You can view Tensorboard's visualizations using the command ```tensorboard --logdir=log_trial_2```
    - You may encounter an issue whereby it says ```Port is in use: 6006``` if you run tensorboard twice on different trials.
    - Simply run the command ```lsof -i:6006``` or whatever the port number is.
    - Then run the command ```kill -9 <PID>``` where the PID is the number you can find when you run the command above.
    - Simply run the command to launch Tensorboard again ```tensorboard --logdir=log_trial_2```


## Detailed Report
To guide you through, I have made a detailed report. You can refer to the report [here](https://github.com/ritchieng/NumNum/blob/master/NumNum/report/report.pdf). Also, it is actually in this repository.

## Academic Journals and Resources
1. [Multi-digit recognition](https://arxiv.org/abs/1312.6082)
2. [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/)

## Licensing
This is an open source project governed by the license in this repository.