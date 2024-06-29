Recognition of the vehicle identification number
---
## Synopsis
The police department of Kharkiv needs to digitalize an old vehicle database. Every
card from the database has special field with 17 boxes that is filled by handwritten
VIN-code characters - one character in each box. VIN or vehicle identification
number is a unique code, including a serial number, used by the automotive industry
to identify individual vehicles. The cards are already scanned and the VIN-code
boxes are already recognized. My task was to design a neural network that can
classify small squared black&white image (VIN character boxes) with single
handwritten character on it.

## Data
For training CNN as an example of handwritten characters from VIN-code I'm using EMNIST dataset wich is de-facto extended famous MNIST dataset. It was downloaded [here](https://www.nist.gov/itl/products-and-services/emnist-dataset) and you also need to download this dataset if you want to run this model.

## Training
This is a sample code for training a neural network using the Keras library. The code demonstrates the process of training a convolutional neural network (CNN) on a dataset of handwritten characters. Here's an overview of the steps involved in the training process:

 ###### Data Preparation:

   1. The training and test datasets are loaded from CSV files using pandas.
   2. The dataset is preprocessed by removing unwanted classes and reshaping the input images.
   3. The data is normalized by dividing the pixel values by 255.0 to scale them between 0 and 1.
###### Model Creation:

1. A sequential model is created using the Sequential class from Keras.
2. Convolutional layers (Conv2D) are added to extract features from the input images.
3. Max pooling layers (MaxPooling2D) are added to reduce the spatial dimensions of the features.
4. Dropout layers (Dropout) are added to prevent overfitting by randomly disabling some neurons during training.
5. The feature maps are flattened using a Flatten layer to be fed into a fully connected layer.
6. Fully connected layers (Dense) are added to perform the classification.
7. The final layer uses the softmax activation function to output class probabilities.
###### Model Compilation:

1. The model is compiled with the appropriate loss function (categorical_crossentropy for multiclass classification), optimizer (adam), and evaluation metric (accuracy).
###### Model Training:

1. The model is trained on the training data using the fit function.
2. The training data is divided into batches (batch_size) to update the model weights incrementally.
3. The training is performed for a specified number of epochs, which is the number of times the model iterates over the entire dataset.
###### Model Evaluation:

1. The trained model is evaluated on the test data using the evaluate function.
2. The test loss and accuracy are calculated and printed.
###### Prediction and Visualization:

1. The model is used to make predictions on a subset of test images using the predict function.
2. The predicted labels and corresponding true labels are displayed along with the test images using matplotlib.
###### Model Saving:

1. The trained model is saved to a file using the save function for future use.
## Results
After integration of the CNN into numbers detection and recognition pipeline I could apply it to handwritten characters.

## How to run
Libs requirements:
1. tensorflow
2. keras
3. pandas
4. numpy
5. matplotlib
6. pillow

Input: images in the input folder
Output: output is in CSV format, the CSV file will be saved on your folder in project.

