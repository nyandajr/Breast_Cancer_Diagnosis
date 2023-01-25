
# Breast Cancer Detection
This is a machine learning model that is trained to detect breast cancer using the breast cancer dataset from sklearn.datasets. The model uses logistic regression to classify the tumors as malignant or benign.

Installation
Install the required packages by running pip install -r requirements.txt
Usage
Run the script python breast_cancer_detection.py
Data
The dataset used in this model is the breast cancer dataset from sklearn.datasets. The data is loaded into a dataframe and various statistical measures are taken. The data is then split into training and test sets, with the model being trained on the training set and evaluated on the test set.

Model
The model used in this script is logistic regression, which is trained to classify the tumors as malignant or benign. The model is trained using the training data and evaluated on the test data, with the accuracy being reported.

Input
The input data is a tuple of 30 features of the breast cancer dataset. You can replace this input_data with your own input to predict the output.

Output
The output is the accuracy on training and test data. The model also gives the prediction output for the given input data.
