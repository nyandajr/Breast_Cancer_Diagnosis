import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

###### Loading data set

from sklearn.datasets import load_breast_cancer
breast_cancer_dataset = load_breast_cancer()

###### Loading data into data frame

df = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

###### Adding the target column to the df

df['label'] = breast_cancer_dataset.target

###### Separating features and target

X = df.drop(columns='label',axis = 1)
Y = df['label']

###### splitting the dataset into training set and testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

###### Model training using Logistic regression

model = LogisticRegression()

# training the Logistic Regression model using Training data
model.fit(X_train, Y_train)

###### Model Evaluation and Accuracy score

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

st.markdown("<h1 style='text-align: center;'>AI Breast Cancer Diagnosis App</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center;'>Enter the patient data using the sliders below</h24>", unsafe_allow_html=True)





# Add a slider for each feature in the input data
input_data = [st.slider(f'{feature}', min_value, max_value) for feature, min_value, max_value in zip(breast_cancer_dataset.feature_names, X.min(), X.max())]

# convert input_data to a numpy array
input_data = np.array(input_data)
prediction = model.predict(input_data.reshape(1,-1))
if prediction == 0:
    st.write("Patient is benign")
else:
    st.write("Patient is malignant, hence he or she needs immediate care support.⚠️")
    
st.subheader("EVALUATION METRICS") 
st.write(f'1) Accuracy on training data is {round(training_data_accuracy*100)}%')
st.write(f'2) Accuracy on testing data is {round(test_data_accuracy*100)}%')

st.empty()
st.empty()


# Adding a footer
footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Copyright © Nyanda Jr</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)



