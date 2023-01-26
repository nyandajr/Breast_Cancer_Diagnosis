from sklearn.datasets import load_breast_cancer
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Breast Cancer Classification", page_icon=":guardsman:", layout="wide")

st.title("Breast Cancer Classification AI App")

st.sidebar.header("Dataset Summary")

# Load the breast cancer dataset
breast_cancer_dataset = load_breast_cancer()
df = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
df["label"] = breast_cancer_dataset.target

# Show the dataset info
# st.sidebar.subheader("Dataset info")
# st.sidebar.write(df.info())

# Show the number of rows and columns in the dataset
st.sidebar.subheader("Dataset dimensions")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")

# Show the distribution of the target variable
st.sidebar.subheader("Target variable distribution")
st.sidebar.write(df["label"].value_counts())

# Get the features and target
X = df.drop(columns="label", axis=1)
Y = df["label"]
df['label'].replace({0:'benign',1:'malignant'},inplace=True)
df['label'].replace({'benign':0,'malignant':1},inplace=True)


# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Get the accuracy on the training and testing sets
training_accuracy = accuracy_score(Y_train, model.predict(X_train))
testing_accuracy = accuracy_score(Y_test, model.predict(X_test))

# Display the model's accuracy

# Input data to predict
# Input data to predict
# Input data to predict
# Input data to predict
input_data = st.multiselect("Select the features", df.columns[:-1], key="input")
input_data = [st.number_input(f"Enter {i}", min_value=df[i].min(), max_value=df[i].max(), value=df[i].mean(), step=0.01) for i in input_data]
if st.button("Click here to predict"):
    if len(input_data) < 30:
        st.error("You must select at least 30 features to make a prediction, so far you have selected less.")

    else:
        input_data = np.array(input_data).reshape(1,-1)
        prediction = model.predict(input_data)
        if prediction[0] == 0:
            st.write("The breast cancer is Benign.")
        else:
            st.write("The bre is Malignant.")


st.header("Model Evaluation")
st.write("Accuracy on training data:", round(training_accuracy*100), "%")
st.write("Accuracy on testing data:", round(testing_accuracy*100), "%")
import streamlit as st

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
<p>Coyright @Nyanda Jr</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)
