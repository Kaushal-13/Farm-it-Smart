from __future__ import print_function
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import metrics, tree
import pickle
import warnings
warnings.filterwarnings('ignore')


PATH = 'Pages/Crop_Recommendation_Dataset/Crops.csv'
df = pd.read_csv(PATH)

features = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
target = df['label']
labels = df['label']

feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


# RF = RandomForestClassifier(n_estimators=20, random_state=0)
# RF.fit(Xtrain, Ytrain)

# predicted_values = RF.predict(Xtest)

# x = metrics.accuracy_score(Ytest, predicted_values)
# acc.append(x)
# model.append('RF')
# print("RF's Accuracy is: ", x)

# print(classification_report(Ytest, predicted_values))

# RF_pkl_filename = 'RandomForest.pkl'
# # Open the file to save as pkl file
# RF_Model_pkl = open(RF_pkl_filename, 'wb')
# pickle.dump(RF, RF_Model_pkl)
# # Close the pickle instances
# RF_Model_pkl.close()


@st.cache_data
def load_model(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model


file_path = 'RandomForest.pkl'  # Replace with your file path
loaded_model = load_model(file_path=file_path)


def main():
    st.title('Input The Following Values')

    # Input fields for each column
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    with col1:
        value1 = st.text_input(feature_names[0], key='value1')

    with col2:
        value2 = st.text_input(feature_names[1], key='value2')

    with col3:
        value3 = st.text_input(feature_names[2], key='value3')

    with col4:
        value4 = st.text_input(feature_names[3], key='value4')

    with col5:
        value5 = st.text_input(feature_names[4], key='value5')

    with col6:
        value6 = st.text_input(feature_names[5], key='value6')

    with col7:
        value7 = st.text_input(feature_names[6], key='value7')
    # Create a button to add the values to an Excel-like table
    if st.button("Add to Excel-like table"):
        global df2
        if 'df2' not in globals():
            df2 = pd.DataFrame(
                columns=feature_names)
        values = [value1, value2, value3, value4, value5, value6, value7]

        data = {col: [val] for col, val in zip(feature_names, values)}
        # Append the input values to the DataFrame
        df2 = df2.append(data, ignore_index=True)

        # Display the updated DataFrame
        st.write("Excel-like table:")
        st.write(df2)
        arr = np.array(values)
        arr = arr.reshape(1, -1)
        prediction = loaded_model.predict(arr)
        st.write(prediction)


if __name__ == "__main__":
    main()
