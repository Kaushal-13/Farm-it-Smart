import streamlit as st
from streamlit_folium import folium_static, st_folium
import folium
import pickle
import numpy as np
from req import getCountry, getWeatherDetails
from GetSoilData import getData


def load_model(file_path):
    with open(file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    return loaded_model


file_path = 'RandomForest.pkl'  # Replace with your file path
loaded_model = load_model(file_path=file_path)


def calculate_average(lis, start, end):
    if not lis:
        return 0  # Handling an empty list to avoid division by zero
    lst = lis[start:end+1]
    total = sum(lst)
    average = total / len(lst)
    return average


m = folium.Map(width=1000)
m.add_child(folium.LatLngPopup())
map_data = st_folium(m)


# List of Gregorian months
months = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December"
]

# Create dropdowns for start and end months
start_month = st.selectbox("Select start month:", months)
end_month = st.selectbox("Select end month:", months)

month_dict = {month: index for index, month in enumerate(months)}

final_data = {

}

file_path = 'RandomForest.pkl'  # Replace with your file path
with open(file_path, 'rb') as file:
    loaded_model = pickle.load(file)

if st.button("Confirm Location"):
    location = map_data['last_clicked']
    st.write(location)
    lat = location['lat']
    lon = location['lng']
    country = getCountry(lat, lon)
    data = getWeatherDetails(lat, lon)

    start = month_dict[start_month]
    end = month_dict[end_month]

    for key in data.keys():
        final_data[key] = calculate_average(data[key], start=start, end=end)
    final_data['rainfall'] = sum(data['rainfall'][start:end+1])
    a = getData(country=country)
    print(a)
    final_data.update(a)

if ("N" in final_data.keys()):
    st.write(final_data)
    values = [float(final_data['N']), float(final_data['P']), float(final_data['K']),
              final_data['temp'], final_data['humid'], final_data['pH'], final_data['rainfall']]
    arr = np.array(values)
    arr = arr.reshape(1, -1)
    prediction = loaded_model.predict(arr)
    st.write(prediction)
