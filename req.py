import requests


def calculate_relative_humidity(temp_celsius, dew_point_celsius):
    # Constants for water vapor calculation
    a = 17.27
    b = 237.7

    # Calculate actual vapor pressure
    gamma = (a * dew_point_celsius) / (b + dew_point_celsius)
    actual_vapor_pressure = 6.11 * 10**(gamma)

    # Calculate saturated vapor pressure at temperature
    gamma_temp = (a * temp_celsius) / (b + temp_celsius)
    saturated_vapor_pressure = 6.11 * 10**(gamma_temp)

    # Calculate relative humidity
    relative_humidity = 100 * \
        (actual_vapor_pressure / saturated_vapor_pressure)

    return relative_humidity


def getWeatherDetails(lat, lon):
    base_url = 'https://api.weatherbit.io/v2.0/normals?'
    api_key = '477d0500ff1c408a8c8fbab009178291'
    params = {
        'lat': lat,
        'lon': lon,
        'key': api_key,
        'start_day': '01-01',
        'end_day': '12-31',
        'tp': 'monthly'
    }
    response = requests.get(base_url, params=params)

    if (response.status_code == 200):
        data = response.json()
        rainfall = []
        temp = []
        dewpt = []
        humid = []
        for val in data['data']:
            rainfall.append(val['precip'])
            temp.append(val['temp'])
            dewpt.append(val['dewpt'])
        for item1, item2 in zip(temp, dewpt):
            humid.append(calculate_relative_humidity(item1, item2))
        print(humid)
        print(dewpt)
        return {
            'temp': temp,
            'humid': humid,
            'rainfall': rainfall
        }

    else:
        return {
            'error': "API error"
        }


def getCountry(lat, long):
    api_key = 'a202c3e9066d4682937272e5510a474a'
    url = 'https://api.geoapify.com/v1/geocode/reverse?'

    params = {
        'lat': lat,
        'lon': long,
        'apiKey': api_key
    }

    response = requests.get(url, params=params)
    print(response)
    data = response.json()

    data = data['features'][0]['properties']['country']
    return data

# Writing the list of dictionaries to a file in JSON format


# Replace 'YOUR_API_KEY' with your OpenWeatherMap API key
# Replace with desired latitude  # Replace with desired longitude

# API endpoint for historical weather data


# Parameters for the API request


# response = requests.get(base_url, params=params)
# print(response)
