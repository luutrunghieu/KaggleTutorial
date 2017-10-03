import pandas as pd
melbourne_data = pd.read_csv("melb_data.csv")
filtered_melbourne_data = melbourne_data.dropna(axis=0)
melbourne_y = filtered_melbourne_data.Price
melbourne_X = filtered_melbourne_data['Rooms','Bathroom','Landsize', 'BuildingArea', 'YearBuilt','Lattitude','Longtitude']

iowa_data = pd.read_csv("train.csv")
