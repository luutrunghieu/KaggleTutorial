import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melbourne_data = pd.read_csv("melb_data.csv")
filtered_melbourne_data = melbourne_data.dropna(axis=0)
melbourne_y = filtered_melbourne_data.Price
columns = ['Rooms','Bathroom','Landsize', 'BuildingArea', 'YearBuilt','Lattitude','Longtitude']
melbourne_X = filtered_melbourne_data[columns]
melb_train_X, melb_val_X, melb_train_y, melb_val_y = train_test_split(melbourne_X, melbourne_y, random_state = 0)
melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(melb_train_X,melb_train_y)
predicted_home_prices = melbourne_model.predict(melb_val_X)
print(mean_absolute_error(melb_val_y,predicted_home_prices))

iowa_data = pd.read_csv("train.csv")
