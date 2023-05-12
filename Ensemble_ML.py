import tensorflow as tf
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy import genfromtxt

# Import the training datasets
feature_file = 'Data/Data_for_ML/training_data/feature'
label_file = 'Data/Data_for_ML/training_data/label_sub12_dndz'

X = genfromtxt(feature_file)
y = genfromtxt(label_file)

# Half sample test
# X = X[1::2]
# y = y[1::2]

# Normalize the data to reduce the dynamical range.
# This uses a minmaxscalar where a minimum and maximum are specified.
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X)
X = scaler_feat.transform(X)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y)
y = scaler_label.transform(y)

# Define the base models
level0 = list()
level0.append(tf.keras.models.load_model('Models/Ensemble_model_1', compile=False))
level0.append(tf.keras.models.load_model('Models/Ensemble_model_2', compile=False))
level0.append(tf.keras.models.load_model('Models/Ensemble_model_3', compile=False))
level0.append(tf.keras.models.load_model('Models/Ensemble_model_4', compile=False))
level0.append(tf.keras.models.load_model('Models/Ensemble_model_5', compile=False))

# Define the meta learning model
level1 = LinearRegression()

# Define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1)

# Fit the model on all available data
model.fit(X, y)
