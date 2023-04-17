# This code will test if the normalisation and inverse transform returns to the original value
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Import the test data
feature_file = 'Data/Data_for_ML/testing_data/feature'
label_file = 'Data/Data_for_ML/testing_data/label_sub6'

X_test = genfromtxt(feature_file)
y_test = genfromtxt(label_file)

# Load scalar fits
scaler_feat = MinMaxScaler(feature_range=(0, 1))
scaler_feat.fit(X_test)
X_test_norm = scaler_feat.transform(X_test)
# Use standard scalar for the label data
scaler_label = StandardScaler()
scaler_label.fit(y_test)
y_test_norm = scaler_label.transform(y_test)

# De-normalize the feature and truth data
X_test_return = scaler_feat.inverse_transform(X_test_norm)
y_test_return = scaler_label.inverse_transform(y_test_norm)

print('Do the first set of parameters match?','\n', X_test[1] == X_test_return[1])
#print(X_test == X_test_return)

print('Original feature: ', '\n', X_test[1])
print('Normalised and de-normalised data: ', '\n', X_test_return[1])