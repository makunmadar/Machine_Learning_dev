"""Testing the networks custom MAE loss function """
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import numpy as np


def masked_mae(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)
    masked_error = tf.abs(y_true - y_pred) * mask
    masked_mae = tf.reduce_sum(masked_error) / tf.reduce_sum(mask)
    return masked_mae


y_true = np.array([4.12, 3.98, 3.86, 3.78, 3.74, 3.73, 3.72, 3.71, 3.69, 3.67, 3.67, 3.65, 3.61, 0., -5.53,
                  -4.17, -2.58, -1.69, -1.4, -1.21, -0.98, -0.81])

y_pred = np.array([4.1208053, 3.977257, 3.8577914, 3.789491, 3.729642, 3.7177243, 3.7058907, 3.6989448,
                  3.6786685, 3.659298, 3.6334503, 3.6065345, 3.5603452, 0.02877389, -5.139916, -4.018825,
                  -2.648102, -1.6970917, -1.3797486, -1.1996487, -0.98092175, -0.801238])

y_predk = y_pred[y_true != 0]
y_truek = y_true[y_true != 0]

print("Custom loss: ", masked_mae(y_true, y_pred))
print("sklearn loss: ", mean_absolute_error(y_truek, y_predk))
