# %%

import numpy as np
import tensorflow as tf
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt

directory = "/home/james/Documents/AirSim/supercomputer_recording/"

# feature_data_ds = tf.data.experimental.make_csv_dataset(
#     directory + "nn_data.csv",
#     batch_size=640
#     num_epochs=1)

# caching = feature_data_ds.cache().shuffle(1000)

data_filename = directory + "nn_data_part1.csv"

column_names = ["depth","PXx", "PXy","Velx", "Vely", "Velz", "Wx", "Wy", "Wz","F1x", "F1y", "F1vx", "F1vy", "F2x", "F2y", "F2vx", "F2vy", "F3x", "F3y", "F3vx", "F3vy", "F4x", "F4y", "F4vx", "F4vy"]

raw_dataset = pd.read_csv(data_filename, header=0, sep=",")

dataset = raw_dataset.copy()

def cap_to_hundred(input):
    return np.clip(input, 0, 100)

dataset["depth"] = dataset["depth"].apply(cap_to_hundred)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop("depth")
test_labels = test_features.pop("depth")

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(np.array(train_features))

def build_and_compile_model(norm):
  model = tf.keras.Sequential([
      norm,
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
#%%

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=200
)

#%%
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
#   plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error')
  plt.legend()
  plt.grid(True)
plot_loss(history)
# %%
