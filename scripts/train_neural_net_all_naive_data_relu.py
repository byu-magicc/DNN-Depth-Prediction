# %%

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import os

directory = "/home/james/Documents/AirSim/supercomputer_recording/"

# feature_data_ds = tf.data.experimental.make_csv_dataset(
#     directory + "nn_data.csv",
#     batch_size=640
#     num_epochs=1)

# caching = feature_data_ds.cache().shuffle(1000)

data_filename = directory + "naive_nn_data.csv"

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
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(64, activation='relu'),
      layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_model = build_and_compile_model(normalizer)
#%%
checkpoint_path = "training_naive/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                               save_weights_only=True,
                                               verbose=1)

history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    epochs=200
)

tf.keras.models.save_model(dnn_model, "training_soph", overwrite=True, include_optimizer=True)

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
test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions, 0.1)
plt.xlabel('True Values [m]')
plt.ylabel('Predictions [m]')
lims = [0, 100]
plt.xlim(lims)
plt.ylim(lims)
error = 5
_ = plt.plot(lims, lims, "k")
_ = plt.plot(lims, [error, 100 + error], "b")
_ = plt.plot(lims, [-error, 100-error], "r")
# %%
errors = test_labels - test_predictions
count = 0
for e in errors:
  if np.abs(e) < error:
    count += 1
num_test_features = len(test_features)
print(str(count) + "/" + str(num_test_features) + " (" + str((count+0.0)/num_test_features*100) + "%) are within " + str(error) + " of their true value")
# %%
