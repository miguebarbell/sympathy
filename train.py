# import everything
from models import build_model_autotune
from tensorflow.keras.callbacks import EarlyStopping
import helper
from imutils import paths
import config
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
from datetime import datetime
import os
import keras_tuner as kt

from tensorflow.keras.utils import to_categorical

patch_sklearn()

#data loading
# list all images
image_paths = list(paths.list_images(config.DATASET))

# init preprocessors
image_pp = helper.ImagePP(28, 28, gray=True)
image2array_pp = helper.Image2ArrayPP()

# load the dataset, show loading images every 500 and update it
print("[INFO] loading images...")
data_loader = helper.DataLoader(preprocessors=[image_pp, image2array_pp])
(data, labels) = data_loader.load(image_paths, verbose=500)

# normalize the pixels
data = data.astype("float") / 255.0

#check the data
# imbalance = helper.check_imbalanced_classes(labels)
labels = LabelBinarizer().fit_transform(labels)
labels = to_categorical(labels, len(np.unique(labels)))
labels_quantity = dict()
for label in np.unique(labels):
    labels_quantity[label] = np.sum(labels == label)

max_labels_quantity = max(labels_quantity.values())
class_weight = dict()
for key in labels_quantity:
    class_weight[key] = max_labels_quantity / labels_quantity[key]

# train the data
# split data with stratify, making sure that have the same ratio of labels in the test and train data
(train_X, test_X, train_y, test_y) = train_test_split(data, labels, test_size=0.2, stratify=labels)
# import the model with the parameters in config.py

es = EarlyStopping(
    monitor="val_accuracy",
    # monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

es2 = EarlyStopping(
    monitor="val_accuracy",
    # monitor="val_loss",
    patience=np.round((config.EARLY_STOPPING_PATIENCE*1.5), 0).astype(int),
    restore_best_weights=True)
# check if we will be using the hyperband tuner
if config.TUNER == "hyperband":
    # instantiate the hyperband tuner object
    print("[INFO] instantiating a hyperband tuner object...")
    tuner = kt.Hyperband(
        build_model_autotune,
        objective="val_accuracy",
        max_epochs=config.EPOCHS,
        factor=3,
        seed=42,
        directory=config.OUTPUT_PATH,
        project_name=config.TUNER)

# check if we will be using the random search tuner
elif config.TUNER == "random":
    # instantiate the random search tuner object
    print("[INFO] instantiating a random search tuner object...")
    tuner = kt.RandomSearch(
        build_model_autotune,
        objective="val_accuracy",
        # max_trials=3,
        max_trials=10,
        seed=42,
        directory=config.OUTPUT_PATH,
        project_name=config.TUNER)

# otherwise, we will be using the bayesian optimization tuner
else:
    # instantiate the bayesian optimization tuner object
    print("[INFO] instantiating a bayesian optimization tuner object...")
    tuner = kt.BayesianOptimization(
        build_model_autotune,
        objective="val_accuracy",
        max_trials=10,
        # max_trials=3,
        seed=42,
        directory=config.OUTPUT_PATH,
        project_name=config.TUNER)

# perform the hyperparameter search
print("[INFO] performing hyperparameter search...")
tuner.search(
    x=train_X, y=train_y,
    validation_data=(test_X, test_y),
    batch_size=config.BS,
    callbacks=[es],
    epochs=config.EPOCHS
)

# grab the best hyperparameters
bestHP = tuner.get_best_hyperparameters(num_trials=1)[0]
print("[INFO] optimal number of filters in conv_1 layer: {}".format(
    bestHP.get("conv_1")))
print("[INFO] optimal number of filters in conv_2 layer: {}".format(
    bestHP.get("conv_2")))
print("[INFO] optimal number of units in dense layer: {}".format(
    bestHP.get("dense_units")))
print("[INFO] optimal learning rate: {:.4f}".format(
    bestHP.get("learning_rate")))

# build the best model and train it
print("[INFO] training the best model...")
model = tuner.hypermodel.build(bestHP)
H = model.fit(x=train_X, y=train_y,
              validation_data=(test_X, test_y), batch_size=config.BS,
              epochs=config.EPOCHS*2, callbacks=[es2], verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=test_X, batch_size=32)
label_names = os.listdir(config.DATASET)
print(classification_report(test_y.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=label_names))
"""
model = build_model()
model.compile(loss=config.LOSS, optimizer=config.OPTIMIZER, metrics=config.METRICS)

print("[INFO] training the model")
H = model.fit(x=train_X, y=train_y, validation_data=(test_X, test_y), batch_size=config.BS, epochs=config.EPOCHS,
              callbacks=[es], verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=test_X, batch_size=config.BS)
label_names = os.listdir(config.DATASET)
print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))
"""
# saving the model and the plot
if config.MODEL_SAVE:
    time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
    # todo: save the metrics to the same folder
    model.save(f"{config.OUTPUT_PATH}/best_at{time}.hdf5")
    helper.save_plot(H, f"{config.OUTPUT_PATH}/best_at{time}plot.png")
