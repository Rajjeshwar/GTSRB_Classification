#!/usr/bin/env python
# coding: utf-8




import tensorflow as tf
from sklearn import model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    MaxPooling2D,
    Conv2D,
    Flatten,
    BatchNormalization,
    Input,
    Average,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model

import sklearn.metrics as metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import gradio as gr
import seaborn as sns
import pandas as pd
import numpy as np
from numpy import save
from numpy import load





def load_dataset():
    X_train = load("X_GTS_train_copy.npy")
    Y_train = load("Y_GTS_train_copy.npy")
    X_test = load("X_GTS_test_copy.npy")
    Y_test = load("Y_GTS_test_copy.npy")
    string_class_labels = load("Class_labels_copy.npy")
    Y_train = Y_train.astype(np.float)
    Y_test = Y_test.astype(np.float)

    X_train, X_val, Y_train, Y_val = model_selection.train_test_split(
        X_train, Y_train, test_size=0.3, random_state=42, shuffle=True
    )

    return X_train, Y_train, X_test, Y_test, string_class_labels, X_val, Y_val




X_train, Y_train, X_test, Y_test, string_class_labels, X_val, Y_val = load_dataset()


# ### Image augmentation parameters




datagen = ImageDataGenerator(
    shear_range=0.15,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=False,
    fill_mode="nearest",
)


# ### Model and training




input_shape = X_train.shape[1:]
model_input = Input(shape=input_shape)





def func_cnn(model_input):

    x = Conv2D(filters=16, kernel_size=(3, 3), activation="relu")(model_input)
    x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization(axis=-1)(x)

    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)

    x = Dropout(rate=0.5)(x)

    x = Dense(43, activation="softmax")(x)

    cnn_model = Model(model_input, x, name="conv_pool_cnn")

    return cnn_model





conv_pool_cnn_model = func_cnn(model_input)





def train(model, loss, optimizer, metrics, batch_size, epochs):
    model.compile(loss=loss, optimizer=optimizer, metrics=[metrics])
    history = model.fit(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(X_val, Y_val),
    )
    return history





history_cnn = train(
    conv_pool_cnn_model, "sparse_categorical_crossentropy", "adam", "accuracy", 32, 30
)


# ### Plotting the loss and the accuracy




plt.plot(history_cnn.history["accuracy"])
plt.plot(history_cnn.history["val_accuracy"])
plt.plot(history_cnn.history["loss"])
plt.legend(["accuracy", "val_accuracy", "loss"])





conv_pool_cnn_model.summary()





score_cnn = conv_pool_cnn_model.evaluate(X_test, Y_test)


# ### Calculating log loss for CNN model




pred_Cnn = conv_pool_cnn_model.predict(X_test)





log_loss_score = metrics.log_loss(Y_test, pred_Cnn)
print("Log loss score: {}".format(log_loss_score))


# ### Confusion matrix




classes_x = np.argmax(pred_Cnn, axis=1)





cf = confusion_matrix(Y_test, classes_x)





df_cm = pd.DataFrame(cf, index=string_class_labels, columns=string_class_labels)
plt.figure(figsize=(20, 20))
sns.heatmap(df_cm, annot=True)


# ### Classification report




print(classification_report(Y_test, classes_x))


# ## Using transfer learning on VGG19




def func_transfer(model_input):

    vgg19_model = tf.keras.applications.VGG19(
        include_top=False,
        weights="imagenet",
        input_tensor=model_input,
        pooling=max,
        classes=43,
        classifier_activation="softmax",
    )

    for layer in vgg19_model.layers:
        layer.trainable = True

    x = BatchNormalization()(vgg19_model.output)
    x = Flatten()(x)
    x = Dense(1024, activation="sigmoid")(x)

    x = Dense(43, activation="softmax")(x)

    vg_model = Model(inputs=model_input, outputs=x, name="vgg19_transfer")

    return vg_model





vgg19_transfer_model = func_transfer(model_input)





history_vg = train(
    vgg19_transfer_model, "sparse_categorical_crossentropy", "adam", "accuracy", 32, 30
)




vgg19_transfer_model.summary()





plt.plot(history_vg.history["accuracy"])
plt.plot(history_vg.history["val_accuracy"])
plt.plot(history_vg.history["loss"])
plt.legend(["accuracy", "val_accuracy", "loss"])





score_vg = vgg19_transfer_model.evaluate(X_test, Y_test)


# ### Log loss for vgg19 transfer architecture model




pred_vg = vgg19_transfer_model.predict(X_test)





log_loss_score_vg = metrics.log_loss(Y_test, pred_vg)
print("Log loss score: {}".format(log_loss_score_vg))


# ### Confusion matrix for vgg19 model




classes_x_vg = np.argmax(pred_vg, axis=1)
cf_vg = confusion_matrix(Y_test, classes_x_vg)

df_cm_vg = pd.DataFrame(cf_vg, index=string_class_labels, columns=string_class_labels)
plt.figure(figsize=(20, 20))
sns.heatmap(df_cm_vg, annot=True)


# ### Classification report




print(classification_report(Y_test, classes_x_vg))


# ### Average ensemble model




def ensemble_model(model_1, model_2):
    models_list = []
    models_list.append(model_1)
    models_list.append(model_2)

    model_output_list = [model(model_input) for model in models_list]
    ensemble_output = tf.keras.layers.Average()(model_output_list)
    ensemble_model = tf.keras.Model(inputs=model_input, outputs=ensemble_output)

    return ensemble_model





ensemble_model = ensemble_model(conv_pool_cnn_model, vgg19_transfer_model)





ensemble_model.summary()


# ### Training




history_ensemble = train(
    ensemble_model, "sparse_categorical_crossentropy", "adam", "accuracy", 32, 30
)


# ### Evaluate and predict on test set




score_ensemble = ensemble_model.evaluate(X_test, Y_test)





pred_ens = ensemble_model.predict(X_test)


# ### Log loss




log_loss_score_ens = metrics.log_loss(Y_test, pred_ens)
print("Log loss score: {}".format(log_loss_score_ens))


# ### Soft voting




### Here we predict the output based on the max probability for each class label obtained by combining the outputs of the CNN and the VG16 based model

softvoting = pred_vg + pred_Cnn
log_loss_softvoting = metrics.log_loss(Y_test, softvoting)
print("Log loss score: {}".format(log_loss_softvoting))



