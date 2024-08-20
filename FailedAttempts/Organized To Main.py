# Import needed libraries
import os
import itertools
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, \
    BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.metrics import Recall
import tensorflow.keras.backend as K
import warnings

warnings.filterwarnings('ignore')

def loading_the_data(data_dir):
    filepaths = []
    labels = []
    folds = os.listdir(data_dir)

    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)

    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')

    df = pd.concat([Fseries, Lseries], axis=1)
    return df


def change_label_names(df, column_name):
    index = {'lung_aca': 'Lung_adenocarcinoma', 'lung_n': 'Lung_benign_tissue',
             'lung_scc': 'Lung squamous_cell_carcinoma'}
    df[column_name] = df[column_name].replace(index)


def custom_autopct(pct, data_balance):
    total = sum(data_balance)
    val = int(round(pct * total / 100.0))
    return "{:.1f}%\n({:d})".format(pct, val)


def plot_data_balance(df):
    data_balance = df.labels.value_counts()
    plt.pie(data_balance, labels=data_balance.index, autopct=lambda pct: custom_autopct(pct, data_balance),
            colors=["#2092E6", "#6D8CE6", "#20D0E6"])
    plt.title("Training data balance")
    plt.axis("equal")
    plt.show()


def create_data_generators(train_df, valid_df, test_df, img_size, batch_size):
    tr_gen = ImageDataGenerator(rescale=1. / 255)
    ts_gen = ImageDataGenerator(rescale=1. / 255)

    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=True,
                                           batch_size=batch_size)
    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                           class_mode='categorical', color_mode='rgb', shuffle=True,
                                           batch_size=batch_size)
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels', target_size=img_size,
                                          class_mode='categorical', color_mode='rgb', shuffle=False,
                                          batch_size=batch_size)

    return train_gen, valid_gen, test_gen


def plot_sample_images(train_gen, batch_size):
    g_dict = train_gen.class_indices
    classes = list(g_dict.keys())
    images, labels = next(train_gen)

    plt.figure(figsize=(20, 20))
    for i in range(batch_size):
        plt.subplot(6, 6, i + 1)
        image = images[i]
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='black', fontsize=16)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def model_performance(history, epochs):
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    Epochs = [i + 1 for i in range(len(tr_acc))]

    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


def model_evaluation(model, train_gen, valid_gen, test_gen):
    train_score = model.evaluate(train_gen, verbose=1)
    valid_score = model.evaluate(valid_gen, verbose=1)
    test_score = model.evaluate(test_gen, verbose=1)

    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])
    print("Train Recall: ", train_score[2])
    print("Train IOU: ", train_score[3])
    print('-' * 20)
    print("Validation Loss: ", valid_score[0])
    print("Validation Accuracy: ", valid_score[1])
    print("Validation Recall: ", valid_score[2])
    print("Validation IOU: ", valid_score[3])
    print('-' * 20)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])
    print("Test Recall: ", test_score[2])
    print("Test IOU: ", test_score[3])


def get_pred(model, test_gen):
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    return y_pred


def plot_confusion_matrix(test_gen, y_pred):
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())
    cm = confusion_matrix(test_gen.classes, y_pred)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


def conv_block(filters, act='relu'):
    block = Sequential()
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(BatchNormalization())
    block.add(MaxPooling2D())
    return block


def dense_block(units, dropout_rate, act='relu'):
    block = Sequential()
    block.add(Dense(units, activation=act))
    block.add(BatchNormalization())
    block.add(Dropout(dropout_rate))
    return block


def iou_metric(y_true, y_pred):
    y_true = K.cast(y_true, K.floatx())
    y_pred = K.cast(y_pred, K.floatx())
    y_pred_binary = K.round(y_pred)
    intersection = K.sum(y_true * y_pred_binary)
    union = K.sum(y_true) + K.sum(y_pred_binary) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())


def cnn_model_structure(img_shape, class_counts):
    cnn_model = Sequential()

    cnn_model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", input_shape=img_shape))
    cnn_model.add(BatchNormalization())
    cnn_model.add(MaxPooling2D())

    cnn_model.add(conv_block(32))
    cnn_model.add(conv_block(64))
    cnn_model.add(conv_block(128))
    cnn_model.add(conv_block(256))

    cnn_model.add(Flatten())

    cnn_model.add(dense_block(128, 0.5))
    cnn_model.add(dense_block(64, 0.3))
    cnn_model.add(dense_block(32, 0.2))

    cnn_model.add(Dense(class_counts, activation="softmax"))

    cnn_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy',
                      metrics=['accuracy', Recall(name='recall'), iou_metric])

    return cnn_model


def efficientnetb3_model_structure(img_shape, class_counts):
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=img_shape, pooling=None)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(class_counts, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output_layer)
    model.compile(optimizer='Adamax', loss='categorical_crossentropy', metrics=['accuracy', Recall(name='recall'),
                                                                                 iou_metric])
    return model


def plot_model_accuracy_loss(model_history):
    epochs_range = range(len(model_history.history['accuracy']))
    acc = model_history.history['accuracy']
    val_acc = model_history.history['val_accuracy']
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()


def main(model_type='cnn'):
    data_dir = 'lung_colon_image_set/lung_image_sets'
    df = loading_the_data(data_dir)
    change_label_names(df, 'labels')
    plot_data_balance(df)

    img_size = (224, 224)
    batch_size = 8

    # Split the data into training, validation, and testing sets
    train_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=0, stratify=df['labels'])
    valid_df, test_df = train_test_split(test_df, train_size=0.5, shuffle=True, random_state=0, stratify=test_df['labels'])

    train_gen, valid_gen, test_gen = create_data_generators(train_df, valid_df, test_df, img_size, batch_size)
    plot_sample_images(train_gen, batch_size)

    if model_type == 'cnn':
        model = cnn_model_structure((*img_size, 3), len(df['labels'].unique()))
    elif model_type == 'efficientnetb3':
        model = efficientnetb3_model_structure((*img_size, 3), len(df['labels'].unique()))
    else:
        raise ValueError("Invalid model_type. Choose either 'cnn' or 'efficientnetb3'.")

    history = model.fit(train_gen, validation_data=valid_gen, epochs=1, verbose=1)

    model_performance(history, 50)
    model_evaluation(model, train_gen, valid_gen, test_gen)

    y_pred = get_pred(model, test_gen)
    plot_confusion_matrix(test_gen, y_pred)


if __name__ == "__main__":
    main(model_type='efficientnetb3')  # You can switch between 'cnn' and 'efficientnetb3'
