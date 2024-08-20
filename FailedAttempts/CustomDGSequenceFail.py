# Import needed libraries
# import system libs
import os
import itertools

# import data handling tools
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle  # Import shuffle function

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img, load_img
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3

# ignore the warnings
import warnings
warnings.filterwarnings('ignore')

# import OpenCV for image processing
import cv2

# Function to add Gaussian noise to an image
def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 0.1
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 1)
    return noisy

# Function to remove noise using median filter
def remove_noise(image):
    denoised = cv2.medianBlur((image * 255).astype(np.uint8), 3)
    denoised = denoised.astype(np.float32) / 255.0
    return denoised


# Modify the image data generator to include noise addition and removal
class CustomDataGenerator(keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size, image_size, class_indices, shuffle=True):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.class_indices = class_indices
        self.shuffle = shuffle
        self.on_epoch_end()

        # Initialize lists to store processed images
        self.processed_images = [self.process_image(filename) for filename in self.image_filenames]

    def __len__(self):
        return int(np.floor(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, index):
        batch_filenames = self.image_filenames[index * self.batch_size:(index + 1) * self.batch_size]
        batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]

        # Find indices of batch_filenames in self.image_filenames
        indices = np.array([np.where(self.image_filenames == filename)[0][0] for filename in batch_filenames])

        images = np.array([self.processed_images[idx] for idx in indices])
        labels = np.array([self.class_indices[label] for label in batch_labels])
        labels = keras.utils.to_categorical(labels, num_classes=len(self.class_indices))

        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)
            self.processed_images = [self.process_image(filename) for filename in self.image_filenames]

    def process_image(self, filename):
        image = load_img(filename, target_size=self.image_size)
        image = img_to_array(image) / 255.0

        # Add noise and then remove it
        noisy_image = add_noise(image)
        denoised_image = remove_noise(noisy_image)

        return denoised_image


# Read the training dataset into the dataframe
# loading the dataset
def loading_the_data(data_dir):
    # Generate data paths with labels
    filepaths = []
    labels = []

    # Get folder names
    folds = os.listdir(data_dir)

    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)

            filepaths.append(fpath)
            labels.append(fold)

    # Concatenate data paths with labels into one DataFrame
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')

    df = pd.concat([Fseries, Lseries], axis=1)

    return df

# change label names to its original names
def change_label_names(df, column_name):
    index = {'lung_aca': 'Lung_adenocarcinoma', 'lung_n': 'Lung_benign_tissue',
             'lung_scc': 'Lung squamous_cell_carcinoma'}

    df[column_name] = df[column_name].replace(index)

# loading the data
data_dir = 'lung_colon_image_set/lung_image_sets'
df = loading_the_data(data_dir)
change_label_names(df, 'labels')
print(df)

# Data preprocessing
# first we will check if the training data is balanced or not
data_balance = df.labels.value_counts()

def custom_autopct(pct):
    total = sum(data_balance)
    val = int(round(pct * total / 100.0))
    return "{:.1f}%\n({:d})".format(pct, val)

# pie chart for data balance
plt.pie(data_balance, labels=data_balance.index, autopct=custom_autopct, colors=["#2092E6", "#6D8CE6", "#20D0E6"])
plt.title("Training data balance")
plt.axis("equal")
plt.show()

# It is balanced, now we will split our data to train, val and test set
# data --> 80% train data && 20% (test, val)
train_df, ts_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=42)

# test data --> 10% train data && 10% (test, val)
valid_df, test_df = train_test_split(ts_df, train_size=0.5, shuffle=True, random_state=42)

# Create image data generator
# in this step we will convert the whole data to numpy arrays
# cropped image size
batch_size = 32
img_size = (224, 224)


# Define class indices
class_indices = {label: idx for idx, label in enumerate(df['labels'].unique())}

# Custom data generators
train_gen = CustomDataGenerator(train_df['filepaths'].values, train_df['labels'].values, batch_size=batch_size, image_size=img_size, class_indices=class_indices)
valid_gen = CustomDataGenerator(valid_df['filepaths'].values, valid_df['labels'].values, batch_size=batch_size, image_size=img_size, class_indices=class_indices)
test_gen = CustomDataGenerator(test_df['filepaths'].values, test_df['labels'].values, batch_size=batch_size, image_size=img_size, class_indices=class_indices, shuffle=False)

# Display sample from train data
images, labels = train_gen.__getitem__(0)

# plotting the patch size samples
plt.figure(figsize=(20, 20))

for i in range(batch_size):
    plt.subplot(6, 6, i + 1)
    image = images[i]
    plt.imshow(image)
    index = np.argmax(labels[i])  # get image index
    class_name = list(class_indices.keys())[index]  # get class of image
    plt.title(class_name, color='black', fontsize=16)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Create needed functions
# Displaying the model performance
def model_performance(history, Epochs):
    # Define needed variables
    tr_acc = history.history['accuracy']
    tr_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    Epochs = [i + 1 for i in range(len(tr_acc))]

    # Plot training history
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

# Evaluate the model
def model_evaluation(model):
    train_score = model.evaluate(train_gen, verbose=1)
    valid_score = model.evaluate(valid_gen, verbose=1)
    test_score = model.evaluate(test_gen, verbose=1)

    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])
    print('-' * 20)
    print("Validation Loss: ", valid_score[0])
    print("Validation Accuracy: ", valid_score[1])
    print('-' * 20)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

# Get Predictions
def get_pred(model, test_gen):
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)

    return y_pred

# Confusion Matrix
def plot_confusion_matrix(test_gen, y_pred):
    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())

    # Display the confusion matrix
    cm = confusion_matrix(test_gen.labels, y_pred)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.show()

# Defining a convolutional NN block for a sequential CNN model
def conv_block(filters, act='relu'):
    block = Sequential()
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(Conv2D(filters, 3, activation=act, padding='same'))
    block.add(BatchNormalization())
    block.add(MaxPooling2D())

    return block


# Defining a dense NN block for a sequential CNN model
def dense_block(units, dropout_rate, act='relu'):
    block = Sequential()
    block.add(Dense(units, activation=act))
    block.add(BatchNormalization())
    block.add(Dropout(dropout_rate))

    return block


# Model Structure
# CNN model
# create Model structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

class_counts = len(list(train_gen.class_indices.keys()))  # to define number of classes in dense layer

# Model architecture
cnn_model = Sequential()

# first conv block
cnn_model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same", activation="relu", input_shape=img_shape))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D())

# second conv block
cnn_model.add(conv_block(32))

# third conv block
cnn_model.add(conv_block(64))

# fourth conv block
cnn_model.add(conv_block(128))

# fifth conv block
cnn_model.add(conv_block(256))

# flatten layer
cnn_model.add(Flatten())

# first dense block
cnn_model.add(dense_block(128, 0.5))

# second dense block
cnn_model.add(dense_block(64, 0.3))

# third dense block
cnn_model.add(dense_block(32, 0.2))

# output layer
cnn_model.add(Dense(class_counts, activation="softmax"))

cnn_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

cnn_model.summary()

# train the model
epochs = 1  # number of all epochs in training

history = cnn_model.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False)

# Display model performance
model_performance(history, epochs)

# Model evaluation
model_evaluation(cnn_model)

# get predictions
y_pred = get_pred(cnn_model, test_gen)

# plot the confusion matrix
plot_confusion_matrix(test_gen, y_pred)
'''
# EfficientNetB3
# get the pre-trained model (EfficientNetB3)
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=img_shape, pooling=None)

# fine-tune EfficientNetB3 (Adding some custom layers on top)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = dense_block(128, 0.5)(x)
x = dense_block(32, 0.2)(x)
predictions = Dense(class_counts, activation="softmax")(x)  # output layer with softmax activation

# the model
EfficientNetB3_model = Model(inputs=base_model.input, outputs=predictions)

EfficientNetB3_model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

EfficientNetB3_model.summary()

# train the model
EfficientNetB3_history = EfficientNetB3_model.fit(train_gen, epochs=epochs, verbose=1, validation_data=valid_gen, shuffle=False)

# Display model performance
model_performance(EfficientNetB3_history, epochs)

# Model evaluation
model_evaluation(EfficientNetB3_model)

# get predictions
y_pred = get_pred(EfficientNetB3_model, test_gen)

# plot the confusion matrix
plot_confusion_matrix(test_gen, y_pred)'''
