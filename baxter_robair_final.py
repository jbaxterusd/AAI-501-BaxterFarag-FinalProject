import pandas as pd
import numpy as np
import tensorflow as tf
import pydicom
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def correct_file_name(row):
    corrected_file_path = row.replace('000000.dcm', '1-1.dcm')
    return corrected_file_path

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    dicom = pydicom.dcmread(image_path)
    img = dicom.pixel_array 
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, -1)
    img = tf.image.grayscale_to_rgb(img)
    img = tf.image.resize_with_pad(img, target_size[0], target_size[1])
    img = img / tf.reduce_max(img)
    return img

def extract_embeddings(model, images):
    intermediate_layer_model = Model(inputs=model.input, 
                                     outputs=model.get_layer(index=-3).output)
    embeddings = intermediate_layer_model.predict(images)
    return embeddings

metadatatrain = pd.read_csv('mass_case_description_train_set.csv')
metadatatest = pd.read_csv('mass_case_description_test_set.csv')

path1 = metadatatrain['pathology']
newpath1 = []

path2 = metadatatest['pathology']
newpath2 = [] 

for row in path1: 
    if row =='BENIGN':
        newpath1.append(0)
    if row == 'BENIGN_WITHOUT_CALLBACK':
        newpath1.append(0)
    if row == 'MALIGNANT':
        newpath1.append(1)

for row in path2: 
    if row =='BENIGN':
        newpath2.append(0)
    if row == 'BENIGN_WITHOUT_CALLBACK':
        newpath2.append(0)
    if row == 'MALIGNANT':
        newpath2.append(1)

metadatatrain['pathology'] = newpath1
metadatatest['pathology'] = newpath2

print(metadatatrain.head())

testtrain = metadatatrain.head(100)
testtest = metadatatest.head(100)

train_images = metadatatrain['image file path']
train_labels = metadatatrain['pathology']
test_images = metadatatest['image file path']
test_labels = metadatatest['pathology']

train_images = train_images.apply(correct_file_name)
test_images = test_images.apply(correct_file_name)

trainingimages = np.array([load_and_preprocess_image(img_path) for img_path in 
                           train_images])

labels = np.array(train_labels)

testingimages = np.array([load_and_preprocess_image(img_path).numpy() for
                          img_path in test_images])

test_labels = np.array(test_labels)

base_model = ResNet50(weights="imagenet", include_top=False, 
                      input_shape=(224, 224, 3))

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(trainingimages, labels, 
                    validation_data=(testingimages, test_labels), 
                    epochs=10, 
                    batch_size=32)

test_loss, test_accuracy = model.evaluate(testingimages, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

train_embeddings = extract_embeddings(model, trainingimages)
test_embeddings = extract_embeddings(model, testingimages)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_embeddings, labels)

rf_predictions = rf_classifier.predict(test_embeddings)

rf_accuracy = accuracy_score(test_labels, rf_predictions)

print(f"Random Forest Test Accuracy: {rf_accuracy * 100:.2f}%")






