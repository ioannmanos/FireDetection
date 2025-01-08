import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

# Ignore specific warning message
import warnings
warnings.filterwarnings("ignore", message="Your `PyDataset` class should call `super\(\)\.__init__\(\*\*kwargs\)` in its constructor\.")

# Custom Preprocessing Functions
def random_crop(image):
    return tf.image.random_crop(image, size=[224, 224, 3])

def cutout(image, mask_size=50):
    image = image.numpy()  # Convert to NumPy array
    height, width = image.shape[0], image.shape[1]
    center_x = np.random.randint(width)
    center_y = np.random.randint(height)
    mask_x1 = np.clip(center_x - mask_size // 2, 0, width)
    mask_y1 = np.clip(center_y - mask_size // 2, 0, height)
    mask_x2 = np.clip(center_x + mask_size // 2, 0, width)
    mask_y2 = np.clip(center_y + mask_size // 2, 0, height)
    image[mask_y1:mask_y2, mask_x1:mask_x2, :] = 0
    return image

def preprocess_image(image):
    image = tf.py_function(cutout, [image], [tf.float32])[0]
    image.set_shape([224, 224, 3])  # Ensure the shape is maintained after cutout
    image = random_crop(image)
    return image

# Define ImageDataGenerator for data preprocessing and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_image  # Apply custom preprocessing
)

test_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_dir = 'C:/Users/user/source/repos/FireDetection/FOREST_FIRE_DATASET/train'
val_dir = 'C:/Users/user/source/repos/FireDetection/FOREST_FIRE_DATASET/validation'
test_dir = 'C:/Users/user/source/repos/FireDetection/FOREST_FIRE_DATASET/test'

# Set up data generators for loading and preprocessing images
batch_size = 32
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # Set target size to 224x224
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),  # Set target size to 224x224
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),  # Set target size to 224x224
    batch_size=batch_size,
    class_mode='binary'
)

# Generate a batch of images from the train generator
augmented_images, augmented_labels = next(train_generator)

# Define the number of images to display
num_images_to_display = 10

# Plot augmented images
fig, axes = plt.subplots(1, num_images_to_display, figsize=(20, 20))

for i in range(num_images_to_display):
    ax = axes[i]
    ax.imshow(augmented_images[i])
    ax.axis('off')
    ax.set_title(f"Label: {int(augmented_labels[i])}")

plt.show()

# Load pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers to prevent them from being updated during training
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the pre-trained VGG16 model
x = Flatten()(base_model.output)
x = Dense(128, kernel_regularizer=l2(0.01))(x)  # Add L2 regularization
x = BatchNormalization()(x)  # Add batch normalization
x = tf.keras.layers.ReLU()(x)  # Use ReLU activation after batch normalization
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(1, activation='sigmoid')(x)

# Create a new model by specifying input and output layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with Adam optimizer and binary cross-entropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)

# Define the directory path for saving model weights
checkpoint_dir = "C:/Users/user/source/repos/FireDetection/FOREST_FIRE_DATASET/training_checkpoint_VGG/"
os.makedirs(checkpoint_dir, exist_ok=True)

# Define model checkpoint callback to save the best model based on validation loss
checkpoint_path = os.path.join(checkpoint_dir, "new_best_model.h5")
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', verbose=1)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20,  
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[early_stopping, checkpoint_callback, reduce_lr]
)

# Save the trained model
model.save('VGG_trained_model.h5')

# Unfreeze some layers in the base model for f ine-tuning
for layer in base_model.layers[-8:]:  # Unfreeze the last 8 layers
    layer.trainable = True

# Compile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuning
fine_tune_history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=10,   #Fine-tune for additional epochs
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[early_stopping, checkpoint_callback, reduce_lr]
)

# Save the fine-tuned model
model.save('VGG_finetuned_model.h5')

# Evaluate the fine-tuned model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
print("Test Loss after fine-tuning:", test_loss)
print("Test Accuracy after fine-tuning:", test_accuracy)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'] + fine_tune_history.history['accuracy'])
plt.plot(history.history['val_accuracy'] + fine_tune_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'] + fine_tune_history.history['loss'])
plt.plot(history.history['val_loss'] + fine_tune_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Access and print the labels of the test_generator
test_labels = test_generator.classes
print("Test Labels:")
print(test_labels)

# Predict using the test generator
test_generator.reset()
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Classification report
report = classification_report(test_labels, predicted_classes, target_names=['Non-fire', 'Fire'], output_dict=True)
print(classification_report(test_labels, predicted_classes, target_names=['Non-fire', 'Fire']))

# Confusion matrix
cm = confusion_matrix(test_labels, predicted_classes)
print("Confusion Matrix:")
print(cm)

# Visualize the classification report metrics
report_df = pd.DataFrame(report).transpose()

# Verify the structure of the DataFrame
print(report_df)

# Plot precision, recall, and f1-score
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Ensure the DataFrame contains the required metrics
required_metrics = ['precision', 'recall', 'f1-score']
assert all(metric in report_df.columns for metric in required_metrics),"DataFrame is missing required metrics."

sns.barplot(x=report_df.index[:-3], y=report_df.loc[:,'precision'][:-3], ax=axes[0])
axes[0].set_title('Precision')
axes[0].set_ylim(0, 1)
axes[0].set_xticklabels(report_df.index[:-3], rotation=45)

sns.barplot(x=report_df.index[:-3], y=report_df.loc[:,'recall'][:-3], ax=axes[1])
axes[1].set_title('Recall')
axes[1].set_ylim(0, 1)
axes[1].set_xticklabels(report_df.index[:-3], rotation=45)

sns.barplot(x=report_df.index[:-3], y=report_df.loc[:,'f1-score'][:-3], ax=axes[2])
axes[2].set_title('F1-score')
axes[2].set_ylim(0, 1)
axes[2].set_xticklabels(report_df.index[:-3], rotation=45)

plt.show()

# Normalize and plot confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=['Non-fire', 'Fire'], yticklabels=['Non-fire', 'Fire'])
plt.title('Normalized Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Plot ROC curve and AUC
fpr, tpr, _ = roc_curve(test_labels, predictions)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

