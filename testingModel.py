"""

import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def test_model_on_images(model_path, test_folder):
    model = load_model(model_path)
    image_paths = [os.path.join(test_folder, img) for img in os.listdir(test_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]
    for img_path in image_paths:
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        predicted_label = 'Fire' if prediction[0][0] < 0.5 else 'Non-Fire'
        confidence = 1 - prediction[0][0] if predicted_label == 'Fire' else prediction[0][0]
        print(f"Image: {img_path}, Prediction: {predicted_label}, Confidence: {confidence:.2f}")

# Example usage
if __name__ == "__main__":
    model_path = 'C:/Users/user/source/repos/FireDetection/VGG_finetuned_model.h5'
    test_folder = 'C:/Users/user/source/repos/FireDetection/testing fire detection/'
    test_model_on_images(model_path, test_folder)

"""
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pathlib import Path

def test_and_visualize_to_output(model_path, test_folder, output_folder, batch_size=12):
    """
    Tests the model, visualizes predictions in multiple figures, and saves outputs to a folder.
    Args:
        model_path (str): Path to the pre-trained model.
        test_folder (str): Path to the folder containing test images.
        output_folder (str): Path to the output folder to save results.
        batch_size (int): Number of images to process in each batch.
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Load the model
    model = load_model(model_path)

    # Collect all image paths from the test folder
    image_paths = [os.path.join(test_folder, img) for img in os.listdir(test_folder) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Initialize variables for storing predictions and results
    all_images = []
    all_titles = []

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            img = load_img(img_path, target_size=(224, 224))  # Resize image
            img_array = img_to_array(img) / 255.0  # Normalize image
            batch_images.append(img_array)

        # Convert list of images to a numpy array
        batch_images = np.array(batch_images)

        # Predict the batch
        predictions = model.predict(batch_images)

        for img_path, pred in zip(batch_paths, predictions):
            predicted_label = 'Fire' if pred[0] < 0.5 else 'Non-Fire'
            confidence = 1 - pred[0] if predicted_label == 'Fire' else pred[0]
            title = f"{predicted_label}\nConfidence: {confidence:.2f}"

            # Save predictions to a text file
            img_name = os.path.basename(img_path)
            with open(os.path.join(output_folder, "predictions.txt"), "a") as f:
                f.write(f"Image: {img_name}, Prediction: {predicted_label}, Confidence: {confidence:.2f}\n")

            # Append image and title for visualization
            all_images.append(load_img(img_path, target_size=(224, 224)))
            all_titles.append(title)

    # Split images into smaller groups for visualization
    cols = 5  # Number of columns per figure
    max_images_per_figure = 25 # Maximum number of images per figure
    for fig_idx in range(0, len(all_images), max_images_per_figure):
        current_images = all_images[fig_idx:fig_idx + max_images_per_figure]
        current_titles = all_titles[fig_idx:fig_idx + max_images_per_figure]

        # Determine rows dynamically for the current figure
        rows = (len(current_images) + cols - 1) // cols
        plt.figure(figsize=(20, rows * 4))  # Adjust figure size dynamically

        # Plot images in the current figure
        for i, img in enumerate(current_images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(current_titles[i], fontsize=10)

        plt.tight_layout()

        # Save the current figure
        fig_name = f"predictions_visualization_{fig_idx // max_images_per_figure + 1}.png"
        plt.savefig(os.path.join(output_folder, fig_name))
        plt.close()

if __name__ == "__main__":
    model_path = 'C:/Users/user/source/repos/FireDetection/VGG_finetuned_model.h5'
    test_folder = 'C:/Users/user/source/repos/FireDetection/Training/nofire'
    output_folder = 'C:/Users/user/source/repos/FireDetection/output_nofire/'
    #test_folder = 'C:/Users/user/source/repos/FireDetection/Training/fire'
    #output_folder = 'C:/Users/user/source/repos/FireDetection/output_fire_new/'
    # Run the function to process, save, and visualize predictions
    test_and_visualize_to_output(model_path, test_folder, output_folder, batch_size=12)

