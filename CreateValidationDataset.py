import os
from sklearn.model_selection import train_test_split
import shutil

# Define constants for file paths
BASE_DIR = 'C:/Users/user/source/repos/FireDetection/'
DATASET_DIR = os.path.join(BASE_DIR, 'FOREST_FIRE_DATASET')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
VAL_DIR = os.path.join(DATASET_DIR, 'validation')

# Create directories if they don't exist
os.makedirs(VAL_DIR, exist_ok=True)

# Create a validation folder
subfolders = ['fire', 'non fire']

# Create corresponding subfolders in the validation data folder
for subfolder in subfolders:
    os.makedirs(os.path.join(VAL_DIR, subfolder), exist_ok=True)

# Copy validation samples from train to validation folder
for subfolder in subfolders:
    subfolder_path = os.path.join(TRAIN_DIR, subfolder)
    validation_subfolder_path = os.path.join(VAL_DIR, subfolder)
    # List all files in the subfolder
    file_names = os.listdir(subfolder_path)
    # Split file names into train and validation sets
    _, validation_files = train_test_split(file_names, test_size=0.2, random_state=42)
    # Copy validation files to the corresponding subfolder in the validation data folder
    for file_name in validation_files:
        source = os.path.join(subfolder_path, file_name)
        destination = os.path.join(validation_subfolder_path, file_name)
        shutil.copy(source, destination)

print("Validation data has been created successfully.")
