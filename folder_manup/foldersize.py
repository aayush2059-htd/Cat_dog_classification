import os
import hashlib
import shutil

# Function to calculate the hash of an image file (SHA-256)


# Function to get the size of all files in a directory
def get_folder_size(directory_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

directory_path = 'cats'

# Calculate folder size before removing duplicates
folder_size_before = get_folder_size(directory_path)
print(f"Folder size : {folder_size_before} bytes")