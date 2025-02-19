import os
import hashlib
import shutil

# Function to calculate the hash of an image file (SHA-256)
def get_image_hash(file_path):
    hash_sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):  # Read in chunks to handle large files
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

# Function to get the size of all files in a directory
def get_folder_size(directory_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
    return total_size

# Generate a file list from 0.jpg to 21249.jpg
file_list = [f'{i}.jpg' for i in range(39961)]  # This will create file names from 0.jpg to 21249.jpg

# Folder path
directory_path = 'cats/'

# Calculate folder size before removing duplicates
folder_size_before = get_folder_size(directory_path)
print(f"Folder size before removing duplicates: {folder_size_before} bytes")

# Dictionary to store file hashes and remove duplicates
file_hashes = {}
files_to_remove = []

# Identify duplicates based on file content (hash comparison)
for file in file_list:
    file_path = os.path.join(directory_path, file)
    if os.path.exists(file_path):  # Ensure the file exists
        file_hash = get_image_hash(file_path)

        if file_hash in file_hashes:
            # If hash is already in the dictionary, mark this file for removal
            files_to_remove.append(file_path)
        else:
            # If it's the first time we see this hash, store it
            file_hashes[file_hash] = file_path

# Remove duplicate images
for file_path in files_to_remove:
    if os.path.exists(file_path):
        os.remove(file_path)  # Remove the duplicate file
        print(f"Removed duplicate image: {file_path}")

# Calculate folder size after removing duplicates
folder_size_after = get_folder_size(directory_path)
print(f"Folder size after removing duplicates: {folder_size_after} bytes")

# Optional: Renaming files with unique names (if needed)
for index, (file_hash, file_path) in enumerate(file_hashes.items()):
    new_file_name = f"unique_{index}_{os.path.basename(file_path)}"
    new_file_path = os.path.join(directory_path, new_file_name)
    shutil.copy(file_path, new_file_path)  # Copy to new unique name
    print(f"Renamed {os.path.basename(file_path)} to {new_file_name}")
