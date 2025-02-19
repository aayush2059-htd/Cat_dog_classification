import os

# Folder path where images are stored
directory_path = "cats/"

# Iterate through the range of file names
for i in range(12498):  # 0 to 21249
    file_path = os.path.join(directory_path, f"{i}.jpg")
    
    if os.path.exists(file_path):  # Check if file exists
        os.remove(file_path)  # Delete the file
        print(f"Deleted: {file_path}")

print("All files deleted successfully!")
