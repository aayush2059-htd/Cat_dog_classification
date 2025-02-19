import os

folder = "cats/"
files = sorted(os.listdir(folder))  # Sort to maintain order
start_index = 0

for i, filename in enumerate(files):
    if filename.endswith(".jpg"):  # Change extension if needed
        new_name = f"{start_index + i}.jpg"
        os.rename(os.path.join(folder, filename), os.path.join(folder, new_name))

print("Renaming completed!")
