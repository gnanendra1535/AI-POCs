import os
import shutil

desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
folder_path = os.path.join(desktop_path, "Text Mining and NLP")

# 1. Create "Greetings.txt" and write text into it
greetings_file = os.path.join(desktop_path, "Greetings.txt")
with open(greetings_file, "w") as file:
    file.write("Welcome to Text Mining and Natural Language Processing")

print(f'File created: {greetings_file}')

# 2. Move file to "Text Mining and NLP" folder
moved_file_path = shutil.move(greetings_file, folder_path)
print(f'File moved to: {moved_file_path}')

# 3. Rename file to "Welcome.txt"
new_file_path = os.path.join(folder_path, "Welcome.txt")
os.rename(os.path.join(folder_path, "Greetings.txt"), new_file_path)
print(f'File renamed to: {new_file_path}')