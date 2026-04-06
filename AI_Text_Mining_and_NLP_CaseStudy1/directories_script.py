import os

# 1. Print current working directory
print("Current working directory:", os.getcwd())

# 2. Create a new directory on Desktop named "Text Mining and NLP"
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
new_dir = os.path.join(desktop_path, "Text Mining and NLP")

# Create directory if it doesn't exist
os.makedirs(new_dir, exist_ok=True)
print(f'Directory created (or already exists): {new_dir}')

# 3. Change current working directory to the new directory
os.chdir(new_dir)
print("Now current working directory is:", os.getcwd())