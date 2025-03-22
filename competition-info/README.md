# Introduction
Welcome to the CEC 2025 Programming Competition! Everything you need is in this GitHub, and we recommend the following steps to get started:

1. Read the rest of this README.md
2. Read the tutorial linked in `githubHowToUseGithubDesktop.txt`
3. Read the tutorial linked in `githubHowToFork.txt`
4. Fork this repo.
5. Clone your forked repo locally on your machine, or use your alternative repo.
6. Add us as Collaborators.
7. Start coding!

# Getting Help
Please follow the steps below if you need help:

1. Check the included documentation for logistics and scoring questions
2. Check the Programming Competition Case for competition questions
3. Check the Discord for previously answered questions
4. If you have done all of that, ask the Directors for help. If we can answer your question, it will be posted on the Discord in English.

# Important Notes
In your README.md please specify:

- How to run your code
- What language and version your code uses (e.g., Python 3.11)
- A list of required packages (e.g., Pandas, NumPy), with versions if needed (e.g., pytorch==2.1.116). A `requirements.txt` would also suffice.
- Model files (with links if not located on git branch) that need to be downloaded to run your code. See `\Testing Information\specific_model_file_download.md` for more information on this.
- If needed, what OS your code should be run on. Any specifications of this sort not included in your README cannot be assumed to be on the Directors’ machine(s).

### Model File Download Instructions:

If your code requires specific files for model data (e.g., model weights stored in HDF5, Pickle, JSON format, etc.), please specify these dependencies in the README file. Uploading large files directly to GitHub may cause delays. If this is the case, you can use a public OneDrive, Google Drive link, or another service to provide access. Teams will face penalties if files are not uploaded on time (file modification dates will be checked for OneDrive and Google Drive). 

Teams will face penalties ranging from  -5pts to -20pts for late model file submissions. The reason for late submission must be stated, and confirmed by directors. In severe cases, competition directors will not use this data (file modification dates will be checked on OneDrive and Google Drive). If your file is more than 1hr late (in upload time), Directors will opt to not use the model data, this will not result in a penalty. This does not apply to code, and presentation submissions. Directors will determine what is classified as a ‘model file’ upon reviewing competitor submissions. 

Alternative download methods are allowed, as long as the files are uploaded within the challenge’s time constraints and the upload time is visible.


# Info Files
The following files provide all the information regarding the competition. This includes:
- The Case Document
- Presentation
- Example Output
- Testing information
- GitHub Info

# Accessing Dataset
We have given you access to the images through the OneDrive link here:

https://dalu-my.sharepoint.com/:u:/g/personal/or942416_dal_ca/EUw8w3GiFVlIkPkwHZGgUhoBnqsPx9cRXZrVvGMeSh_VjA?e=Iuqni0 

You'll notice that the dataset is to be saved as a 7z file. You will need to install either 7-zip, Keka, or p7zip to extract the dataset.

###  Windows
7-zip can be downloaded for Windows using the following link:
https://www.7-zip.org

### macOS
Keka can be downloaded for MacOS using the following link:
https://www.keka.io/en/

### Ubuntu
On Ubuntu, please install p7zip (or any other 7z supported software) by running the following commands in your terminal:
```
sudo apt update
sudo apt install p7zip-full
```


When extracting the file, make sure to use 7-zip, p7zip, or Keka.

### PASSWORD

The password for the 7z file is:

```
gT5&dK9zR2wQ!aP0eY3B#6vL1zXhF8j
```

### Directory structure

The folder you have downloaded will contain three subfolders, a 'yes' folder, 'no' folder and 'CEC_test' folder. The 'yes' and 'no' folders contain images corresponding to healthy and unhealthy MRI scans, while the 'CEC_test' folder will be used by the directors for testing purposes.
```
/CEC_2025
│
├── /yes/
│   ├── yes__1.png
│   ├── yes__2.png
│   └── ...
│
├── /no/
│   ├── no__1.png
│   ├── no__2.png
│   └── ...
│
└── /CEC_test/
    ├── test__1.png
    ├── test__2.png
    └── ...
```

# Potentially Useful Info:

## Retrieve Dataset
Below is an example code (in Python, however it's not mandatory for you to use Python) which accesseses this folder and then runs through using the root directory. 

Note: This references the environment variable CEC_2025_dataset which we encourage you to setup on your local computers. In this repository you can check `\Testing Information\setting_up_environment_variable.md` for more information on using environment variables. 

```ruby
import os
from os import path

dataset_dir = os.getenv('CEC_2025_dataset')  # Retrieves the path from the environment variable

# Initialize lists to hold data
image_paths = []
targets = []

total_images = 0

# Map for target labels
label_map = {'no': 0, 'yes': 1, 'CEC_test': 2} 

# Loop through subdirectories in the dataset directory
for subdir in os.listdir(dataset_dir):
    subdir_path = path.join(dataset_dir, subdir)

    if os.path.isdir(subdir_path):
        subdir_path_list = os.listdir(subdir_path)
        for image in subdir_path_list:
            image_paths.append(path.join(subdir_path, image))
            targets.append(label_map[subdir])

        total_images += len(subdir_path_list)
        print(f"Number of images in '{subdir}' directory: {len(subdir_path_list)}")

print(f'Total number of images: {total_images}')
```
This should ouput something like this:
```
Number of images in 'no' directory: 8205
Number of images in 'CEC_test' directory: 1
Number of images in 'yes' directory: 9310
Total number of images: 17516
```
## Checking File Paths
You may also want to check the file paths of your images:
```ruby
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'image_path': image_paths,
    'target': targets
}, index=np.arange(0, total_images))

print(df.head())
```
Which should output something like this:
```
                                          image_path  target
0  /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
1  /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
2  /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
3  /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
4  /Users/orionwiersma/Documents/CEC_2025/no/no_...       0
```
