# Setting Up `CEC_2025_dataset` Environment Variable

As a part of making testing more efficient and so that nothing is lost while trying to run test scripts on the directors laptops, it is strongly encouraged that you perform environment variable mapping.

This guide explains how to set up an environment variable named `CEC_2025_dataset` to point to the folder containing `CEC_2025`, so that you can use it in your code without hardcoding the full path. You are also encouraged to use environment variables for any files referenced that are required to be downloaded on directors machines.

---

## 1. **Windows Instructions**

### **Step 1: Set the Environment Variable**

1. Open **Start** and search for **"Environment Variables"**.
2. Click on **Edit the system environment variables**.
3. In the **System Properties** window, click on **Environment Variables...** at the bottom.
4. Under **User variables**, click **New...** to create a new environment variable.
5. In the **Variable name** field, enter `CEC_2025_dataset`.
6. In the **Variable value** field, enter the full path to the folder containing `CEC_2025` (e.g., `C:\Users\YourUsername\Documents\CEC_2025`).
7. Click **OK** to save the variable.
8. Click **OK** again to close the **Environment Variables** window.
9. To test if this worked try running the following command in Power Shell
```
$env:CEC_2025_dataset
```

This should ouput something like this:
```
C:\Users\YourUsername\Documents\CEC_2025
```

Note: Windows often requires a full reboot to apply environment variable changes system-wide, especially for Python and other applications to recognize them. As a result its best to reboot after running steps 1-8.

### **Step 2: Testing the Environment Variable in Python (not required)**

While the environment variable should now work, it'd help to test it. Below is some Python code which you can use to reference the `CEC_2025_dataset` folder:


```python
import os

# Get the path to the folder from the environment variable
dataset_folder = os.getenv('CEC_2025_dataset')

if dataset_folder:
    # Print or list files in the folder
    print(f"Dataset folder is located at: {dataset_folder}")
else:
    print("Please set the 'CEC_2025_dataset' environment variable.")
```

## 2. **macOS Instructions**

### **Step 1: Set the Environment Variable**

1. Open Terminal
2. Type the following command to open your `.bash_profile` (if using Bash) or `.zshrc ` (if using Zsh):
```
nano ~/.bash_profile  # For Bash users
```
Or if you are using Zsh:
```
nano ~/.zshrc  # For Zsh users
```
3. Add the following line at the end of the file, replacing the path with the actual location of your `CEC_2025` folder:
*Note: You will need to go down to the end of your file, make sure to change the file path to the correct one on your machine.
```
export CEC_2025_dataset="/Users/YourUsername/Documents/CEC_2025"
```
4. Save and exit the file by pressing Ctrl + X, then press Y to confirm, and hit Enter.

```
source ~/.bash_profile  # For Bash users
```
Or if you are using Zsh:
```
source ~/.zshrc  # For Zsh users
```
5. Test the Variable
You can test the variable by trying the following commad:
```
echo $CEC_2025_dataset
```
Which should output something like this:
```
C:\Users\YourUsername\Documents\CEC_2025
```

### **Step 2: Testing the Environment Variable in Python (not required)**
While the environment variable should now work, it'd help to test it. Below is some Python code which you can use to reference the `CEC_2025_dataset` folder:

```python
import os

# Get the path to the folder from the environment variable
dataset_folder = os.getenv('CEC_2025_dataset')

if dataset_folder:
    # Print or list files in the folder
    print(f"Dataset folder is located at: {dataset_folder}")
else:
    print("Please set the 'CEC_2025_dataset' environment variable.")

```
# Special Note

Some IDE configurations (like PyCharm, or Jupyter) require additional setup for environment variables because they don’t automatically inherit environment variables from the system shell. You may need to do some extra mapping for those IDE's.

