# Testing
Your code will be tested by comparing the results of your algorithm to the correct results, with each team getting a percentage value for how accurate there code was. A folder labeled 'CEC_test' will have a set amount of test images (that only the directors have access to) corresponding to 'yes' or 'no'. The file format for these images will be PNG, with the names of each image being 'test_xxx.png'. Note: "x" in this case represents the number of files tested, similar to the yes and no directories. Make sure you have a testing script referencing this folder located at `/CEC_2025/CEC_test`. This can be done very simply by employing environment variables that you can expect the Directors will have on their machines. See `/Testing Information/setting_up_environment_variable.md` for more info.  

Results will be compared with the correct results in Excel. It is necessary that competitors output their results to a CSV, XLSX, or Google Sheet file (or similar). While other UIs are encouraged for aesthetic purposes, a .csv file or anything similar for outputting raw results is necessary for testing purposes! You can see the sample output in the info folder for more details relating to this! 

### Testing Output
See CEC_test_output_ex.csv file for an example as to how we would prefer raw data to be outputted.

See mock_testing_file.csv to see how your code will be tested. Here we will compare your test output with the correct results.

### Accuracy -> Penalties/Bonus!
Teams who are able to construct an algorythm with >=95% accuracy will receive bonus points, while those unable to meet benchmark threshold with <65% accuracy will be subject to point deductions.

### All you need to do is...
As long as your code outputs the test image being tested (1st column) and the result, which is either 'yes' or 'no' (2nd column), your code will be ready for testing. You are required to make the testing script.
