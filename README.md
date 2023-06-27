# Task
This is a python script that takes in 3 distict datasets<br>

- Training dataset (400,5)
- Testing dataset (100,2)
- Ideal functions dataset (400,51)

It then finds 4 ideal functions that are the best fit for the training data<br> 
by using least square method(linear regression) to pick which ones have the<br>
least deviation (mean-squared-error)<br>
The four ideal datasets are then checked against our dataset <br>
and mapped if the deviation produced does not exceed the deviation<br>
found between the ideal functions chosen and the training set
<br>

## Build
Having python and pip in your system is a requirement to run this script<br>
To use the script simply: 
 - <strong>Clone this repository using command below or download the zip file from above</strong><br>
    git clone https://github.com/Nushynells/develop.git
 - <strong>Navigate to the cloned folder</strong><br>
    cd task/ <br>
 - <strong>Install the required python files</strong><br>
    I would recommend creating a virtual environment to avoid interfereing with system packages. <br>
    Learn how <a href="https://docs.python.org/3/library/venv.html#creating-virtual-environments">here</a>...<br>
    Activate it and install the requirements file using the command below<br>
    pip install -r requirements.txt
    
## Use
 <strong>The script needs 3 csv files with data as specified above. </strong>
 <strong>If that condition is met, simply run the main.py file while passing arguments as specified below. </strong>
  - -n or --train for the training file
  - -t or --test for the testing file
  - -l or --ideal for the ideal functions file
  
  ## Example
  - python -m main.py -n path/to/train.csv -t path/to/test.csv -l path/to/ideal.csv
  <br>or <br>
  - python -m main.py --train path/to/train.csv --test path/to/test.csv --ideal path/to/ideal.csv

  Beware that the program will open a bunch of plots on your browser
 
