import pandas as pd
import sys, getopt
from database import Model


def start(train_path, test_path, ideal_path):
    print("...Starting...")
    # Lets call our model class
    # It contains all the functions we are going to use
    model = Model()
    # Let's read the data provided to us and catch any errors in it
    try:
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        ideal_data = pd.read_csv(ideal_path)
    except:
        print("...An error occurred while trying to read files...")
        print("Please make sure the files passed are in a proper csv format")
    else:
        # No error is caught so we proceed to populate the database created with our test and ideal data
        model.populate("train_table", train_data)

        model.populate("ideal_table", ideal_data)
        # Using the given datasets we get the four ideal functions and store them in a list
        functions = model.get_ideal_functions(
            model.read_table("train_table"), model.read_table("ideal_table")
        )
        # We then pass our test data and our four ideal functions to be added to the database
        model.add_test_data(functions, test_data)

        print("...Completed...")
        print("...Reading Data...")
        # Let's plot our data
        print("...Plotting Data...")
        model.plot()


def verify_files(*args) -> bool:
    """Verify that files provided are csv

    Returns:
        bool: True or False
    """
    for arg in args:
        # check if the file has an extension .csv
        if arg.split(".")[-1] != "csv":
            return False
    return True


def main(argv):
    # Declare empty variables to hold our arguments
    test_set = ""
    train_set = ""
    ideal_functions = ""
    try:
        # Get the options passed through the terminal
        opts, args = getopt.getopt(argv, "hn:t:l:", ["train=", "test=", "ideal="])

    except getopt.GetoptError:
        print("main.py -n <train_set> -t <test_set> -l <ideal_functions>")
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("main.py -n <train_set> -t <test_set> -l <ideal_functions>")
            sys.exit()
        elif opt in ("-n", "--train"):
            train_set = arg
        elif opt in ("-t", "--test"):
            test_set = arg
        elif opt in ("-l", "--ideal"):
            ideal_functions = arg
    if not train_set or not test_set or not ideal_functions:
        print("All 3 datasets are required to proceed")

    else:
        if verify_files(train_set, test_set, ideal_functions):
            start(train_set, test_set, ideal_functions)
        else:
            print("Only csv files are acceptable")


if __name__ == "__main__":
    main(sys.argv[1:])
