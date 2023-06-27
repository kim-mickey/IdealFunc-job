from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from bokeh.plotting import Figure
from bokeh.io import show, output_file
from bokeh.layouts import gridplot


class Database:
    """Database Class with actions such as creation, population and reading of tables"""

    def __init__(self) -> None:
        # we create an sqlite database called database.sqlite
        self.engine = create_engine("sqlite+pysqlite:///database.sqlite")
        self.models = []

    def populate(self, name: str, df: pd.DataFrame):
        """Populate Database with a table ,
        by passing a table name and a DataFrame as aguments.
        If the table exists it will we replaced.

        Args:
            name (str): Name to save the table under
            df (pd.DataFrame):  Dataframe used to populate database
        """
        print(f"...Added {name} to database")
        with self.engine.execution_options(autocommit=True).connect() as conn:
            df.to_sql(
                name=name,
                con=conn,
                index=False,
                if_exists="replace",
            )

    def read_table(self, name: str) -> pd.DataFrame:
        """Read table data into a dataframe

        Args:
            name (str): The name of the table to read from
        """
        # use the with keyword to ensure the connection closses with the function
        with self.engine.execution_options(autocommit=True).connect() as conn:
            # select the table with the name passed and return it as a DataFrame
            result = conn.execute(text(f"select * from '{name}'"))
            df = pd.DataFrame(result.fetchall())
            return df


class Model(Database):
    """Model class to hold all our data

    Args:
        Database (Database): Parent class
    """

    def __init__(self) -> None:
        super().__init__()
        self.deviations = []
        self.functions = []
        self.predictions = []

    def get_ideal_functions(
        self, train_data: pd.DataFrame, ideal_data: pd.DataFrame
    ) -> list:
        """Get Ideal functions

        Args:
            train_data (pd.DataFrame): The training dataset
            ideal_data (pd.DataFrame): The ideal functions dataset

        Returns:
            list: columns with the best fit
        """
        X = train_data[["x"]].copy()  # Lets copy over dataframe with our X values

        for col in train_data.columns[1:]:
            # since the column names differ, we need to rename them to y
            # failure to do so raises an error since we already fitted our model with y
            y = train_data.rename(columns={col: "y"}).copy()["y"]

            # We will use linear regression since we are looking for the line of best fit
            # we instantiate and fit the model
            model = LinearRegression()
            model.fit(X, y)
            # we then pass the model and our ideal data to the function find_function
            # we then append the result to our functions list
            best_fit = self.find_function(model, ideal_data)
            self.models.append(model)
            self.functions.append(best_fit)

        return self.functions

    def find_function(self, model: LinearRegression, ideal_data: pd.DataFrame) -> str:
        """Find an ideal function for the model

        Args:
            model (LinearRegression): Our fitted model
            ideal_data (pd.DataFrame): The ideal functions dataset

        Returns:
            str: ideal function column name
        """
        # Create a list to hold our scores
        scores = []
        # Get X
        X = ideal_data[["x"]].copy()
        pred = model.predict(X)
        self.predictions.append(pred)
        for col in ideal_data.columns[1:]:
            # we are going to score every ideal function and add them to a list
            # we are going to use mean_squared error as it gets the deviation squared
            y = ideal_data.rename(columns={col: "y"}).copy()["y"]
            scores.append(mean_squared_error(y, pred))
        # we then return the column with the lowest score since it has the least errors
        column = ideal_data.columns[scores.index(min(scores))]
        # Lets append our deviations to a class variable to be accessed later
        self.deviations.append(min(scores))
        return column

    def add_test_data(self, functions: list, test_data: pd.DataFrame):
        """Add Test data to the database

        Args:
            functions (list): Ideal functions selected
            test_data (pd.Dataframe): The test dataset
        """
        # Create a dataframe to hold our columns
        df = pd.DataFrame(columns=["x", "y", "y_delta", "y_ideal"], dtype="float64")
        # Add the columns to a database table
        self.populate("test_table", df)
        print("...Adding test_table data, please wait...")
        ideals = self.read_table("ideal_table").set_index("x")[functions]
        # Pass the individual values from each row into the
        # mapping function to calculate the ideal function and y deviation
        # and passes all of that into the database table we've just created
        test_data.apply(
            lambda num: self.mapping(num, ideals.loc[ideals.index == num["x"]]), axis=1
        )

    def mapping(self, num, functions):
        """Map individual test case to the four ideal functions and add them to the database

        Args:
        num (list): List of 'x' and 'y' value from a dataframe row
        """
        deviations = []
        for col in functions.columns:
            dev = mean_squared_error([num["y"]], functions[col].values)
            deviations.append(dev)
        min_deviation = min(deviations)
        mapped_function = functions.iloc[:, deviations.index(min_deviation)]

        with self.engine.execution_options(autocommit=True).connect() as conn:

            if (
                min_deviation - self.deviations[deviations.index(min_deviation)]
                > 2**0.5
            ):
                conn.execute(
                    text(
                        "INSERT INTO test_table (x, y, y_delta , y_ideal) VALUES (:x,:y,:y_delta ,:y_ideal)"
                    ),
                    [
                        {
                            "x": num["x"],
                            "y": num["y"],
                            "y_delta": min_deviation,
                            "y_ideal": mapped_function.values[0],
                        }
                    ],
                )
            else:
                conn.execute(
                    text(
                        "INSERT INTO test_table (x, y, y_delta , y_ideal) VALUES (:x,:y,:y_delta ,:y_ideal)"
                    ),
                    [
                        {
                            "x": num["x"],
                            "y": num["y"],
                            "y_delta": 0,
                            "y_ideal": num["y"],
                        }
                    ],
                )

    def plot(self):
        output_file('index.html')
        train_df = self.read_table("train_table")
        test_df = self.read_table("test_table")
        ideal_df = self.read_table("ideal_table")

        # create a new plot for showing our training data and the ideal functions chosen
        s1 = self.plot_training_ideal(
            train_df.set_index("x"), ideal_df.set_index("x")[self.functions]
        )

        # create another one for the test data
        s2 = self.plot_test_data()
        
        # put all the plots in a gridplot and pass it to the show function to display
        p = gridplot(
            s1+s2,
            merge_tools=False,
            sizing_mode="scale_both",
        )

        # show the results
        show(p)

    def plot_training_ideal(
        self, train_df: pd.DataFrame, ideal_df: pd.DataFrame
    ) -> list:
        """Plot training data with chosen ideal functions

        Args:
            train_df (pd.DataFrame): dataframe with our training data
            ideal_df (pd.DataFrame): dataframe with the chosen ideal functions

        Returns:
            list: plots of values
        """
        # create a list to hold our plots
        plots = [[], []]
        for col in train_df.columns:

            index = train_df.columns.get_loc(col)
            ideal = ideal_df
            # lets create our bokeh figure to hold the plots
            fig = Figure(
                title=f"Training data {col} and chosen ideal function {self.functions[index]}",
                margin=1,
                x_axis_label="X",
                y_axis_label="Y",
                tools="pan,hover,box_zoom,wheel_zoom,reset,save",
            )

            # plot the training data y column
            fig.circle(
                x="x",
                y=col,
                size=2,
                legend_label=f"{col} - training",
                color="blue",
                source=train_df,
            )
            # plot the predicted values
            fig.square(
                x="x",
                y=self.functions[index],
                legend_label=f"{self.functions[index]} - ideal",
                size=1,
                color="green",
                source=ideal,
            )
            # lets retrieve the model belonging to this column and use it to show the predicted values

            fig.line(
                x=train_df.index,
                y=self.predictions[index],
                legend_label="predicted",
                color="red",
            )
            # plot the chosen ideal function
            # set plot legend details
            fig.legend.location = "top_left"
            fig.legend.click_policy = "hide"
            if index >= 2:
                plots[1].append(fig)
            else:
                plots[0].append(fig)

        return plots

    def plot_test_data(self) -> list:
        """Plot testing data with y deviation and y ideal function

        Returns:
            list: plots of values
        """
        test_df = self.read_table('test_table').set_index('x')
        
        # create a list to hold our plots
        plots = [[], []]
        for col in test_df.columns:
            index = test_df.columns.get_loc(col)
            # lets create our bokeh figure to hold the plots
            fig = Figure(
                title=f"Testing Data",
                margin=10,
                x_axis_label="X",
                y_axis_label="Y",
                tools="pan,hover,wheel_zoom,reset,save",
            )

            # plot the y column vs x
            fig.circle(
                x="x",
                y=col,
                size=2,
                legend_label=f"{col}",
                color="blue",
                source=test_df,
            )
            
            # set plot legend details
            fig.legend.location = "top_left"
            fig.legend.click_policy = "hide"
            if index >= 2:
                plots[1].append(fig)
            else:
                plots[0].append(fig)

        return plots
