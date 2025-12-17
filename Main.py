import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from openpyxl import load_workbook
from math import sin, exp
from sklearn.metrics import r2_score
import os, time
from sklearn.metrics import accuracy_score

path = "my_equations/11_9_25.2"
run_id = "11_9_25.2"

def Training_Set(var):
    df = pd.read_excel('AllSAMPLES.xlsx')

    #Number of rows to read for the training set (70% of the total data) 
    Training_Set = 3500

    # df.head(Num_rows)

    df = df.iloc[:Training_Set]
    df.columns = ["Tin", "Q", "flow_shale", "flow_steam", "length", "Pressue", "Status","Feasability",]


    # 2. Separate X (Features) and y (Target)
    # X contains all columns *except* 'Numeral' and 'Status'

    X = df[var].to_numpy() 

    # X = df[["Q", "length"]].to_numpy()

    # y contains only the 'Numeral' column
    Y = df["Feasability"].to_numpy()

    
    # model = PySRRegressor.from_file(run_directory="my_equations/10_6_25/checkpoint.pkl")
    model = PySRRegressor(
        maxsize=50,
        populations=50,
        niterations=3000,  # < Increase me for better results
        binary_operators=["+", "*", "-", "/"],
        unary_operators=[
            "exp",       
            "square",
            "sqrt",
            "inv",
            # ^ Custom operator (julia syntax)
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        # ^ Define operator for SymPy as well
        elementwise_loss='SigmoidLoss()',
        # ^ Custom loss function (julia syntax)
        output_directory= "my_equations",  # Where to save the equations
        run_id=run_id,  # Save equations to disk
        warm_start=True,  # Load from output_directory if possible
        model_selection="accuracy", # Select the most accurate model at the end
        nested_constraints={
            "square": {"square": 2, "sqrt": 4},
            "sqrt": {"square": 4, "sqrt": 2},
            "exp": {"exp": 0, "inv": 0},
        },
        maxdepth=10,
        complexity_of_constants=2,
        early_stop_condition=(
        "stop_if(loss, complexity) = loss < 0.03 && complexity < 10"
        # Stop early if we find a good and simple equation
        ),
    )

	        # "log": {"exp": 1, "log": 0},
            # "sin": {"sin": 0, "cos": 0, "tan": 0}, 
            # "cos": {"sin": 0, "cos": 0, "tan": 0},
            # "tan": {"sin": 0, "cos": 0, "tan": 0},
    model.fit(X, Y)

    # print(model)

    # print(model.latex())
    



def Validation_Set(var):
    
    df = pd.read_excel('AllSAMPLES.xlsx')

    #Number of rows to read for the training set (70% of the total data) 
    Training_Set = 3500
    #Number of rows to read for the Validation set (15% of the total data) 
    Validation_Set = 750

    # df.head(Num_rows)

    df = df.iloc[Training_Set:(Validation_Set + Training_Set)]
    df.columns = ["Tin", "Q", "flow_shale", "flow_steam", "length", "Pressue", "Status","Feasability",]


    # 2. Separate X (Features) and y (Target)
    
    X = df[var].to_numpy() 
    # X = df[["Q", "length"]].to_numpy()

    # y contains only the 'Feasability' column
    Y = df["Feasability"].to_numpy() 


    model = PySRRegressor.from_file(run_directory=path)
    model.set_params(extra_sympy_mappings="inv(x) = 1/x",)
   


def Test_Set(var):
    
    df = pd.read_excel('AllSAMPLES.xlsx')

    #Number of rows to read for the training set (70% of the total data) 
    Training_Set = 3500
    #Number of rows to read for the Validation set (15% of the total data) 
    Validation_Set = 750

    # df.head(Num_rows)

    df = df.iloc[(Validation_Set + Training_Set):]
    df.columns = ["Tin", "Q", "flow_shale", "flow_steam", "length", "Pressue", "Status","Feasability",]

    X_test = df[var].to_numpy() 
    y_test = df["Feasability"].to_numpy()

    
    model = PySRRegressor.from_file(run_directory=path)
    model.set_params(extra_sympy_mappings="inv(x) = 1/x",)
    lambda_func = model.get_best()

    y_pred = model.predict(X_test)
    # print(y_pred)
    
    # 1. Get the raw prediction scores (z values)
    # y_pred is the raw output (z) of the symbolic equation: f(X)
    raw_scores_z = model.predict(X_test)
    
    
    # 3. Apply a threshold (typically 0.5) to get binary predicted classes (0 or 1)
    # Use .astype(int) to convert True/False to 1/0
    
    predicted_classes = np.sign(raw_scores_z)
    predicted_classes[predicted_classes == 0] = -1  # Treat 0 as -1 for binary classification
    
    # 4. Calculate the Accuracy
    accuracy = accuracy_score(y_test, predicted_classes)

    print(f"Accuracy Score: {accuracy:.4f}\n")

    return accuracy, lambda_func

    
def print_results(accuracy, best_lambda_func, time_elapsed, var):

    folder_name = "Result_Sigmoid_Function"

    var_list = []
    for i in range(len(var)):
        var_list.append(f"X{i}: {var[i]}")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # Creates the folder if it doesn't exist

    # Step 2: Create a text file inside the folder
    result = f"\nAccuracy Score for the best PySR equation from {path}: {accuracy}\nVariables Used: {var_list}\nTime Elapsed: {time_elapsed:.2f} seconds\nEquation Used: {best_lambda_func}\n"
    dashes = "-" * 90
    
    txt = result + dashes
    file_path = os.path.join(folder_name, "Results.txt")
    with open(file_path, "a") as file:
        file.write(txt + "\n")

def start(variables):
    time_start = time.time()
    Training_Set(variables)
    Validation_Set(variables)
    accuracy, best_lambda_func = Test_Set(variables)
    time_end = time.time()
    time_elapsed = time_end - time_start
    print_results(accuracy, best_lambda_func, time_elapsed, variables)


def main():
    # Variables to choose from: "Tin", "Q", "flow_shale", "flow_steam", "length", "Pressue"
    all_variables = ["Tin", "Q", "flow_shale","flow_steam","length", "Pressue"]
    # time_start = time.time()
    # Training_Set(variables)
    # Validation_Set(variables)
    # accuracy, best_lambda_func = Test_Set(variables)
    # time_end = time.time()
    # time_elapsed = time_end - time_start
    # print_results(accuracy, best_lambda_func, time_elapsed, variables)
    # var_1 = ["Tin", "Q", "flow_shale","flow_steam","length"]
    # var_2 = ["Tin", "Q", "flow_shale","flow_steam","length", "Pressue"]
    # var_3 = ["Q", "flow_shale",]
    start(all_variables)
    


if __name__ == "__main__":
    main()