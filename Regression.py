import numpy as np
import pandas as pd
import os, time, random
from pysr import PySRRegressor
from math import sin, exp
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split



def load_and_split_data():
    
    df = pd.read_excel('AllSAMPLES.xlsx')
    
    random_int = random.randint(0, 2**9)
    # Rename columns here so it is consistent for all sets
    df.columns = ["Tin", "Q", "flow_shale", "flow_steam", "length", "Pressure", "Status", "Feasability"]

    # 1. Split into Train (55%) and Temp (45%)
    df_train, df_temp = train_test_split(df, test_size=0.45, random_state=random_int, shuffle=True)

    # 2. Split Temp (45%) into Validation (15%) and Test (30%)
    # We use 0.67 because 67% of 45% = 30% of total
    df_val, df_test = train_test_split(df_temp, test_size=0.67, random_state=42, shuffle=True)

    return df_train, df_val, df_test, random_int

def Training_Set(df, var, run_id, path):

    # 1. Separate X (Features) and y (Target)
    # X contains passed columns *except* 'Feasability' and 'Status'
    X = df[var].to_numpy() 
    # y contains only the 'Feasability' column
    Y = df["Feasability"].to_numpy()

    model = PySRRegressor(
        maxsize=50,
        populations=50,
        niterations=1000,  #< Increase me for better results
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
        maxdepth=30,
        complexity_of_constants=4,
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
    


def Validation_Set(df, var, path):
    # 1. Separate X (Features) and y (Target)
    X_val = df[var].to_numpy() 
    y_val = df["Feasability"].to_numpy() 

    model = PySRRegressor.from_file(run_directory=path)
    model.set_params(extra_sympy_mappings="inv(x) = 1/x",)
    raw_scores_z = model.predict(X_val)
    
    # 2. Apply the same threshold as your Test Set
    predicted_classes = np.sign(raw_scores_z)
    predicted_classes[predicted_classes == 0] = -1 
    
    # 3. Calculate Accuracy
    acc = accuracy_score(y_val, predicted_classes)

    return acc
   


def Test_Set(df, var, matrix_status, path):

    X_test = df[var].to_numpy() 
    y_test = df["Feasability"].to_numpy()

    model = PySRRegressor.from_file(run_directory=path)
    model.set_params(extra_sympy_mappings="inv(x) = 1/x",)
    lambda_func = model.get_best()

   
    # 1. Get the raw prediction scores (z values)
    # y_pred is the raw output (z) of the symbolic equation: f(X)
    raw_scores_z = model.predict(X_test)
    
    
    # 2. Apply a threshold (typically 0.5) to get binary predicted classes (0 or 1)
    # Use .astype(int) to convert True/False to 1/0
    # probabilities = 1 / (1 + np.exp(-raw_scores_z))
    # predicted_classes = probabilities.round()             <-- Used for 0 and 1



    predicted_classes = np.sign(raw_scores_z)
    predicted_classes[predicted_classes == 0] = -1  # Treat 0 as -1 for binary classification
    
    # 4. Calculate the Accuracy
    accuracy = accuracy_score(y_test, predicted_classes)

    print(f"Accuracy Score: {accuracy:.4f}")
    if matrix_status:
        tn, fp, fn, tp = confusion_matrix(y_test, predicted_classes).ravel()
        print("-" * 30)
        print(f"Type 1 Errors (False Positives): {fp}")
        print(f"   (Model said Feasible, Reality was NOT)")
        print(f"Type 2 Errors (False Negatives): {fn}")
        print(f"   (Model said NOT Feasible, Reality was Feasible)")
        print("-" * 30)
        print(f"Correctly Identified Feasible:     {tp}")
        print(f"Correctly Identified Not Feasible: {tn}\n")

    return accuracy, lambda_func

    
def print_results(accuracy, best_lambda_func, time_elapsed, var, rand_state, v_score, path):

    folder_name = "Result_Sigmoid_Function"

    var_list = []
    for i in range(len(var)):
        var_list.append(f"X{i}: {var[i]}")

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)  # Creates the folder if it doesn't exist

    # Step 2: Create a text file inside the folder
    result = f"\nAccuracy Score for the best PySR equation from {path}: {accuracy}\nValidation Score:{v_score}\nVariables Used: {var_list}\nRandom State Used: {rand_state}\nTime Elapsed: {time_elapsed:.2f} seconds\nEquation Used: {best_lambda_func}\n"
    dashes = "-" * 90
    
    txt = result + dashes
    file_path = os.path.join(folder_name, "2_7_2026_Results.txt")
    with open(file_path, "a") as file:
        file.write(txt + "\n")

def start(variables, run_id, path, num_repeat):
    
    for i in range(num_repeat):
        if path[-1] != ".":
           path = path[:-1] + str(i)
           run_id = run_id[:-1]+ str(i)
        else:
            path = path + str(i)
            run_id = run_id + str(i)

        time_start = time.time()

        df_train, df_val, df_test, random_state_used = load_and_split_data()

        Training_Set(df_train, variables, run_id, path)
        validation_score= Validation_Set(df_val, variables, path)
        accuracy, best_lambda_func = Test_Set(df_test, variables, False, path)

        time_end = time.time()
        time_elapsed = time_end - time_start
        print_results(accuracy, best_lambda_func, time_elapsed, variables, random_state_used, validation_score, path)

def printlatexequation(folder_path):
    model = PySRRegressor.from_file(run_directory=folder_path)
    print(f"{model.latex()}\n")


def main():
    path = "my_equations/2_7_26"
    run_id = "2_7_26"
    num_repeat = 5
    sleep_time = num_repeat * 1300

    # all_variables = ["Tin", "Q", "flow_shale","flow_steam","length", "Pressure"]
    no_Tin = ["Q", "flow_shale","flow_steam","length", "Pressure"]
    
    start(no_Tin, run_id, path, num_repeat)





    
    


if __name__ == "__main__":
    main()