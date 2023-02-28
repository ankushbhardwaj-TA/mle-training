import logging
import os
import pickle

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor


def train_scores(output_dir, model_name, mse, rmse, mape):
    """
    Saves the training scores of a machine learning model to a file in the specified output
    directory.

    Args:
        output_dir (str): The directory where the output file will be saved.
        model_name (str): The name of the machine learning model.
        mse (float): The mean squared error of the model on the training data.
        rmse (float): The root mean squared error of the model on the training data.
        mape (float): The mean absolute percentage error of the model on the training data.

    Returns:
        None. Saves the train scores in output_dir.
    """
    output_path = os.path.join(output_dir, f"{model_name}_scores.txt")
    with open(output_path, "w") as f:
        f.write(f"{model_name} train scores:\n")
        f.write(f"MSE: {mse:.2f}\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAPE: {mape:.2f}\n")


def train_model(input_folder, output_folder, output_score_folder):
    """
    Trains a Linear Regression model, a Decision Tree Regressor, a RandomizedSearchCV model, and a
    GridSearchCV model on the provided training dataset. The trained models are saved as pickle
    files in the output folder. Model performances are evaluated using mean squared error (MSE),
    root mean squared error (RMSE), and mean absolute percentage error (MAPE), and the results are
    saved as text files in the output score folder.

    Arguments:
        input_folder (str): Path to the folder containing the input training dataset.
        output_folder (str): Path to the folder where the trained models will be saved as pickle
        files.
        output_score_folder (str): Path to the folder where the evaluation results will be saved as
        text files.

    Returns:
        None. Generates informative log files and trains the model, saves them as pickle files.
    """

    # Load the train dataset
    strat_train_set = pd.read_csv(os.path.join(input_folder, r"train.csv"))

    strat_train_set["rooms_per_household"] = (
        strat_train_set["total_rooms"] / strat_train_set["households"]
    )
    strat_train_set["bedrooms_per_room"] = (
        strat_train_set["total_bedrooms"] / strat_train_set["total_rooms"]
    )
    strat_train_set["population_per_household"] = (
        strat_train_set["population"] / strat_train_set["households"]
    )

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    ).copy()  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    # Train the model - Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    # Save the model to a pickle file - Linear Regression
    with open(os.path.join(output_folder, "Linear_Regression_Model.pkl"), "wb") as f:
        pickle.dump(lin_reg, f)

    # Model performance - Linear Regression
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    lin_mape = mean_absolute_percentage_error(housing_labels, housing_predictions)
    train_scores(output_score_folder, "Linear_Regression_Model", lin_mse, lin_rmse, lin_mape)

    # Train the model - Decision Tree Regressor
    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    # Save the model to a pickle file - Decision Tree Regressor
    with open(os.path.join(output_folder, "Decision_Tree_Regressor.pkl"), "wb") as f:
        pickle.dump(tree_reg, f)

    # Model performance - Decision Tree Regressor
    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    tree_mape = mean_absolute_percentage_error(housing_labels, housing_predictions)
    train_scores(output_score_folder, "Decision_Tree_Regressor", tree_mse, tree_rmse, tree_mape)

    # Train the model - RandomizedSearchCV
    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }
    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    rnd_search.fit(housing_prepared, housing_labels)

    # Save the model to a pickle file - RandomizedSearchCV
    with open(os.path.join(output_folder, "Randomized_Search_CV.pkl"), "wb") as f:
        pickle.dump(rnd_search, f)

    # Model performance - RandomizedSearchCV
    cvres = rnd_search.cv_results_
    logging.info("root negative mean score vs parameters for Randomized Search CV")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logging.info(f"{np.sqrt(-mean_score)}, {params}")
    housing_predictions = rnd_search.predict(housing_prepared)
    rnd_search_mse = mean_squared_error(housing_labels, housing_predictions)
    rnd_search_rmse = np.sqrt(rnd_search_mse)
    rnd_search_mape = mean_absolute_percentage_error(housing_labels, housing_predictions)
    train_scores(
        output_score_folder,
        "Randomized_Search_CV",
        rnd_search_mse,
        rnd_search_rmse,
        rnd_search_mape,
    )

    # Train the model - GridSearchCV
    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(housing_prepared, housing_labels)
    best_grid_search = grid_search.best_estimator_

    # Save the model to a pickle file - GridSearchCV
    with open(os.path.join(output_folder, "Best_Estimator_Grid_Search_CV.pkl"), "wb") as f:
        pickle.dump(best_grid_search, f)

    # Model performance - GridSearchCV
    cvres = grid_search.cv_results_
    logging.info("root negative mean score vs parameters for Grid Search CV")
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        logging.info(f"{np.sqrt(-mean_score)}, {params}")
    housing_predictions = best_grid_search.predict(housing_prepared)
    best_grid_search_mse = mean_squared_error(housing_labels, housing_predictions)
    best_grid_search_rmse = np.sqrt(best_grid_search_mse)
    best_grid_search_mape = mean_absolute_percentage_error(housing_labels, housing_predictions)
    train_scores(
        output_score_folder,
        "Best_Estimator_Grid_Search_CV",
        best_grid_search_mse,
        best_grid_search_rmse,
        best_grid_search_mape,
    )
    mlflow.log_artifacts(output_folder)
