# read this one
# xgboost rolling yield optimized 10.08.2026

###############################################################################
# Libraries
###############################################################################

from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

from hyperopt import (
    hp,
    fmin,
    tpe,
    Trials,
    STATUS_OK,
    space_eval
)

from hyperopt.pyll import scope


###############################################################################
# Main settings
###############################################################################

random.seed(42)
np.random.seed(42)

# Number of Hyperopt parameter combinations tested for every outer test year
MAX_EVALS = 20 #change to 20 later

# Maximum number of boosting rounds allowed
MAX_N_ESTIMATORS = 8000

# Stop when validation RMSE has not improved for this many rounds
EARLY_STOPPING_ROUNDS = 200


###############################################################################
# Start measuring total runtime
###############################################################################

total_start_time = time.time()


###############################################################################
# Load complete yield dataset
###############################################################################

Final_dataset = pd.read_csv(
    "/Users/manavi/Desktop/Manuscript_review/"
    "Datasets/Final_merged_March19.csv"
)

print(
    "Original dataset shape:",
    Final_dataset.shape
)


###############################################################################
# Remove rows with missing response
###############################################################################

Final_dataset = (
    Final_dataset
    .dropna(
        subset=["Average_BLUE"]
    )
    .reset_index(drop=True)
)

print(
    "Dataset shape after removing missing responses:",
    Final_dataset.shape
)

print("\nObservations per year:")

print(
    Final_dataset["Year"]
    .value_counts()
    .sort_index()
)


###############################################################################
# Create grouping variable, predictors and response
###############################################################################

years = Final_dataset[
    "Year"
].to_numpy()

unique_years = sorted(
    Final_dataset[
        "Year"
    ].unique()
)


X = Final_dataset.drop(
    columns=[
        "Year",
        "Treatment",
        "Hybrid",
        "Field_Location",
        "Env",
        "Average_BLUE"
    ]
)


y = Final_dataset[
    "Average_BLUE"
].copy()


###############################################################################
# Check that all predictor columns are numeric
###############################################################################

non_numeric_columns = (
    X.select_dtypes(
        exclude=np.number
    )
    .columns
    .tolist()
)

if len(non_numeric_columns) > 0:

    raise TypeError(
        "These predictor columns are not numeric:\n"
        f"{non_numeric_columns}"
    )


###############################################################################
# Convert to float32 to reduce memory use
###############################################################################

X = X.astype(
    np.float32
)

y = y.astype(
    np.float32
)


print(
    "\nX shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print(
    "Years:",
    np.unique(years)
)


###############################################################################
# Output folder
###############################################################################

output_folder = Path(
    "/Users/manavi/Desktop/Manuscript_review/"
    "xgboost_optimized_rolling_year_yield"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

print(
    "\nResults will be saved in:"
)

print(
    output_folder
)


###############################################################################
# Hyperparameter search space
#
# n_estimators is not optimized here.
# It has a fixed upper limit of 8000.
# Early stopping determines the effective number of trees.
###############################################################################

space = {

    # Continuous, log-uniform
    "learning_rate": hp.loguniform(
        "learning_rate",
        np.log(0.005),
        np.log(0.30)
    ),

    # Integer
    "max_depth": scope.int(
        hp.quniform(
            "max_depth",
            2,
            15,
            1
        )
    ),

    # Continuous, log-uniform
    "min_child_weight": hp.loguniform(
        "min_child_weight",
        np.log(1),
        np.log(20)
    ),

    # Continuous, uniform
    "subsample": hp.uniform(
        "subsample",
        0.10,
        1.00
    ),

    # Continuous, uniform
    "colsample_bytree": hp.uniform(
        "colsample_bytree",
        0.10,
        1.00
    ),

    # Continuous, log-uniform
    "gamma": hp.loguniform(
        "gamma",
        np.log(0.01),
        np.log(0.50)
    ),

    # Continuous, log-uniform
    "reg_alpha": hp.loguniform(
        "reg_alpha",
        np.log(10),
        np.log(200)
    ),

    # Continuous, log-uniform
    "reg_lambda": hp.loguniform(
        "reg_lambda",
        np.log(1),
        np.log(10)
    )
}


###############################################################################
# Outer rolling-year cross-validation
#
# Nested rolling optimization requires at least TWO years in the outer
# training set so that at least one rolling inner train-validation split
# can be created.
#
# Example:
#   Outer fold 1: Year1 + Year2 -> Year3
#   Outer fold 2: Year1 + Year2 + Year3 -> Year4
#   Outer fold 3: Year1 + Year2 + Year3 + Year4 -> Year5
###############################################################################

if len(unique_years) < 3:

    raise ValueError(
        "At least three unique years are required for nested rolling-year "
        "cross-validation."
    )


###############################################################################
# Objects for storing results
###############################################################################

results = []

best_parameters_all_folds = []

all_y_test = []

all_y_pred = []

all_hybrid_env = []

all_test_years = []


###############################################################################
# Begin outer rolling-year loop
###############################################################################

for outer_fold, i in enumerate(
    range(
        1,
        len(unique_years) - 1
    ),
    start=1
):

    fold_start_time = time.time()

    print("\n" + "=" * 75)

    print(
        f"OUTER FOLD {outer_fold}"
    )

    print("=" * 75)


    ###########################################################################
    # Expanding outer training years and next-year test set
    ###########################################################################

    outer_training_years = unique_years[
        :i + 1
    ]

    test_year = unique_years[
        i + 1
    ]


    train_index = Final_dataset[
        "Year"
    ].isin(
        outer_training_years
    )

    test_index = (
        Final_dataset[
            "Year"
        ]
        ==
        test_year
    )


    outer_train_idx = np.flatnonzero(
        train_index.to_numpy()
    )

    outer_test_idx = np.flatnonzero(
        test_index.to_numpy()
    )


    ###########################################################################
    # Create outer training and test datasets
    ###########################################################################

    X_train = X.iloc[
        outer_train_idx
    ].copy()

    X_test = X.iloc[
        outer_test_idx
    ].copy()


    y_train = y.iloc[
        outer_train_idx
    ].copy()

    y_test = y.iloc[
        outer_test_idx
    ].copy()


    ###########################################################################
    # Outer-training years
    ###########################################################################

    train_years = years[
        outer_train_idx
    ]


    print(
        "Outer training years:",
        np.unique(train_years)
    )

    print(
        "Outer test year:",
        test_year
    )

    print(
        "Outer training shape:",
        X_train.shape
    )

    print(
        "Outer test shape:",
        X_test.shape
    )


    ###########################################################################
    # Objective function
    #
    # Hyperopt supplies one parameter combination.
    # It is evaluated using INNER ROLLING-YEAR CV.
    ###########################################################################

    def objective(params):

        # Create a copy so Hyperopt's original dictionary is not changed
        params = params.copy()

        # max_depth must be an integer
        params["max_depth"] = int(
            params["max_depth"]
        )


        #######################################################################
        # Inner rolling-year setup
        #######################################################################

        inner_unique_years = sorted(
            np.unique(
                train_years
            )
        )


        #######################################################################
        # Store inner-fold results
        #######################################################################

        inner_fold_mse = []

        inner_best_iterations = []


        #######################################################################
        # Inner rolling-year loop
        #
        # Example for outer training years Year1, Year2, Year3:
        #   Inner fold 1: Year1 -> Year2
        #   Inner fold 2: Year1 + Year2 -> Year3
        #######################################################################

        for inner_i in range(
            len(inner_unique_years) - 1
        ):

            inner_training_years = inner_unique_years[
                :inner_i + 1
            ]

            inner_validation_year = inner_unique_years[
                inner_i + 1
            ]


            inner_train_mask = np.isin(
                train_years,
                inner_training_years
            )

            inner_validation_mask = (
                train_years
                ==
                inner_validation_year
            )


            inner_train_idx = np.flatnonzero(
                inner_train_mask
            )

            inner_validation_idx = np.flatnonzero(
                inner_validation_mask
            )


            ###################################################################
            # Inner training data
            ###################################################################

            X_inner_train = X_train.iloc[
                inner_train_idx
            ]

            y_inner_train = y_train.iloc[
                inner_train_idx
            ]


            ###################################################################
            # Inner validation data
            ###################################################################

            X_inner_validation = X_train.iloc[
                inner_validation_idx
            ]

            y_inner_validation = y_train.iloc[
                inner_validation_idx
            ]


            ###################################################################
            # Temporary XGBoost model
            ###################################################################

            temporary_model = xgb.XGBRegressor(

                n_estimators=MAX_N_ESTIMATORS,

                objective="reg:squarederror",

                eval_metric="rmse",

                early_stopping_rounds=
                    EARLY_STOPPING_ROUNDS,

                tree_method="hist",

                random_state=42,

                verbosity=0,

                **params
            )


            ###################################################################
            # Train temporary model
            #
            # The held-out inner year controls early stopping.
            ###################################################################

            temporary_model.fit(

                X_inner_train,

                y_inner_train,

                eval_set=[
                    (
                        X_inner_validation,
                        y_inner_validation
                    )
                ],

                verbose=False
            )


            ###################################################################
            # Predict inner validation year
            ###################################################################

            inner_validation_predictions = (
                temporary_model.predict(
                    X_inner_validation
                )
            )


            ###################################################################
            # Calculate inner validation MSE
            ###################################################################

            validation_mse = mean_squared_error(
                y_inner_validation,
                inner_validation_predictions
            )


            inner_fold_mse.append(
                validation_mse
            )


            ###################################################################
            # Obtain number of trees selected by early stopping
            #
            # best_iteration is zero-based, so add 1.
            ###################################################################

            model_best_iteration = getattr(
                temporary_model,
                "best_iteration",
                MAX_N_ESTIMATORS - 1
            )


            number_of_trees = int(
                model_best_iteration + 1
            )


            inner_best_iterations.append(
                number_of_trees
            )


        #######################################################################
        # Mean MSE across inner rolling validation years
        #######################################################################

        mean_inner_mse = float(
            np.mean(
                inner_fold_mse
            )
        )


        #######################################################################
        # Median selected tree number across inner folds
        #######################################################################

        median_best_iteration = int(
            np.median(
                inner_best_iterations
            )
        )


        #######################################################################
        # Return results to Hyperopt
        #######################################################################

        return {

            "loss":
                mean_inner_mse,

            "status":
                STATUS_OK,

            "best_iteration":
                median_best_iteration
        }


    ###########################################################################
    # Run Hyperopt for the current outer fold
    ###########################################################################

    trials = Trials()


    best_raw = fmin(

        fn=objective,

        space=space,

        algo=tpe.suggest,

        max_evals=MAX_EVALS,

        trials=trials,

        rstate=np.random.default_rng(
            42 + outer_fold
        )
    )


    ###########################################################################
    # Convert Hyperopt result into usable parameter values
    ###########################################################################

    best_params = space_eval(
        space,
        best_raw
    )


    # Ensure max_depth is stored as an integer
    best_params["max_depth"] = int(
        best_params["max_depth"]
    )


    ###########################################################################
    # Retrieve best Hyperopt trial
    ###########################################################################

    best_trial = trials.best_trial


    ###########################################################################
    # Retrieve number of trees selected by early stopping
    ###########################################################################

    best_n_estimators = int(
        best_trial[
            "result"
        ][
            "best_iteration"
        ]
    )


    ###########################################################################
    # Retrieve best inner LOYO MSE
    ###########################################################################

    best_inner_mse = float(
        best_trial[
            "result"
        ][
            "loss"
        ]
    )


    ###########################################################################
    # The fixed candidate maximum is 8000 trees
    ###########################################################################

    candidate_max_estimators = (
        MAX_N_ESTIMATORS
    )


    ###########################################################################
    # Parameters used in the final model
    ###########################################################################

    best_params_final = best_params.copy()


    print("\nBest XGBoost parameters:")

    print(
        best_params_final
    )

    print(
        "Maximum trees allowed:",
        candidate_max_estimators
    )

    print(
        "Early-stopping selected trees:",
        best_n_estimators
    )

    print(
        "Best mean inner rolling-year MSE:",
        best_inner_mse
    )


    ###########################################################################
    # Fit final model using all outer-training years
    ###########################################################################

    final_model = xgb.XGBRegressor(

        n_estimators=
            best_n_estimators,

        objective=
            "reg:squarederror",

        eval_metric=
            "rmse",

        tree_method=
            "hist",

        random_state=
            42,

        verbosity=
            0,

        **best_params_final
    )


    final_model.fit(
        X_train,
        y_train
    )


    ###########################################################################
    # Save fitted final model immediately
    ###########################################################################

    model_path = (
        output_folder
        /
        (
            f"xgboost_optimized_rolling_year_yield_"
            f"test_year_{test_year}.json"
        )
    )


    final_model.save_model(
        str(model_path)
    )


    print(
        "Fitted model saved:",
        model_path
    )


    ###########################################################################
    # Predict untouched outer-test year
    ###########################################################################

    y_pred = final_model.predict(
        X_test
    )


    ###########################################################################
    # Calculate outer-test RMSE
    ###########################################################################

    outer_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )


    ###########################################################################
    # Calculate outer-test Pearson correlation
    ###########################################################################

    if (
        len(y_test) > 1
        and np.std(y_test) > 0
        and np.std(y_pred) > 0
    ):

        outer_correlation = float(
            pearsonr(
                y_test,
                y_pred
            )[0]
        )

    else:

        outer_correlation = np.nan


    print(
        f"\nFinal performance for held-out year {test_year}:"
    )

    print(
        f"RMSE = {outer_rmse:.3f}"
    )

    print(
        f"Pearson r = {outer_correlation:.3f}"
    )


    ###########################################################################
    # Obtain Hybrid and Env for outer-test observations
    ###########################################################################

    hybrid_env = (
        Final_dataset.iloc[
            outer_test_idx
        ][
            [
                "Hybrid",
                "Env"
            ]
        ]
        .astype(str)
        .agg(
            "_".join,
            axis=1
        )
        .to_numpy()
    )


    ###########################################################################
    # Store outer-fold results
    ###########################################################################

    results.append({

        "Year":
            test_year,

        "RMSE":
            outer_rmse,

        "Correlation":
            outer_correlation,

        "Best_inner_MSE":
            best_inner_mse,

        "Maximum_estimators":
            candidate_max_estimators,

        "Best_n_estimators":
            best_n_estimators
    })


    ###########################################################################
    # Store best parameters for current outer fold
    ###########################################################################

    best_parameters_all_folds.append({

        "Test_year":
            test_year,

        "Best_inner_MSE":
            best_inner_mse,

        "Maximum_estimators":
            candidate_max_estimators,

        "Best_n_estimators":
            best_n_estimators,

        **best_params_final
    })


    ###########################################################################
    # Create and save outer-fold predictions
    ###########################################################################

    fold_predictions = pd.DataFrame({

        "Hybrid_Env":
            hybrid_env,

        "Year":
            test_year,

        "Actual":
            y_test.to_numpy(),

        "Predicted":
            y_pred
    })


    fold_predictions.to_csv(

        output_folder
        /
        (
            f"fold_{test_year}_yield_"
            "optimized_xgboost_rolling_year.csv"
        ),

        index=False
    )


    ###########################################################################
    # Store values for pooled calculations
    ###########################################################################

    all_y_test.append(
        y_test.to_numpy()
    )

    all_y_pred.append(
        y_pred
    )

    all_hybrid_env.append(
        hybrid_env
    )

    all_test_years.append(
        np.repeat(
            test_year,
            len(y_test)
        )
    )


    ###########################################################################
    # Save checkpoint results after every completed rolling-year outer fold
    ###########################################################################

    pd.DataFrame(
        results
    ).to_csv(

        output_folder
        /
        "rolling_year_results_checkpoint.csv",

        index=False
    )


    pd.DataFrame(
        best_parameters_all_folds
    ).to_csv(

        output_folder
        /
        "rolling_year_best_parameters_checkpoint.csv",

        index=False
    )


    ###########################################################################
    # Print outer-fold runtime
    ###########################################################################

    fold_runtime_minutes = (
        time.time()
        -
        fold_start_time
    ) / 60


    print(
        f"\nOuter-fold runtime: "
        f"{fold_runtime_minutes:.2f} minutes"
    )


###############################################################################
# Convert outer-fold results to DataFrame
###############################################################################

results_df = pd.DataFrame(
    results
)


###############################################################################
# Calculate mean across year-specific metrics
###############################################################################

mean_rmse = results_df[
    "RMSE"
].mean()


mean_correlation = results_df[
    "Correlation"
].mean()


mean_row = pd.DataFrame(
    [
        {
            "Year":
                "Mean_across_years",

            "RMSE":
                mean_rmse,

            "Correlation":
                mean_correlation,

            "Best_inner_MSE":
                np.nan,

            "Maximum_estimators":
                np.nan,

            "Best_n_estimators":
                np.nan
        }
    ]
)


###############################################################################
# Combine all outer-test predictions
###############################################################################

final_y_true = np.concatenate(
    all_y_test
)


final_y_pred = np.concatenate(
    all_y_pred
)


final_hybrid_env = np.concatenate(
    all_hybrid_env
)


final_test_years = np.concatenate(
    all_test_years
)


###############################################################################
# Calculate pooled RMSE
###############################################################################

pooled_rmse = np.sqrt(
    mean_squared_error(
        final_y_true,
        final_y_pred
    )
)


###############################################################################
# Calculate pooled Pearson correlation
###############################################################################

if (
    np.std(final_y_true) > 0
    and np.std(final_y_pred) > 0
):

    pooled_correlation = float(
        pearsonr(
            final_y_true,
            final_y_pred
        )[0]
    )

else:

    pooled_correlation = np.nan


pooled_row = pd.DataFrame(
    [
        {
            "Year":
                "Pooled_predictions",

            "RMSE":
                pooled_rmse,

            "Correlation":
                pooled_correlation,

            "Best_inner_MSE":
                np.nan,

            "Maximum_estimators":
                np.nan,

            "Best_n_estimators":
                np.nan
        }
    ]
)


###############################################################################
# Add mean and pooled rows
###############################################################################

results_df = pd.concat(
    [
        results_df,
        mean_row,
        pooled_row
    ],
    ignore_index=True
)


###############################################################################
# Save final rolling-year outer-fold results
###############################################################################

results_df.to_csv(

    output_folder
    /
    "rolling_year_results_yield_optimized_xgboost.csv",

    index=False
)


###############################################################################
# Save best parameters for every rolling-year outer fold
###############################################################################

best_parameters_df = pd.DataFrame(
    best_parameters_all_folds
)


best_parameters_df.to_csv(

    output_folder
    /
    "best_parameters_each_rolling_year_fold.csv",

    index=False
)


###############################################################################
# Create final prediction DataFrame
###############################################################################

final_predictions_df = pd.DataFrame({

    "Hybrid_Env":
        final_hybrid_env,

    "Year":
        final_test_years,

    "Actual":
        final_y_true,

    "Predicted":
        final_y_pred
})


###############################################################################
# Save all outer-test predictions
###############################################################################

final_predictions_df.to_csv(

    output_folder
    /
    "rolling_year_final_predictions_optimized_xgboost_yield.csv",

    index=False
)


###############################################################################
# Print final results
###############################################################################

print("\n" + "=" * 75)

print("FINAL RESULTS")

print("=" * 75)

print(
    results_df
)


###############################################################################
# Print total runtime
###############################################################################

total_runtime_minutes = (
    time.time()
    -
    total_start_time
) / 60


print(
    f"\nTotal runtime: "
    f"{total_runtime_minutes:.2f} minutes"
)



print(
    "\nResults saved in:"
)

print(
    output_folder
)