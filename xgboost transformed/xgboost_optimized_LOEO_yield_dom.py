###############################################################################
# 25.08.2026
#
# Yield
# Dominance transformed XGBoost
# LOEO grouped cross-validation
#
# Outer CV:
# Fixed splits from MASTER_CV_LOCATION_SPLITS.csv
#
# Inner CV:
# GroupKFold on outer-training groups for Hyperopt
###############################################################################


from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import GroupKFold
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
# Settings
###############################################################################

random.seed(42)
np.random.seed(42)

NUM_ITERATIONS = 5
N_GROUPS = 5

GROUP_SOURCE_COLUMN = "Field_Location"

MAX_EVALS = 20
MAX_N_ESTIMATORS = 8000
EARLY_STOPPING_ROUNDS = 200

# Number of SNP columns at beginning of X
N_SNP_COLUMNS = 2397


###############################################################################
# Start runtime
###############################################################################

total_start_time = time.time()


###############################################################################
# Yield height dataset
###############################################################################

DATA_PATH = (
    "/Users/manavi/Desktop/Manuscript_review/"
    "Datasets/Final_merged_March19.csv"
)


###############################################################################
# MASTER CV splits created previously
###############################################################################

MASTER_SPLIT_PATH = (
    "/Users/manavi/Desktop/Manuscript_review/"
    "xgboost_optimized_leave_group_env_out_yield/"
    "MASTER_CV_LOCATION_SPLITS.csv"
)


###############################################################################
# Output folder
###############################################################################

OUTPUT_FOLDER = Path(
    "/Users/manavi/Desktop/Manuscript_review/"
    "xgboost_optimized_leave_group_env_out_dominance_yield"
)

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


###############################################################################
# Load yield data
###############################################################################

Final_dataset = pd.read_csv(
    DATA_PATH
)

print(
    "Original dataset shape:",
    Final_dataset.shape
)


###############################################################################
# Remove missing response
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


###############################################################################
# Load MASTER CV splits
###############################################################################

master_splits = pd.read_csv(
    MASTER_SPLIT_PATH
)

Final_dataset[
    "Field_Location"
] = Final_dataset[
    "Field_Location"
].astype(str)

master_splits[
    "Field_Location"
] = master_splits[
    "Field_Location"
].astype(str)


###############################################################################
# Create X and y
###############################################################################

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
# Check predictors
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
# Convert to float32
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
    "Unique field locations:",
    Final_dataset[
        "Field_Location"
    ].nunique()
)

print(
    "Results folder:",
    OUTPUT_FOLDER
)


###############################################################################
# Hyperparameter search space
###############################################################################

space = {

    "learning_rate": hp.loguniform(
        "learning_rate",
        np.log(0.005),
        np.log(0.30)
    ),

    "max_depth": scope.int(
        hp.quniform(
            "max_depth",
            2,
            15,
            1
        )
    ),

    "min_child_weight": hp.loguniform(
        "min_child_weight",
        np.log(1),
        np.log(20)
    ),

    "subsample": hp.uniform(
        "subsample",
        0.10,
        1.00
    ),

    "colsample_bytree": hp.uniform(
        "colsample_bytree",
        0.10,
        1.00
    ),

    "gamma": hp.loguniform(
        "gamma",
        np.log(0.01),
        np.log(0.50)
    ),

    "reg_alpha": hp.loguniform(
        "reg_alpha",
        np.log(10),
        np.log(200)
    ),

    "reg_lambda": hp.loguniform(
        "reg_lambda",
        np.log(1),
        np.log(10)
    )
}


###############################################################################
# Storage
###############################################################################

all_fold_results = []

all_iteration_results = []

best_parameters_all_folds = []

all_predictions = []


###############################################################################
# Outer LOEO iterations
###############################################################################

for iteration in range(
    1,
    NUM_ITERATIONS + 1
):

    iteration_start_time = time.time()

    print(
        "\n" + "#" * 80
    )

    print(
        f"ITERATION {iteration}"
    )

    print(
        "#" * 80
    )


    iteration_y_test = []

    iteration_y_pred = []


    ############################################################################
    # Outer folds
    ############################################################################

    for outer_fold in range(
        1,
        N_GROUPS + 1
    ):

        fold_start_time = time.time()

        print(
            "\n" + "=" * 75
        )

        print(
            f"ITERATION {iteration} - OUTER FOLD {outer_fold}"
        )

        print(
            "=" * 75
        )


        #######################################################################
        # Read the previously saved outer split
        #######################################################################

        current_split = master_splits[
            (
                master_splits[
                    "Iteration"
                ] == iteration
            )
            &
            (
                master_splits[
                    "Fold"
                ] == outer_fold
            )
        ]


        #######################################################################
        # Training locations
        #######################################################################

        train_locations = current_split.loc[
            current_split[
                "Split"
            ] == "train",
            "Field_Location"
        ].tolist()


        #######################################################################
        # Test locations
        #######################################################################

        test_locations = current_split.loc[
            current_split[
                "Split"
            ] == "test",
            "Field_Location"
        ].tolist()


        #######################################################################
        # Find observation indices
        #######################################################################

        outer_train_idx = np.where(

            Final_dataset[
                "Field_Location"
            ].isin(
                train_locations
            )

        )[0]


        outer_test_idx = np.where(

            Final_dataset[
                "Field_Location"
            ].isin(
                test_locations
            )

        )[0]


        #######################################################################
        # Field location -> group mapping
        #######################################################################

        location_to_group = dict(
            zip(
                current_split[
                    "Field_Location"
                ],
                current_split[
                    "Group"
                ]
            )
        )


        #######################################################################
        # Outer train and test data
        #######################################################################

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


        #######################################################################
        # Training group labels
        #
        # Used for INNER GroupKFold
        #######################################################################

        train_groups = (

            Final_dataset
            .iloc[
                outer_train_idx
            ][
                "Field_Location"
            ]

            .map(
                location_to_group
            )

            .to_numpy()
        )


        #######################################################################
        # Outer test group
        #######################################################################

        test_group = current_split.loc[
            current_split[
                "Split"
            ] == "test",
            "Group"
        ].unique()[0]


        print(
            "Outer training groups:",
            np.unique(
                train_groups
            )
        )

        print(
            "Outer test group:",
            test_group
        )

        print(
            "Outer training shape:",
            X_train.shape
        )

        print(
            "Outer test shape:",
            X_test.shape
        )


        #######################################################################
        # DOMINANCE TRANSFORMATION
        #
        # d is calculated using COMPLETE OUTER-TRAINING SET only.
        #
        # Same d is applied to:
        #   X_train
        #   X_test
        #
        # Environmental variables are unchanged.
        #######################################################################


        #######################################################################
        # SNP matrices
        #######################################################################

        xtrain_snp = X_train.iloc[
            :,
            :N_SNP_COLUMNS
        ].to_numpy(
            copy=True
        )


        features_train = X_train.iloc[
            :,
            :N_SNP_COLUMNS
        ].to_numpy(
            copy=True
        )


        features_test = X_test.iloc[
            :,
            :N_SNP_COLUMNS
        ].to_numpy(
            copy=True
        )


        #######################################################################
        # Mean phenotype for genotype 0, 1 and 2
        #######################################################################

        snp_mean = np.zeros(
            (
                xtrain_snp.shape[1],
                3
            )
        )


        print(
            "Number of SNPs for dominance transformation:",
            xtrain_snp.shape[1]
        )


        #######################################################################
        # Calculate genotype means using outer training phenotype only
        #######################################################################

        for snp in range(
            xtrain_snp.shape[1]
        ):

            for genotype in range(3):

                snp_mean[
                    snp,
                    genotype
                ] = np.nanmean(

                    y_train[
                        xtrain_snp[
                            :,
                            snp
                        ]
                        ==
                        genotype
                    ]
                )


                # Replace missing means with 0
                snp_mean[
                    np.isnan(
                        snp_mean
                    )
                ] = 0.0


        #######################################################################
        # If genotype 0 mean > genotype 2 mean
        #######################################################################

        my_choose = (
            snp_mean[
                :,
                0
            ]
            >
            snp_mean[
                :,
                2
            ]
        )


        #######################################################################
        # Swap genotype-0 and genotype-2 means
        #######################################################################

        temp = snp_mean[
            my_choose,
            0
        ].copy()


        snp_mean[
            my_choose,
            0
        ] = snp_mean[
            my_choose,
            2
        ]


        snp_mean[
            my_choose,
            2
        ] = temp


        #######################################################################
        # Flip genotype coding in temporary SNP matrix
        #######################################################################

        xtrain_snp[
            :,
            my_choose
        ] = np.abs(

            xtrain_snp[
                :,
                my_choose
            ]

            - 2
        )


        #######################################################################
        # Calculate degree of dominance d
        #######################################################################

        d = (

            (
                snp_mean[
                    :,
                    1
                ]
                -
                snp_mean[
                    :,
                    0
                ]
            )

            /

            (
                snp_mean[
                    :,
                    2
                ]
                -
                snp_mean[
                    :,
                    0
                ]
            )

            * 2
        )


        #######################################################################
        # Replace missing d with 1
        #######################################################################

        my_choose_na = np.isnan(
            d
        )


        d[
            my_choose_na
        ] = 1


        #######################################################################
        # Save dominance values BEFORE capping
        #######################################################################

        d_before_capping = d.copy()


        dominance_before_capping = pd.DataFrame({

            "SNP":
                X_train.columns[
                    :N_SNP_COLUMNS
                ],

            "Mean_genotype_0":
                snp_mean[
                    :,
                    0
                ],

            "Mean_genotype_1":
                snp_mean[
                    :,
                    1
                ],

            "Mean_genotype_2":
                snp_mean[
                    :,
                    2
                ],

            "d_before_capping":
                d_before_capping
        })


        dominance_before_capping.to_csv(

            OUTPUT_FOLDER

            / (

                "Dominance_Calculation_Before_Capping_"

                f"Iteration_{iteration}_"

                f"Fold_{outer_fold}_"

                f"Test_{test_group}_yield.csv"
            ),

            index=False
        )


        #######################################################################
        # Cap d at 2
        #######################################################################

        my_choose1 = (
            d >= 2
        )

        d[
            my_choose1
        ] = 2


        #######################################################################
        # Cap d at 0
        #######################################################################

        my_choose2 = (
            d <= 0
        )

        d[
            my_choose2
        ] = 0


        #######################################################################
        # Save dominance values AFTER capping
        #######################################################################

        dominance_after_capping = pd.DataFrame({

            "SNP":
                X_train.columns[
                    :N_SNP_COLUMNS
                ],

            "Mean_genotype_0":
                snp_mean[
                    :,
                    0
                ],

            "Mean_genotype_1":
                snp_mean[
                    :,
                    1
                ],

            "Mean_genotype_2":
                snp_mean[
                    :,
                    2
                ],

            "d_after_capping":
                d
        })


        dominance_after_capping.to_csv(

            OUTPUT_FOLDER

            / (

                "Dominance_Calculation_After_Capping_"

                f"Iteration_{iteration}_"

                f"Fold_{outer_fold}_"

                f"Test_{test_group}_yield.csv"
            ),

            index=False
        )


        #######################################################################
        # Replace heterozygous genotype 1 by d in OUTER TRAIN
        #######################################################################

        for snp in range(
            features_train.shape[1]
        ):

            my_choose = (

                features_train[
                    :,
                    snp
                ]

                == 1
            )


            features_train[
                my_choose,
                snp
            ] = d[
                snp
            ]


        #######################################################################
        # Apply SAME d to OUTER TEST
        #######################################################################

        for snp in range(
            features_test.shape[1]
        ):

            my_choose = (

                features_test[
                    :,
                    snp
                ]

                == 1
            )


            features_test[
                my_choose,
                snp
            ] = d[
                snp
            ]


        #######################################################################
        # Put transformed SNPs back into X
        #######################################################################

        X_train.iloc[
            :,
            :N_SNP_COLUMNS
        ] = features_train


        X_test.iloc[
            :,
            :N_SNP_COLUMNS
        ] = features_test


        #######################################################################
        # HYPEROPT OBJECTIVE
        #
        # Same LOEO idea as before:
        #
        # The outer test group stays completely excluded.
        #
        # Hyperopt uses GroupKFold on the remaining outer-training groups.
        #######################################################################

        def objective(params):

            params = params.copy()

            params[
                "max_depth"
            ] = int(
                params[
                    "max_depth"
                ]
            )


            inner_fold_mse = []

            inner_best_iterations = []


            unique_inner_groups = np.unique(
                train_groups
            )


            inner_group_kfold = GroupKFold(
                n_splits=len(
                    unique_inner_groups
                )
            )


            ###################################################################
            # Inner grouped CV
            ###################################################################

            for (
                inner_train_idx,
                inner_validation_idx
            ) in inner_group_kfold.split(

                X_train,

                y_train,

                groups=train_groups
            ):


                ################################################################
                # Inner training data
                ################################################################

                X_inner_train = X_train.iloc[
                    inner_train_idx
                ]


                y_inner_train = y_train.iloc[
                    inner_train_idx
                ]


                ################################################################
                # Inner validation data
                ################################################################

                X_inner_validation = X_train.iloc[
                    inner_validation_idx
                ]


                y_inner_validation = y_train.iloc[
                    inner_validation_idx
                ]


                ################################################################
                # Temporary model
                ################################################################

                temporary_model = xgb.XGBRegressor(

                    n_estimators=
                        MAX_N_ESTIMATORS,

                    objective=
                        "reg:squarederror",

                    eval_metric=
                        "rmse",

                    early_stopping_rounds=
                        EARLY_STOPPING_ROUNDS,

                    tree_method=
                        "hist",

                    random_state=
                        42,

                    verbosity=
                        0,

                    **params
                )


                ################################################################
                # Fit temporary model
                ################################################################

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


                ################################################################
                # Predict inner validation
                ################################################################

                validation_predictions = (
                    temporary_model.predict(
                        X_inner_validation
                    )
                )


                ################################################################
                # Validation MSE
                ################################################################

                validation_mse = (
                    mean_squared_error(

                        y_inner_validation,

                        validation_predictions
                    )
                )


                inner_fold_mse.append(
                    validation_mse
                )


                ################################################################
                # Number of trees from early stopping
                ################################################################

                best_iteration = getattr(

                    temporary_model,

                    "best_iteration",

                    MAX_N_ESTIMATORS - 1
                )


                inner_best_iterations.append(

                    int(
                        best_iteration + 1
                    )
                )


            ###################################################################
            # Return Hyperopt loss
            ###################################################################

            return {

                "loss":
                    float(
                        np.mean(
                            inner_fold_mse
                        )
                    ),

                "status":
                    STATUS_OK,

                "best_iteration":
                    int(
                        np.median(
                            inner_best_iterations
                        )
                    )
            }


        #######################################################################
        # Run Hyperopt
        #######################################################################

        trials = Trials()


        best_raw = fmin(

            fn=objective,

            space=space,

            algo=tpe.suggest,

            max_evals=MAX_EVALS,

            trials=trials,

            rstate=np.random.default_rng(

                42
                +
                iteration * 100
                +
                outer_fold
            )
        )


        #######################################################################
        # Best parameters
        #######################################################################

        best_params = space_eval(
            space,
            best_raw
        )


        best_params[
            "max_depth"
        ] = int(
            best_params[
                "max_depth"
            ]
        )


        best_trial = trials.best_trial


        #######################################################################
        # Best number of trees
        #######################################################################

        best_n_estimators = int(

            best_trial[
                "result"
            ][
                "best_iteration"
            ]
        )


        #######################################################################
        # Best inner MSE
        #######################################################################

        best_inner_mse = float(

            best_trial[
                "result"
            ][
                "loss"
            ]
        )


        best_params_final = best_params.copy()


        print(
            "\nBest XGBoost parameters:"
        )

        print(
            best_params_final
        )

        print(
            "Maximum trees allowed:",
            MAX_N_ESTIMATORS
        )

        print(
            "Early-stopping selected trees:",
            best_n_estimators
        )

        print(
            "Best mean inner group-CV MSE:",
            best_inner_mse
        )


        #######################################################################
        # Fit final model on complete OUTER training set
        #######################################################################

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


        #######################################################################
        # Save fitted model
        #######################################################################

        model_path = (

            OUTPUT_FOLDER

            / (

                "xgboost_optimized_leave_group_out_"

                "dominance_yield"

                f"iteration_{iteration}_"

                f"fold_{outer_fold}_"

                f"test_{test_group}.json"
            )
        )


        final_model.save_model(
            str(
                model_path
            )
        )


        #######################################################################
        # Predict OUTER test group
        #######################################################################

        y_pred = final_model.predict(
            X_test
        )


        #######################################################################
        # Outer MSE
        #######################################################################

        outer_mse = mean_squared_error(
            y_test,
            y_pred
        )


        #######################################################################
        # Outer RMSE
        #######################################################################

        outer_rmse = np.sqrt(
            outer_mse
        )


        #######################################################################
        # Outer Pearson correlation
        #######################################################################

        if (
            len(y_test) > 1
            and
            np.std(y_test) > 0
            and
            np.std(y_pred) > 0
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
            f"\nPerformance for held-out group {test_group}:"
        )

        print(
            f"MSE = {outer_mse:.4f}"
        )

        print(
            f"RMSE = {outer_rmse:.4f}"
        )

        print(
            f"Pearson r = {outer_correlation:.4f}"
        )


        #######################################################################
        # Store fold results
        #######################################################################

        all_fold_results.append({

            "Iteration":
                iteration,

            "Fold":
                outer_fold,

            "Test_group":
                test_group,

            "MSE":
                outer_mse,

            "RMSE":
                outer_rmse,

            "Correlation":
                outer_correlation,

            "Best_inner_MSE":
                best_inner_mse,

            "Maximum_estimators":
                MAX_N_ESTIMATORS,

            "Best_n_estimators":
                best_n_estimators
        })


        #######################################################################
        # Store best parameters
        #######################################################################

        best_parameters_all_folds.append({

            "Iteration":
                iteration,

            "Fold":
                outer_fold,

            "Test_group":
                test_group,

            "Best_inner_MSE":
                best_inner_mse,

            "Maximum_estimators":
                MAX_N_ESTIMATORS,

            "Best_n_estimators":
                best_n_estimators,

            **best_params_final
        })


        #######################################################################
        # Outer test metadata
        #######################################################################

        test_metadata = Final_dataset.iloc[
            outer_test_idx
        ]


        hybrid_env = (

            test_metadata[
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


        #######################################################################
        # Save fold predictions
        #######################################################################

        fold_predictions = pd.DataFrame({

            "Hybrid_Env":
                hybrid_env,

            "Hybrid":
                test_metadata[
                    "Hybrid"
                ]
                .astype(str)
                .to_numpy(),

            "Env":
                test_metadata[
                    "Env"
                ]
                .astype(str)
                .to_numpy(),

            "Field_Location":
                test_metadata[
                    "Field_Location"
                ]
                .astype(str)
                .to_numpy(),

            "Year":
                test_metadata[
                    "Year"
                ]
                .to_numpy(),

            "Iteration":
                iteration,

            "Fold":
                outer_fold,

            "Test_group":
                test_group,

            "Actual":
                y_test.to_numpy(),

            "Predicted":
                y_pred
        })


        fold_predictions.to_csv(

            OUTPUT_FOLDER

            / (

                f"iteration_{iteration}_"

                f"fold_{outer_fold}_"

                f"test_{test_group}_"

                "dominance_yield_predictions.csv"
            ),

            index=False
        )


        all_predictions.append(
            fold_predictions
        )


        iteration_y_test.append(
            y_test.to_numpy()
        )


        iteration_y_pred.append(
            y_pred
        )


        #######################################################################
        # Checkpoint files
        #######################################################################

        pd.DataFrame(
            all_fold_results
        ).to_csv(

            OUTPUT_FOLDER
            / "leave_group_out_dominance_results_checkpoint_yield.csv",

            index=False
        )


        pd.DataFrame(
            best_parameters_all_folds
        ).to_csv(

            OUTPUT_FOLDER
            / "leave_group_out_dominance_best_parameters_checkpoint_yield.csv",

            index=False
        )


        pd.concat(
            all_predictions,
            ignore_index=True
        ).to_csv(

            OUTPUT_FOLDER
            / "leave_group_out_dominance_predictions_checkpoint_yield.csv",

            index=False
        )


        #######################################################################
        # Fold runtime
        #######################################################################

        fold_runtime_minutes = (

            time.time()
            -
            fold_start_time

        ) / 60


        print(

            f"\nOuter-fold runtime: "

            f"{fold_runtime_minutes:.2f} minutes"
        )


    ############################################################################
    # Iteration summary
    ############################################################################

    current_fold_results = pd.DataFrame(

        [

            result

            for result in all_fold_results

            if result[
                "Iteration"
            ] == iteration
        ]
    )


    mean_fold_mse = (
        current_fold_results[
            "MSE"
        ].mean()
    )


    mean_fold_rmse = (
        current_fold_results[
            "RMSE"
        ].mean()
    )


    mean_fold_correlation = (
        current_fold_results[
            "Correlation"
        ].mean()
    )


    ###########################################################################
    # Pool predictions from all 5 test groups
    ###########################################################################

    iteration_true = np.concatenate(
        iteration_y_test
    )


    iteration_pred = np.concatenate(
        iteration_y_pred
    )


    pooled_iteration_mse = (
        mean_squared_error(
            iteration_true,
            iteration_pred
        )
    )


    pooled_iteration_rmse = np.sqrt(
        pooled_iteration_mse
    )


    if (
        np.std(
            iteration_true
        ) > 0
        and
        np.std(
            iteration_pred
        ) > 0
    ):

        pooled_iteration_correlation = float(

            pearsonr(

                iteration_true,

                iteration_pred
            )[0]
        )

    else:

        pooled_iteration_correlation = np.nan


    ###########################################################################
    # Store iteration results
    ###########################################################################

    all_iteration_results.append({

        "Iteration":
            iteration,

        "Mean_fold_MSE":
            mean_fold_mse,

        "Mean_fold_RMSE":
            mean_fold_rmse,

        "Mean_fold_Correlation":
            mean_fold_correlation,

        "Pooled_prediction_MSE":
            pooled_iteration_mse,

        "Pooled_prediction_RMSE":
            pooled_iteration_rmse,

        "Pooled_prediction_Correlation":
            pooled_iteration_correlation
    })


    ###########################################################################
    # Save iteration checkpoint
    ###########################################################################

    pd.DataFrame(
        all_iteration_results
    ).to_csv(

        OUTPUT_FOLDER

        / (
            "leave_group_out_dominance_"
            "iteration_summary_checkpoint_yield.csv"
        ),

        index=False
    )


    ###########################################################################
    # Runtime
    ###########################################################################

    iteration_runtime_minutes = (

        time.time()
        -
        iteration_start_time

    ) / 60


    print(
        "\n" + "-" * 75
    )

    print(
        f"ITERATION {iteration} SUMMARY"
    )

    print(
        "-" * 75
    )

    print(
        f"Mean fold RMSE: "
        f"{mean_fold_rmse:.4f}"
    )

    print(
        f"Mean fold correlation: "
        f"{mean_fold_correlation:.4f}"
    )

    print(
        f"Pooled prediction RMSE: "
        f"{pooled_iteration_rmse:.4f}"
    )

    print(
        f"Pooled prediction correlation: "
        f"{pooled_iteration_correlation:.4f}"
    )

    print(
        f"Iteration runtime: "
        f"{iteration_runtime_minutes:.2f} minutes"
    )


###############################################################################
# Final DataFrames
###############################################################################

fold_results_df = pd.DataFrame(
    all_fold_results
)


iteration_results_df = pd.DataFrame(
    all_iteration_results
)


best_parameters_df = pd.DataFrame(
    best_parameters_all_folds
)


all_predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


###############################################################################
# Save fold results
###############################################################################

fold_results_df.to_csv(

    OUTPUT_FOLDER

    / (
        "leave_group_out_fold_results_"
        "optimized_xgboost_dominance_yield.csv"
    ),

    index=False
)


###############################################################################
# Save iteration summaries
###############################################################################

iteration_results_df.to_csv(

    OUTPUT_FOLDER

    / (
        "leave_group_out_iteration_summary_"
        "optimized_xgboost_dominance_yield.csv"
    ),

    index=False
)


###############################################################################
# Save best parameters
###############################################################################

best_parameters_df.to_csv(

    OUTPUT_FOLDER

    / (
        "leave_group_out_"
        "best_parameters_each_fold_"
        "dominance_yield.csv"
    ),

    index=False
)


###############################################################################
# Save all predictions
###############################################################################

all_predictions_df.to_csv(

    OUTPUT_FOLDER

    / (
        "leave_group_out_all_predictions_"
        "optimized_xgboost_dominance_yield.csv"
    ),

    index=False
)


###############################################################################
# Overall summary
###############################################################################

mean_iteration_pooled_rmse = (

    iteration_results_df[
        "Pooled_prediction_RMSE"
    ].mean()
)


mean_iteration_pooled_correlation = (

    iteration_results_df[
        "Pooled_prediction_Correlation"
    ].mean()
)


###############################################################################
# Pool all predictions from all iterations
###############################################################################

overall_true = (
    all_predictions_df[
        "Actual"
    ]
    .to_numpy()
)


overall_pred = (
    all_predictions_df[
        "Predicted"
    ]
    .to_numpy()
)


overall_pooled_rmse = np.sqrt(

    mean_squared_error(
        overall_true,
        overall_pred
    )
)


if (
    np.std(
        overall_true
    ) > 0
    and
    np.std(
        overall_pred
    ) > 0
):

    overall_pooled_correlation = float(

        pearsonr(
            overall_true,
            overall_pred
        )[0]
    )

else:

    overall_pooled_correlation = np.nan


###############################################################################
# Overall summary dataframe
###############################################################################

overall_summary_df = pd.DataFrame(

    [

        {

            "Summary":
                "Mean_across_iteration_pooled_metrics",

            "RMSE":
                mean_iteration_pooled_rmse,

            "Correlation":
                mean_iteration_pooled_correlation
        },

        {

            "Summary":
                "Pooled_all_predictions_all_iterations",

            "RMSE":
                overall_pooled_rmse,

            "Correlation":
                overall_pooled_correlation
        }
    ]
)


###############################################################################
# Save overall summary
###############################################################################

overall_summary_df.to_csv(

    OUTPUT_FOLDER

    / (
        "leave_group_out_overall_summary_"
        "optimized_xgboost_dominance_yield.csv"
    ),

    index=False
)


###############################################################################
# Print final results
###############################################################################

print(
    "\n" + "=" * 80
)

print(
    "FINAL ITERATION SUMMARIES"
)

print(
    "=" * 80
)

print(
    iteration_results_df
)


print(
    "\n" + "=" * 80
)

print(
    "OVERALL SUMMARY"
)

print(
    "=" * 80
)

print(
    overall_summary_df
)


###############################################################################
# Total runtime
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
    OUTPUT_FOLDER
)