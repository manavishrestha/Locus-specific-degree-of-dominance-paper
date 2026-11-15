#01.10.2026

#Bayesian with Master CV

#LOEO yield

rm(list = ls())

setwd("/Users/manavi/Desktop/Manuscript_review/")

library(BGLR)
library(data.table)
library(dplyr)

set.seed(42)

output_dir <- "/Users/manavi/Desktop/Manuscript_review/Bayesian_LOEO_new_CV_yield"

dir.create(
  output_dir,
  showWarnings = FALSE,
  recursive = TRUE
)
# ============================================================
# Read data
# ============================================================


Final_merged <- read.csv("Datasets/Final_merged_March19.csv")


# Master CV splits
Master_CV <- read.csv(
  "/Users/manavi/Desktop/Manuscript_review/xgboost_optimized_leave_group_env_out_yield/MASTER_CV_LOCATION_SPLITS.csv"
)

# ============================================================
# Data partitions
# ============================================================

blues <- Final_merged[, 1:5]
geno <- as.matrix(Final_merged[, 6:2402])
env <- Final_merged[, 2403:3056]
meta <- Final_merged[, 3058:3059]
weather <- Final_merged[, 3060:3075]

# Target
y <- blues[, 4]  # Average_BLUE

merged_rest <- data.frame(
  env,
  meta,
  weather
)

# ============================================================
# CV settings
# ============================================================

n_groups <- 5
num_iterations <- 5

all_results <- list()

# ============================================================
# Outer CV
# ============================================================

for (iteration in 1:num_iterations) {
  
  cat(sprintf("\nIteration %d:\n", iteration))
  
  mse_scores <- c()
  rmse_scores <- c()
  correlation_scores <- c()
  
  for (fold in 1:n_groups) {
    
    # --------------------------------------------------------
    # Get THIS iteration/fold from MASTER CV
    # --------------------------------------------------------
    
    cv_fold <- Master_CV %>%
      filter(
        Iteration == iteration,
        Fold == fold
      )
    
    # Locations defined as training locations in MASTER CV
    train_locations <- cv_fold %>%
      filter(Split == "train") %>%
      pull(Field_Location) %>%
      unique()
    
    # Locations defined as test locations in MASTER CV
    test_locations <- cv_fold %>%
      filter(Split == "test") %>%
      pull(Field_Location) %>%
      unique()
    
    # Test group, just for checking/printing
    test_group <- unique(cv_fold$Test_group)
    
    # --------------------------------------------------------
    # Convert location split to observation-level indices
    # --------------------------------------------------------
    
    train_idx <- Final_merged$Field_Location %in% train_locations
    test_idx <- Final_merged$Field_Location %in% test_locations
    
    cat(sprintf("  Fold %d:\n", fold))
    cat(sprintf("    Test group: %s\n", paste(test_group, collapse = ", ")))
    cat(sprintf("    Training locations: %d\n", length(train_locations)))
    cat(sprintf("    Test locations: %d\n", length(test_locations)))
    cat(sprintf("    Training observations: %d\n", sum(train_idx)))
    cat(sprintf("    Test observations: %d\n", sum(test_idx)))
    
    # --------------------------------------------------------
    # Mask test observations
    # --------------------------------------------------------
    
    y_train <- y
    y_train[test_idx] <- NA
    
    # --------------------------------------------------------
    # BGLR model
    # --------------------------------------------------------
    
    ETA <- list(
      list(
        X = geno,
        model = "BayesA"
      ),
      list(
        X = merged_rest,
        model = "BRR"
      )
    )
    
    model <- BGLR(
      y = y_train,
      ETA = ETA,
      nIter = 5000,
      burnIn = 500,
      verbose = FALSE
    )
    
    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    
    model_file <- sprintf(
      "iteration_%d_fold_%d_model.rds",
      iteration,
      fold
    )
    
    saveRDS(
      model,
      model_file
    )
    
    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    
    y_pred <- model$yHat[test_idx]
    y_true <- y[test_idx]
    
    pred_results <- data.frame(
      Iteration = iteration,
      Fold = fold,
      Actual = y_true,
      Predicted = y_pred
    )
    
    # Save predictions
    pred_file <- file.path(
      output_dir,
      sprintf("iteration_%d_fold_%d_predictions.csv", iteration, fold)
    )
    
    write.csv(
      pred_results,
      pred_file,
      row.names = FALSE
    )
    
    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------
    
    mse <- mean(
      (y_true - y_pred)^2
    )
    
    rmse <- sqrt(mse)
    
    correlation <- cor(
      y_true,
      y_pred
    )
    
    mse_scores <- c(
      mse_scores,
      mse
    )
    
    rmse_scores <- c(
      rmse_scores,
      rmse
    )
    
    correlation_scores <- c(
      correlation_scores,
      correlation
    )
    
    cat(sprintf(
      "    MSE: %.4f\n    RMSE: %.4f\n    Correlation: %.4f\n",
      mse,
      rmse,
      correlation
    ))
  }
  
  # ========================================================
  # Summarize iteration
  # ========================================================
  
  pooled_mse <- mean(mse_scores)
  
  pooled_rmse <- mean(rmse_scores)
  
  pooled_correlation <- mean(correlation_scores)
  
  iteration_results <- data.frame(
    Fold = paste0(
      "Fold_",
      1:n_groups
    ),
    MSE = mse_scores,
    RMSE = rmse_scores,
    Correlation = correlation_scores
  )
  
  iteration_results <- rbind(
    iteration_results,
    data.frame(
      Fold = "Pooled",
      MSE = pooled_mse,
      RMSE = pooled_rmse,
      Correlation = pooled_correlation
    )
  )
  
  all_results[[iteration]] <- iteration_results
  
  cat(sprintf(
    "\nIteration %d Summary: MSE=%.4f, RMSE=%.4f, Cor=%.4f\n%s\n",
    iteration,
    pooled_mse,
    pooled_rmse,
    pooled_correlation,
    strrep("=", 40)
  ))
}

# ============================================================
# Combine and save all iterations
# ============================================================

final_results <- bind_rows(
  all_results,
  .id = "Iteration"
)

write.csv(
  final_results,
  file.path(output_dir, "Iteration_eval_results.csv"),
  row.names = FALSE
)

# Print final results
print(final_results)