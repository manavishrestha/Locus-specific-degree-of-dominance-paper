#01.10.2026

#Bayesian with Master CV

#LOEO dom yield
rm(list = ls())

setwd("/Users/manavi/Desktop/Manuscript_review/")

library(BGLR)
library(data.table)
library(dplyr)

set.seed(42)

# ============================================================
# Output foldr
# ============================================================

output_dir <- "/Users/manavi/Desktop/Manuscript_review/Bayesian_Dom_LOEO_new_CV_yield"

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

blues <- Final_merged[,1:5]
geno <- as.matrix(Final_merged[,6:2402])
env <- Final_merged[,2403:3056]
meta <- Final_merged[,3058:3059]
weather <- Final_merged[,3060:3075]

# Target
y <- blues[,4]  # Average_BLUE

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
    
    # Locations defined as training locations
    train_locations <- cv_fold %>%
      filter(Split == "train") %>%
      pull(Field_Location) %>%
      unique()
    
    # Locations defined as test locations
    test_locations <- cv_fold %>%
      filter(Split == "test") %>%
      pull(Field_Location) %>%
      unique()
    
    test_group <- unique(cv_fold$Test_group)
    
    # --------------------------------------------------------
    # Convert locations to observation indices
    # --------------------------------------------------------
    
    train_index <- Final_merged$Field_Location %in% train_locations
    test_index <- Final_merged$Field_Location %in% test_locations
    
    train_groups <- unique(cv_fold$Group[cv_fold$Split == "train"])
    
    cat(sprintf("  Fold %d:\n", fold))
    cat("Train Group(s):", paste(train_groups, collapse = ", "),
        "| Test Group:", paste(test_group, collapse = ", "), "\n")
    cat("Training locations:", length(train_locations), "\n")
    cat("Test locations:", length(test_locations), "\n")
    cat("Training observations:", sum(train_index), "\n")
    cat("Test observations:", sum(test_index), "\n")
    
    # --------------------------------------------------------
    # Mask test observations
    # --------------------------------------------------------
    
    y_train <- y
    y_train[test_index] <- NA
    
    geno_train <- geno[train_index, ]
    geno_test <- geno[test_index, ]
    
    # ========================================================
    # DOMINANCE TRANSFORMATION
    # ===================================================================
    
    # Check heterozygotes before
    het_before <- sum(geno == 1, na.rm = TRUE)
    
    # Dominance transformation
    snp_mean = array(0, dim = c(ncol(geno_train), 3))
    
    for (snp in 1:ncol(geno_train)) {
      snp_mean[snp,1] = mean(
        y_train[geno_train[,snp] == 0],
        na.rm = TRUE
      )
      
      snp_mean[snp,2] = mean(
        y_train[geno_train[,snp] == 1],
        na.rm = TRUE
      )
      
      snp_mean[snp,3] = mean(
        y_train[geno_train[,snp] == 2],
        na.rm = TRUE
      )
    }
    
    snp_mean[is.na(snp_mean)] <- 0
    
    snp_mean_temp = snp_mean
    
    to_flip = snp_mean_temp[,1] > snp_mean_temp[,3]
    
    snp_mean[to_flip,1] = snp_mean_temp[to_flip,3]
    snp_mean[to_flip,3] = snp_mean_temp[to_flip,1]
    
    geno_train[,to_flip] = abs(geno_train[,to_flip] - 2)
    geno_test[,to_flip] = abs(geno_test[,to_flip] - 2)
    
    d = (snp_mean[,2] - snp_mean[,1]) /
      (snp_mean[,3] - snp_mean[,1]) * 2
    
    d[is.na(d)] = 1
    d[d >= 2] = 2
    d[d <= 0] = 0
    
    for (snp in 1:ncol(geno_train)) {
      
      het_train = which(
        geno_train[,snp] == 1
      )
      
      geno_train[het_train, snp] = d[snp]
      
      het_test = which(
        geno_test[,snp] == 1
      )
      
      geno_test[het_test, snp] = d[snp]
    }
    
    het_after <- sum(
      geno_train == 1,
      na.rm = TRUE
    ) +
      sum(
        geno_test == 1,
        na.rm = TRUE
      )
    
    cat(
      "Heterozygotes coded as 1 before:",
      het_before,
      "| after:",
      het_after,
      "\n"
    )
    
    geno_full <- geno
    
    geno_full[train_index, ] <- geno_train
    geno_full[test_index, ] <- geno_test
    
    # ========================================================
    # BGLR
    # ========================================================
    
    ETA <- list(
      list(
        X = geno_full,
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
    # Predictions
    # --------------------------------------------------------
    
    y_pred <- model$yHat[test_index]
    y_true <- y[test_index]
    
    # --------------------------------------------------------
    # Metrics
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
    
    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------
    
    pred_results <- data.frame(
      Iteration = iteration,
      Fold = fold,
      Actual = y_true,
      Predicted = y_pred
    )
    
    write.csv(
      pred_results,
      file.path(
        output_dir,
        sprintf(
          "dom_env_iter_%d_fold_%d_results_yield.csv",
          iteration,
          fold
        )
      ),
      row.names = FALSE
    )
    
    cat(sprintf(
      "    MSE: %.4f\n    RMSE: %.4f\n    Correlation: %.4f\n",
      mse,
      rmse,
      correlation
    ))
  }
  
  # ========================================================
  # Summary for iteration
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
    "Iteration %d Summary: MSE=%.4f, RMSE=%.4f, Cor=%.4f\n%s\n",
    iteration,
    pooled_mse,
    pooled_rmse,
    pooled_correlation,
    strrep("=", 40)
  ))
}

# ============================================================
# Save final results
# ============================================================

final_results <- bind_rows(
  all_results,
  .id = "Iteration"
)

write.csv(
  final_results,
  file.path(
    output_dir,
    "dom_env_all_iterations_summary_yield.csv"
  ),
  row.names = FALSE
)

print(final_results)

# ============================================================
# Extract only pooled rows
# ============================================================

pooled_df <- final_results %>%
  filter(Fold == "Pooled")

print(pooled_df)

write.csv(
  pooled_df,
  file.path(
    output_dir,
    "dom_pooled_entries_yeild.csv"
  ),
  row.names = FALSE
)

# ============================================================
# Mean across iterations
# ============================================================

mean_row <- pooled_df %>%
  summarise(
    Iteration = "joint",
    Fold = "mean",
    MSE = mean(MSE),
    RMSE = mean(RMSE),
    Correlation = mean(Correlation)
  )

# Append mean row
pooled_df <- bind_rows(
  pooled_df,
  mean_row
)

write.csv(
  pooled_df,
  file.path(
    output_dir,
    "dom_mean_entries_yield.csv"
  ),
  row.names = FALSE
)

