library(caret)
library(data.table)
library(MASS)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)

# ============================================
# DATA LOADING AND PREPARATION
# ============================================

data <- as.data.frame(read.csv('/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/dhcp_marsfet_harmonized_params_only.csv'))

# ============================================
# 5-FOLD CROSS-VALIDATION SETUP
# ============================================

set.seed(123)  # For reproducibility
folds <- createFolds(data$gestational_age, k = 5, list = TRUE, returnTrain = FALSE)

# Initialize storage for cross-validation results
cv_results <- data.frame()
all_fitted_models <- list()

# ============================================
# FEATURE RENAMING FUNCTION
# ============================================

rename_features <- function(df) {
  colnames(df)[colnames(df) == "B4_band_relative_power_harm"] <- "B4 Band Relative Power"
  colnames(df)[colnames(df) == "B5_band_relative_power_harm"] <- "B5 Band Relative Power"
  colnames(df)[colnames(df) == "B6_band_relative_power_harm"] <- "B6 Band Relative Power"
  return(df)
}

# ============================================
# HELPER FUNCTIONS FOR EVALUATION
# ============================================

params_to_scores <- function(y, params){
  if(ncol(params) == 2){
    params <- cbind(params, rep(0, nrow(params)))
    params <- cbind(params, rep(log(1), nrow(params)))
  }
  
  log_densities <- gamlss.dist::dSHASHo2(y,
                                         mu=params[,1],
                                         sigma=exp(params[,2]),
                                         nu=params[,3],
                                         tau=exp(params[,4]),
                                         log=T)
  
  data.frame('log_densities' = log_densities)
}

# ============================================
# DEFINE Y FEATURES
# ============================================

y_values <- list("B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

# ============================================
# MAIN CROSS-VALIDATION LOOP
# ============================================

cat("Starting 5-fold cross-validation...\n")
cat("================================================\n")

for (feature in y_values) {
  cat("\nProcessing feature:", feature, "\n")
  cat("--------------------------------\n")
  
  # Store fold-specific results for this feature
  fold_results <- data.frame()
  
  for (fold_num in 1:5) {
    cat("  Fold", fold_num, "of 5...\n")
    
    # Split data
    test_indices <- folds[[fold_num]]
    train_indices <- setdiff(1:nrow(data), test_indices)
    
    df_train <- data[train_indices, ]
    df_test <- data[test_indices, ]
    
    # Rename features
    df_train <- rename_features(df_train)
    df_test <- rename_features(df_test)
    
    # Extract x and y
    x_train <- df_train$gestational_age
    y_train <- df_train[[feature]]
    x_test <- df_test$gestational_age
    y_test <- df_test[[feature]]
    
    # Skip if insufficient data
    if (length(y_train) < 20 || length(y_test) < 5) {
      cat("    Skipping - insufficient data\n")
      next
    }
    
    # Fit models (single set only)
    tryCatch({
      # Model 1a: Linear
      m1a <- gam(list(y_train ~ x_train,
                      ~ 1),
                 family = gaulss(),
                 optimizer = 'efs',
                 data = df_train)
      
      # Model 1b: Smooth mean
      m1b <- gam(list(y_train ~ s(x_train),
                      ~ 1),
                 family = gaulss(),
                 optimizer = 'efs',
                 data = df_train)
      
      # Model 1c: Smooth mean and variance
      m1c <- gam(list(y_train ~ s(x_train),
                      ~ s(x_train)),
                 family = gaulss(),
                 optimizer = 'efs',
                 data = df_train)
      
      # Model 1d: SHASH with constant shape
      m1d <- gam(list(y_train ~ s(x_train),
                      ~ s(x_train),
                      ~ 1,
                      ~ 1),
                 family = shash(),
                 optimizer = 'efs',
                 data = df_train)
      
      # Model 1e: SHASH full
      m1e <- gam(list(y_train ~ s(x_train),
                      ~ s(x_train),
                      ~ s(x_train),
                      ~ s(x_train)),
                 family = shash(),
                 optimizer = 'efs',
                 data = df_train)
      
      # Evaluate on test set
      test_data <- data.frame(x_train = x_test, y_train = y_test)
      
      models <- list(m1a = m1a, m1b = m1b, m1c = m1c, m1d = m1d, m1e = m1e)
      model_names <- c("m1a", "m1b", "m1c", "m1d", "m1e")
      
      for (i in 1:length(models)) {
        model <- models[[i]]
        model_name <- model_names[i]
        
        # Calculate BIC and AIC
        bic_val <- BIC(model)
        aic_val <- AIC(model)
        
        # Calculate LogScore on test set
        test_pred <- predict(model, newdata = test_data)
        test_scores <- params_to_scores(y_test, test_pred)
        log_score <- mean(test_scores$log_densities, na.rm = TRUE)
        
        # Store results
        fold_results <- rbind(fold_results, data.frame(
          Feature = feature,
          Model = model_name,
          Fold = fold_num,
          BIC = bic_val,
          AIC = aic_val,
          LogScore = log_score
        ))
      }
      
    }, error = function(e) {
      cat("    Error in fold", fold_num, ":", e$message, "\n")
    })
  }
  
  # Aggregate results across folds for this feature
  if (nrow(fold_results) > 0) {
    feature_summary <- fold_results %>%
      group_by(Feature, Model) %>%
      summarise(
        BIC_mean = mean(BIC, na.rm = TRUE),
        BIC_sd = sd(BIC, na.rm = TRUE),
        AIC_mean = mean(AIC, na.rm = TRUE),
        AIC_sd = sd(AIC, na.rm = TRUE),
        LogScore_mean = mean(LogScore, na.rm = TRUE),
        LogScore_sd = sd(LogScore, na.rm = TRUE),
        n_folds = n(),
        .groups = 'drop'
      )
    
    cv_results <- rbind(cv_results, feature_summary)
    
    # Print summary for this feature
    cat("\n  Summary for", feature, ":\n")
    print(feature_summary %>% 
            arrange(desc(LogScore_mean)) %>%
            select(Model, BIC_mean, AIC_mean, LogScore_mean))
  }
}

# ============================================
# SELECT BEST MODELS
# ============================================

cat("\n================================================\n")
cat("SELECTING BEST MODELS\n")
cat("================================================\n")

# Select best model for each feature based on LogScore (higher is better)
best_models <- cv_results %>%
  group_by(Feature) %>%
  slice_max(LogScore_mean, n = 1, with_ties = FALSE) %>%
  ungroup()

cat("\nBest models selected based on LogScore:\n")
print(table(best_models$Model))

# # Save CV results
# write.csv(cv_results, 
#           "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/cv_results_summary.csv",
#           row.names = FALSE)

# write.csv(best_models,
#           "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/best_models_cv.csv",
#           row.names = FALSE)

# ============================================
# TRAIN FINAL MODELS ON FULL DATASET
# ============================================

cat("\n================================================\n")
cat("TRAINING FINAL MODELS ON FULL DATASET\n")
cat("================================================\n")

# Split full dataset for final training (70/30 split)
set.seed(456)
train_idx_final <- createDataPartition(data$gestational_age, p = 0.7, list = FALSE)
df_train_final <- rename_features(data[train_idx_final, ])
df_test_final <- rename_features(data[-train_idx_final, ])

final_models <- list()
final_results <- data.frame()

for (i in 1:nrow(best_models)) {
  feature <- best_models$Feature[i]
  model_type <- best_models$Model[i]
  
  cat("\nTraining final model for", feature, "using", model_type, "\n")
  
  x_train <- df_train_final$gestational_age
  y_train <- df_train_final[[feature]]
  x_test <- df_test_final$gestational_age
  y_test <- df_test_final[[feature]]
  
  tryCatch({
    # Train the selected model type
    if (model_type == "m1a") {
      final_model <- gam(list(y_train ~ x_train, ~ 1),
                         family = gaulss(), optimizer = 'efs', data = df_train_final)
    } else if (model_type == "m1b") {
      final_model <- gam(list(y_train ~ s(x_train), ~ 1),
                         family = gaulss(), optimizer = 'efs', data = df_train_final)
    } else if (model_type == "m1c") {
      final_model <- gam(list(y_train ~ s(x_train), ~ s(x_train)),
                         family = gaulss(), optimizer = 'efs', data = df_train_final)
    } else if (model_type == "m1d") {
      final_model <- gam(list(y_train ~ s(x_train), ~ s(x_train), ~ 1, ~ 1),
                         family = shash(), optimizer = 'efs', data = df_train_final)
    } else if (model_type == "m1e") {
      final_model <- gam(list(y_train ~ s(x_train), ~ s(x_train), ~ s(x_train), ~ s(x_train)),
                         family = shash(), optimizer = 'efs', data = df_train_final)
    }
    
    # Store model
    final_models[[feature]] <- final_model
    
    # Evaluate on test set
    test_data <- data.frame(x_train = x_test, y_train = y_test)
    test_pred <- predict(final_model, newdata = test_data)
    test_scores <- params_to_scores(y_test, test_pred)
    
    # Store final performance
    final_results <- rbind(final_results, data.frame(
      Feature = feature,
      Model = model_type,
      BIC = BIC(final_model),
      AIC = AIC(final_model),
      LogScore_test = mean(test_scores$log_densities, na.rm = TRUE),
      CV_LogScore_mean = best_models$LogScore_mean[i],
      CV_LogScore_sd = best_models$LogScore_sd[i]
    ))
    
  }, error = function(e) {
    cat("  Error training final model:", e$message, "\n")
  })
}

# Save final results
# write.csv(final_results,
#           "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/final_model_performance.csv",
#           row.names = FALSE)

# Save final models
# saveRDS(final_models,
#         "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/final_fitted_models.rds")


cat("\n================================================\n")
cat("GENERATING COMBINED NORMATIVE PLOT\n")
cat("================================================\n")

dir.create("/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/normative_plots/abstract",
           showWarnings = FALSE, recursive = TRUE)

# Rename features in the full dataset
data <- rename_features(data)

# Define colors for each band
feature_colors <- c(
  "B4 Band Relative Power" = "blue",
  "B5 Band Relative Power" = "green",
  "B6 Band Relative Power" = "red"
)


# Initialize empty data frames for combined plotting
all_data <- data.frame()
all_quantiles <- data.frame()

# Function to get quantiles
params_to_quantiles <- function(params, quantiles) {
  if(ncol(params) == 2) {
    params <- cbind(params, matrix(0, nrow = nrow(params), ncol = 2))
  }
  return(as.data.frame(sapply(quantiles,
                              function(q) {
                                gamlss.dist::qSHASHo2(q, params[,1],
                                                      exp(params[,2]),
                                                      params[,3],
                                                      exp(params[,4]))
                              })))
}

quantiles <- pnorm(c(-2, -1, 0, 1, 2))
quantile_names <- c("Q2.3", "Q15.9", "Q50", "Q84.1", "Q97.7")

# Collect data for all features
for (feature in names(final_models)) {
  cat("\nProcessing:", feature, "\n")
  
  model <- final_models[[feature]]
  
  # Combine all data points (no training/test distinction)
  feature_data <- data.frame(
    x = data$gestational_age,
    y = data[[feature]],
    feature = feature
  )
  all_data <- rbind(all_data, feature_data)
  
  # Generate predictions for smooth curves
  x_range <- seq(min(data$gestational_age), max(data$gestational_age), length.out = 200)
  pred_data <- data.frame(x_train = x_range, y_train = rep(0, 200))
  
  preds <- predict(model, newdata = pred_data)
  quantile_preds <- params_to_quantiles(preds, quantiles)
  colnames(quantile_preds) <- quantile_names
  
  quantile_preds_long <- reshape2::melt(quantile_preds, variable.name = "quantile", value.name = "value")
  quantile_preds_long$x <- rep(x_range, 5)
  quantile_preds_long$feature <- feature
  
  all_quantiles <- rbind(all_quantiles, quantile_preds_long)
}

# Create grouping variable explicitly
all_quantiles$group_id <- paste(all_quantiles$quantile, all_quantiles$feature, sep = "_")

# Separate median from other quantiles for different styling
median_quantiles <- all_quantiles[all_quantiles$quantile == "Q50", ]
other_quantiles <- all_quantiles[all_quantiles$quantile != "Q50", ]

# Create combined plot
p_combined <- ggplot() +
  # Plot non-median quantiles with lighter alpha
  geom_line(data = other_quantiles,
            aes(x = x, y = value, group = group_id, color = feature),
            linewidth = 0.5, alpha = 0.3) +
  # Plot median quantile with bold lines
  geom_line(data = median_quantiles,
            aes(x = x, y = value, color = feature),
            linewidth = 1.2, alpha = 0.8) +
  # Plot all data points
  geom_point(data = all_data,
             aes(x = x, y = y, color = feature),
             size = 1.5, alpha = 0.4) +
  scale_color_manual(values = feature_colors,
                     name = "Band Relative Power") +
  scale_x_continuous(breaks = seq(20, 40, by = 2)) +  # Add this line for 2-week intervals
  labs(title = "Combined Normative Models: Band Relative Power",
       x = "Gestational Age (weeks)",
       y = "Relative Power") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        legend.position = "right",
        legend.title = element_text(face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10))

# Save combined plot
filename_combined <- "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/normative_plots/abstract/combined_band_relative power_normative.png"
ggsave(filename_combined, p_combined, width = 12, height = 7, dpi = 300)

cat("\nCombined plot saved to:", filename_combined, "\n")
# ============================================
# FINAL SUMMARY
# ============================================

cat("\n================================================\n")
cat("ANALYSIS COMPLETE!\n")
cat("================================================\n")
cat("\nSummary of best models:\n")
print(table(final_results$Model))
cat("\nMean performance across features:\n")
cat("Mean BIC:", round(mean(final_results$BIC), 2), "\n")
cat("Mean AIC:", round(mean(final_results$AIC), 2), "\n")
cat("Mean LogScore:", round(mean(final_results$LogScore_test, na.rm = TRUE), 3), "\n")
cat("\nAll results saved to /home/INT/dienye.h/gamlss_normative_paper-main/harmonization/\n")