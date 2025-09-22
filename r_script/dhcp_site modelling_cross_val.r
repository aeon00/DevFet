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

data <- as.data.frame(read.csv('/home/INT/dienye.h/python_files/combined_dataset/dhcp_qc_filtered.csv'))

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
  colnames(df)[colnames(df) == "surface_area_cm2"] <- "Surface Area cm2"
  colnames(df)[colnames(df) == "analyze_folding_power"] <- "Folding Power"
  colnames(df)[colnames(df) == "B4_vertex_percentage"] <- "B4 Vertex Percentage"
  colnames(df)[colnames(df) == "B5_vertex_percentage"] <- "B5 Vertex Percentage"
  colnames(df)[colnames(df) == "B6_vertex_percentage"] <- "B6 Vertex Percentage"
  colnames(df)[colnames(df) == "band_parcels_B4"] <- "Band_parcels B4"
  colnames(df)[colnames(df) == "band_parcels_B5"] <- "Band Parcels B5"
  colnames(df)[colnames(df) == "band_parcels_B6"] <- "Band Parcels B6"
  colnames(df)[colnames(df) == "volume_ml"] <- "Hemispheric Volume"
  colnames(df)[colnames(df) == "gyrification_index"] <- "Gyrification Index"
  colnames(df)[colnames(df) == "hull_area"] <- "Hull Area"
  colnames(df)[colnames(df) == "B4_surface_area"] <- "B4 Surface Area"
  colnames(df)[colnames(df) == "B5_surface_area"] <- "B5 Surface Area"
  colnames(df)[colnames(df) == "B6_surface_area"] <- "B6 Surface Area"
  colnames(df)[colnames(df) == "B4_surface_area_percentage"] <- "B4 Surface Area Percentage"
  colnames(df)[colnames(df) == "B5_surface_area_percentage"] <- "B5 Surface Area Percentage"
  colnames(df)[colnames(df) == "B6_surface_area_percentage"] <- "B6 Surface Area Percentage"
  colnames(df)[colnames(df) == "band_power_B4"] <- "B4 Band Power"
  colnames(df)[colnames(df) == "band_power_B5"] <- "B5 Band Power"
  colnames(df)[colnames(df) == "band_power_B6"] <- "B6 Band Power"
  colnames(df)[colnames(df) == "B4_band_relative_power"] <- "B4 Band Relative Power"
  colnames(df)[colnames(df) == "B5_band_relative_power"] <- "B5 Band Relative Power"
  colnames(df)[colnames(df) == "B6_band_relative_power"] <- "B6 Band Relative Power"
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

y_values <- list("Surface Area cm2", "Folding Power", "B4 Vertex Percentage", "B5 Vertex Percentage",
                "B6 Vertex Percentage", "Band_parcels B4", "Band Parcels B5", "Band Parcels B6", 
                "Hemispheric Volume", "Gyrification Index", "Hull Area", "B4 Surface Area", 
                "B5 Surface Area", "B6 Surface Area", "B4 Surface Area Percentage", 
                "B5 Surface Area Percentage", "B6 Surface Area Percentage", 
                "B4 Band Power", "B5 Band Power", "B6 Band Power",
                "B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

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

# Save CV results
write.csv(cv_results, 
          "/home/INT/dienye.h/gamlss_normative_paper-main/cv_results_summary.csv",
          row.names = FALSE)

write.csv(best_models,
          "/home/INT/dienye.h/gamlss_normative_paper-main/best_models_cv.csv",
          row.names = FALSE)

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
write.csv(final_results,
          "/home/INT/dienye.h/gamlss_normative_paper-main/final_model_performance.csv",
          row.names = FALSE)

# Save final models
saveRDS(final_models,
        "/home/INT/dienye.h/gamlss_normative_paper-main/final_fitted_models.rds")

# ============================================
# GENERATE NORMATIVE PLOTS
# ============================================

cat("\n================================================\n")
cat("GENERATING NORMATIVE PLOTS\n")
cat("================================================\n")

dir.create("/home/INT/dienye.h/gamlss_normative_paper-main/normative_plots",
           showWarnings = FALSE, recursive = TRUE)

for (feature in names(final_models)) {
  cat("\nCreating plots for:", feature, "\n")
  
  model <- final_models[[feature]]
  model_type <- final_results$Model[final_results$Feature == feature]
  
  # Get predictions for plotting
  plot_data_train <- data.frame(
    x = df_train_final$gestational_age,
    y = df_train_final[[feature]],
    dataset = "Training"
  )
  
  plot_data_test <- data.frame(
    x = df_test_final$gestational_age,
    y = df_test_final[[feature]],
    dataset = "Test"
  )
  
  combined_data <- rbind(plot_data_train, plot_data_test)
  
  # Generate quantile predictions
  train_pred <- predict(model, newdata = data.frame(x_train = plot_data_train$x,
                                                    y_train = plot_data_train$y))
  
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
  quantile_labels <- c("2.3%", "15.9%", "50%", "84.1%", "97.7%")
  quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")
  
  train_quantiles <- params_to_quantiles(train_pred, quantiles)
  train_quantiles_long <- reshape2::melt(train_quantiles, variable.name = "quantile", value.name = "value")
  train_quantiles_long$x <- rep(plot_data_train$x, 5)
  
  # Create plot
  p <- ggplot(combined_data) +
    geom_line(data = train_quantiles_long,
              aes(x = x, y = value, group = quantile, linetype = quantile),
              linewidth = 0.7, color = "gray30", alpha = 0.8) +
    geom_point(aes(x = x, y = y, color = dataset), size = 2, alpha = 0.6) +
    scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    labs(title = paste("Normative Model:", feature),
         subtitle = paste("Model:", model_type, 
                         "| BIC:", round(final_results$BIC[final_results$Feature == feature], 1),
                         "| AIC:", round(final_results$AIC[final_results$Feature == feature], 1),
                         "| LogScore:", round(final_results$LogScore_test[final_results$Feature == feature], 3)),
         x = "Gestational Age (weeks)",
         y = feature) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5),
          legend.position = "right")
  
  # Save plot
  filename <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/normative_plots/",
                    gsub(" ", "_", feature), "_normative.png")
  ggsave(filename, p, width = 10, height = 6, dpi = 300)
}

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
cat("\nAll results saved to /home/INT/dienye.h/gamlss_normative_paper-main/\n")