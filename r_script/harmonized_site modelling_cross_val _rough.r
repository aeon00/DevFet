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
# 1. DEFINE ALL FILE PATHS
# ============================================
# Define these once here. Nothing is hardcoded below this block!

# Input Data
DATA_PATH <- "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/dhcp_ref_harmonized_log_transform_all_sites_harm_parms_only.csv"

# Output Directories & Files
BASE_OUT_DIR <- "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref"
PLOT_OUT_DIR <- file.path(BASE_OUT_DIR, "normative_plots")

# CSV Exports
ALL_ROUNDS_CSV <- file.path(BASE_OUT_DIR, "all_20_rounds_raw_results.csv") 
CV_RESULTS_CSV <- file.path(BASE_OUT_DIR, "cv_results_summary.csv")
BEST_MODELS_CSV <- file.path(BASE_OUT_DIR, "best_models_cv.csv")
FINAL_RESULTS_CSV <- file.path(BASE_OUT_DIR, "final_model_performance.csv")
FINAL_MODELS_RDS <- file.path(BASE_OUT_DIR, "final_fitted_models.rds")

# Create output directory for plots if it doesn't exist
dir.create(PLOT_OUT_DIR, showWarnings = FALSE, recursive = TRUE)


# ============================================
# 2. DATA LOADING AND RENAMING
# ============================================
data <- as.data.frame(read.csv(DATA_PATH))

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

data <- rename_features(data)

# ============================================
# 3. HELPER FUNCTIONS
# ============================================
params_to_scores <- function(y, params){
  if(ncol(params) == 2){
    params <- cbind(params, rep(0, nrow(params)))
    params <- cbind(params, rep(log(1), nrow(params)))
  }
  log_densities <- gamlss.dist::dSHASHo2(y, mu=params[,1], sigma=exp(params[,2]), nu=params[,3], tau=exp(params[,4]), log=T)
  data.frame('log_densities' = log_densities)
}

params_to_quantiles <- function(params, quantiles) {
  if(ncol(params) == 2) {
    params <- cbind(params, matrix(0, nrow = nrow(params), ncol = 2))
  }
  return(as.data.frame(sapply(quantiles, function(q) {
    gamlss.dist::qSHASHo2(q, params[,1], exp(params[,2]), params[,3], exp(params[,4]))
  })))
}

# ============================================
# 4. DEFINE Y FEATURES (All 23 metrics)
# ============================================
y_values <- list("Surface Area cm2", "Folding Power", "B4 Vertex Percentage", "B5 Vertex Percentage",
                 "B6 Vertex Percentage", "Band_parcels B4", "Band Parcels B5", "Band Parcels B6", 
                 "Hemispheric Volume", "Gyrification Index", "Hull Area", "B4 Surface Area", 
                 "B5 Surface Area", "B6 Surface Area", "B4 Surface Area Percentage", 
                 "B5 Surface Area Percentage", "B6 Surface Area Percentage", 
                 "B4 Band Power", "B5 Band Power", "B6 Band Power",
                 "B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

# ============================================
# 5. 20-ROUND BOOTSTRAP LOOP (Model Selection)
# ============================================
cat("\n================================================\n")
cat("Starting 20 rounds of STRATIFIED 80/20 splits...\n")
cat("================================================\n")

set.seed(123)
cv_results <- data.frame()
all_rounds_data <- data.frame() # Stores granular data across all rounds

for (feature in y_values) {
  cat("\nProcessing feature:", feature, "\n")
  round_results <- data.frame()
  
  for (round_num in 1:20) {
    
    # 1. Create a Stratified Split based on gestational_age
    train_indices <- createDataPartition(data$gestational_age, p = 0.8, groups=15, list = FALSE)
    train_indices <- as.vector(train_indices)
    
    # 2. Assign remaining 20% to test set
    test_indices <- setdiff(seq_len(nrow(data)), train_indices)
    
    df_train <- data[train_indices, ]
    df_test <- data[test_indices, ]
    
    x_train <- df_train$gestational_age
    y_train <- df_train[[feature]]
    x_test <- df_test$gestational_age
    y_test <- df_test[[feature]]
    
    if (length(y_train) < 20 || length(y_test) < 5) { next }
    
    tryCatch({
      # A. TARGET TRANSFORMATION
      if (grepl("relative.*power|percentage", feature, ignore.case = TRUE)) {
        if (max(y_train, na.rm = TRUE) > 1.5) {
          y_train <- y_train / 100
          y_test <- y_test / 100
        }
        n_train <- length(y_train); y_train_sq <- (y_train * (n_train - 1) + 0.5) / n_train
        n_test <- length(y_test); y_test_sq <- (y_test * (n_test - 1) + 0.5) / n_test
        
        df_train$y_train_target <- qlogis(y_train_sq)
        y_test_target <- qlogis(y_test_sq)
      } else {
        df_train$y_train_target <- y_train
        y_test_target <- y_test
      }
      
      # B. TRAIN MODELS
      m1a <- gam(list(y_train_target ~ x_train, ~ 1), family = gaulss(), optimizer = 'efs', data = df_train)
      m1b <- gam(list(y_train_target ~ s(x_train), ~ 1), family = gaulss(), optimizer = 'efs', data = df_train)
      m1c <- gam(list(y_train_target ~ s(x_train), ~ s(x_train)), family = gaulss(), optimizer = 'efs', data = df_train)
      m1d <- gam(list(y_train_target ~ s(x_train), ~ s(x_train), ~ 1, ~ 1), family = shash(), optimizer = 'efs', data = df_train)
      m1e <- gam(list(y_train_target ~ s(x_train), ~ s(x_train), ~ s(x_train), ~ s(x_train)), family = shash(), optimizer = 'efs', data = df_train)
      
      models <- list(m1a = m1a, m1b = m1b, m1c = m1c, m1d = m1d, m1e = m1e)
      model_names <- c("m1a", "m1b", "m1c", "m1d", "m1e")
      test_data <- data.frame(x_train = x_test)
      
      # C. EVALUATION
      for (i in 1:length(models)) {
        model <- models[[i]]
        test_pred <- predict(model, newdata = test_data)
        test_scores <- params_to_scores(y_test_target, test_pred)
        
        round_results <- rbind(round_results, data.frame(
          Feature = feature,
          Model = model_names[i],
          Round = round_num,
          BIC = BIC(model),
          AIC = AIC(model),
          LogScore = mean(test_scores$log_densities, na.rm = TRUE)
        ))
      }
    }, error = function(e) { cat("    Error in round", round_num, ":", e$message, "\n") })
  }
  
  if (nrow(round_results) > 0) {
    all_rounds_data <- rbind(all_rounds_data, round_results)
    
    feature_summary <- round_results %>%
      group_by(Feature, Model) %>%
      summarise(BIC_mean = mean(BIC, na.rm = TRUE), AIC_mean = mean(AIC, na.rm = TRUE),
                LogScore_mean = mean(LogScore, na.rm = TRUE), LogScore_sd = sd(LogScore, na.rm = TRUE), .groups = 'drop')
    cv_results <- rbind(cv_results, feature_summary)
    
    cat("\n  Summary for", feature, ":\n")
    print(feature_summary %>% 
            arrange(desc(LogScore_mean)) %>%
            select(Model, BIC_mean, AIC_mean, LogScore_mean))
  }
}

best_models <- cv_results %>% group_by(Feature) %>% slice_max(LogScore_mean, n = 1, with_ties = FALSE) %>% ungroup()

write.csv(all_rounds_data, ALL_ROUNDS_CSV, row.names = FALSE)
write.csv(cv_results, CV_RESULTS_CSV, row.names = FALSE)
write.csv(best_models, BEST_MODELS_CSV, row.names = FALSE)

# ============================================
# 6. TRAIN FINAL MODELS
# ============================================
cat("\n================================================\n")
cat("TRAINING FINAL MODELS ON FULL DATASET\n")
cat("================================================\n")

set.seed(456)
train_idx_final <- createDataPartition(data$gestational_age, p = 0.8, groups=15, list = FALSE)
df_train_final <- data[train_idx_final, ]
df_test_final <- data[-train_idx_final, ]

final_models <- list()
final_results <- data.frame()

for (i in 1:nrow(best_models)) {
  feature <- best_models$Feature[i]
  model_type <- best_models$Model[i]
  
  x_train <- df_train_final$gestational_age
  y_train <- df_train_final[[feature]]
  x_test <- df_test_final$gestational_age
  y_test <- df_test_final[[feature]]
  
  tryCatch({
    # A. TARGET TRANSFORMATION
    if (grepl("relative.*power|percentage", feature, ignore.case = TRUE)) {
      if (max(y_train, na.rm = TRUE) > 1.5) {
        y_train <- y_train / 100
        y_test <- y_test / 100
      }
      n_train <- length(y_train); y_train_sq <- (y_train * (n_train - 1) + 0.5) / n_train
      n_test <- length(y_test); y_test_sq <- (y_test * (n_test - 1) + 0.5) / n_test
      
      df_train_final$y_train_target <- qlogis(y_train_sq)
      y_test_target <- qlogis(y_test_sq)
    } else {
      df_train_final$y_train_target <- y_train
      y_test_target <- y_test
    }
    
    # B. TRAIN
    if (model_type == "m1a") { final_model <- gam(list(y_train_target ~ x_train, ~ 1), family = gaulss(), optimizer = 'efs', data = df_train_final) }
    else if (model_type == "m1b") { final_model <- gam(list(y_train_target ~ s(x_train), ~ 1), family = gaulss(), optimizer = 'efs', data = df_train_final) }
    else if (model_type == "m1c") { final_model <- gam(list(y_train_target ~ s(x_train), ~ s(x_train)), family = gaulss(), optimizer = 'efs', data = df_train_final) }
    else if (model_type == "m1d") { final_model <- gam(list(y_train_target ~ s(x_train), ~ s(x_train), ~ 1, ~ 1), family = shash(), optimizer = 'efs', data = df_train_final) }
    else if (model_type == "m1e") { final_model <- gam(list(y_train_target ~ s(x_train), ~ s(x_train), ~ s(x_train), ~ s(x_train)), family = shash(), optimizer = 'efs', data = df_train_final) }
    
    final_models[[feature]] <- final_model
    
    # C. EVALUATE
    test_data <- data.frame(x_train = x_test)
    test_pred <- predict(final_model, newdata = test_data)
    test_scores <- params_to_scores(y_test_target, test_pred)
    
    final_results <- rbind(final_results, data.frame(
      Feature = feature, Model = model_type, BIC = BIC(final_model), AIC = AIC(final_model),
      LogScore_test = mean(test_scores$log_densities, na.rm = TRUE)
    ))
  }, error = function(e) { cat("  Error training final model:", e$message, "\n") })
}

write.csv(final_results, FINAL_RESULTS_CSV, row.names = FALSE)
saveRDS(final_models, FINAL_MODELS_RDS)


# ============================================
# 7. GENERATE INDIVIDUAL NORMATIVE PLOTS
# ============================================
cat("\n================================================\n")
cat("GENERATING INDIVIDUAL NORMATIVE PLOTS\n")
cat("================================================\n")

quantiles <- pnorm(c(-2, -1, 0, 1, 2))
quantile_names <- c("2.3%", "15.9%", "50%", "84.1%", "97.7%")
quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")

for (feature in names(final_models)) {
  cat("Rendering plot for:", feature, "\n")
  
  model <- final_models[[feature]]
  model_type <- final_results$Model[final_results$Feature == feature]
  
  # Prepare actual data points for the scatter plot, distinguished by Training vs Test
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
  
  # Generate continuous curve range for smooth lines
  x_range <- seq(min(data$gestational_age), max(data$gestational_age), length.out = 200)
  pred_data <- data.frame(x_train = x_range)
  
  preds <- predict(model, newdata = pred_data)
  quantile_preds <- params_to_quantiles(preds, quantiles)
  
  # THE PLOT FIX: Inverse logit and 0-100 scaling for bounded variables
  if (grepl("relative.*power|percentage", feature, ignore.case = TRUE)) {
    quantile_preds_matrix <- plogis(as.matrix(quantile_preds))
    if (max(data[[feature]], na.rm = TRUE) > 1.5) {
      quantile_preds <- as.data.frame(quantile_preds_matrix * 100)
    } else {
      quantile_preds <- as.data.frame(quantile_preds_matrix)
    }
  }
  
  colnames(quantile_preds) <- quantile_names
  quantile_preds_long <- reshape2::melt(quantile_preds, variable.name = "quantile", value.name = "value")
  quantile_preds_long$x <- rep(x_range, 5)
  
  # Create plot
  p <- ggplot() +
    # Quantile lines
    geom_line(data = quantile_preds_long,
              aes(x = x, y = value, linetype = quantile),
              linewidth = 0.7, color = "gray30", alpha = 0.8) +
    # Raw data points
    geom_point(data = combined_data,
               aes(x = x, y = y, color = dataset),
               size = 2, alpha = 0.5) +
    scale_color_manual(values = c("Training" = "blue", "Test" = "red"), name = "Dataset") +
    scale_linetype_manual(values = setNames(quantile_linetypes, quantile_names), name = "Quantiles") +
    labs(title = paste("Normative Model:", feature),
         subtitle = paste("Model:", model_type, 
                          "| BIC:", round(final_results$BIC[final_results$Feature == feature], 1),
                          "| AIC:", round(final_results$AIC[final_results$Feature == feature], 1),
                          "| LogScore:", round(final_results$LogScore_test[final_results$Feature == feature], 3)),
         x = "PMA (weeks)",
         y = feature) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5),
          legend.position = "right")
  
  # Save individual plot safely
  safe_filename <- gsub(" ", "_", feature)
  ggsave(file.path(PLOT_OUT_DIR, paste0(safe_filename, "_normative.png")), p, width = 10, height = 6, dpi = 300)
}

cat("\nPIPELINE COMPLETE! Models trained, saved, and all individual plots generated successfully.\n")