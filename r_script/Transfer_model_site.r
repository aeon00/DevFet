library(mgcv)
library(gamlss)
library(gamlss.dist)
library(ggplot2)
library(dplyr)
library(reshape2)
library(patchwork)
library(cowplot)
library(tidyr)

# ============================================
# STEP 1: LOAD TRAINED MODELS AND DATA
# ============================================

cat("Loading trained models and preparing for site transfer...\n")
cat("================================================\n")

# Load the final fitted models from your previous analysis
final_models <- readRDS("/home/INT/dienye.h/gamlss_normative_paper-main/final_fitted_models.rds")

# Load new site data (you'll need to specify the path to your new site data)
# For demonstration, I'll show the structure needed
new_site_data <- read.csv('/home/INT/dienye.h/python_files/combined_dataset/marsfet_qc_filtered.csv')  # UPDATE THIS PATH

# Function to rename features (same as before)
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

new_site_data <- rename_features(new_site_data)

# Split new site data for calibration (use stratified sampling across gestational ages)
set.seed(789)

# Create age bins for stratified sampling
new_site_data$age_bin <- cut(new_site_data$gestational_age, 
                              breaks = quantile(new_site_data$gestational_age, 
                                              probs = seq(0, 1, by = 0.2), 
                                              na.rm = TRUE),
                              include.lowest = TRUE)

# Sample proportionally from each age bin
n_calibration <- min(100, floor(0.2 * nrow(new_site_data)))
samples_per_bin <- ceiling(n_calibration / length(unique(new_site_data$age_bin)))

calibration_idx <- c()
for(bin in unique(new_site_data$age_bin)) {
  bin_indices <- which(new_site_data$age_bin == bin)
  n_to_sample <- min(samples_per_bin, length(bin_indices))
  calibration_idx <- c(calibration_idx, 
                       sample(bin_indices, n_to_sample, replace = FALSE))
}

# Trim to exact calibration size if needed
if(length(calibration_idx) > n_calibration) {
  calibration_idx <- sample(calibration_idx, n_calibration, replace = FALSE)
}

calibration_data <- new_site_data[calibration_idx, ]
validation_data <- new_site_data[-calibration_idx, ]

# Remove the temporary age_bin column
calibration_data$age_bin <- NULL
validation_data$age_bin <- NULL
new_site_data$age_bin <- NULL

cat("\nNew site data summary:\n")
cat("Total samples:", nrow(new_site_data), "\n")
cat("Calibration samples:", nrow(calibration_data), "\n")
cat("Validation samples:", nrow(validation_data), "\n")
cat("Age range in calibration:", 
    range(calibration_data$gestational_age, na.rm = TRUE), "\n")
cat("Age range in full data:", 
    range(new_site_data$gestational_age, na.rm = TRUE), "\n\n")

# ============================================
# STEP 2: RECALIBRATION FUNCTIONS
# ============================================

recalibrate_model <- function(original_model, calibration_data, feature_name) {
  
  cat("Recalibrating model for:", feature_name, "\n")
  
  # Prepare data for prediction
  x_cal <- calibration_data$gestational_age
  y_cal <- calibration_data[[feature_name]]
  
  # Remove NA values
  valid_idx <- !is.na(y_cal) & !is.na(x_cal)
  x_cal <- x_cal[valid_idx]
  y_cal <- y_cal[valid_idx]
  
  if (length(y_cal) < 10) {
    cat("  Warning: Insufficient calibration data for", feature_name, "\n")
    return(NULL)
  }
  
  pred_data <- data.frame(x_train = x_cal, y_train = y_cal)
  
  # Get predictions from original model
  original_predictions <- predict(original_model, newdata = pred_data)
  
  # Check if model is GAM or GAMLSS based on predictions structure
  n_params <- ncol(original_predictions)
  
  # Extract fixed parameters if SHASH model
  if (n_params >= 4) {
    # SHASH model
    fitted_nu <- original_predictions[1, 3]  # nu is constant
    fitted_tau <- original_predictions[1, 4]  # tau is constant
  } else {
    # Gaussian model - add default nu and tau
    fitted_nu <- 0
    fitted_tau <- log(1)
  }
  
  # Create recalibration data
  recal_data <- data.frame(
    y = y_cal,
    predictions_mu = original_predictions[, 1],
    predictions_sigma = original_predictions[, 2]
  )
  
  # Fit recalibration model - only adjust intercepts
  tryCatch({
    recal_model <- gamlss(
      y ~ 1 + offset(predictions_mu),
      sigma.formula = ~ 1 + offset(predictions_sigma),
      nu.fix = TRUE,
      nu.start = fitted_nu,
      tau.fix = TRUE,
      tau.start = exp(fitted_tau),
      family = SHASHo2,
      data = recal_data,
      method = mixed(10, 100),  # CHANGE FROM mixed(5, 50)
      trace = FALSE,
      control = gamlss.control(n.cyc = 100, trace = FALSE)  # ADD THIS LINE
    )
    
    # Create adjusted model by updating intercepts
    adjusted_model <- original_model
    
    # Update intercepts
    adjusted_model$coefficients['(Intercept)'] <- 
      adjusted_model$coefficients['(Intercept)'] + recal_model$mu.coefficients['(Intercept)']
    
    if ('(Intercept).1' %in% names(adjusted_model$coefficients)) {
      adjusted_model$coefficients['(Intercept).1'] <- 
        adjusted_model$coefficients['(Intercept).1'] + recal_model$sigma.coefficients['(Intercept)']
    }
    
    cat("  Mu adjustment:", round(recal_model$mu.coefficients['(Intercept)'], 3), "\n")
    cat("  Sigma adjustment:", round(recal_model$sigma.coefficients['(Intercept)'], 3), "\n")
    
    return(list(
      adjusted_model = adjusted_model,
      mu_adjustment = recal_model$mu.coefficients['(Intercept)'],
      sigma_adjustment = recal_model$sigma.coefficients['(Intercept)']
    ))
    
  }, error = function(e) {
    cat("  Error in recalibration:", e$message, "\n")
    return(NULL)
  })
}

# ============================================
# STEP 3: PERFORM RECALIBRATION
# ============================================

cat("\n================================================\n")
cat("PERFORMING MODEL RECALIBRATION\n")
cat("================================================\n")

recalibrated_models <- list()
recalibration_results <- data.frame()

for (feature_name in names(final_models)) {
  
  result <- recalibrate_model(
    original_model = final_models[[feature_name]],
    calibration_data = calibration_data,
    feature_name = feature_name
  )
  
  if (!is.null(result)) {
    recalibrated_models[[feature_name]] <- result$adjusted_model
    
    recalibration_results <- rbind(recalibration_results, data.frame(
      Feature = feature_name,
      Mu_Adjustment = result$mu_adjustment,
      Sigma_Adjustment = result$sigma_adjustment
    ))
  }
}

# Save recalibration results
write.csv(recalibration_results,
          "/home/INT/dienye.h/gamlss_normative_paper-main/recalibration_adjustments.csv",
          row.names = FALSE)

# ============================================
# STEP 4: EVALUATION FUNCTION
# ============================================

evaluate_model_performance <- function(model, data, feature_name, model_type) {
  
  x <- data$gestational_age
  y <- data[[feature_name]]
  
  valid_idx <- !is.na(y) & !is.na(x)
  x <- x[valid_idx]
  y <- y[valid_idx]
  
  if (length(y) < 10) return(NULL)
  
  pred_data <- data.frame(x_train = x, y_train = y)
  predictions <- predict(model, newdata = pred_data)
  
  # Calculate log score
  if (ncol(predictions) == 2) {
    # Gaussian
    log_densities <- dnorm(y, mean = predictions[,1], 
                          sd = exp(predictions[,2]), log = TRUE)
  } else {
    # SHASH
    log_densities <- dSHASHo2(y,
                              mu = predictions[,1],
                              sigma = exp(predictions[,2]),
                              nu = predictions[,3],
                              tau = exp(predictions[,4]),
                              log = TRUE)
  }
  
  return(data.frame(
    Feature = feature_name,
    Model_Type = model_type,
    LogScore = mean(log_densities, na.rm = TRUE),
    N = length(y)
  ))
}

# ============================================
# STEP 5: COMPARE PERFORMANCE
# ============================================

cat("\n================================================\n")
cat("COMPARING MODEL PERFORMANCE\n")
cat("================================================\n")

performance_comparison <- data.frame()

for (feature_name in names(recalibrated_models)) {
  
  # Original model on validation data
  perf_original <- evaluate_model_performance(
    final_models[[feature_name]],
    validation_data,
    feature_name,
    "Original"
  )
  
  # Recalibrated model on validation data
  perf_recalibrated <- evaluate_model_performance(
    recalibrated_models[[feature_name]],
    validation_data,
    feature_name,
    "Recalibrated"
  )
  
  if (!is.null(perf_original) && !is.null(perf_recalibrated)) {
    performance_comparison <- rbind(performance_comparison, 
                                   perf_original, 
                                   perf_recalibrated)
  }
}

# Print comparison
comparison_summary <- reshape2::dcast(performance_comparison, 
                                      Feature ~ Model_Type, 
                                      value.var = "LogScore") %>%
  mutate(Improvement = Recalibrated - Original) %>%
  arrange(desc(Improvement))

cat("\nPerformance Comparison (LogScore):\n")
print(comparison_summary)

write.csv(comparison_summary,
          "/home/INT/dienye.h/gamlss_normative_paper-main/performance_comparison.csv",
          row.names = FALSE)

# ============================================
# STEP 6: PLOTTING FUNCTIONS
# ============================================

generate_quantile_curves <- function(model, x_range, feature_name) {
  
  x_seq <- seq(min(x_range), max(x_range), length.out = 200)
  pred_data <- data.frame(x_train = x_seq, y_train = rep(mean(x_seq), length(x_seq)))
  
  predictions <- predict(model, newdata = pred_data)
  
  # Generate quantiles
  quantiles <- pnorm(c(-2, -1, 0, 1, 2))
  
  if (ncol(predictions) == 2) {
    # Gaussian
    predictions <- cbind(predictions, matrix(0, nrow = nrow(predictions), ncol = 2))
  }
  
  quantile_curves <- as.data.frame(sapply(quantiles,
    function(q) {
      qSHASHo2(q, 
               predictions[,1],
               exp(predictions[,2]),
               predictions[,3],
               exp(predictions[,4]))
    }))
  
  quantile_curves$x <- x_seq
  quantile_curves_long <- melt(quantile_curves, id.vars = "x")
  
  return(quantile_curves_long)
}

# ============================================
# STEP 7: GENERATE PLOTS
# ============================================

cat("\n================================================\n")
cat("GENERATING COMPARISON PLOTS\n")
cat("================================================\n")

dir.create("/home/INT/dienye.h/gamlss_normative_paper-main/transfer_plots",
           showWarnings = FALSE, recursive = TRUE)

for (feature_name in names(recalibrated_models)) {
  
  cat("\nCreating plots for:", feature_name, "\n")
  
  # Get data
  x_cal <- calibration_data$gestational_age
  y_cal <- calibration_data[[feature_name]]
  x_val <- validation_data$gestational_age
  y_val <- validation_data[[feature_name]]
  
  # Remove NAs
  cal_valid <- !is.na(y_cal) & !is.na(x_cal)
  val_valid <- !is.na(y_val) & !is.na(x_val)
  
  x_cal <- x_cal[cal_valid]
  y_cal <- y_cal[cal_valid]
  x_val <- x_val[val_valid]
  y_val <- y_val[val_valid]
  
  if (length(y_cal) < 5 || length(y_val) < 5) {
    cat("  Skipping - insufficient data\n")
    next
  }
  
  x_range <- range(c(x_cal, x_val), na.rm = TRUE)
  
  # Generate curves
  original_curves <- generate_quantile_curves(final_models[[feature_name]], 
                                              x_range, feature_name)
  original_curves$Model <- "Original"
  
  recalibrated_curves <- generate_quantile_curves(recalibrated_models[[feature_name]], 
                                                  x_range, feature_name)
  recalibrated_curves$Model <- "Recalibrated"
  
  # Define visual parameters
  quantile_linetypes <- c("V1" = "dashed", "V2" = "dashed", "V3" = "solid", 
                          "V4" = "dashed", "V5" = "dashed")
  quantile_labels <- c("2.3%", "15.9%", "50%", "84.1%", "97.7%")
  
  # PLOT 1: Calibration data with both curves
  p1 <- ggplot() +
    geom_line(data = original_curves,
              aes(x = x, y = value, group = variable, linetype = variable),
              color = "blue", alpha = 0.5, linewidth = 0.7) +
    geom_line(data = recalibrated_curves,
              aes(x = x, y = value, group = variable, linetype = variable),
              color = "red", linewidth = 0.7) +
    geom_point(aes(x = x_cal, y = y_cal), 
               color = "black", size = 2, alpha = 0.6) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    labs(title = paste(feature_name, "- Calibration Data"),
         subtitle = "Blue: Original | Red: Recalibrated",
         x = "Gestational Age (weeks)",
         y = feature_name,
         linetype = "Quantile") +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5))
  
  # PLOT 2: Validation data with recalibrated curves
  p2 <- ggplot() +
    geom_line(data = recalibrated_curves,
              aes(x = x, y = value, group = variable, linetype = variable),
              color = "darkgreen", linewidth = 0.8) +
    geom_point(aes(x = x_val, y = y_val), 
               color = "coral", size = 1.5, alpha = 0.5) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    labs(title = paste(feature_name, "- Validation Data"),
         subtitle = "Recalibrated Model Applied",
         x = "Gestational Age (weeks)",
         y = feature_name,
         linetype = "Quantile") +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5))
  
  # PLOT 3: Curves only (clinical template)
  p3 <- ggplot() +
    geom_line(data = recalibrated_curves,
              aes(x = x, y = value, group = variable, linetype = variable),
              color = "black", linewidth = 1) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    labs(title = paste("Normative Curves:", feature_name),
         subtitle = "Clinical Reference (New Site)",
         x = "Gestational Age (weeks)",
         y = feature_name,
         linetype = "Percentile") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
          plot.subtitle = element_text(hjust = 0.5, size = 11),
          panel.grid.minor = element_blank(),
          panel.border = element_rect(colour = "black", fill = NA, size = 1),
          axis.text = element_text(size = 10),
          axis.title = element_text(size = 12),
          legend.title = element_text(size = 11),
          legend.text = element_text(size = 10))
  
  # PLOT 4: Before/After comparison
  cal_data_plot <- data.frame(x = x_cal, y = y_cal, Dataset = "Calibration")
  val_data_plot <- data.frame(x = x_val, y = y_val, Dataset = "Validation")
  combined_data <- rbind(cal_data_plot, val_data_plot)
  
  p4 <- ggplot() +
    geom_line(data = original_curves,
              aes(x = x, y = value, group = variable),
              color = "gray50", linetype = "dotted", linewidth = 0.5) +
    geom_line(data = recalibrated_curves,
              aes(x = x, y = value, group = variable, linetype = variable),
              color = "darkblue", linewidth = 0.8) +
    geom_point(data = combined_data,
               aes(x = x, y = y, color = Dataset, shape = Dataset),
               size = 2, alpha = 0.6) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    scale_color_manual(values = c("Calibration" = "red", "Validation" = "blue")) +
    labs(title = paste(feature_name, "- Full Comparison"),
         subtitle = "Dotted: Original | Solid: Recalibrated",
         x = "Gestational Age (weeks)",
         y = feature_name,
         linetype = "Quantile") +
    theme_bw()
  
  # Save individual plots
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/transfer_plots/",
                gsub(" ", "_", feature_name), "_calibration.png"),
         p1, width = 10, height = 6, dpi = 300)
  
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/transfer_plots/",
                gsub(" ", "_", feature_name), "_validation.png"),
         p2, width = 10, height = 6, dpi = 300)
  
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/transfer_plots/",
                gsub(" ", "_", feature_name), "_clinical_template.png"),
         p3, width = 10, height = 6, dpi = 300)
  
  # Combined plot using patchwork
  p_combined <- (p1 | p2) / (p3 | p4) +
    plot_annotation(
      title = paste("Model Transfer -", feature_name),
      subtitle = paste("Mu adjustment:", 
                      round(recalibration_results$Mu_Adjustment[recalibration_results$Feature == feature_name], 3),
                      "| Sigma adjustment:",
                      round(recalibration_results$Sigma_Adjustment[recalibration_results$Feature == feature_name], 3)),
      theme = theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
                   plot.subtitle = element_text(size = 12, hjust = 0.5))
    )
  
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/transfer_plots/",
                gsub(" ", "_", feature_name), "_combined.png"),
         p_combined, width = 16, height = 12, dpi = 300)
}

# ============================================
# STEP 8: SAVE RECALIBRATED MODELS
# ============================================

saveRDS(recalibrated_models,
        "/home/INT/dienye.h/gamlss_normative_paper-main/recalibrated_models.rds")

cat("\n================================================\n")
cat("MODEL TRANSFER COMPLETE!\n")
cat("================================================\n")
cat("\nFiles saved:\n")
cat("- recalibrated_models.rds (adjusted models for new site)\n")
cat("- recalibration_adjustments.csv (mu and sigma adjustments)\n")
cat("- performance_comparison.csv (performance metrics)\n")
cat("- transfer_plots/ (visualization directory)\n")
cat("\nKey outputs:\n")
cat("- Clinical templates (curves only) for practical use\n")
cat("- Comparison plots showing original vs recalibrated\n")
cat("- Validation plots showing fit on new site data\n")