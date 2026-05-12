# ################################################################################
# # DERIVATIVE ANALYSIS FOR NORMATIVE MODELS (GROWTH RATE & ACCELERATION)
# ################################################################################

# # Load required libraries
# library(mgcv)
# library(gamlss.dist)
# library(ggplot2)
# library(cowplot)
# library(dplyr)
# library(tidyr)

# # ==============================================================================
# # CONFIGURATION
# # ==============================================================================

# # Set paths
# models_file <- "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/final_fitted_models.rds"
# output_dir <- "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/derivative_plots"
# plot_dir <- file.path(output_dir, "plots")

# # Create output directories if they don't exist
# dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
# dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# # Define gestational age range for derivative calculation
# ga_min <- 21  # Minimum gestational age in weeks
# ga_max <- 40  # Maximum gestational age in weeks
# n_points <- 200  # Number of points for smooth derivative curves

# # Derivative calculation parameters
# h_step <- 0.01  # Step size for finite differences

# # Only focusing on the median line (P50)
# percentile_probs <- c(0.5)  
# percentile_names <- c("P50")

# # ==============================================================================
# # FUNCTION DEFINITIONS
# # ==============================================================================

# # Function to calculate derivative using finite differences
# calculate_derivative <- function(predictions, x_values, h) {
#   n <- length(x_values)
#   derivatives <- numeric(n)
  
#   for(i in 1:n) {
#     if(i < n) {
#       # Forward difference
#       derivatives[i] <- (predictions[i+1] - predictions[i]) / (x_values[i+1] - x_values[i])
#     } else {
#       # Backward difference for last point
#       derivatives[i] <- (predictions[i] - predictions[i-1]) / (x_values[i] - x_values[i-1])
#     }
#   }
  
#   return(derivatives)
# }

# # Function to calculate second derivative
# calculate_second_derivative <- function(first_derivatives, x_values) {
#   n <- length(x_values)
#   second_derivatives <- numeric(n)
  
#   for(i in 1:n) {
#     if(i < n) {
#       # Forward difference on first derivative
#       second_derivatives[i] <- (first_derivatives[i+1] - first_derivatives[i]) / 
#         (x_values[i+1] - x_values[i])
#     } else {
#       # Backward difference for last point
#       second_derivatives[i] <- (first_derivatives[i] - first_derivatives[i-1]) / 
#         (x_values[i] - x_values[i-1])
#     }
#   }
  
#   return(second_derivatives)
# }

# # Function to get quantile predictions from GAM model
# predict_quantiles <- function(model, newdata, percentile_probs) {
#   pred <- predict(model, newdata = newdata, type = "response")
#   family_name <- model$family$family
#   n_params <- ncol(pred)
#   quantile_preds <- matrix(NA, nrow = nrow(newdata), ncol = length(percentile_probs))
  
#   for(i in 1:length(percentile_probs)) {
#     if(family_name == "gaulss" || n_params == 2) {
#       quantile_preds[,i] <- qnorm(p = percentile_probs[i], 
#                                   mean = pred[,1], 
#                                   sd = exp(pred[,2]))
#     } else if(family_name == "shash" || n_params == 4) {
#       quantile_preds[,i] <- gamlss.dist::qSHASHo2(p = percentile_probs[i], 
#                                                   mu = pred[,1], 
#                                                   sigma = exp(pred[,2]),
#                                                   nu = pred[,3], 
#                                                   tau = exp(pred[,4]))
#     } else {
#       stop(sprintf("Unsupported family: %s with %d parameters", family_name, n_params))
#     }
#   }
#   return(quantile_preds)
# }

# # Function to analyze one model's median derivatives
# analyze_model_derivatives <- function(model, y_feature, x_seq) {
  
#   cat(sprintf("\nAnalyzing derivatives for: %s\n", y_feature))
  
#   if(length(model$smooth) > 0) {
#     predictor_var <- model$smooth[[1]]$term
#   } else {
#     model_names <- names(model$model)
#     predictor_var <- model_names[2]
#   }
  
#   pred_data <- data.frame(x_seq)
#   names(pred_data) <- predictor_var
  
#   quantile_preds <- predict_quantiles(model, pred_data, percentile_probs)
  
#   q_deriv <- calculate_derivative(quantile_preds[,1], x_seq, h_step)
#   q_deriv2 <- calculate_second_derivative(q_deriv, x_seq)
  
#   quantile_df <- data.frame(
#     age = x_seq,
#     quantile = "P50",
#     value = quantile_preds[,1],
#     first_derivative = q_deriv,
#     second_derivative = q_deriv2,
#     feature_original = y_feature
#   )
  
#   return(quantile_df)
# }

# # ==============================================================================
# # MAIN ANALYSIS LOOP
# # ==============================================================================

# cat("=================================================================\n")
# cat("DERIVATIVE ANALYSIS FOR NORMATIVE MODELS\n")
# cat("=================================================================\n\n")

# cat(sprintf("Loading models from: %s\n", models_file))

# final_models <- tryCatch({
#   readRDS(models_file)
# }, error = function(e) {
#   stop(sprintf("ERROR: Could not load models file: %s\n", e$message))
# })

# cat(sprintf("Successfully loaded %d models\n\n", length(final_models)))

# # Initialize storage for all results
# all_quantile_results <- list()

# # Create gestational age sequence
# x_seq <- seq(ga_min, ga_max, length.out = n_points)

# # Loop through each model in the list
# for(feature_name in names(final_models)) {
  
#   model <- final_models[[feature_name]]
  
#   if(is.null(model) || !inherits(model, "gam")) {
#     cat(sprintf("  Skipping invalid/null model: %s\n", feature_name))
#     next
#   }
  
#   results_df <- tryCatch({
#     analyze_model_derivatives(model, feature_name, x_seq)
#   }, error = function(e) {
#     cat(sprintf("  ERROR analyzing derivatives for %s: %s\n", feature_name, e$message))
#     return(NULL)
#   })
  
#   if(!is.null(results_df)) {
#     all_quantile_results[[feature_name]] <- results_df
#   }
# }

# # Combine all results into a single data frame
# combined_quantile_df <- do.call(rbind, all_quantile_results)

# # ==============================================================================
# # GENERATE OVERLAY PLOTS
# # ==============================================================================

# cat("\n=================================================================\n")
# cat("GENERATING COMBINED PLOTS\n")
# cat("=================================================================\n")

# # Define specific colors for each band
# feature_colors <- c(
#   "B4 Band Power" = "blue",
#   "B5 Band Power" = "green",
#   "B6 Band Power" = "red"
# )

# # Plot 1: Growth Rates (First Derivative)
# p_growth <- ggplot(combined_quantile_df, aes(x = age, y = first_derivative, color = feature_original)) +
#   geom_line(linewidth = 1.2) +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
#   scale_color_manual(values = feature_colors) +
#   labs(title = "Growth rate (first order derivatives)",
#        x = "Gestational Age (weeks)",
#        y = "Growth Rate (units/week)",
#        color = "Feature") +
#   theme_bw() +
#   theme(legend.position = "bottom",
#         plot.title = element_text(face = "bold", hjust = 0.5))

# # Plot 2: Acceleration (Second Derivative)
# p_accel <- ggplot(combined_quantile_df, aes(x = age, y = second_derivative, color = feature_original)) +
#   geom_line(linewidth = 1.2) +
#   geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
#   scale_color_manual(values = feature_colors) +
#   labs(title = "Acceleration (2nd order derivatives)",
#        x = "Gestational Age (weeks)",
#        y = "Acceleration (units/week²)",
#        color = "Feature") +
#   theme_bw() +
#   theme(legend.position = "bottom",
#         plot.title = element_text(face = "bold", hjust = 0.5))

# # Combine the 2 plots side-by-side
# combined_plot <- plot_grid(p_growth, p_accel, ncol = 2, labels = c("A", "B"))

# # Save the plots
# combined_filename <- file.path(plot_dir, "combined_features_derivatives.png")
# ggsave(filename = combined_filename, plot = combined_plot, width = 14, height = 6, dpi = 300)

# ggsave(filename = file.path(plot_dir, "features_growth_rate.png"), plot = p_growth, width = 8, height = 6, dpi = 300)
# ggsave(filename = file.path(plot_dir, "features_acceleration.png"), plot = p_accel, width = 8, height = 6, dpi = 300)

# cat(sprintf("\nCombined derivative plots saved to: %s\n", combined_filename))

# # Save the data to CSV
# csv_filename <- file.path(output_dir, "median_derivatives_combined.csv")
# write.csv(combined_quantile_df, file = csv_filename, row.names = FALSE)
# cat(sprintf("Derivative data saved to: %s\n", csv_filename))

# cat("\n=================================================================\n")
# cat("ANALYSIS COMPLETE\n")
# cat("=================================================================\n\n")


################################################################################
# DERIVATIVE ANALYSIS FOR NORMATIVE MODELS (GROWTH RATE & ACCELERATION)
################################################################################

# Load required libraries
library(mgcv)
library(gamlss.dist)
library(ggplot2)
library(cowplot)
library(dplyr)
library(tidyr)

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================

data_path <- "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/dhcp_ref_harmonized_log_transform_all_sites_harm_parms_only.csv"
models_file <- "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/final_fitted_models.rds"
output_dir <- "/home/INT/dienye.h/python_files/final_harmonization/log_transform/dhcp_ref/derivative_plots"
plot_dir <- file.path(output_dir, "plots")

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

ga_min <- 21  
ga_max <- 39  # CHANGED: Now stops exactly at 39 weeks
n_points <- 200  
h_step <- 0.01  

percentile_probs <- c(0.5)  
percentile_names <- c("P50")

# ==============================================================================
# DATA LOADING & RENAMING
# ==============================================================================
data <- as.data.frame(read.csv(data_path))

rename_features <- function(df) {
  colnames(df)[colnames(df) == "band_power_B4"] <- "B4 Band Power"
  colnames(df)[colnames(df) == "band_power_B5"] <- "B5 Band Power"
  colnames(df)[colnames(df) == "band_power_B6"] <- "B6 Band Power"
  colnames(df)[colnames(df) == "B4_surface_area_percentage"] <- "B4 Surface Area Percentage"
  colnames(df)[colnames(df) == "B5_surface_area_percentage"] <- "B5 Surface Area Percentage"
  colnames(df)[colnames(df) == "B6_surface_area_percentage"] <- "B6 Surface Area Percentage"
  colnames(df)[colnames(df) == "B4_band_relative_power"] <- "B4 Band Relative Power"
  colnames(df)[colnames(df) == "B5_band_relative_power"] <- "B5 Band Relative Power"
  colnames(df)[colnames(df) == "B6_band_relative_power"] <- "B6 Band Relative Power"
  return(df)
}

data <- rename_features(data)

# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================

calculate_derivative <- function(predictions, x_values, h) {
  n <- length(x_values)
  derivatives <- numeric(n)
  
  for(i in 1:n) {
    if(i < n) {
      derivatives[i] <- (predictions[i+1] - predictions[i]) / (x_values[i+1] - x_values[i])
    } else {
      derivatives[i] <- (predictions[i] - predictions[i-1]) / (x_values[i] - x_values[i-1])
    }
  }
  return(derivatives)
}

calculate_second_derivative <- function(first_derivatives, x_values) {
  n <- length(x_values)
  second_derivatives <- numeric(n)
  
  for(i in 1:n) {
    if(i < n) {
      second_derivatives[i] <- (first_derivatives[i+1] - first_derivatives[i]) / (x_values[i+1] - x_values[i])
    } else {
      second_derivatives[i] <- (first_derivatives[i] - first_derivatives[i-1]) / (x_values[i] - x_values[i-1])
    }
  }
  return(second_derivatives)
}

predict_quantiles <- function(model, newdata, percentile_probs) {
  pred <- predict(model, newdata = newdata, type = "response")
  family_name <- model$family$family
  n_params <- ncol(pred)
  quantile_preds <- matrix(NA, nrow = nrow(newdata), ncol = length(percentile_probs))
  
  for(i in 1:length(percentile_probs)) {
    if(family_name == "gaulss" || n_params == 2) {
      quantile_preds[,i] <- qnorm(p = percentile_probs[i], mean = pred[,1], sd = exp(pred[,2]))
    } else if(family_name == "shash" || n_params == 4) {
      quantile_preds[,i] <- gamlss.dist::qSHASHo2(p = percentile_probs[i], mu = pred[,1], sigma = exp(pred[,2]), nu = pred[,3], tau = exp(pred[,4]))
    } else {
      stop(sprintf("Unsupported family: %s with %d parameters", family_name, n_params))
    }
  }
  return(quantile_preds)
}

analyze_model_derivatives <- function(model, y_feature, x_seq, raw_data) {
  
  cat(sprintf("\nAnalyzing derivatives for: %s\n", y_feature))
  
  # Provide both possible predictor names so the GAM never panics
  pred_data <- data.frame(
    gestational_age = x_seq,
    x_train = x_seq
  )
  
  # 1. Get raw model predictions (may be in logit space)
  quantile_preds <- predict_quantiles(model, pred_data, percentile_probs)
  median_curve <- quantile_preds[,1]
  
  # 2. TRANSFORM BACK TO REAL BIOLOGICAL SPACE BEFORE DIFFERENTIATING
  if (grepl("relative.*power|percentage", y_feature, ignore.case = TRUE)) {
    median_curve <- plogis(median_curve)
    
    # Scale to 0-100 if the original data was percentages
    if (max(raw_data[[y_feature]], na.rm = TRUE) > 1.5) {
      median_curve <- median_curve * 100
    }
  }
  
  # 3. Calculate finite differences on the true biological curve
  q_deriv <- calculate_derivative(median_curve, x_seq, h_step)
  q_deriv2 <- calculate_second_derivative(q_deriv, x_seq)
  
  quantile_df <- data.frame(
    age = x_seq,
    quantile = "P50",
    value = median_curve,
    first_derivative = q_deriv,
    second_derivative = q_deriv2,
    feature_original = y_feature
  )
  
  return(quantile_df)
}

# ==============================================================================
# MAIN ANALYSIS LOOP
# ==============================================================================

cat("=================================================================\n")
cat("DERIVATIVE ANALYSIS FOR NORMATIVE MODELS\n")
cat("=================================================================\n\n")

final_models <- tryCatch({ readRDS(models_file) }, error = function(e) { stop(sprintf("ERROR: Could not load models file: %s\n", e$message)) })
cat(sprintf("Successfully loaded %d models\n\n", length(final_models)))

all_quantile_results <- list()
x_seq <- seq(ga_min, ga_max, length.out = n_points)

# Run derivative calculations for the 9 specific features
features_to_run <- c("B4 Band Power", "B5 Band Power", "B6 Band Power",
                     "B4 Surface Area Percentage", "B5 Surface Area Percentage", "B6 Surface Area Percentage",
                     "B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

for(feature_name in features_to_run) {
  model <- final_models[[feature_name]]
  if(is.null(model)) next
  
  results_df <- analyze_model_derivatives(model, feature_name, x_seq, data)
  if(!is.null(results_df)) {
    all_quantile_results[[feature_name]] <- results_df
  }
}

combined_quantile_df <- do.call(rbind, all_quantile_results)

csv_filename <- file.path(output_dir, "median_derivatives_combined.csv")
write.csv(combined_quantile_df, file = csv_filename, row.names = FALSE)
cat(sprintf("\nDerivative data saved to: %s\n", csv_filename))

# ==============================================================================
# GENERATE GROUPED OVERLAY PLOTS
# ==============================================================================

cat("\n=================================================================\n")
cat("GENERATING GROUPED COMBINED PLOTS\n")
cat("=================================================================\n")

feature_colors <- c(
  "B4 Band Power" = "blue", "B5 Band Power" = "green", "B6 Band Power" = "red",
  "B4 Surface Area Percentage" = "blue", "B5 Surface Area Percentage" = "green", "B6 Surface Area Percentage" = "red",
  "B4 Band Relative Power" = "blue", "B5 Band Relative Power" = "green", "B6 Band Relative Power" = "red"
)

plot_groups <- list(
  "Band_Power" = c("B4 Band Power", "B5 Band Power", "B6 Band Power"),
  "Surface_Area_Percentage" = c("B4 Surface Area Percentage", "B5 Surface Area Percentage", "B6 Surface Area Percentage"),
  "Band_Relative_Power" = c("B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")
)

for (group_name in names(plot_groups)) {
  
  target_features <- plot_groups[[group_name]]
  group_df <- combined_quantile_df %>% filter(feature_original %in% target_features)
  
  if(nrow(group_df) == 0) next
  
  # Plot 1: Growth Rates (First Derivative)
  p_growth <- ggplot(group_df, aes(x = age, y = first_derivative, color = feature_original)) +
    geom_line(linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(values = feature_colors) +
    scale_x_continuous(breaks = seq(20, 40, by = 2)) +  # <--- ADDED 2-WEEK INTERVALS
    labs(title = paste(gsub("_", " ", group_name), "- Growth Rate"),
         x = "Gestational Age (weeks)",
         y = "Growth Rate (units/week)",
         color = "Metrics") +
    theme_bw() +
    theme(legend.position = "bottom", plot.title = element_text(face = "bold", hjust = 0.5))
  
  # Plot 2: Acceleration (Second Derivative)
  p_accel <- ggplot(group_df, aes(x = age, y = second_derivative, color = feature_original)) +
    geom_line(linewidth = 1.2) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    scale_color_manual(values = feature_colors) +
    scale_x_continuous(breaks = seq(20, 40, by = 2)) +  # <--- ADDED 2-WEEK INTERVALS
    labs(title = paste(gsub("_", " ", group_name), "- Acceleration"),
         x = "Gestational Age (weeks)",
         y = "Acceleration (units/week²)",
         color = "Metrics") +
    theme_bw() +
    theme(legend.position = "bottom", plot.title = element_text(face = "bold", hjust = 0.5))
  
  # Combine side-by-side
  combined_plot <- plot_grid(p_growth, p_accel, ncol = 2, labels = c("A", "B"))
  
  # Save
  combined_filename <- file.path(plot_dir, paste0("derivatives_", group_name, ".png"))
  ggsave(filename = combined_filename, plot = combined_plot, width = 14, height = 6, dpi = 300)
}

cat("\nANALYSIS COMPLETE. All 3 grouped derivative plots saved successfully.\n")