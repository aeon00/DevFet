################################################################################
# DERIVATIVE ANALYSIS FOR NORMATIVE MODELS
# Author: Your Name
# Date: 2025
# Purpose: Load saved GAM models, calculate derivatives, plot, and save results
################################################################################

# Load required libraries
library(mgcv)
library(gamlss.dist)
library(ggplot2)
library(cowplot)
library(dplyr)
library(tidyr)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Set paths
models_file <- "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/final_fitted_models.rds"
output_dir <- "/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/derivatives_output/"
plot_dir <- file.path(output_dir, "plots")

# Create output directories if they don't exist
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)

# Define gestational age range for derivative calculation
ga_min <- 21  # Minimum gestational age in weeks
ga_max <- 40  # Maximum gestational age in weeks
n_points <- 200  # Number of points for smooth derivative curves

# Derivative calculation parameters
h_step <- 0.01  # Step size for finite differences

# Percentiles for derivative analysis (expressed as probabilities 0-1)
percentile_probs <- c(0.03, 0.16, 0.5, 0.84, 0.97)  # -2SD, -1SD, median, +1SD, +2SD
percentile_names <- c("P3", "P16", "P50", "P84", "P97")

# ==============================================================================
# FUNCTION DEFINITIONS
# ==============================================================================

# Function to calculate derivative using finite differences
calculate_derivative <- function(predictions, x_values, h) {
  n <- length(x_values)
  derivatives <- numeric(n)
  
  for(i in 1:n) {
    if(i < n) {
      # Forward difference
      derivatives[i] <- (predictions[i+1] - predictions[i]) / (x_values[i+1] - x_values[i])
    } else {
      # Backward difference for last point
      derivatives[i] <- (predictions[i] - predictions[i-1]) / (x_values[i] - x_values[i-1])
    }
  }
  
  return(derivatives)
}

# Function to calculate second derivative
calculate_second_derivative <- function(first_derivatives, x_values) {
  n <- length(x_values)
  second_derivatives <- numeric(n)
  
  for(i in 1:n) {
    if(i < n) {
      # Forward difference on first derivative
      second_derivatives[i] <- (first_derivatives[i+1] - first_derivatives[i]) / 
        (x_values[i+1] - x_values[i])
    } else {
      # Backward difference for last point
      second_derivatives[i] <- (first_derivatives[i] - first_derivatives[i-1]) / 
        (x_values[i] - x_values[i-1])
    }
  }
  
  return(second_derivatives)
}

# Function to get quantile predictions from GAM model
predict_quantiles <- function(model, newdata, percentile_probs) {
  # Predict all parameters
  pred <- predict(model, newdata = newdata, type = "response")
  
  # Determine family type and number of parameters
  family_name <- model$family$family
  n_params <- ncol(pred)
  
  # Calculate quantiles
  quantile_preds <- matrix(NA, nrow = nrow(newdata), ncol = length(percentile_probs))
  
  for(i in 1:length(percentile_probs)) {
    if(family_name == "gaulss" || n_params == 2) {
      # Gaussian family with 2 parameters (mu, sigma)
      # Use qnorm directly
      quantile_preds[,i] <- qnorm(p = percentile_probs[i], 
                                   mean = pred[,1], 
                                   sd = exp(pred[,2]))
    } else if(family_name == "shash" || n_params == 4) {
      # SHASH family with 4 parameters (mu, sigma, nu, tau)
      quantile_preds[,i] <- gamlss.dist::qSHASHo2(p = percentile_probs[i], 
                                                   mu = pred[,1], 
                                                   sigma = exp(pred[,2]),
                                                   nu = pred[,3], 
                                                   tau = exp(pred[,4]))
    } else {
      stop(sprintf("Unsupported family: %s with %d parameters", family_name, n_params))
    }
  }
  
  return(quantile_preds)
}

# Function to analyze one model
analyze_model_derivatives <- function(model, y_feature, x_seq, percentile_probs, percentile_names) {
  
  cat(sprintf("\nAnalyzing derivatives for: %s\n", y_feature))
  
  # Get predictor variable name from model's smooth terms
  if(length(model$smooth) > 0) {
    predictor_var <- model$smooth[[1]]$term
    cat(sprintf("  Model predictor variable: %s\n", predictor_var))
  } else {
    # Fallback: check model frame column names
    model_names <- names(model$model)
    predictor_var <- model_names[2]
    cat(sprintf("  Model predictor variable (from model frame): %s\n", predictor_var))
  }
  
  # Create prediction dataframe with correct variable name
  pred_data <- data.frame(x_seq)
  names(pred_data) <- predictor_var
  
  # Get predictions for all parameters
  pred_params <- predict(model, newdata = pred_data, type = "response")
  
  # Calculate derivatives for each parameter
  results_list <- list()
  
  # Mean (mu) derivatives
  if(ncol(pred_params) >= 1) {
    mu_deriv <- calculate_derivative(pred_params[,1], x_seq, h_step)
    mu_deriv2 <- calculate_second_derivative(mu_deriv, x_seq)
    
    results_list$mu <- data.frame(
      age = x_seq,
      parameter = "mu",
      value = pred_params[,1],
      first_derivative = mu_deriv,
      second_derivative = mu_deriv2,
      y_feature = y_feature
    )
  }
  
  # Scale (sigma) derivatives
  if(ncol(pred_params) >= 2) {
    sigma_deriv <- calculate_derivative(pred_params[,2], x_seq, h_step)
    sigma_deriv2 <- calculate_second_derivative(sigma_deriv, x_seq)
    
    results_list$sigma <- data.frame(
      age = x_seq,
      parameter = "sigma",
      value = pred_params[,2],
      first_derivative = sigma_deriv,
      second_derivative = sigma_deriv2,
      y_feature = y_feature
    )
  }
  
  # Skewness (nu) derivatives
  if(ncol(pred_params) >= 3) {
    nu_deriv <- calculate_derivative(pred_params[,3], x_seq, h_step)
    nu_deriv2 <- calculate_second_derivative(nu_deriv, x_seq)
    
    results_list$nu <- data.frame(
      age = x_seq,
      parameter = "nu",
      value = pred_params[,3],
      first_derivative = nu_deriv,
      second_derivative = nu_deriv2,
      y_feature = y_feature
    )
  }
  
  # Kurtosis (tau) derivatives
  if(ncol(pred_params) >= 4) {
    tau_deriv <- calculate_derivative(pred_params[,4], x_seq, h_step)
    tau_deriv2 <- calculate_second_derivative(tau_deriv, x_seq)
    
    results_list$tau <- data.frame(
      age = x_seq,
      parameter = "tau",
      value = pred_params[,4],
      first_derivative = tau_deriv,
      second_derivative = tau_deriv2,
      y_feature = y_feature
    )
  }
  
  # Combine parameter results
  param_results <- do.call(rbind, results_list)
  
  # Calculate quantile derivatives
  quantile_preds <- predict_quantiles(model, pred_data, percentile_probs)
  
  quantile_results <- list()
  for(i in 1:length(percentile_probs)) {
    q_deriv <- calculate_derivative(quantile_preds[,i], x_seq, h_step)
    q_deriv2 <- calculate_second_derivative(q_deriv, x_seq)
    
    quantile_results[[i]] <- data.frame(
      age = x_seq,
      quantile = percentile_names[i],
      quantile_value = percentile_probs[i],
      value = quantile_preds[,i],
      first_derivative = q_deriv,
      second_derivative = q_deriv2,
      y_feature = y_feature
    )
  }
  
  quantile_df <- do.call(rbind, quantile_results)
  
  return(list(
    parameters = param_results,
    quantiles = quantile_df
  ))
}

# Function to create plots
create_derivative_plots <- function(param_data, quantile_data, 
                                    feature_name, safe_feature_name, plot_dir) {
  
    # Plot 1: Parameter values over age
    p1 <- ggplot(param_data, aes(x = age, y = value, color = parameter)) +
    geom_line(linewidth = 1) +
    labs(title = paste0(feature_name, ": Distribution Parameters"),
        x = "Gestational Age (weeks)",
        y = "Parameter Value",
        color = "Parameter") +
    theme_minimal() +
    theme(legend.position = "bottom",
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA))

    # Plot 2: First derivatives of parameters
    p2 <- ggplot(param_data, aes(x = age, y = first_derivative, color = parameter)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = paste0(feature_name, ": Parameter Growth Rates"),
        x = "Gestational Age (weeks)",
        y = "First Derivative (rate of change)",
        color = "Parameter") +
    theme_minimal() +
    theme(legend.position = "bottom",
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA))

    # Plot 3: Second derivatives of parameters
    p3 <- ggplot(param_data, aes(x = age, y = second_derivative, color = parameter)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = paste0(feature_name, ": Parameter Acceleration"),
        x = "Gestational Age (weeks)",
        y = "Second Derivative (acceleration)",
        color = "Parameter") +
    theme_minimal() +
    theme(legend.position = "bottom",
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA))

    # Plot 4: Quantile curves
    p4 <- ggplot(quantile_data, aes(x = age, y = value, color = quantile)) +
    geom_line(linewidth = 1) +
    labs(title = paste0(feature_name, ": Normative Curves"),
        x = "Gestational Age (weeks)",
        y = feature_name,
        color = "Percentile") +
    theme_minimal() +
    theme(legend.position = "bottom",
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA))

    # Plot 5: First derivatives of quantiles (growth rates)
    p5 <- ggplot(quantile_data, aes(x = age, y = first_derivative, color = quantile)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = paste0(feature_name, ": Growth Rates by Percentile"),
        x = "Gestational Age (weeks)",
        y = "Growth Rate (units/week)",
        color = "Percentile") +
    theme_minimal() +
    theme(legend.position = "bottom",
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA))

    # Plot 6: Second derivatives of quantiles (acceleration)
    p6 <- ggplot(quantile_data, aes(x = age, y = second_derivative, color = quantile)) +
    geom_line(linewidth = 1) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    labs(title = paste0(feature_name, ": Growth Acceleration by Percentile"),
        x = "Gestational Age (weeks)",
        y = "Acceleration (units/week²)",
        color = "Percentile") +
    theme_minimal() +
    theme(legend.position = "bottom",
            panel.background = element_rect(fill = "white", color = NA),
            plot.background = element_rect(fill = "white", color = NA))

    # Combine plots with white background
    combined_plot <- plot_grid(p1, p2, p3, p4, p5, p6, 
                                ncol = 2, nrow = 3,
                                labels = c("A", "B", "C", "D", "E", "F")) +
    theme(plot.background = element_rect(fill = "white", color = NA))
  
  # Save combined plot using safe filename
  ggsave(filename = file.path(plot_dir, paste0(safe_feature_name, "_derivatives_all.png")),
         plot = combined_plot,
         width = 14, height = 18, dpi = 300)
  
  # Save individual plots using safe filenames
  ggsave(filename = file.path(plot_dir, paste0(safe_feature_name, "_parameters.png")),
         plot = p1, width = 8, height = 6, dpi = 300)
  
  ggsave(filename = file.path(plot_dir, paste0(safe_feature_name, "_growth_rates.png")),
         plot = p5, width = 8, height = 6, dpi = 300)
  
  ggsave(filename = file.path(plot_dir, paste0(safe_feature_name, "_acceleration.png")),
         plot = p6, width = 8, height = 6, dpi = 300)
  
  cat(sprintf("  Plots saved for %s\n", feature_name))
  
  return(list(p1 = p1, p2 = p2, p3 = p3, p4 = p4, p5 = p5, p6 = p6))
}

# ==============================================================================
# MAIN ANALYSIS LOOP
# ==============================================================================

cat("=================================================================\n")
cat("DERIVATIVE ANALYSIS FOR NORMATIVE MODELS\n")
cat("=================================================================\n\n")

# Load ALL models from the single RDS file
cat(sprintf("Loading models from: %s\n", models_file))

final_models <- tryCatch({
  readRDS(models_file)
}, error = function(e) {
  stop(sprintf("ERROR: Could not load models file: %s\n", e$message))
})

cat(sprintf("Successfully loaded %d models\n\n", length(final_models)))

# Print available models
cat("Available models:\n")
for(i in 1:length(names(final_models))) {
  cat(sprintf("  %d. %s\n", i, names(final_models)[i]))
}
cat("\n")

# Initialize storage for all results
all_param_results <- list()
all_quantile_results <- list()

# Create gestational age sequence
x_seq <- seq(ga_min, ga_max, length.out = n_points)

# Loop through each model in the list
for(feature_name in names(final_models)) {
  
  cat(sprintf("\n--- Processing: %s ---\n", feature_name))
  
  # Extract the model
  model <- final_models[[feature_name]]
  
  # Check if model exists and is valid
  if(is.null(model)) {
    cat(sprintf("  WARNING: Model is NULL: %s\n", feature_name))
    cat("  Skipping...\n")
    next
  }
  
  if(!inherits(model, "gam")) {
    cat(sprintf("  WARNING: Model is not a GAM object: %s\n", feature_name))
    cat("  Skipping...\n")
    next
  }
  
  # Create a safe filename (replace spaces and special characters)
  safe_feature_name <- gsub(" ", "_", feature_name)
  safe_feature_name <- gsub("[^A-Za-z0-9_]", "", safe_feature_name)
  
  # Analyze derivatives
  results <- tryCatch({
    analyze_model_derivatives(model, safe_feature_name, x_seq, 
                              percentile_probs, percentile_names)
  }, error = function(e) {
    cat(sprintf("  ERROR analyzing derivatives: %s\n", e$message))
    cat(sprintf("  Error details: %s\n", as.character(e)))
    return(NULL)
  })
  
  if(is.null(results)) {
    next
  }
  
  # Store results with original feature name
  results$parameters$feature_original <- feature_name
  results$quantiles$feature_original <- feature_name
  
  all_param_results[[feature_name]] <- results$parameters
  all_quantile_results[[feature_name]] <- results$quantiles
  
  # Create and save plots
  cat("  Creating plots...\n")
  plots <- tryCatch({
    create_derivative_plots(results$parameters, results$quantiles, 
                            feature_name, safe_feature_name, plot_dir)
  }, error = function(e) {
    cat(sprintf("  ERROR creating plots: %s\n", e$message))
    return(NULL)
  })
  
  cat(sprintf("  Completed: %s\n", feature_name))
}

# ==============================================================================
# SAVE RESULTS TO CSV
# ==============================================================================

cat("\n=================================================================\n")
cat("SAVING RESULTS\n")
cat("=================================================================\n\n")

# Combine all parameter results
if(length(all_param_results) > 0) {
  combined_param_df <- do.call(rbind, all_param_results)
  param_output_file <- file.path(output_dir, "parameter_derivatives.csv")
  write.csv(combined_param_df, file = param_output_file, row.names = FALSE)
  cat(sprintf("Parameter derivatives saved to: %s\n", param_output_file))
}

# Combine all quantile results
if(length(all_quantile_results) > 0) {
  combined_quantile_df <- do.call(rbind, all_quantile_results)
  quantile_output_file <- file.path(output_dir, "quantile_derivatives.csv")
  write.csv(combined_quantile_df, file = quantile_output_file, row.names = FALSE)
  cat(sprintf("Quantile derivatives saved to: %s\n", quantile_output_file))
}

# ==============================================================================
# SUMMARY STATISTICS
# ==============================================================================

cat("\n=================================================================\n")
cat("SUMMARY STATISTICS\n")
cat("=================================================================\n\n")

if(length(all_quantile_results) > 0) {
  
  for(y_feature in names(all_quantile_results)) {
    
    cat(sprintf("\n%s:\n", y_feature))
    cat("-----------------------------------\n")
    
    q_data <- all_quantile_results[[y_feature]]
    
    # Find maximum growth rate for median
    median_data <- q_data[q_data$quantile == "P50", ]
    max_growth_idx <- which.max(median_data$first_derivative)
    max_growth_age <- median_data$age[max_growth_idx]
    max_growth_rate <- median_data$first_derivative[max_growth_idx]
    
    cat(sprintf("  Maximum growth rate (P50): %.4f units/week at %.1f weeks\n", 
                max_growth_rate, max_growth_age))
    
    # Find inflection point (where second derivative crosses zero)
    sign_changes <- diff(sign(median_data$second_derivative))
    inflection_indices <- which(sign_changes != 0)
    
    if(length(inflection_indices) > 0) {
      cat("  Inflection points (P50):\n")
      for(idx in inflection_indices) {
        cat(sprintf("    Age: %.1f weeks\n", median_data$age[idx]))
      }
    } else {
      cat("  No inflection points detected\n")
    }
    
    # Growth rate range across all percentiles
    growth_range <- range(q_data$first_derivative, na.rm = TRUE)
    cat(sprintf("  Growth rate range (all percentiles): %.4f to %.4f units/week\n",
                growth_range[1], growth_range[2]))
  }
}

cat("\n=================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("=================================================================\n\n")
cat(sprintf("Results saved to: %s\n", output_dir))
cat(sprintf("Plots saved to: %s\n", plot_dir))