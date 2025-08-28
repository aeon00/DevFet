# Single Site GAMLSS Brain Morphology Analysis Code
# Modified from multi-site version for single dataset analysis

# Check and load required packages
required_packages <- c("MASS", "gamlss.dist", "nlme", "mgcv", "reshape2", 
                      "ggplot2", "cowplot", "patchwork", "caret")

# Check for missing packages
missing_packages <- required_packages[!sapply(required_packages, function(pkg) {
  requireNamespace(pkg, quietly = TRUE)
})]

if(length(missing_packages) > 0) {
  cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
  install.packages(missing_packages)
}

# Load required libraries
suppressPackageStartupMessages({
  library(MASS)
  library(gamlss.dist)
  library(nlme)
  library(mgcv)
  library(reshape2)
  library(ggplot2)
  library(cowplot)
  library(patchwork)
  library(caret)
})

# Import specific functions from patchwork to avoid errors
if(requireNamespace("patchwork", quietly = TRUE)) {
  plot_spacer <- patchwork::plot_spacer
  wrap_plots <- patchwork::wrap_plots
} else {
  # Define simple alternatives if patchwork unavailable
  plot_spacer <- function() {
    ggplot() + theme_void()
  }
  wrap_plots <- function(plots, ncol = 2, nrow = NULL) {
    # Simple fallback using cowplot
    if(requireNamespace("cowplot", quietly = TRUE)) {
      return(cowplot::plot_grid(plotlist = plots, ncol = ncol, nrow = nrow))
    } else {
      # Last resort: return first plot
      return(plots[[1]])
    }
  }
}

cat("All required packages loaded successfully.\n")

# Function to normalize numerical columns (z-score normalization)
normalize_columns <- function(df, cols_to_normalize = NULL) {
  # If no columns specified, normalize all numeric columns
  if (is.null(cols_to_normalize)) {
    cols_to_normalize <- names(df)[sapply(df, is.numeric)]
  }
  
  # Normalize specified columns
  df_normalized <- df
  for (col in cols_to_normalize) {
    if (col %in% names(df) && is.numeric(df[[col]])) {
      df_normalized[[paste0(col, "_normalized")]] <- scale(df[[col]])[,1]
    }
  }
  
  return(df_normalized)
}

# Function to normalize data to 0-1 scale for comparison
normalize_data <- function(x) {
  if(length(x) == 0 || all(is.na(x))) return(x)
  x_min <- min(x, na.rm = TRUE)
  x_max <- max(x, na.rm = TRUE)
  if(x_max == x_min) return(rep(0.5, length(x)))  # Handle constant values
  (x - x_min) / (x_max - x_min)
}

# Function to check if file exists and read it safely
safe_read_csv <- function(filepath, description = "") {
  if(!file.exists(filepath)) {
    stop("Data file not found: ", filepath, " (", description, ")")
  }
  
  tryCatch({
    df <- read.csv(filepath, header = TRUE, stringsAsFactors = FALSE)
    cat("Successfully loaded", description, "- Rows:", nrow(df), "Cols:", ncol(df), "\n")
    return(df)
  }, error = function(e) {
    stop("Error reading ", filepath, ": ", e$message)
  })
}

# Read dataset with error checking
cat("Loading dataset...\n")

# Update this path to match your system and dataset
# Choose one of your datasets or specify a new path
data_path <- "/home/INT/dienye.h/python_files/combined_dataset/marsfet_qc_filtered.csv"  # UPDATE THIS PATH

# Alternative: use relative path or allow user input
# data_path <- "your_dataset.csv"

df <- safe_read_csv(data_path, "Brain morphology dataset")

# Basic data exploration
cat("\nDataset Overview:\n")
cat("Dimensions:", nrow(df), "rows ×", ncol(df), "columns\n")
cat("Column names:", paste(head(names(df), 10), collapse = ", "), 
    if(ncol(df) > 10) "..." else "", "\n")

# Check for missing values
cat("\nMissing values per column:\n")
missing_counts <- colSums(is.na(df))
print(missing_counts[missing_counts > 0])

# Remove rows with missing gestational_age
if("gestational_age" %in% names(df)) {
  initial_rows <- nrow(df)
  df <- df[!is.na(df$gestational_age), ]
  cat("Removed", initial_rows - nrow(df), "rows with missing gestational_age\n")
} else {
  stop("gestational_age column not found in the data")
}

# Rename columns for clarity (only if they exist)
column_mapping <- list(
  "surface_area_cm2" = "Surface Area cm2",
  "analyze_folding_power" = "Folding Power",
  "B4_vertex_percentage" = "B4 Vertex Percentage",
  "B5_vertex_percentage" = "B5 Vertex Percentage",
  "B6_vertex_percentage" = "B6 Vertex Percentage",
  "band_parcels_B4" = "Band_parcels B4",
  "band_parcels_B5" = "Band Parcels B5",
  "band_parcels_B6" = "Band Parcels B6",
  "volume_ml" = "Hemispheric Volume",
  "gyrification_index" = "Gyrification Index",
  "hull_area" = "Hull Area",
  "B4_surface_area" = "B4 Surface Area",
  "B5_surface_area" = "B5 Surface Area",
  "B6_surface_area" = "B6 Surface Area",
  "B4_surface_area_percentage" = "B4 Surface Area Percentage",
  "B5_surface_area_percentage" = "B5 Surface Area Percentage",
  "B6_surface_area_percentage" = "B6 Surface Area Percentage",
  "band_power_B4" = "B4 Band Power",
  "band_power_B5" = "B5 Band Power",
  "band_power_B6" = "B6 Band Power",
  "B4_band_relative_power" = "B4 Band Relative Power",
  "B5_band_relative_power" = "B5 Band Relative Power",
  "B6_band_relative_power" = "B6 Band Relative Power"
)

# Apply column renaming only for existing columns
for(old_name in names(column_mapping)) {
  if(old_name %in% names(df)) {
    names(df)[names(df) == old_name] <- column_mapping[[old_name]]
  }
}

# Define y variables for analysis (only include existing columns)
potential_y_values <- c("Surface Area cm2", "Folding Power", "B4 Vertex Percentage", 
                       "B5 Vertex Percentage","B6 Vertex Percentage", "Band_parcels B4", 
                       "Band Parcels B5", "Band Parcels B6", "Hemispheric Volume", 
                       "Gyrification Index", "Hull Area", "B4 Surface Area", 
                       "B5 Surface Area", "B6 Surface Area", "B4 Surface Area Percentage", 
                       "B5 Surface Area Percentage", "B6 Surface Area Percentage", 
                       "B4 Band Power", "B5 Band Power", "B6 Band Power",
                       "B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

# Filter to only existing columns
y_values <- potential_y_values[potential_y_values %in% names(df)]
cat("Y variables to analyze:", length(y_values), "\n")
cat("Variables:", paste(head(y_values, 5), collapse = ", "), 
    if(length(y_values) > 5) "..." else "", "\n")

# Initialize results dataframe
results <- data.frame(Model = character(),
                      Y_feature = character(),
                      BIC = double(),
                      AIC = double(), 
                      stringsAsFactors = FALSE)

# Simplified quantile extraction function for single site
extract_quantiles_robust <- function(model, newdata, quantiles = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)) {
  tryCatch({
    # Get predictions
    pred <- predict(model, newdata = newdata, type = "response")
    
    n_pred <- nrow(newdata)
    n_quantiles <- length(quantiles)
    
    # Initialize result matrix
    result <- matrix(NA, nrow = n_pred, ncol = n_quantiles)
    colnames(result) <- paste0("q", quantiles)
    
    # Handle different prediction formats
    if(is.list(pred)) {
      # GAMLSS with multiple parameters
      mu <- as.numeric(pred[[1]])
      
      # Ensure mu has correct length
      if(length(mu) != n_pred) {
        if(length(mu) == 1) {
          mu <- rep(mu, n_pred)
        } else {
          cat("Warning: mu length (", length(mu), ") doesn't match newdata rows (", n_pred, ")\n")
          mu <- rep(mu[1], n_pred)
        }
      }
      
      if(length(pred) >= 2) {
        sigma <- exp(as.numeric(pred[[2]]))
        if(length(sigma) != n_pred) {
          if(length(sigma) == 1) {
            sigma <- rep(sigma, n_pred)
          } else {
            sigma <- rep(sigma[1], n_pred)
          }
        }
      } else {
        sigma <- rep(1, n_pred)
      }
      
      # Use normal approximation for quantiles
      for(i in seq_along(quantiles)) {
        result[, i] <- qnorm(quantiles[i], mean = mu, sd = sigma)
      }
      
    } else if(is.matrix(pred)) {
      # Matrix format
      if(nrow(pred) != n_pred) {
        cat("Warning: prediction matrix rows (", nrow(pred), ") don't match newdata rows (", n_pred, ")\n")
        if(nrow(pred) == 1) {
          pred <- matrix(rep(pred[1,], n_pred), nrow = n_pred, byrow = TRUE)
        } else {
          pred <- matrix(rep(pred[1,], n_pred), nrow = n_pred, byrow = TRUE)
        }
      }
      
      mu <- pred[, 1]
      if(ncol(pred) >= 2) {
        sigma <- exp(pred[, 2])
      } else {
        sigma <- rep(1, n_pred)
      }
      
      for(i in seq_along(quantiles)) {
        result[, i] <- qnorm(quantiles[i], mean = mu, sd = sigma)
      }
      
    } else {
      # Simple predictions - assume normal with estimated variance
      mu <- as.numeric(pred)
      
      # Ensure mu has correct length
      if(length(mu) != n_pred) {
        if(length(mu) == 1) {
          mu <- rep(mu, n_pred)
        } else {
          cat("Warning: prediction length (", length(mu), ") doesn't match newdata rows (", n_pred, ")\n")
          mu <- rep(mu[1], n_pred)
        }
      }
      
      # Estimate sigma from model
      if(!is.null(model$sig2)) {
        sigma <- rep(sqrt(model$sig2), n_pred)
      } else {
        sigma <- rep(1, n_pred)
      }
      
      for(i in seq_along(quantiles)) {
        result[, i] <- qnorm(quantiles[i], mean = mu, sd = sigma)
      }
    }
    
    # Clean up non-finite values
    result[!is.finite(result)] <- NA
    
    return(as.data.frame(result))
    
  }, error = function(e) {
    cat("Warning: Quantile extraction failed:", e$message, "\n")
    # Return appropriately sized NA dataframe
    n_pred <- nrow(newdata)
    n_quantiles <- length(quantiles)
    result <- matrix(NA, nrow = n_pred, ncol = n_quantiles)
    colnames(result) <- paste0("q", quantiles)
    return(as.data.frame(result))
  })
}

# Store best models for each Y variable
best_models <- list()

# Main analysis loop with progress tracking
cat("\n=== Starting Model Fitting ===\n")
total_vars <- length(y_values)
current_var <- 0

for (i in y_values) {
  current_var <- current_var + 1
  cat("\n[", current_var, "/", total_vars, "] Analyzing:", i, "\n")
  cat(rep("=", 50), "\n", sep = "")
  
  # Prepare data
  x <- df$gestational_age
  y <- df[[i]]
  
  # Skip if variable not found
  if (is.null(y)) {
    cat("Warning: Variable", i, "not found in dataset. Skipping...\n")
    next
  }
  
  # Remove NA values
  complete_cases <- complete.cases(x, y)
  if(sum(complete_cases) < 10) {
    cat("Warning: Insufficient complete cases (", sum(complete_cases), ") for", i, ". Skipping...\n")
    next
  }
  
  x <- x[complete_cases]
  y <- y[complete_cases]
  
  # Create temporary dataframe for modeling
  temp_df <- data.frame(x = x, y = y)
  
  max_vol <- max(y, na.rm = TRUE)
  min_vol <- min(y, na.rm = TRUE)
  max_age <- max(x, na.rm = TRUE)
  min_age <- min(x, na.rm = TRUE)
  
  cat("Data summary - N:", nrow(temp_df), 
      "| Y range: [", round(min_vol, 2), ",", round(max_vol, 2), "]",
      "| Age range: [", round(min_age, 1), ",", round(max_age, 1), "]\n")
  
  # Initialize model storage for single site models
  models <- list()
  model_names <- c("linear", "gam_smooth", "gamlss_normal", "gamlss_shash")
  bic_values <- rep(NA, 4)
  aic_values <- rep(NA, 4)
  
  # Fit models with individual error handling
  cat("Fitting models: ")
  
  # Model 1: Simple linear regression
  tryCatch({
    models[[1]] <- lm(y ~ x, data = temp_df)
    bic_values[1] <- BIC(models[[1]])
    aic_values[1] <- AIC(models[[1]])
    cat("linear ")
  }, error = function(e) {
    cat("linear(failed) ")
  })
  
  # Model 2: GAM with smooth term
  tryCatch({
    models[[2]] <- gam(y ~ s(x), data = temp_df)
    bic_values[2] <- BIC(models[[2]])
    aic_values[2] <- AIC(models[[2]])
    cat("gam ")
  }, error = function(e) {
    cat("gam(failed) ")
  })
  
  # Model 3: GAMLSS with normal distribution and smooth mean
  tryCatch({
    models[[3]] <- gam(list(y ~ s(x), ~ 1), 
                       family = gaulss(), data = temp_df)
    bic_values[3] <- BIC(models[[3]])
    aic_values[3] <- AIC(models[[3]])
    cat("gamlss_normal ")
  }, error = function(e) {
    cat("gamlss_normal(failed) ")
  })
  
  # Model 4: GAMLSS with SHASH distribution
  tryCatch({
    models[[4]] <- gam(list(y ~ s(x),
                           ~ 1,
                           ~ 1,
                           ~ 1), 
                       family = shash(), data = temp_df)
    bic_values[4] <- BIC(models[[4]])
    aic_values[4] <- AIC(models[[4]])
    cat("gamlss_shash ")
  }, error = function(e) {
    cat("gamlss_shash(failed) ")
  })
  
  cat("\n")
  
  # Check if any models fitted successfully
  successful_models <- !is.na(bic_values)
  if(!any(successful_models)) {
    cat("Error: No models fitted successfully for", i, "\n")
    next
  }
  
  # Find best model based on BIC (among successful models)
  valid_bic <- bic_values[successful_models]
  valid_indices <- which(successful_models)
  best_idx_among_valid <- which.min(valid_bic)
  best_model_idx <- valid_indices[best_idx_among_valid]
  
  best_model <- models[[best_model_idx]]
  best_models[[i]] <- best_model
  
  # Store results for all attempted models
  for(j in 1:4) {
    if(successful_models[j]) {
      bic_aic_data <- data.frame(
        Model = model_names[j],
        Y_feature = i,
        BIC = bic_values[j],
        AIC = aic_values[j],
        stringsAsFactors = FALSE
      )
      results <- rbind(results, bic_aic_data)
    }
  }
  
  # Print model comparison for successful models
  cat("Model Comparison for", i, ":\n")
  cat("Model\t\t\tBIC\t\tAIC\t\tStatus\n")
  for (j in 1:4) {
    status <- if(successful_models[j]) "Success" else "Failed"
    bic_str <- if(successful_models[j]) sprintf("%.2f", bic_values[j]) else "NA"
    aic_str <- if(successful_models[j]) sprintf("%.2f", aic_values[j]) else "NA"
    cat(sprintf("%-15s\t%s\t%s\t%s\n", model_names[j], bic_str, aic_str, status))
  }
  cat("Best model:", model_names[best_model_idx], "(lowest BIC among successful models)\n")
  
  # ============================================================================
  # CREATE VISUALIZATION FOR BEST MODEL
  # ============================================================================
  
  tryCatch({
    cat("Creating plot for best model...\n")
    
    # Create prediction data for plotting
    x_seq <- seq(min_age, max_age, length.out = 100)
    pred_df <- data.frame(x = x_seq)
    
    # Create base plot with data points
    p_best <- ggplot(temp_df, aes(x = x, y = y)) +
      geom_point(size = 2, alpha = 0.6, color = "#0072B2")
    
    # Add model predictions based on model type
    if(best_model_idx %in% c(3, 4)) {  # GAMLSS models
      # Get predictions using robust function
      preds <- predict(best_model, newdata = pred_df, type = "response")
      
      # Handle different prediction formats
      if (is.list(preds) && length(preds) > 0) {
        fit_values <- as.numeric(preds[[1]])
        if (length(preds) > 1) {
          se_values <- exp(as.numeric(preds[[2]]))
        } else {
          se_values <- rep(sd(temp_df$y, na.rm = TRUE) * 0.1, length(fit_values))
        }
      } else {
        fit_values <- as.numeric(preds)
        se_values <- rep(sd(temp_df$y, na.rm = TRUE) * 0.1, length(fit_values))
      }
      
      # Create prediction dataframe
      pred_plot_df <- data.frame(
        x = x_seq,
        fit = fit_values,
        se = se_values,
        lower = fit_values - 1.96 * se_values,
        upper = fit_values + 1.96 * se_values
      )
      
      # Add predictions to plot
      p_best <- p_best +
        geom_ribbon(data = pred_plot_df, 
                    aes(x = x, ymin = lower, ymax = upper),
                    alpha = 0.3, fill = "#0072B2", inherit.aes = FALSE) +
        geom_line(data = pred_plot_df,
                  aes(x = x, y = fit),
                  color = "#0072B2", linewidth = 1.2, inherit.aes = FALSE)
      
    } else {  # Linear or GAM models
      # Use built-in prediction with confidence intervals
      p_best <- p_best +
        geom_smooth(method = if(best_model_idx == 1) "lm" else "gam",
                   formula = if(best_model_idx == 1) y ~ x else y ~ s(x),
                   se = TRUE, alpha = 0.3, color = "#0072B2", fill = "#0072B2")
    }
    
    # Finalize plot
    p_best <- p_best +
      labs(title = paste('Best Model (', model_names[best_model_idx], ') for: ', i, sep = ""),
           subtitle = paste('BIC:', round(bic_values[best_model_idx], 2), 
                           '| AIC:', round(aic_values[best_model_idx], 2),
                           '| N:', nrow(temp_df)),
           x = 'Gestational Age (weeks)',
           y = i) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 10)) +
      scale_x_continuous(breaks = seq(ceiling(min_age), floor(max_age), by = 2))
    
    # Save the plot
    filename <- paste0("best_model_", gsub("[^A-Za-z0-9]", "_", i), ".png")
    ggsave(filename, plot = p_best, width = 10, height = 6, units = 'in', dpi = 300, bg = "white")
    cat("Plot saved as:", filename, "\n")
    
    # Display the plot
    print(p_best)
    
  }, error = function(e) {
    cat("Error creating model plot for", i, ":", e$message, "\n")
    
    # Fallback: Simple scatter plot with smooth
    tryCatch({
      p_simple <- ggplot(temp_df, aes(x = x, y = y)) +
        geom_point(size = 2, alpha = 0.6, color = "#0072B2") +
        geom_smooth(method = "loess", se = TRUE, alpha = 0.3, color = "#0072B2") +
        labs(title = paste('Smooth curve for:', i),
             subtitle = 'Fallback plot - model predictions failed',
             x = 'Gestational Age (weeks)',
             y = i) +
        theme_minimal()
      
      filename <- paste0("smooth_", gsub("[^A-Za-z0-9]", "_", i), ".png")
      ggsave(filename, plot = p_simple, width = 10, height = 6, units = 'in', dpi = 300, bg = "white")
      cat("Fallback plot saved as:", filename, "\n")
      
    }, error = function(e2) {
      cat("Even fallback plot failed:", e2$message, "\n")
    })
  })
  
  # Additional diagnostic information
  cat("\nModel Diagnostics:\n")
  
  # Check data characteristics
  cor_age_y <- cor(temp_df$x, temp_df$y, use = "complete.obs")
  cat("Age-outcome correlation:", round(cor_age_y, 3), "\n")
  
  # Check if relationship is essentially flat
  if(abs(cor_age_y) < 0.1) {
    cat("*** WARNING: Very weak age relationship (|r| < 0.1) ***\n")
    cat("This variable may not change substantially with gestational age.\n")
  }
  
  # Check effective degrees of freedom for best model
  if(!is.null(best_model$edf)) {
    total_edf <- sum(best_model$edf, na.rm = TRUE)
    cat("Best model total EDF:", round(total_edf, 2), "\n")
    if(total_edf < 2) {
      cat("*** WARNING: Model is essentially linear (EDF < 2) ***\n")
    }
  }
  
  # Data distribution check
  y_range <- max(temp_df$y) - min(temp_df$y)
  y_mean <- mean(temp_df$y)
  cv <- sd(temp_df$y) / abs(y_mean)
  cat("Coefficient of variation:", round(cv, 3), "\n")
  
  if(cv < 0.1) {
    cat("*** WARNING: Low variability (CV < 0.1) - limited dynamic range ***\n")
  }
  
  # Clean up large objects to manage memory
  if(current_var %% 5 == 0) {
    gc()  # Garbage collection every 5 variables
  }
}

# Save results table
if(nrow(results) > 0) {
  write.csv(results, "single_site_model_results.csv", row.names = FALSE)
  cat("\nModel comparison results saved as: single_site_model_results.csv\n")
  
  # Create summary plot comparing BIC across all models and variables
  tryCatch({
    model_summary <- aggregate(BIC ~ Model, data = results, FUN = function(x) {
      c(mean = mean(x, na.rm = TRUE), 
        se = sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x))),
        count = sum(!is.na(x)))
    })
    
    # Flatten the result
    model_summary_df <- data.frame(
      Model = model_summary$Model,
      mean_BIC = model_summary$BIC[, "mean"],
      se_BIC = model_summary$BIC[, "se"],
      count = model_summary$BIC[, "count"]
    )
    
    p_summary <- ggplot(model_summary_df, aes(x = Model, y = mean_BIC)) +
      geom_col(fill = "steelblue", alpha = 0.7) +
      geom_errorbar(aes(ymin = mean_BIC - se_BIC, ymax = mean_BIC + se_BIC),
                    width = 0.2) +
      geom_text(aes(label = paste0("n=", count)), vjust = -0.5, size = 3) +
      labs(title = "Average BIC Across All Variables by Model",
           subtitle = "Error bars show standard error, n = number of successful fits",
           x = "Model",
           y = "Mean BIC") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            plot.subtitle = element_text(hjust = 0.5, size = 10))
    
    ggsave("single_site_model_summary.png", plot = p_summary, width = 10, height = 6, 
           units = 'in', dpi = 300, bg = "white")
    cat("Summary plot saved as: single_site_model_summary.png\n")
    
  }, error = function(e) {
    cat("Error creating summary plot:", e$message, "\n")
  })
} else {
  cat("Warning: No successful model fits found. No results to save.\n")
}

# ============================================================================
# NORMATIVE MODELING ANALYSIS
# ============================================================================

cat("\n=== NORMATIVE MODELING ANALYSIS ===\n")

# Split data for normative modeling (only if we have sufficient data)
if(nrow(df) >= 20 && length(best_models) > 0) {
  set.seed(123)
  
  # Create training/test split
  train_idx <- createDataPartition(1:nrow(df), p = 0.7, list = FALSE)
  df_train <- df[train_idx, ]
  df_test <- df[-train_idx, ]
  
  cat("Training set size:", nrow(df_train), "\n")
  cat("Test set size:", nrow(df_test), "\n")
  
  # Helper functions for normative modeling
  params_to_scores <- function(target_variable, predictions) {
    if(length(target_variable) == 0 || all(is.na(target_variable))) {
      return(list(z_randomized = numeric(0), log_densities = numeric(0), 
                  mu = numeric(0), sigma = numeric(0)))
    }
    
    if(is.list(predictions)) {
      mu <- as.numeric(predictions[[1]])
      if(length(predictions) >= 2) {
        sigma <- exp(as.numeric(predictions[[2]]))
      } else {
        sigma <- rep(1, length(mu))
      }
    } else if(is.matrix(predictions) && ncol(predictions) >= 2) {
      mu <- predictions[,1]
      sigma <- exp(predictions[,2])
    } else {
      mu <- as.numeric(predictions)
      sigma <- rep(1, length(mu))
    }
    
    # Ensure positive sigma
    sigma[sigma <= 0] <- 1
    sigma[!is.finite(sigma)] <- 1
    
    z_randomized <- (target_variable - mu) / sigma
    log_densities <- dnorm(target_variable, mean = mu, sd = sigma, log = TRUE)
    
    return(list(
      z_randomized = z_randomized,
      log_densities = log_densities,
      mu = mu,
      sigma = sigma
    ))
  }
  
  calibration_descriptives <- function(z_scores) {
    z_clean <- z_scores[is.finite(z_scores)]
    
    if(length(z_clean) < 5) {
      return(list(skew = NA, kurtosis = NA, W = NA, p_value = NA))
    }
    
    n <- length(z_clean)
    mean_z <- mean(z_clean)
    sd_z <- sd(z_clean)
    
    if(sd_z == 0) {
      return(list(skew = NA, kurtosis = NA, W = NA, p_value = NA))
    }
    
    skew <- (sum((z_clean - mean_z)^3) / n) / (sd_z^3)
    kurtosis <- (sum((z_clean - mean_z)^4) / n) / (sd_z^4) - 3
    
    # Shapiro-Wilk test (sample if too large)
    if(length(z_clean) > 5000) {
      test_sample <- sample(z_clean, 5000)
    } else {
      test_sample <- z_clean
    }
    
    tryCatch({
      shapiro_result <- shapiro.test(test_sample)
      W <- shapiro_result$statistic
      p_value <- shapiro_result$p.value
    }, error = function(e) {
      W <- NA
      p_value <- NA
    })
    
    return(list(skew = skew, kurtosis = kurtosis, W = W, p_value = p_value))
  }
  
  # Create normative curves for best models
  cat("\n=== CREATING NORMATIVE CURVES ===\n")
  
  # Use only training data for age range
  train_age_range <- seq(min(df_train$gestational_age, na.rm = TRUE), 
                        max(df_train$gestational_age, na.rm = TRUE), 
                        length.out = 100)
  
  normative_results <- data.frame(
    Variable = character(),
    LogScore = numeric(),
    Skewness = numeric(),
    Kurtosis = numeric(),
    ShapiroW = numeric(),
    ShapiroP = numeric(),
    stringsAsFactors = FALSE
  )
  
  for(var in names(best_models)) {
    if(!is.null(best_models[[var]]) && var %in% names(df_train) && var %in% names(df_test)) {
      cat("Creating normative curves for", var, "\n")
      
      tryCatch({
        # Create prediction data for training
        newdata <- data.frame(x = train_age_range)
        
        # Get predictions
        pred <- predict(best_models[[var]], newdata = newdata, type = "response")
        
        # Extract parameters for percentile calculation
        if(is.list(pred) && length(pred) >= 2) {
          mu <- as.numeric(pred[[1]])
          sigma <- exp(as.numeric(pred[[2]]))
        } else if(is.matrix(pred) && ncol(pred) >= 2) {
          mu <- pred[,1]
          sigma <- exp(pred[,2])
        } else {
          mu <- as.numeric(pred)
          # Estimate sigma from training residuals
          temp_train <- df_train[complete.cases(df_train$gestational_age, df_train[[var]]), ]
          if(nrow(temp_train) > 0) {
            sigma <- rep(sd(temp_train[[var]], na.rm = TRUE), length(mu))
          } else {
            sigma <- rep(1, length(mu))
          }
        }
        
        # Ensure positive sigma
        sigma[sigma <= 0 | !is.finite(sigma)] <- 1
        
        # Calculate percentiles
        percentiles <- data.frame(
          gestational_age = train_age_range,
          p05 = qnorm(0.05, mu, sigma),
          p25 = qnorm(0.25, mu, sigma),
          p50 = qnorm(0.50, mu, sigma),
          p75 = qnorm(0.75, mu, sigma),
          p95 = qnorm(0.95, mu, sigma)
        )
        
        # Create normative plot
        test_data <- df_test[complete.cases(df_test$gestational_age, df_test[[var]]), ]
        
        p_norm <- ggplot() +
          geom_ribbon(data = percentiles, 
                     aes(x = gestational_age, ymin = p05, ymax = p95), 
                     alpha = 0.2, fill = "blue") +
          geom_ribbon(data = percentiles, 
                     aes(x = gestational_age, ymin = p25, ymax = p75), 
                     alpha = 0.3, fill = "blue") +
          geom_line(data = percentiles, aes(x = gestational_age, y = p50), 
                   color = "blue", linewidth = 1.2) +
          labs(title = paste("Normative Curves:", var),
               subtitle = "Blue = Training norms (5th, 25th, 50th, 75th, 95th percentiles)",
               x = "Gestational Age (weeks)", 
               y = var) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
                plot.subtitle = element_text(hjust = 0.5, size = 10))
        
        # Add test data if available
        if(nrow(test_data) > 0) {
          p_norm <- p_norm +
            geom_point(data = test_data, 
                      aes(x = gestational_age, y = .data[[var]]), 
                      color = "red", alpha = 0.6, size = 1.5) +
            labs(subtitle = paste("Blue = Training norms, Red = Test data (n=", nrow(test_data), ")", sep=""))
        }
        
        filename <- paste0("normative_curves_", gsub("[^A-Za-z0-9]", "_", var), ".png")
        ggsave(filename, p_norm, width = 10, height = 6, dpi = 300, bg = "white")
        cat("Saved:", filename, "\n")
        
        # Calculate test set metrics for all test data
        test_complete <- df_test[complete.cases(df_test$gestational_age, df_test[[var]]), ]
        if(nrow(test_complete) > 0) {
          # Prepare test data for prediction
          test_pred_data <- data.frame(x = test_complete$gestational_age)
          
          test_pred <- predict(best_models[[var]], newdata = test_pred_data, type = "response")
          scores <- params_to_scores(test_complete[[var]], test_pred)
          descriptives <- calibration_descriptives(scores$z_randomized)
          
          # Store results
          normative_results <- rbind(normative_results, data.frame(
            Variable = var,
            LogScore = mean(scores$log_densities, na.rm = TRUE),
            Skewness = descriptives$skew,
            Kurtosis = descriptives$kurtosis,
            ShapiroW = descriptives$W,
            ShapiroP = descriptives$p_value,
            stringsAsFactors = FALSE
          ))
          
          cat("Test set metrics (n=", nrow(test_complete), "):\n")
          cat("  Log score:", round(mean(scores$log_densities, na.rm = TRUE), 3), "\n")
          cat("  Skewness:", round(descriptives$skew, 3), "\n")
          cat("  Kurtosis:", round(descriptives$kurtosis, 3), "\n")
          cat("  Shapiro-Wilk W:", round(descriptives$W, 3), "\n")
          cat("  Shapiro-Wilk p:", round(descriptives$p_value, 3), "\n")
        }
        
      }, error = function(e) {
        cat("Error creating normative curves for", var, ":", e$message, "\n")
      })
    }
  }
  
  # Save normative results
  if(nrow(normative_results) > 0) {
    write.csv(normative_results, "single_site_normative_results.csv", row.names = FALSE)
    cat("\nNormative modeling results saved as: single_site_normative_results.csv\n")
  }
  
} else {
  cat("Warning: Insufficient data for normative modeling (need ≥20 observations and ≥1 successful model)\n")
}

# ============================================================================
# CREATE FINAL SUMMARY
# ============================================================================

cat("\n=== ANALYSIS SUMMARY ===\n")

# Data summary
cat("Dataset Summary:\n")
cat("  Total observations:", nrow(df), "\n")
cat("  Age range:", round(min(df$gestational_age, na.rm = TRUE), 1), 
    "to", round(max(df$gestational_age, na.rm = TRUE), 1), "weeks\n")

# Model summary
cat("\nModel Fitting Summary:\n")
cat("  Variables attempted:", length(y_values), "\n")
cat("  Variables with successful models:", length(best_models), "\n")
cat("  Total successful model fits:", nrow(results), "\n")

if(nrow(results) > 0) {
  model_success_rate <- table(results$Model)
  cat("  Success rate by model:\n")
  for(model in names(model_success_rate)) {
    cat("    ", model, ":", model_success_rate[model], "/", length(y_values), 
        "(", round(100 * model_success_rate[model] / length(y_values), 1), "%)\n")
  }
  
  # Best model frequency
  if(length(best_models) > 0) {
    best_model_names <- sapply(names(best_models), function(var) {
      if(!is.null(best_models[[var]]) && var %in% results$Y_feature) {
        var_results <- results[results$Y_feature == var, ]
        if(nrow(var_results) > 0) {
          return(var_results$Model[which.min(var_results$BIC)])
        }
      }
      return(NA)
    })
    best_model_names <- best_model_names[!is.na(best_model_names)]
    
    if(length(best_model_names) > 0) {
      best_freq <- table(best_model_names)
      cat("  Most frequently selected as best model:\n")
      for(model in names(sort(best_freq, decreasing = TRUE))) {
        cat("    ", model, ":", best_freq[model], "times\n")
      }
    }
  }
}

# File output summary
cat("\nFiles Created:\n")
created_files <- c(
  "single_site_model_results.csv",
  "single_site_model_summary.png",
  "single_site_normative_results.csv"
)

# Add individual model plots
if(length(best_models) > 0) {
  for(var in names(best_models)) {
    safe_name <- gsub("[^A-Za-z0-9]", "_", var)
    created_files <- c(created_files, paste0("best_model_", safe_name, ".png"))
  }
}

# Add normative curve plots
if(exists("normative_results") && nrow(normative_results) > 0) {
  for(var in normative_results$Variable) {
    safe_name <- gsub("[^A-Za-z0-9]", "_", var)
    created_files <- c(created_files, paste0("normative_curves_", safe_name, ".png"))
  }
}

# Check which files actually exist
existing_files <- created_files[file.exists(created_files)]
cat("  Successfully created", length(existing_files), "files:\n")
for(file in head(existing_files, 10)) {
  cat("    -", file, "\n")
}
if(length(existing_files) > 10) {
  cat("    ... and", length(existing_files) - 10, "more files\n")
}

# Final recommendations
cat("\nRecommendations:\n")
if(nrow(results) == 0) {
  cat("  - No models fitted successfully. Check data quality and model specifications.\n")
  cat("  - Verify that required columns exist in the dataset.\n")
  cat("  - Consider simpler model structures if convergence issues persist.\n")
} else {
  cat("  - Review single_site_model_results.csv for detailed BIC/AIC comparisons.\n")
  cat("  - Examine individual model plots for goodness of fit.\n")
  if(exists("normative_results") && nrow(normative_results) > 0) {
    cat("  - Check normative modeling results for calibration quality.\n")
    cat("  - Variables with Shapiro-Wilk p > 0.05 show good normality of residuals.\n")
  }
  cat("  - Consider additional covariates if needed for better model fit.\n")
}

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Total runtime: ", round(proc.time()[3], 1), " seconds\n")

# Clean up workspace (optional)
# rm(list = setdiff(ls(), c("df", "best_models", "results", "normative_results")))
gc()  # Final garbage collection