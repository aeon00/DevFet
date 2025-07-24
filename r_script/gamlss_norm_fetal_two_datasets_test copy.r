library(MASS)
library(gamlss)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)
library(ggplot2)
library(cowplot)
library(patchwork)
library(dplyr)

# ============================================================================
# PART 1: DATA PREPARATION
# ============================================================================

# Read and prepare data from both datasets
df1 <- read.csv("/home/INT/dienye.h/python_files/dhcp_dataset_info/combined_results.csv", 
                header=TRUE, stringsAsFactors = FALSE)
df2 <- read.csv("/home/INT/dienye.h/python_files/devfetfiles/filtered_qc_3_and_above_Copie.csv", 
                header=TRUE, stringsAsFactors = FALSE)

# Add dataset/site identifier
df1$site <- "Site1"
df2$site <- "Site2"

# Function to standardize column names
standardize_columns <- function(df) {
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

# Standardize column names for both datasets
df1 <- standardize_columns(df1)
df2 <- standardize_columns(df2)

# Select only common columns to avoid rbind error
common_cols <- intersect(colnames(df1), colnames(df2))
df1_common <- df1[, common_cols]
df2_common <- df2[, common_cols]

# Combine datasets
df_combined <- rbind(df1_common, df2_common)

# Convert site to factor for modeling
df_combined$site <- as.factor(df_combined$site)

# Select key variables for analysis
selected_variables <- c("B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

# ============================================================================
# PART 2: HYPERPARAMETER TUNING FOR GAMLSS (CORRECTED VERSION)
# ============================================================================

# Function to perform hyperparameter tuning for GAMLSS
tune_gamlss <- function(data, response_var, predictor_var, site_var) {
  
  cat("\n=== Starting Hyperparameter Tuning for", response_var, "===\n")
  
  # Distribution families to try
  families <- list(
    "shash" = shash(),
    "BCT" = BCT(),
    "BCPE" = BCPE(),
    "NO" = NO()
  )
  
  # Degrees of freedom options for smoothers
  df_options <- c(3, 5, 7, 9)
  
  # Whether to include site as fixed effect (GAMLSS doesn't have re() in standard version)
  site_options <- c(TRUE, FALSE)
  
  # Initialize results storage
  results <- data.frame()
  
  # Prepare data for modeling
  model_data <- data.frame(
    y = data[[response_var]],
    x = data[[predictor_var]],
    site = data[[site_var]]
  )
  
  # Remove NA values
  model_data <- na.omit(model_data)
  
  # Check if we have data
  if(nrow(model_data) == 0) {
    cat("Error: No data available after removing NAs\n")
    return(list(results = results, best_config = NULL))
  }
  
  # Grid search over hyperparameters
  for (family_name in names(families)) {
    for (df in df_options) {
      for (use_site in site_options) {
        
        cat("Testing: Family =", family_name, ", df =", df, ", Site effect =", use_site, "\n")
        
        tryCatch({
          
          # Build formula based on whether to include site effects
          if (use_site) {
            # Model with site as fixed effect (since re() is causing issues)
            formula_mu <- as.formula(paste0("y ~ pb(x, df=", df, ") + site"))
            formula_sigma <- as.formula(paste0("~ pb(x, df=", ceiling(df/2), ")"))
          } else {
            # Model without site effects
            formula_mu <- as.formula(paste0("y ~ pb(x, df=", df, ")"))
            formula_sigma <- as.formula(paste0("~ pb(x, df=", ceiling(df/2), ")"))
          }
          
          # For distributions with more parameters
          if (family_name %in% c("shash", "BCT", "BCPE")) {
            formula_nu <- ~ 1  # Keep nu constant
            formula_tau <- ~ 1 # Keep tau constant
            
            # Fit model
            model <- gamlss(
              formula = formula_mu,
              sigma.formula = formula_sigma,
              nu.formula = formula_nu,
              tau.formula = formula_tau,
              family = families[[family_name]],
              data = model_data,
              control = gamlss.control(trace = FALSE, n.cyc = 50)
            )
          } else {
            # For 2-parameter distributions
            model <- gamlss(
              formula = formula_mu,
              sigma.formula = formula_sigma,
              family = families[[family_name]],
              data = model_data,
              control = gamlss.control(trace = FALSE, n.cyc = 50)
            )
          }
          
          # Extract model fit statistics
          aic <- AIC(model)
          bic <- BIC(model)
          gdev <- deviance(model)
          
          # Store results
          result_row <- data.frame(
            variable = response_var,
            family = family_name,
            df = df,
            site_effect = use_site,
            AIC = aic,
            BIC = bic,
            deviance = gdev
          )
          
          results <- rbind(results, result_row)
          
        }, error = function(e) {
          cat("  Error:", e$message, "\n")
        })
      }
    }
  }
  
  # Find best model based on AIC if we have results
  if(nrow(results) > 0) {
    best_idx <- which.min(results$AIC)
    best_config <- results[best_idx, ]
    
    cat("\nBest configuration for", response_var, ":\n")
    print(best_config)
  } else {
    best_config <- NULL
    cat("\nNo successful models for", response_var, "\n")
  }
  
  return(list(results = results, best_config = best_config))
}

# ============================================================================
# PART 3: FIT OPTIMIZED GAMLSS MODELS
# ============================================================================

# Function to fit GAMLSS with optimal hyperparameters
fit_optimized_gamlss <- function(data, response_var, predictor_var, site_var, config) {
  
  if(is.null(config)) {
    cat("No configuration available for", response_var, "\n")
    return(NULL)
  }
  
  # Prepare data
  model_data <- data.frame(
    y = data[[response_var]],
    x = data[[predictor_var]],
    site = data[[site_var]]
  )
  model_data <- na.omit(model_data)
  
  # Get optimal configuration
  family_name <- as.character(config$family)
  df <- config$df
  use_site <- config$site_effect
  
  # Select distribution family
  family <- switch(family_name,
    "shash" = shash(),
    "BCT" = BCT(),
    "BCPE" = BCPE(),
    "NO" = NO()
  )
  
  # Build formulas
  if (use_site) {
    formula_mu <- as.formula(paste0("y ~ pb(x, df=", df, ") + site"))
    formula_sigma <- as.formula(paste0("~ pb(x, df=", ceiling(df/2), ")"))
  } else {
    formula_mu <- as.formula(paste0("y ~ pb(x, df=", df, ")"))
    formula_sigma <- as.formula(paste0("~ pb(x, df=", ceiling(df/2), ")"))
  }
  
  # Fit model
  if (family_name %in% c("shash", "BCT", "BCPE")) {
    model <- gamlss(
      formula = formula_mu,
      sigma.formula = formula_sigma,
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = family,
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
  } else {
    model <- gamlss(
      formula = formula_mu,
      sigma.formula = formula_sigma,
      family = family,
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
  }
  
  return(list(model = model, data = model_data))
}

# ============================================================================
# PART 4: SIMPLER APPROACH - SINGLE MODEL WITH SITE EFFECTS
# ============================================================================

# If the tuning approach fails, here's a simpler direct approach
fit_simple_gamlss_with_sites <- function(data, response_var, predictor_var, site_var) {
  
  cat("\nFitting simple GAMLSS model for", response_var, "with site effects\n")
  
  # Prepare data
  model_data <- data.frame(
    y = data[[response_var]],
    x = data[[predictor_var]],
    site = as.factor(data[[site_var]])
  )
  
  # Remove NAs
  model_data <- na.omit(model_data)
  
  if(nrow(model_data) == 0) {
    cat("No data available for", response_var, "\n")
    return(NULL)
  }
  
  # Try different approaches
  models <- list()
  
  # Approach 1: Site as fixed effect with flexible distribution
  tryCatch({
    cat("Trying shash distribution with site effects...\n")
    models$shash_site <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = shash(),
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
    cat("Success with shash!\n")
  }, error = function(e) {
    cat("Error with shash:", e$message, "\n")
  })
  
  # Approach 2: Normal distribution with site effects (more stable)
  tryCatch({
    cat("Trying normal distribution with site effects...\n")
    models$normal_site <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      family = NO(),
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
    cat("Success with normal!\n")
  }, error = function(e) {
    cat("Error with normal:", e$message, "\n")
  })
  
  # Approach 3: BCT distribution
  tryCatch({
    cat("Trying BCT distribution with site effects...\n")
    models$bct_site <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = BCT(),
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
    cat("Success with BCT!\n")
  }, error = function(e) {
    cat("Error with BCT:", e$message, "\n")
  })
  
  # Select best model based on AIC
  if(length(models) > 0) {
    aics <- sapply(models, AIC)
    best_model_name <- names(which.min(aics))
    cat("\nBest model:", best_model_name, "with AIC =", min(aics), "\n")
    return(list(model = models[[best_model_name]], data = model_data))
  } else {
    cat("All models failed for", response_var, "\n")
    return(NULL)
  }
}

# ============================================================================
# PART 5: MAIN ANALYSIS PIPELINE (SIMPLIFIED)
# ============================================================================

# Try the simpler approach
best_models <- list()

for (var in selected_variables) {
  cat("\n", strrep("=", 60), "\n")
  cat("Processing variable:", var, "\n")
  cat(strrep("=", 60), "\n")
  
  # Check if variable exists
  if(!(var %in% colnames(df_combined))) {
    cat("Warning: Variable", var, "not found in combined dataset\n")
    next
  }
  
  # Check if gestational_age exists
  if(!("gestational_age" %in% colnames(df_combined))) {
    cat("Warning: 'gestational_age' column not found\n")
    cat("Available columns:", paste(colnames(df_combined), collapse=", "), "\n")
    # Try to find alternative column names
    ga_cols <- grep("age|gestational|GA", colnames(df_combined), value=TRUE, ignore.case=TRUE)
    if(length(ga_cols) > 0) {
      cat("Possible gestational age columns:", paste(ga_cols, collapse=", "), "\n")
      # Use the first match
      predictor_col <- ga_cols[1]
      cat("Using", predictor_col, "as predictor\n")
    } else {
      cat("No gestational age column found, skipping\n")
      next
    }
  } else {
    predictor_col <- "gestational_age"
  }
  
  # Fit model using simple approach
  model_fit <- fit_simple_gamlss_with_sites(
    data = df_combined,
    response_var = var,
    predictor_var = predictor_col,
    site_var = "site"
  )
  
  if(!is.null(model_fit)) {
    best_models[[var]] <- model_fit
  }
}

# ============================================================================
# PART 6: EXTRACT PREDICTIONS
# ============================================================================

extract_predictions_simple <- function(model, data) {
  
  # Create prediction grid
  x_seq <- seq(min(data$x), max(data$x), length.out = 100)
  sites <- unique(data$site)
  
  # Initialize storage
  all_predictions <- data.frame()
  
  for (site in sites) {
    # Create new data
    newdata <- data.frame(
      x = x_seq,
      site = factor(site, levels = levels(data$site))
    )
    
    # Get predictions
    pred <- predictAll(model, newdata = newdata, type = "response")
    
    # Calculate quantiles based on distribution
    if(model$family$family[1] == "shash") {
      predictions <- data.frame(
        x = x_seq,
        site = site,
        q0.025 = qSHASH(0.025, mu = pred$mu, sigma = pred$sigma, nu = pred$nu, tau = pred$tau),
        q0.5 = qSHASH(0.5, mu = pred$mu, sigma = pred$sigma, nu = pred$nu, tau = pred$tau),
        q0.975 = qSHASH(0.975, mu = pred$mu, sigma = pred$sigma, nu = pred$nu, tau = pred$tau)
      )
    } else if(model$family$family[1] == "BCT") {
      predictions <- data.frame(
        x = x_seq,
        site = site,
        q0.025 = qBCT(0.025, mu = pred$mu, sigma = pred$sigma, nu = pred$nu, tau = pred$tau),
        q0.5 = qBCT(0.5, mu = pred$mu, sigma = pred$sigma, nu = pred$nu, tau = pred$tau),
        q0.975 = qBCT(0.975, mu = pred$mu, sigma = pred$sigma, nu = pred$nu, tau = pred$tau)
      )
    } else {
      predictions <- data.frame(
        x = x_seq,
        site = site,
        q0.025 = qNO(0.025, mu = pred$mu, sigma = pred$sigma),
        q0.5 = qNO(0.5, mu = pred$mu, sigma = pred$sigma),
        q0.975 = qNO(0.975, mu = pred$mu, sigma = pred$sigma)
      )
    }
    
    all_predictions <- rbind(all_predictions, predictions)
  }
  
  return(all_predictions)
}

# ============================================================================
# PART 7: VISUALIZATION
# ============================================================================

if(length(best_models) > 0) {
  
  # Prepare data for plotting
  plot_data <- data.frame()
  plot_predictions <- data.frame()
  
  for (var in names(best_models)) {
    # Get model and data
    model_fit <- best_models[[var]]
    model <- model_fit$model
    model_data <- model_fit$data
    
    # Add actual data points
    var_data <- data.frame(
      x = model_data$x,
      y = model_data$y,
      site = model_data$site,
      variable = var
    )
    plot_data <- rbind(plot_data, var_data)
    
    # Get predictions
    predictions <- extract_predictions_simple(model, model_data)
    predictions$variable <- var
    plot_predictions <- rbind(plot_predictions, predictions)
  }
  
  # Define colors
  custom_colors <- c(
    "B4 Band Relative Power" = "#00BFC4",
    "B5 Band Relative Power" = "#00BA38",
    "B6 Band Relative Power" = "#F8766D"
  )
  
  # Create plot
  p_multisite <- ggplot() +
    # Add points
    geom_point(data = plot_data,
               aes(x = x, y = y, color = variable, shape = site),
               size = 1.5, alpha = 0.6) +
    
    # Add 95% confidence intervals
    geom_ribbon(data = plot_predictions,
                aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
                alpha = 0.15) +
    
    # Add median lines
    geom_line(data = plot_predictions,
              aes(x = x, y = q0.5, color = variable, linetype = site),
              linewidth = 1.2) +
    
    # Facet by variable
    facet_wrap(~ variable, scales = "free_y", ncol = 1) +
    
    # Styling
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    scale_shape_manual(values = c("Site1" = 16, "Site2" = 17)) +
    scale_linetype_manual(values = c("Site1" = "solid", "Site2" = "dashed")) +
    
    labs(title = "GAMLSS Models with Multi-Site Effects",
         subtitle = "Single model per variable accounting for site-specific variations",
         x = "Gestational Age (weeks)",
         y = "Response Value",
         color = "Variable",
         fill = "Variable",
         shape = "Site",
         linetype = "Site") +
    
    theme_cowplot() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      plot.subtitle = element_text(hjust = 0.5, size = 11),
      legend.position = "bottom",
      strip.text = element_text(size = 10, face = "bold")
    )
  
  # Display plot
  print(p_multisite)
  
  # Save plot
  ggsave("/home/INT/dienye.h/python_files/GAMLSS/GAMLSS_MultiSite_Corrected.png", 
         p_multisite, width = 12, height = 14, units = 'in', bg = "white", dpi = 300)
  
  cat("\n=== ANALYSIS COMPLETE ===\n")
  cat("Models fitted successfully!\n")
  cat("Plot saved to: /home/INT/dienye.h/python_files/GAMLSS/GAMLSS_MultiSite_Corrected.png\n")
  
} else {
  cat("\n=== WARNING ===\n")
  cat("No models were successfully fitted.\n")
  cat("Please check your data and column names.\n")
}