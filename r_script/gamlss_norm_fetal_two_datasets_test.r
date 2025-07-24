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

# Add dataset/site identifier - this will be our random effect variable
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

# Combine datasets into one
df_combined <- rbind(df1, df2)

# Convert site to factor for modeling
df_combined$site <- as.factor(df_combined$site)

# Select key variables for analysis
selected_variables <- c("B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

# ============================================================================
# PART 2: HYPERPARAMETER TUNING FOR GAMLSS
# ============================================================================

# Function to perform hyperparameter tuning for GAMLSS
tune_gamlss <- function(data, response_var, predictor_var, site_var) {
  
  cat("\n=== Starting Hyperparameter Tuning for", response_var, "===\n")
  
  # Define hyperparameter grid
  # For GAMLSS, key hyperparameters include:
  # 1. Distribution family (shash, BCT, BCPE, etc.)
  # 2. Smoothing parameter selection method (GAIC, ML, GCV)
  # 3. Degrees of freedom for smoothers
  # 4. Whether to include random effects for site
  
  # Distribution families to try
  families <- list(
    "shash" = shash(),    # Sinh-Arcsinh (flexible for skewness and kurtosis)
    "BCT" = BCT(),        # Box-Cox t (handles skewness and heavy tails)
    "BCPE" = BCPE(),      # Box-Cox Power Exponential (very flexible)
    "NO" = NO()           # Normal (baseline comparison)
  )
  
  # Degrees of freedom options for smoothers
  df_options <- c(3, 5, 7, 9)
  
  # Whether to include site as random effect
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
  
  # Grid search over hyperparameters
  for (family_name in names(families)) {
    for (df in df_options) {
      for (use_site in site_options) {
        
        cat("Testing: Family =", family_name, ", df =", df, ", Site effect =", use_site, "\n")
        
        tryCatch({
          
          # Build formula based on whether to include site effects
          if (use_site) {
            # Model with site as random effect
            # GAMLSS allows random effects through re() function
            formula_mu <- as.formula(paste0("y ~ s(x, df=", df, ") + re(random=~1|site)"))
            formula_sigma <- as.formula(paste0("~ s(x, df=", ceiling(df/2), ")"))
          } else {
            # Model without site effects
            formula_mu <- as.formula(paste0("y ~ s(x, df=", df, ")"))
            formula_sigma <- as.formula(paste0("~ s(x, df=", ceiling(df/2), ")"))
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
              control = gamlss.control(trace = FALSE)
            )
          } else {
            # For 2-parameter distributions
            model <- gamlss(
              formula = formula_mu,
              sigma.formula = formula_sigma,
              family = families[[family_name]],
              data = model_data,
              control = gamlss.control(trace = FALSE)
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
            deviance = gdev,
            n_params = length(coef(model))
          )
          
          results <- rbind(results, result_row)
          
        }, error = function(e) {
          cat("  Error:", e$message, "\n")
        })
      }
    }
  }
  
  # Find best model based on AIC
  best_idx <- which.min(results$AIC)
  best_config <- results[best_idx, ]
  
  cat("\nBest configuration for", response_var, ":\n")
  print(best_config)
  
  return(list(results = results, best_config = best_config))
}

# ============================================================================
# PART 3: FIT OPTIMIZED GAMLSS MODELS WITH SITE EFFECTS
# ============================================================================

# Function to fit GAMLSS with optimal hyperparameters and site effects
fit_optimized_gamlss <- function(data, response_var, predictor_var, site_var, config) {
  
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
    formula_mu <- as.formula(paste0("y ~ s(x, df=", df, ") + re(random=~1|site)"))
    formula_sigma <- as.formula(paste0("~ s(x, df=", ceiling(df/2), ")"))
  } else {
    formula_mu <- as.formula(paste0("y ~ s(x, df=", df, ")"))
    formula_sigma <- as.formula(paste0("~ s(x, df=", ceiling(df/2), ")"))
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
# PART 4: EXTRACT PREDICTIONS AND CONFIDENCE INTERVALS
# ============================================================================

# Function to extract predictions with site-specific adjustments
extract_predictions_with_sites <- function(model, data, quantiles = c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)) {
  
  # Create prediction grid
  x_seq <- seq(min(data$x), max(data$x), length.out = 100)
  sites <- unique(data$site)
  
  # Initialize storage for predictions
  all_predictions <- data.frame()
  
  for (site in sites) {
    # Create new data for predictions
    newdata <- data.frame(
      x = x_seq,
      site = factor(rep(site, length(x_seq)), levels = levels(data$site))
    )
    
    # Get predictions for all parameters
    pred_params <- predictAll(model, newdata = newdata, type = "response")
    
    # Extract quantile function
    qfun <- model$family$qf
    
    # Calculate quantiles
    predictions_quantiles <- data.frame(x = x_seq, site = site)
    
    for (q in quantiles) {
      quant_name <- paste0("q", q)
      
      # Calculate quantiles using distribution parameters
      if (model$family$nopar == 4) {
        predictions_quantiles[[quant_name]] <- qfun(p = q, 
                                                    mu = pred_params$mu,
                                                    sigma = pred_params$sigma,
                                                    nu = pred_params$nu,
                                                    tau = pred_params$tau)
      } else if (model$family$nopar == 3) {
        predictions_quantiles[[quant_name]] <- qfun(p = q,
                                                    mu = pred_params$mu,
                                                    sigma = pred_params$sigma,
                                                    nu = pred_params$nu)
      } else {
        predictions_quantiles[[quant_name]] <- qfun(p = q,
                                                    mu = pred_params$mu,
                                                    sigma = pred_params$sigma)
      }
    }
    
    all_predictions <- rbind(all_predictions, predictions_quantiles)
  }
  
  return(all_predictions)
}

# ============================================================================
# PART 5: MAIN ANALYSIS PIPELINE
# ============================================================================

# Run hyperparameter tuning for each variable
tuning_results <- list()
best_models <- list()

for (var in selected_variables) {
  cat("\n", strrep("=", 60), "\n")
  cat("Processing variable:", var, "\n")
  cat(strrep("=", 60), "\n")
  
  # Run hyperparameter tuning
  tuning <- tune_gamlss(
    data = df_combined,
    response_var = var,
    predictor_var = "gestational_age",
    site_var = "site"
  )
  
  tuning_results[[var]] <- tuning
  
  # Fit model with best configuration
  model_fit <- fit_optimized_gamlss(
    data = df_combined,
    response_var = var,
    predictor_var = "gestational_age",
    site_var = "site",
    config = tuning$best_config
  )
  
  best_models[[var]] <- model_fit
}

# ============================================================================
# PART 6: VISUALIZATION
# ============================================================================

# Prepare data for plotting
plot_data <- data.frame()
plot_predictions <- data.frame()

for (var in selected_variables) {
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
  predictions <- extract_predictions_with_sites(model, model_data)
  predictions$variable <- var
  plot_predictions <- rbind(plot_predictions, predictions)
}

# Define colors
custom_colors <- c(
  "B4 Band Relative Power" = "#00BFC4",
  "B5 Band Relative Power" = "#00BA38",
  "B6 Band Relative Power" = "#F8766D"
)

# Create plot with site-specific effects
p_multisite <- ggplot() +
  # Add points
  geom_point(data = plot_data,
             aes(x = x, y = y, color = variable, shape = site),
             size = 1.5, alpha = 0.6) +
  
  # Add 95% confidence intervals
  geom_ribbon(data = plot_predictions,
              aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
              alpha = 0.15) +
  
  # Add median lines for each site
  geom_line(data = plot_predictions,
            aes(x = x, y = q0.5, color = variable, linetype = site),
            linewidth = 1.2) +
  
  # Facet by variable for clarity
  facet_wrap(~ variable, scales = "free_y", ncol = 1) +
  
  # Styling
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +
  scale_shape_manual(values = c("Site1" = 16, "Site2" = 17)) +
  scale_linetype_manual(values = c("Site1" = "solid", "Site2" = "dashed")) +
  
  labs(title = "Optimized GAMLSS Models with Multi-Site Effects",
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
ggsave("/home/INT/dienye.h/python_files/GAMLSS/Optimized_GAMLSS_MultiSite.png", 
       p_multisite, width = 12, height = 14, units = 'in', bg = "white", dpi = 300)

# ============================================================================
# PART 7: MODEL DIAGNOSTICS AND SUMMARY
# ============================================================================

# Print detailed summary for each variable
cat("\n", strrep("=", 60), "\n")
cat("MODEL SUMMARIES AND DIAGNOSTICS\n")
cat(strrep("=", 60), "\n")

for (var in selected_variables) {
  cat("\n### Variable:", var, "###\n")
  
  # Get best configuration
  best_config <- tuning_results[[var]]$best_config
  cat("\nOptimal Configuration:\n")
  print(best_config)
  
  # Get model
  model <- best_models[[var]]$model
  
  # Print model summary
  cat("\nModel Summary:\n")
  print(summary(model))
  
  # Extract site effects if included
  if (best_config$site_effect) {
    cat("\nSite Effects:\n")
    # Extract random effects
    ranef <- getSmo(model)
    if (!is.null(ranef)) {
      print(ranef)
    }
  }
  
  # Model diagnostics
  cat("\nModel Fit Statistics:\n")
  cat("AIC:", AIC(model), "\n")
  cat("BIC:", BIC(model), "\n")
  cat("Global Deviance:", deviance(model), "\n")
  cat("Degrees of Freedom:", model$df.fit, "\n")
}

# ============================================================================
# PART 8: SAVE HYPERPARAMETER TUNING RESULTS
# ============================================================================

# Combine all tuning results
all_tuning_results <- data.frame()
for (var in names(tuning_results)) {
  results <- tuning_results[[var]]$results
  all_tuning_results <- rbind(all_tuning_results, results)
}

# Save tuning results
write.csv(all_tuning_results, 
          "/home/INT/dienye.h/python_files/GAMLSS/hyperparameter_tuning_results.csv",
          row.names = FALSE)

cat("\n", strrep("=", 60), "\n")
cat("ANALYSIS COMPLETE!\n")
cat("- Models fitted with optimal hyperparameters\n")
cat("- Site-specific effects included where beneficial\n")
cat("- Results and plots saved\n")
cat(strrep("=", 60), "\n")