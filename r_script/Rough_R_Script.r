# ============================================================================
# GAMLSS WITH JSU DISTRIBUTION - OPTIMAL IMPLEMENTATION
# ============================================================================

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
# PART 1: DATA PREPARATION (keeping your existing code)
# ============================================================================

# Read and prepare data from both datasets
df1 <- read.csv("/home/INT/dienye.h/python_files/dhcp_dataset_info/combined_results.csv", 
                header=TRUE, stringsAsFactors = FALSE)
df2 <- read.csv("/home/INT/dienye.h/python_files/devfetfiles/filtered_qc_3_and_above.csv", 
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

# Select only common columns
common_cols <- intersect(colnames(df1), colnames(df2))
df1_common <- df1[, common_cols]
df2_common <- df2[, common_cols]

# Combine datasets
df_combined <- rbind(df1_common, df2_common)

# Convert site to factor
df_combined$site <- as.factor(df_combined$site)

# Select key variables
selected_variables <- c("B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

# ============================================================================
# PART 2: COMPREHENSIVE MODEL FITTING WITH JSU AND OTHER DISTRIBUTIONS
# ============================================================================

fit_comprehensive_gamlss <- function(data, response_var, predictor_var, site_var) {
  
  cat("\n=== Fitting models for", response_var, "===\n")
  
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
  
  cat("Sample size:", nrow(model_data), "\n")
  cat("Sites:", levels(model_data$site), "\n")
  
  # Store all models
  models <- list()
  
  # Priority 1: JSU (Best SHASH alternative)
  cat("\n1. Trying JSU (Johnson's SU) distribution...\n")
  tryCatch({
    # Start with simpler model for initial values
    init_model <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ 1,
      family = NO(),
      data = model_data,
      trace = FALSE
    )
    
    # Fit JSU with starting values
    models$jsu <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = JSU(),
      data = model_data,
      start.from = init_model,
      control = gamlss.control(trace = FALSE, n.cyc = 100)
    )
    cat("   Success with JSU! AIC =", AIC(models$jsu), "\n")
    cat("   Converged:", models$jsu$converged, "\n")
  }, error = function(e) {
    cat("   Error with JSU:", e$message, "\n")
  })
  
  # Priority 2: SHASH with mixed algorithm (might work better)
  cat("\n2. Trying SHASH with mixed algorithm...\n")
  tryCatch({
    models$shash_mixed <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = SHASH(),
      data = model_data,
      method = mixed(10, 50),
      control = gamlss.control(trace = FALSE)
    )
    cat("   Success with SHASH (mixed)! AIC =", AIC(models$shash_mixed), "\n")
    cat("   Converged:", models$shash_mixed$converged, "\n")
  }, error = function(e) {
    cat("   Error with SHASH:", e$message, "\n")
  })
  
  # Priority 3: BCT (Already working well)
  cat("\n3. Trying BCT distribution...\n")
  tryCatch({
    models$bct <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = BCT(),
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
    cat("   Success with BCT! AIC =", AIC(models$bct), "\n")
  }, error = function(e) {
    cat("   Error with BCT:", e$message, "\n")
  })
  
  # Priority 4: SEP3 (Skew Exponential Power)
  cat("\n4. Trying SEP3 distribution...\n")
  tryCatch({
    models$sep3 <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = SEP3(),
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
    cat("   Success with SEP3! AIC =", AIC(models$sep3), "\n")
  }, error = function(e) {
    cat("   Error with SEP3:", e$message, "\n")
  })
  
  # Priority 5: BCPE (Box-Cox Power Exponential)
  cat("\n5. Trying BCPE distribution...\n")
  tryCatch({
    models$bcpe <- gamlss(
      y ~ pb(x) + site,
      sigma.formula = ~ pb(x),
      nu.formula = ~ 1,
      tau.formula = ~ 1,
      family = BCPE(),
      data = model_data,
      control = gamlss.control(trace = FALSE)
    )
    cat("   Success with BCPE! AIC =", AIC(models$bcpe), "\n")
  }, error = function(e) {
    cat("   Error with BCPE:", e$message, "\n")
  })
  
  # Select best model
  if(length(models) > 0) {
    # Get AICs for all converged models
    aics <- sapply(models, function(m) {
      if(!is.null(m$converged) && !m$converged) {
        return(AIC(m) + 100)  # Penalty for non-convergence
      }
      return(AIC(m))
    })
    
    best_model_name <- names(which.min(aics))
    
    cat("\n=== Model comparison ===\n")
    comparison <- data.frame(
      Model = names(aics),
      AIC = aics,
      Delta_AIC = aics - min(aics)
    )
    comparison <- comparison[order(comparison$AIC), ]
    print(comparison)
    
    cat("\nBest model:", best_model_name, "\n")
    
    return(list(
      model = models[[best_model_name]], 
      data = model_data, 
      all_models = models,
      best_distribution = best_model_name
    ))
  } else {
    cat("All models failed for", response_var, "\n")
    return(NULL)
  }
}

# ============================================================================
# PART 3: FIT MODELS FOR ALL VARIABLES
# ============================================================================

best_models <- list()

for (var in selected_variables) {
  cat("\n", strrep("=", 60), "\n")
  cat("Processing variable:", var, "\n")
  cat(strrep("=", 60), "\n")
  
  # Check if variable exists
  if(!(var %in% colnames(df_combined))) {
    cat("Warning: Variable", var, "not found\n")
    next
  }
  
  # Check for gestational age column
  ga_col <- if("gestational_age" %in% colnames(df_combined)) {
    "gestational_age"
  } else {
    # Find alternative column names
    ga_cols <- grep("age|gestational|GA", colnames(df_combined), value=TRUE, ignore.case=TRUE)
    if(length(ga_cols) > 0) ga_cols[1] else NULL
  }
  
  if(is.null(ga_col)) {
    cat("No gestational age column found\n")
    next
  }
  
  # Fit models
  model_fit <- fit_comprehensive_gamlss(
    data = df_combined,
    response_var = var,
    predictor_var = ga_col,
    site_var = "site"
  )
  
  if(!is.null(model_fit)) {
    best_models[[var]] <- model_fit
  }
}

# ============================================================================
# PART 4: EXTRACT PREDICTIONS
# ============================================================================

extract_predictions_flexible <- function(model, data, distribution_name) {
  
  # Create prediction grid
  x_seq <- seq(min(data$x), max(data$x), length.out = 100)
  sites <- unique(data$site)
  
  # Get the quantile function for the distribution
  qfun <- switch(distribution_name,
    "jsu" = qJSU,
    "shash_mixed" = qSHASH,
    "bct" = qBCT,
    "sep3" = qSEP3,
    "bcpe" = qBCPE,
    qNO  # Default
  )
  
  # Initialize storage
  all_predictions <- data.frame()
  
  for (site in sites) {
    # Create new data
    newdata <- data.frame(
      x = x_seq,
      site = factor(site, levels = levels(data$site))
    )
    
    tryCatch({
      # Get predictions
      pred <- predictAll(model, newdata = newdata, type = "response")
      
      # Calculate quantiles
      predictions <- data.frame(
        x = x_seq,
        site = site,
        q0.025 = qfun(0.025, mu = pred$mu, sigma = pred$sigma, 
                      nu = pred$nu, tau = pred$tau),
        q0.5 = qfun(0.5, mu = pred$mu, sigma = pred$sigma, 
                    nu = pred$nu, tau = pred$tau),
        q0.975 = qfun(0.975, mu = pred$mu, sigma = pred$sigma, 
                      nu = pred$nu, tau = pred$tau)
      )
      
      all_predictions <- rbind(all_predictions, predictions)
      
    }, error = function(e) {
      cat("Error in predictions for site", site, ":", e$message, "\n")
    })
  }
  
  return(all_predictions)
}

# ============================================================================
# PART 5: VISUALIZATION
# ============================================================================

if(length(best_models) > 0) {
  
  # Prepare data for plotting
  plot_data <- data.frame()
  plot_predictions <- data.frame()
  
  for (var in names(best_models)) {
    cat("\nGenerating predictions for", var, "\n")
    
    model_fit <- best_models[[var]]
    model <- model_fit$model
    model_data <- model_fit$data
    distribution <- model_fit$best_distribution
    
    # Add actual data points
    var_data <- data.frame(
      x = model_data$x,
      y = model_data$y,
      site = model_data$site,
      variable = var,
      distribution = distribution
    )
    plot_data <- rbind(plot_data, var_data)
    
    # Get predictions
    predictions <- extract_predictions_flexible(model, model_data, distribution)
    if(nrow(predictions) > 0) {
      predictions$variable <- var
      predictions$distribution <- distribution
      plot_predictions <- rbind(plot_predictions, predictions)
    }
  }
  
  # Define colors
  custom_colors <- c(
    "B4 Band Relative Power" = "#00BFC4",
    "B5 Band Relative Power" = "#00BA38",
    "B6 Band Relative Power" = "#F8766D"
  )
  
  # Create main plot
  p_main <- ggplot() +
    # Add points
    geom_point(data = plot_data,
               aes(x = x, y = y, color = variable, shape = site),
               size = 1.2, alpha = 0.5) +
    
    # Add confidence intervals
    geom_ribbon(data = plot_predictions,
                aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
                alpha = 0.15) +
    
    # Add median lines
    geom_line(data = plot_predictions,
              aes(x = x, y = q0.5, color = variable, linetype = site),
              linewidth = 1) +
    
    # Styling
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    scale_shape_manual(values = c("Site1" = 16, "Site2" = 17)) +
    scale_linetype_manual(values = c("Site1" = "solid", "Site2" = "dashed")) +
    
    labs(title = "GAMLSS Models with Optimal Distributions",
         subtitle = "Using JSU/BCT distributions with site-specific effects",
         x = "Gestational Age (weeks)",
         y = "Band Relative Power",
         color = "Variable",
         fill = "Variable",
         shape = "Site",
         linetype = "Site") +
    
    scale_x_continuous(breaks = seq(22, 44, by = 2)) +
    
    theme_cowplot() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5, size = 11),
      legend.position = "bottom",
      legend.box = "horizontal"
    )
  
  # Display and save
  print(p_main)
  ggsave("/home/INT/dienye.h/python_files/GAMLSS/GAMLSS_Optimal_Distributions.png", 
         p_main, width = 10, height = 8, units = 'in', bg = "white", dpi = 300)
  
  # Create faceted version
  p_facet <- ggplot() +
    geom_point(data = plot_data,
               aes(x = x, y = y, color = variable, shape = site),
               size = 1.5, alpha = 0.6) +
    geom_ribbon(data = plot_predictions,
                aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
                alpha = 0.15) +
    geom_line(data = plot_predictions,
              aes(x = x, y = q0.5, color = variable, linetype = site),
              linewidth = 1.2) +
    facet_wrap(~ variable, scales = "free_y", ncol = 1) +
    scale_color_manual(values = custom_colors) +
    scale_fill_manual(values = custom_colors) +
    scale_shape_manual(values = c("Site1" = 16, "Site2" = 17)) +
    scale_linetype_manual(values = c("Site1" = "solid", "Site2" = "dashed")) +
    labs(title = "GAMLSS Models by Variable",
         subtitle = "With optimal distribution for each variable",
         x = "Gestational Age (weeks)",
         y = "Response Value") +
    theme_cowplot() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "bottom"
    )
  
  print(p_facet)
  ggsave("/home/INT/dienye.h/python_files/GAMLSS/GAMLSS_Optimal_Faceted.png", 
         p_facet, width = 12, height = 14, units = 'in', bg = "white", dpi = 300)
}

# ============================================================================
# PART 6: FINAL SUMMARY
# ============================================================================

cat("\n", strrep("=", 60), "\n")
cat("FINAL MODEL SUMMARY\n")
cat(strrep("=", 60), "\n")

summary_table <- data.frame()

for (var in names(best_models)) {
  model_fit <- best_models[[var]]
  model <- model_fit$model
  
  # Get site effects
  coef_all <- coef(model)
  site_coef <- coef_all[grep("site", names(coef_all))]
  
  summary_row <- data.frame(
    Variable = var,
    Distribution = model_fit$best_distribution,
    AIC = AIC(model),
    BIC = BIC(model),
    Converged = ifelse(is.null(model$converged), TRUE, model$converged),
    Site_Effect = ifelse(length(site_coef) > 0, round(site_coef[1], 4), NA),
    N = nrow(model_fit$data)
  )
  
  summary_table <- rbind(summary_table, summary_row)
}

print(summary_table)

# Save summary
write.csv(summary_table, 
          "/home/INT/dienye.h/python_files/GAMLSS/model_summary_optimal.csv",
          row.names = FALSE)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Models fitted with optimal distributions\n")
cat("JSU (Johnson's SU) used as primary flexible distribution\n")
cat("Results saved successfully\n")