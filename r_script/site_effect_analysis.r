# ============================================
# SITE EFFECT ANALYSIS AND UNIFIED CURVES
# ============================================

library(mgcv)
library(ggplot2)
library(dplyr)
library(patchwork)


# Load the saved models and data
all_fitted_models <- readRDS("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/all_fitted_models.rds")
results <- read.csv("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/model_results.csv")
data_splits <- readRDS("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/data_splits.rds")

df1 <- data_splits$df1
df_test_allsites <- data_splits$df_test

# Now you can proceed with p-value analysis, comprehensive evaluation, etc.
cat("Loaded", length(all_fitted_models), "fitted models\n")
cat("Ready for analysis!\n")

# ============================================
# PART 1: EXTRACT P-VALUES FOR SITE EFFECTS
# ============================================

# Function to extract site effect significance from models
extract_site_significance <- function(model, model_name) {
  
  result <- data.frame(
    model = model_name,
    site_parametric_p = NA,
    site_smooth_p = NA,
    site_effect_size = NA,
    site_variance_explained = NA
  )
  
  tryCatch({
    model_summary <- summary(model)
    
    # For fixed effects models (m2a-m2e)
    if (grepl("m2", model_name)) {
      # In GAM models with multiple linear predictors, we need to check each one
      # For gaulss family: first predictor is mean, second is variance
      # For shash family: four predictors (mean, variance, skewness, kurtosis)
      
      if (!is.null(model_summary$p.table)) {
        # This is for simple GAM
        p_table <- model_summary$p.table
      } else if (!is.null(model_summary$p.coeff)) {
        # This is for GAM with multiple predictors
        p_table <- model_summary$p.coeff[[1]]  # First predictor (mean)
      } else {
        # Try another approach - look at the model coefficients directly
        coef_summary <- summary(model)
        if (is.list(coef_summary)) {
          # For models with multiple linear predictors
          p_table <- coef_summary[[1]]$p.table
        }
      }
      
      if (!is.null(p_table) && nrow(p_table) > 0) {
        site_rows <- grep("site", rownames(p_table), ignore.case = TRUE)
        if (length(site_rows) > 0) {
          result$site_parametric_p <- min(p_table[site_rows, ncol(p_table)])  # Last column is usually p-value
          result$site_effect_size <- mean(abs(p_table[site_rows, 1]))  # First column is estimate
        }
      }
    }
    
    # For random effects models (m3a-m3e) - this part usually works
    if (grepl("m3", model_name)) {
      s_table <- model_summary$s.table
      if (!is.null(s_table)) {
        site_rows <- grep("site", rownames(s_table), ignore.case = TRUE)
        if (length(site_rows) > 0) {
          result$site_smooth_p <- min(s_table[site_rows, ncol(s_table)])
          result$site_variance_explained <- sum(s_table[site_rows, "edf"])
        }
      }
    }
    
  }, error = function(e) {
    cat("Error extracting significance for", model_name, ":", e$message, "\n")
  })
  
  return(result)
}

# Function to perform likelihood ratio test between models
compare_models_lrt <- function(model_no_site, model_with_site) {
  # Likelihood ratio test
  lrt <- anova(model_no_site, model_with_site, test = "LRT")
  
  # Extract p-value
  if (!is.null(lrt) && nrow(lrt) > 1) {
    p_value <- lrt[2, "Pr(>Chi)"]
    chi_stat <- lrt[2, "Deviance"]
    df <- lrt[2, "Df"]
    
    return(data.frame(
      LRT_statistic = chi_stat,
      LRT_df = df,
      LRT_pvalue = p_value,
      significant = p_value < 0.05
    ))
  }
  
  return(data.frame(LRT_statistic = NA, LRT_df = NA, LRT_pvalue = NA, significant = NA))
}

# ============================================
# PART 2: ANALYZE SITE EFFECTS FOR ALL FEATURES
# ============================================

# Create a comprehensive site effect analysis
site_effect_results <- data.frame()

for (feature in y_values) {
  cat("\nAnalyzing site effects for:", feature, "\n")
  
  # Get models for this feature
  # Assuming you have stored: m1b (no site), m2b (fixed site), m3b (random site)
  # as the smooth mean models which are often best
  
  model_no_site <- all_fitted_models[[paste(feature, "m1b", sep="_")]]
  model_fixed_site <- all_fitted_models[[paste(feature, "m2b", sep="_")]]
  model_random_site <- all_fitted_models[[paste(feature, "m3b", sep="_")]]
  
  # Compare no site vs fixed site
  lrt_fixed <- compare_models_lrt(model_no_site, model_fixed_site)
  
  # Compare no site vs random site
  lrt_random <- compare_models_lrt(model_no_site, model_random_site)
  
  # Extract site significance from models
  fixed_sig <- extract_site_significance(model_fixed_site, "m2b")
  random_sig <- extract_site_significance(model_random_site, "m3b")
  
  # Combine results
  result_row <- data.frame(
    Feature = feature,
    Fixed_Site_LRT_p = lrt_fixed$LRT_pvalue,
    Random_Site_LRT_p = lrt_random$LRT_pvalue,
    Fixed_Site_Param_p = fixed_sig$site_parametric_p,
    Random_Site_Smooth_p = random_sig$site_smooth_p,
    Site_Significant = lrt_fixed$LRT_pvalue < 0.05 | lrt_random$LRT_pvalue < 0.05
  )
  
  site_effect_results <- rbind(site_effect_results, result_row)
}

print(site_effect_results)
write.csv(site_effect_results, 
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/site_effect_analysis.csv",
          row.names = FALSE)

# ============================================
# PART 3: CREATE UNIFIED NORMATIVE CURVES
# ============================================

# Three approaches for unified curves:

# APPROACH 1: Average the predictions across sites
create_unified_curves_averaged <- function(feature_name, model, df_train, df_test) {
  
  # Create prediction grid
  age_grid <- seq(min(df_train$gestational_age), 
                  max(df_train$gestational_age), 
                  length.out = 100)
  
  # Predict for each site
  pred_dhcp <- data.frame(
    x = age_grid,
    site_id = factor("dHCP", levels = levels(df_train$site_id))
  )
  
  pred_marsfet <- data.frame(
    x = age_grid,
    site_id = factor("MarsFet", levels = levels(df_train$site_id))
  )
  
  # Get predictions
  params_dhcp <- predict(model, newdata = pred_dhcp)
  params_marsfet <- predict(model, newdata = pred_marsfet)
  
  # Average the parameters (on the linear predictor scale)
  params_averaged <- (params_dhcp + params_marsfet) / 2
  
  # Convert to quantiles
  quantiles <- pnorm(c(-2, -1, 0, 1, 2))
  
  if(ncol(params_averaged) == 2){
    params_averaged <- cbind(params_averaged, matrix(0, nrow = nrow(params_averaged), ncol = 2))
  }
  
  quantiles_df <- as.data.frame(sapply(quantiles,
                                       function(q){gamlss.dist::qSHASHo2(q, 
                                                                         params_averaged[,1],
                                                                         exp(params_averaged[,2]),
                                                                         params_averaged[,3],
                                                                         exp(params_averaged[,4]))}))
  
  quantiles_df$x <- age_grid
  
  return(quantiles_df)
}

# APPROACH 2: Use the larger/reference site as the standard
create_unified_curves_reference <- function(feature_name, model, df_train, reference_site = "dHCP") {
  
  # Create prediction grid for reference site only
  age_grid <- seq(min(df_train$gestational_age), 
                  max(df_train$gestational_age), 
                  length.out = 100)
  
  pred_data <- data.frame(
    x = age_grid,
    site_id = factor(reference_site, levels = levels(df_train$site_id))
  )
  
  # Get predictions for reference site
  params_ref <- predict(model, newdata = pred_data)
  
  # Convert to quantiles
  quantiles <- pnorm(c(-2, -1, 0, 1, 2))
  
  if(ncol(params_ref) == 2){
    params_ref <- cbind(params_ref, matrix(0, nrow = nrow(params_ref), ncol = 2))
  }
  
  quantiles_df <- as.data.frame(sapply(quantiles,
                                       function(q){gamlss.dist::qSHASHo2(q, 
                                                                         params_ref[,1],
                                                                         exp(params_ref[,2]),
                                                                         params_ref[,3],
                                                                         exp(params_ref[,4]))}))
  
  quantiles_df$x <- age_grid
  
  return(quantiles_df)
}

# APPROACH 3: Remove site effects and use marginal predictions
create_unified_curves_marginal <- function(feature_name, df_train) {
  
  # Fit a model without site effects for unified curves
  x <- df_train$gestational_age
  y <- df_train[[feature_name]]
  
  # Fit simple smooth model without sites
  model_unified <- gam(list(y ~ s(x), ~ 1),
                       family = gaulss(),
                       optimizer = 'efs',
                       data = df_train)
  
  # Create prediction grid
  age_grid <- seq(min(x), max(x), length.out = 100)
  pred_data <- data.frame(x = age_grid)
  
  # Get predictions
  params_unified <- predict(model_unified, newdata = pred_data)
  
  # Convert to quantiles
  quantiles <- pnorm(c(-2, -1, 0, 1, 2))
  
  quantiles_df <- as.data.frame(sapply(quantiles,
                                       function(q){qnorm(p=q, 
                                                        mean=params_unified[,1], 
                                                        sd=exp(params_unified[,2]))}))
  
  quantiles_df$x <- age_grid
  
  return(quantiles_df)
}

# ============================================
# PART 4: GENERATE UNIFIED PLOTS
# ============================================

# Example for one feature - apply to all in your loop
for (i in 1:nrow(best_models)) {
  
  feature_name <- best_models$Y_feature[i]
  model_name <- best_models$Model[i]
  
  cat("\nCreating unified curves for:", feature_name, "\n")
  
  # Get the best model
  model_key <- paste(feature_name, model_name, sep="_")
  best_model <- all_fitted_models[[model_key]]
  
  if (is.null(best_model)) next
  
  # Prepare data
  train_plot_data <- data.frame(
    x = df1$gestational_age,
    y = df1[[feature_name]],
    site_id = df1$site_id,
    dataset = "Training"
  )
  
  test_plot_data <- data.frame(
    x = df_test$gestational_age,
    y = df_test[[feature_name]],
    site_id = df_test$site_id,
    dataset = "Test"
  )
  
  combined_data <- rbind(train_plot_data, test_plot_data)
  
  # Generate unified curves using different approaches
  
  # Check if model has site effects
  has_site_effects <- grepl("m2|m3", model_name)
  
  if (has_site_effects) {
    # Approach 1: Averaged
    unified_curves_avg <- create_unified_curves_averaged(feature_name, best_model, df1, df_test)
    
    # Approach 2: Reference site (dHCP as it's larger)
    unified_curves_ref <- create_unified_curves_reference(feature_name, best_model, df1, "dHCP")
  } else {
    # If no site effects in model, use the model predictions directly
    age_grid <- seq(min(df1$gestational_age), max(df1$gestational_age), length.out = 100)
    pred_data <- data.frame(x = age_grid)
    params <- predict(best_model, newdata = pred_data)
    
    if(ncol(params) == 2){
      params <- cbind(params, matrix(0, nrow = nrow(params), ncol = 2))
    }
    
    quantiles <- pnorm(c(-2, -1, 0, 1, 2))
    unified_curves_ref <- as.data.frame(sapply(quantiles,
                                               function(q){gamlss.dist::qSHASHo2(q, 
                                                                                 params[,1],
                                                                                 exp(params[,2]),
                                                                                 params[,3],
                                                                                 exp(params[,4]))}))
    unified_curves_ref$x <- age_grid
  }
  
  # Approach 3: Marginal (always available)
  unified_curves_marginal <- create_unified_curves_marginal(feature_name, df1)
  
  # Reshape for plotting
  unified_long_ref <- reshape2::melt(unified_curves_ref, id.vars = "x", 
                                     variable.name = "quantile", value.name = "value")
  unified_long_marginal <- reshape2::melt(unified_curves_marginal, id.vars = "x",
                                          variable.name = "quantile", value.name = "value")
  
  quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")
  quantile_labels <- c("2.3%", "15.9%", "50%", "84.1%", "97.7%")
  
  # Plot 1: Reference Site Approach
  p_unified_ref <- ggplot(combined_data) +
    geom_line(data = unified_long_ref,
              aes(x = x, y = value, linetype = quantile),
              linewidth = 0.8, color = "darkblue") +
    geom_point(aes(x = x, y = y, color = interaction(site_id, dataset)),
               size = 1.5, alpha = 0.4) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    scale_color_manual(values = c("dHCP.Training" = "lightblue",
                                  "MarsFet.Training" = "lightgreen",
                                  "dHCP.Test" = "salmon",
                                  "MarsFet.Test" = "orange")) +
    labs(title = paste(feature_name, "- Unified Curves (Reference: dHCP)"),
         subtitle = paste("Model:", model_name, 
                         "| Site Effect p:", 
                         round(site_effect_results$Fixed_Site_LRT_p[site_effect_results$Feature == feature_name], 3)),
         x = "Gestational Age (weeks)",
         y = feature_name) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Plot 2: Marginal Approach (no site effects)
  p_unified_marginal <- ggplot(combined_data) +
    geom_line(data = unified_long_marginal,
              aes(x = x, y = value, linetype = quantile),
              linewidth = 0.8, color = "darkred") +
    geom_point(aes(x = x, y = y, shape = site_id, alpha = dataset),
               color = "gray40", size = 1.5) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    scale_alpha_manual(values = c("Training" = 0.6, "Test" = 0.3)) +
    scale_shape_manual(values = c("dHCP" = 17, "MarsFet" = 16)) +
    labs(title = paste(feature_name, "- Unified Curves (Pooled/Marginal)"),
         subtitle = "Site effects removed - single normative reference",
         x = "Gestational Age (weeks)",
         y = feature_name) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Combined plot
  p_combined_unified <- p_unified_ref / p_unified_marginal +
    plot_annotation(
      title = paste("Unified Normative Curves -", feature_name),
      subtitle = paste("Site Effect Significant:", 
                      site_effect_results$Site_Significant[site_effect_results$Feature == feature_name])
    )
  
  # Save plots
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves_unified/",
                gsub(" ", "_", feature_name), "_unified_curves.png"),
         p_combined_unified, width = 12, height = 10, dpi = 300)
}

# ============================================
# PART 5: STATISTICAL SUMMARY OF SITE EFFECTS
# ============================================

cat("\n================================================\n")
cat("SITE EFFECT STATISTICAL SUMMARY\n")
cat("================================================\n")

# Count significant site effects
n_significant <- sum(site_effect_results$Site_Significant, na.rm = TRUE)
n_total <- nrow(site_effect_results)

cat("Features with significant site effects:", n_significant, "/", n_total, 
    "(", round(100*n_significant/n_total, 1), "%)\n\n")

# Show which approach is best for each feature
recommendations <- site_effect_results %>%
  mutate(
    Recommendation = case_when(
      !Site_Significant ~ "Use marginal/pooled model (no site adjustment needed)",
      Site_Significant & Fixed_Site_LRT_p < Random_Site_LRT_p ~ "Use fixed site effects",
      Site_Significant ~ "Use random site effects or site-specific models",
      TRUE ~ "Further investigation needed"
    )
  )

print(recommendations[, c("Feature", "Site_Significant", "Recommendation")])

write.csv(recommendations,
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/site_modeling_recommendations.csv",
          row.names = FALSE)


# ============================================
# GROUP COMPARISON STATISTICS FOR NORMATIVE MODELING
# ============================================

library(dplyr)
library(tidyr)

# Function to calculate comprehensive group comparison statistics
# Modified function with better error handling
calculate_group_statistics <- function(z_scores_g1, z_scores_g2, 
                                      group1_name, group2_name,
                                      feature_name) {
  
  # Remove NAs
  z_scores_g1 <- z_scores_g1[!is.na(z_scores_g1) & !is.infinite(z_scores_g1)]
  z_scores_g2 <- z_scores_g2[!is.na(z_scores_g2) & !is.infinite(z_scores_g2)]
  
  # Sample sizes
  n_g1 <- length(z_scores_g1)
  n_g2 <- length(z_scores_g2)
  
  # Convert z-scores to centiles
  centiles_g1 <- pnorm(z_scores_g1)
  centiles_g2 <- pnorm(z_scores_g2)
  
  # Median centiles
  median_centile_g1 <- median(centiles_g1, na.rm = TRUE)
  median_centile_g2 <- median(centiles_g2, na.rm = TRUE)
  
  # Mann-Whitney U test with error handling
  p_uncorrected <- NA
  p_lower_ci <- NA
  p_upper_ci <- NA
  
  if (n_g1 >= 2 && n_g2 >= 2) {
    tryCatch({
      # Try with confidence intervals
      mw_test <- wilcox.test(z_scores_g1, z_scores_g2, conf.int = TRUE)
      p_uncorrected <- mw_test$p.value
      
      if (!is.null(mw_test$conf.int)) {
        p_lower_ci <- mw_test$conf.int[1]
        p_upper_ci <- mw_test$conf.int[2]
      }
    }, error = function(e) {
      # If CI calculation fails, try without CI
      tryCatch({
        mw_test_no_ci <- wilcox.test(z_scores_g1, z_scores_g2, conf.int = FALSE)
        p_uncorrected <<- mw_test_no_ci$p.value
        p_lower_ci <<- NA
        p_upper_ci <<- NA
        cat("Warning: Could not calculate CI for", feature_name, "- using p-value only\n")
      }, error = function(e2) {
        cat("Warning: Wilcoxon test failed for", feature_name, "\n")
        p_uncorrected <<- NA
      })
    })
  }
  
  # Cohen's D with more robust calculation
  mean_diff <- NA
  cohens_d <- NA
  cohens_d_lower <- NA
  cohens_d_upper <- NA
  
  if (n_g1 >= 2 && n_g2 >= 2) {
    mean_g1 <- mean(z_scores_g1)
    mean_g2 <- mean(z_scores_g2)
    var_g1 <- var(z_scores_g1)
    var_g2 <- var(z_scores_g2)
    
    # Check for zero variance
    if (var_g1 > 0 && var_g2 > 0) {
      mean_diff <- mean_g1 - mean_g2
      
      # Pooled standard deviation
      pooled_sd <- sqrt(((n_g1 - 1) * var_g1 + (n_g2 - 1) * var_g2) / 
                        (n_g1 + n_g2 - 2))
      
      if (pooled_sd > 0) {
        cohens_d <- mean_diff / pooled_sd
        
        # CI for Cohen's D
        se_d <- sqrt((n_g1 + n_g2) / (n_g1 * n_g2) + 
                     cohens_d^2 / (2 * (n_g1 + n_g2)))
        
        cohens_d_lower <- cohens_d - 1.96 * se_d
        cohens_d_upper <- cohens_d + 1.96 * se_d
      }
    }
  }
  
  # Return results
  data.frame(
    Group1 = group1_name,
    Group2 = group2_name,
    Feature = feature_name,
    Sample_size_G1 = n_g1,
    Sample_size_G2 = n_g2,
    Median_centile_G1 = median_centile_g1,
    Median_centile_G2 = median_centile_g2,
    P_lower_CI = p_lower_ci,
    P_upper_CI = p_upper_ci,
    P_uncorrected = p_uncorrected,
    Cohens_D = cohens_d,
    Cohens_D_lower_CI = cohens_d_lower,
    Cohens_D_upper_CI = cohens_d_upper,
    Mean_group_difference = mean_diff
  )
}

# Function to apply multiple comparison correction
apply_fdr_correction <- function(comparison_table) {
  # Benjamini-Hochberg FDR correction
  comparison_table$P_corrected_BH_FDR <- p.adjust(comparison_table$P_uncorrected, 
                                                   method = "BH")
  
  # Add significance markers
  comparison_table$Significance <- case_when(
    comparison_table$P_corrected_BH_FDR < 0.001 ~ "***",
    comparison_table$P_corrected_BH_FDR < 0.01 ~ "**",
    comparison_table$P_corrected_BH_FDR < 0.05 ~ "*",
    TRUE ~ "NS"
  )
  
  return(comparison_table)
}

# ============================================
# APPLY TO YOUR DATA - SITE COMPARISONS
# ============================================

# Function to compare sites for all features
compare_sites_comprehensive <- function(all_fitted_models, df_test, y_values) {
  
  comparison_results <- data.frame()
  
  for (feature in y_values) {
    cat("\nAnalyzing:", feature, "\n")
    
    # Get the best model for this feature (you can modify this based on your selection)
    # For now, using m1b (smooth mean, no site) as example
    model_key <- paste(feature, "m1b", sep="_")
    model <- all_fitted_models[[model_key]]
    
    if (is.null(model)) {
      cat("Model not found for", feature, "\n")
      next
    }
    
    # Prepare test data
    test_pred_data <- data.frame(
      x = df_test$gestational_age,
      y = df_test[[feature]],
      site_id = df_test$site_id
    )
    
    # Get predictions and z-scores
    test_predictions <- predict(model, newdata = test_pred_data)
    test_scores <- params_to_scores(test_pred_data$y, test_predictions)
    
    # Split z-scores by site
    z_dhcp <- test_scores$z_randomized[df_test$site_id == "dHCP"]
    z_marsfet <- test_scores$z_randomized[df_test$site_id == "MarsFet"]
    
    # Calculate comprehensive statistics
    stats <- calculate_group_statistics(
      z_scores_g1 = z_dhcp,
      z_scores_g2 = z_marsfet,
      group1_name = "dHCP",
      group2_name = "MarsFet",
      feature_name = feature
    )
    
    comparison_results <- rbind(comparison_results, stats)
  }
  
  # Apply FDR correction across all comparisons
  comparison_results <- apply_fdr_correction(comparison_results)
  
  return(comparison_results)
}

# ============================================
# CREATE VISUALIZATION OF EFFECT SIZES
# ============================================

create_effect_size_plot <- function(comparison_results) {
  
  # Order by effect size
  comparison_results <- comparison_results %>%
    arrange(Cohens_D)
  
  comparison_results$Feature <- factor(comparison_results$Feature, 
                                       levels = comparison_results$Feature)
  
  p <- ggplot(comparison_results, aes(x = Feature, y = Cohens_D)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_hline(yintercept = c(-0.2, 0.2), linetype = "dotted", color = "gray70") +
    geom_hline(yintercept = c(-0.5, 0.5), linetype = "dotted", color = "gray70") +
    geom_hline(yintercept = c(-0.8, 0.8), linetype = "dotted", color = "gray70") +
    geom_errorbar(aes(ymin = Cohens_D_lower_CI, ymax = Cohens_D_upper_CI),
                  width = 0.2, color = "gray40") +
    geom_point(aes(color = Significance), size = 3) +
    scale_color_manual(values = c("***" = "red", 
                                  "**" = "orange",
                                  "*" = "gold",
                                  "NS" = "gray60")) +
    coord_flip() +
    labs(title = "Site Effects: Cohen's D Effect Sizes",
         subtitle = "dHCP vs MarsFet (positive = dHCP higher)",
         x = "Brain Feature",
         y = "Cohen's D (Standardized Effect Size)") +
    theme_bw() +
    annotate("text", x = 1, y = 0.2, label = "Small", size = 3, color = "gray50") +
    annotate("text", x = 1, y = 0.5, label = "Medium", size = 3, color = "gray50") +
    annotate("text", x = 1, y = 0.8, label = "Large", size = 3, color = "gray50")
  
  return(p)
}

# ============================================
# EXPLANATION OF METRICS
# ============================================

explain_metrics <- function() {
  cat("\n================================================\n")
  cat("EXPLANATION OF STATISTICAL METRICS\n")
  cat("================================================\n\n")
  
  cat("1. MEDIAN CENTILE (Percentile):\n")
  cat("   - Shows where the median of each group falls in the normative distribution\n")
  cat("   - Values near 0.5 = typical for the reference population\n")
  cat("   - Values < 0.5 = below average\n")
  cat("   - Values > 0.5 = above average\n")
  cat("   - Example: 0.035 means the median is at the 3.5th percentile (very low)\n\n")
  
  cat("2. P-VALUES:\n")
  cat("   - P uncorrected: Raw p-value from Mann-Whitney U test\n")
  cat("   - P corrected (BH-FDR): Benjamini-Hochberg False Discovery Rate correction\n")
  cat("   - Controls for multiple comparisons across features\n")
  cat("   - P < 0.05 indicates significant group differences\n\n")
  
  cat("3. COHEN'S D:\n")
  cat("   - Standardized measure of effect size (group difference / pooled SD)\n")
  cat("   - Interpretation:\n")
  cat("     |d| < 0.2: Negligible effect\n")
  cat("     |d| = 0.2-0.5: Small effect\n")
  cat("     |d| = 0.5-0.8: Medium effect\n")
  cat("     |d| > 0.8: Large effect\n")
  cat("   - Positive d: Group 1 > Group 2\n")
  cat("   - Negative d: Group 1 < Group 2\n\n")
  
  cat("4. CONFIDENCE INTERVALS:\n")
  cat("   - Show uncertainty in the estimates\n")
  cat("   - If CI for Cohen's D includes 0, effect may not be meaningful\n")
  cat("   - Narrower CIs indicate more precise estimates\n\n")
  
  cat("5. MEAN GROUP DIFFERENCE:\n")
  cat("   - Raw difference in z-scores between groups\n")
  cat("   - In normative modeling, this shows how far groups deviate\n")
  cat("   - from the reference population and from each other\n\n")
  
  cat("WHY THESE METRICS MATTER FOR NORMATIVE MODELING:\n")
  cat("- They quantify how different populations compare to normative references\n")
  cat("- Help identify which brain features show site/scanner effects\n")
  cat("- Guide decisions about harmonization or site-specific models\n")
  cat("- Essential for clinical translation and multi-site studies\n")
}

# ============================================
# RUN THE ANALYSIS
# ============================================

# Example usage:
comparison_results <- compare_sites_comprehensive(all_fitted_models, df_test, y_values)
print(comparison_results)

# Save results
write.csv(comparison_results, 
          "site_comparison_statistics.csv",
          row.names = FALSE)

# Create visualization
p_effects <- create_effect_size_plot(comparison_results)
ggsave("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/site_effect_sizes.png", p_effects, width = 10, height = 8)

# Print explanation
explain_metrics()