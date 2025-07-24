library(MASS)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)
library(ggplot2)
library(cowplot)
library(patchwork)

# Read and prepare data
df2 <- read.csv("/home/INT/dienye.h/python_files/dhcp_dataset_info/combined_results.csv", 
                header=TRUE, stringsAsFactors = FALSE)
df3 <- read.csv("/home/INT/dienye.h/python_files/devfetfiles/filtered_qc_3_and_above_Copie.csv", 
                header=TRUE, stringsAsFactors = FALSE)

# Add dataset/site identifier
df2$Cohort <- "dHCP"
df3$Cohort <- "marsfet"

# Combine datasets
df1 <- rbind(df2, df3)

# Column renaming (same as original)
colnames(df1)[colnames(df1) == "surface_area_cm2"] <- "Surface Area cm2"
colnames(df1)[colnames(df1) == "analyze_folding_power"] <- "Folding Power"
colnames(df1)[colnames(df1) == "B4_vertex_percentage"] <- "B4 Vertex Percentage"
colnames(df1)[colnames(df1) == "B5_vertex_percentage"] <- "B5 Vertex Percentage"
colnames(df1)[colnames(df1) == "B6_vertex_percentage"] <- "B6 Vertex Percentage"
colnames(df1)[colnames(df1) == "band_parcels_B4"] <- "Band_parcels B4"
colnames(df1)[colnames(df1) == "band_parcels_B5"] <- "Band Parcels B5"
colnames(df1)[colnames(df1) == "band_parcels_B6"] <- "Band Parcels B6"
colnames(df1)[colnames(df1) == "volume_ml"] <- "Hemispheric Volume"
colnames(df1)[colnames(df1) == "gyrification_index"] <- "Gyrification Index"
colnames(df1)[colnames(df1) == "hull_area"] <- "Hull Area"
colnames(df1)[colnames(df1) == "B4_surface_area"] <- "B4 Surface Area"
colnames(df1)[colnames(df1) == "B5_surface_area"] <- "B5 Surface Area"
colnames(df1)[colnames(df1) == "B6_surface_area"] <- "B6 Surface Area"
colnames(df1)[colnames(df1) == "B4_surface_area_percentage"] <- "B4 Surface Area Percentage"
colnames(df1)[colnames(df1) == "B5_surface_area_percentage"] <- "B5 Surface Area Percentage"
colnames(df1)[colnames(df1) == "B6_surface_area_percentage"] <- "B6 Surface Area Percentage"
colnames(df1)[colnames(df1) == "band_power_B4"] <- "B4 Band Power"
colnames(df1)[colnames(df1) == "band_power_B5"] <- "B5 Band Power"
colnames(df1)[colnames(df1) == "band_power_B6"] <- "B6 Band Power"
colnames(df1)[colnames(df1) == "B4_band_relative_power"] <- "B4 Band Relative Power"
colnames(df1)[colnames(df1) == "B5_band_relative_power"] <- "B5 Band Relative Power"
colnames(df1)[colnames(df1) == "B6_band_relative_power"] <- "B6 Band Relative Power"

# Select key variables for combined plotting (you can modify this list)
selected_variables <- c("B4 Band Power", "B5 Band Power", "B6 Band Power")

# Function to normalize data to 0-1 scale for comparison
normalize_data <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

# Function to fit GAMLSS model and extract quantiles
fit_gamlss_and_extract_quantiles <- function(x, y, cohort, variable_name) {
  
  # Create data frame with cohort
  model_data <- data.frame(x = x, y = y, cohort = factor(cohort))

  # Fit GAMLSS model (same as m4 in your original code)
  m4 <- gam(list(y ~ s(x) + cohort, 
                 ~ s(x) + cohort, 
                 ~ 1, 
                 ~ 1), 
            family=shash(), 
            data=model_data)
  
  # Get predictions for each cohort level
  x_seq <- seq(min(x, na.rm = TRUE), max(x, na.rm = TRUE), length.out = 100)
  cohort_levels <- levels(factor(cohort))
  
  # Predict for each cohort
  all_predictions <- data.frame()
  
  for (coh in cohort_levels) {
    pred_data <- data.frame(x = x_seq, cohort = factor(coh, levels = cohort_levels))
    predictions_params <- predict(m4, newdata = pred_data)
    
    qshash <- m4$family$qf
    quantiles <- c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
    
    predictions_quantiles <- as.data.frame(sapply(quantiles, 
                                                 function(q){
                                                   qshash(p = q, mu = predictions_params)
                                                 }))
    
    colnames(predictions_quantiles) <- paste0("q", quantiles)
    predictions_quantiles$x <- x_seq
    predictions_quantiles$variable <- variable_name
    predictions_quantiles$cohort <- coh
    
    all_predictions <- rbind(all_predictions, predictions_quantiles)
  }
  
  return(all_predictions)
}

# CORRECTED main processing loop
combined_quantiles <- data.frame()
combined_points <- data.frame()

for (var in selected_variables) {
  cat("Processing variable:", var, "\n")
  
  x <- df1$gestational_age
  y <- df1[[var]]
  y_norm <- normalize_data(y)
  cohort <- df1$Cohort  # Use the site column as cohort
  
  # Get quantiles for this variable with cohort
  var_quantiles <- fit_gamlss_and_extract_quantiles(x, y_norm, cohort, var)
  combined_quantiles <- rbind(combined_quantiles, var_quantiles)
  
  # Prepare point data
  var_points <- data.frame(
    x = x,
    y = y_norm,
    variable = var,
    cohort = cohort
  )
  combined_points <- rbind(combined_points, var_points)
}


# DIAGNOSTIC CHECKS FIRST
cat("=== DIAGNOSTIC INFORMATION ===\n")
cat("Cohort distribution:\n")
print(table(df1$Cohort))

cat("\nGestational age range:\n")
print(summary(df1$gestational_age))

cat("\nData points by gestational age bins:\n")
print(table(cut(df1$gestational_age, breaks = 8)))

cat("\nCohort effects in quantiles data:\n")
if("cohort" %in% colnames(combined_quantiles)) {
  print(table(combined_quantiles$cohort))
} else {
  cat("WARNING: No cohort column found in quantiles data!\n")
}

# Enhanced color scheme with cohort distinction
custom_colors <- c(
  "B4 Band Power" = "#00BFC4",  # Teal
  "B5 Band Power" = "#00BA38",  # Green 
  "B6 Band Power" = "#F8766D"   # Red
)

# Create cohort-aware plots

# 1. MAIN PLOT: Faceted by cohort to show differences clearly
p_combined_cohorts <- ggplot() +
  # Add points with cohort distinction
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable), 
             size = 0.8, alpha = 0.7) +
  
  # Add 95% confidence interval ribbons (cohort-specific if available)
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
              alpha = 0.2) +
  
  # Add median lines (cohort-specific if available)
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable),
            linewidth = 1.2) +
  
  # Facet by cohort to show differences
  facet_wrap(~cohort, labeller = label_both) +
  
  # Color schemes
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +
  
  # Styling
  labs(title = "GAMLSS Models: Brain Morphology Variables by Cohort",
       subtitle = "Normalized data with 95% confidence intervals - Cohort comparison",
       x = "Gestational Age (weeks)",
       y = "Normalized Values (0-1 scale)",
       color = "Variable",
       fill = "Variable") +
  
  scale_x_continuous(breaks = seq(22, 44, by = 4)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  
  theme_cowplot() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.text = element_text(size = 8),
    strip.text = element_text(size = 10)
  ) +
  
  guides(
    color = guide_legend(override.aes = list(alpha = 1, size = 2)),
    fill = guide_legend(override.aes = list(alpha = 0.3))
  )

# 2. OVERLAY PLOT: Both cohorts on same plot with FIXED LEGENDS
p_combined_overlay <- ggplot() +
  # Add points with shape distinction for cohorts
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable, shape = cohort), 
             size = 1.2, alpha = 0.6) +
  
  # Add confidence interval ribbons
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
              alpha = 0.15) +
  
  # Add median lines with different line types for cohorts
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable, linetype = cohort),
            linewidth = 1.2) +
  
  # FIXED: Better color and shape schemes with proper labels
  scale_color_manual(values = custom_colors, name = "Brain Measure") +
  scale_fill_manual(values = custom_colors, name = "Brain Measure") +
  scale_shape_manual(values = c("dHCP" = 16, "marsfet" = 17), 
                     name = "Cohort",
                     labels = c("dHCP" = "dHCP (circles)", "marsfet" = "Marseille (triangles)")) +
  scale_linetype_manual(values = c("dHCP" = "solid", "marsfet" = "dashed"), 
                        name = "Cohort",
                        labels = c("dHCP" = "dHCP (solid line)", "marsfet" = "Marseille (dashed line)")) +
  
  # Styling
  labs(title = "GAMLSS Models: Brain Morphology Variables",
       subtitle = "Normalized data with 95% CI - Cohort overlay comparison",
       x = "Gestational Age (weeks)",
       y = "Normalized Values (0-1 scale)") +
  
  scale_x_continuous(breaks = seq(22, 44, by = 2)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  
  theme_cowplot() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    axis.text = element_text(size = 9),
    legend.box = "horizontal"
  ) +
  
  # FIXED: Better legend arrangement
  guides(
    color = guide_legend(override.aes = list(alpha = 1, size = 2.5),
                         title.position = "top", 
                         nrow = 1,
                         order = 1),
    fill = "none",  # Remove duplicate fill legend
    shape = guide_legend(override.aes = list(alpha = 1, size = 2.5),
                         title.position = "top",
                         nrow = 1,
                         order = 2),
    linetype = guide_legend(override.aes = list(alpha = 1, size = 1),
                            title.position = "top",
                            nrow = 1,
                            order = 3)
  )

# 3. ALTERNATIVE OVERLAY PLOT: Cleaner version with just one cohort legend
p_combined_overlay_clean <- ggplot() +
  # Add points with shape distinction for cohorts
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable, shape = cohort), 
             size = 1.2, alpha = 0.6) +
  
  # Add confidence interval ribbons
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
              alpha = 0.15) +
  
  # Add median lines with different line types for cohorts
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable, linetype = cohort),
            linewidth = 1.2) +
  
  # Color and shape schemes
  scale_color_manual(values = custom_colors, name = "Brain Measure") +
  scale_fill_manual(values = custom_colors, guide = "none") +  # Hide fill legend
  scale_shape_manual(values = c("dHCP" = 16, "marsfet" = 17), 
                     name = "Dataset",
                     labels = c("dHCP", "Marseille")) +
  scale_linetype_manual(values = c("dHCP" = "solid", "marsfet" = "dashed"), 
                        guide = "none") +  # Hide linetype legend (same info as shape)
  
  # Styling
  labs(title = "GAMLSS Models: Brain Morphology Variables by Dataset",
       subtitle = "Normalized data with 95% CI (circles = dHCP, triangles = Marseille, solid/dashed lines)",
       x = "Gestational Age (weeks)",
       y = "Normalized Values (0-1 scale)") +
  
  scale_x_continuous(breaks = seq(22, 44, by = 2)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  
  theme_cowplot() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_text(size = 11, face = "bold"),
    legend.text = element_text(size = 10),
    axis.text = element_text(size = 9),
    legend.box = "horizontal"
  ) +
  
  guides(
    color = guide_legend(override.aes = list(alpha = 1, size = 2.5), 
                         title.position = "top", nrow = 1),
    shape = guide_legend(override.aes = list(alpha = 1, size = 2.5), 
                         title.position = "top", nrow = 1)
  )

# 4. ORIGINAL PLOT: Everything together without cohort differentiation
p_combined_original <- ggplot() +
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable), 
             size = 1, alpha = 0.6) +
  
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
              alpha = 0.2) +
  
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable),
            linewidth = 1.2) +
  
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +
  
  labs(title = "GAMLSS Models: Multiple Brain Morphology Variables",
       subtitle = "Normalized data with 95% confidence intervals (combined cohorts)",
       x = "Gestational Age (weeks)",
       y = "Normalized Values (0-1 scale)",
       color = "Variable",
       fill = "Variable") +
  
  scale_x_continuous(breaks = seq(22, 44, by = 2)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  
  theme_cowplot() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.text = element_text(size = 9)
  ) +
  
  guides(
    color = guide_legend(override.aes = list(alpha = 1, size = 2)),
    fill = guide_legend(override.aes = list(alpha = 0.3))
  )

# 5. DATA DISTRIBUTION PLOT
p_distribution <- ggplot(df1, aes(x = gestational_age)) +
  geom_histogram(aes(fill = Cohort), bins = 30, alpha = 0.7, position = "identity") +
  facet_wrap(~Cohort, ncol = 1) +
  labs(title = "Data Distribution by Gestational Age",
       x = "Gestational Age (weeks)",
       y = "Count") +
  theme_cowplot() +
  theme(legend.position = "none")

# Display plots
cat("\n=== DISPLAYING PLOTS ===\n")
print("1. Cohort-faceted plot:")
print(p_combined_cohorts)

print("2. Cohort overlay plot (detailed legends):")
print(p_combined_overlay)

print("3. Cohort overlay plot (clean version):")
print(p_combined_overlay_clean)

print("4. Original combined plot (no cohort distinction):")
print(p_combined_original)

# Save all plots
ggsave("/home/INT/dienye.h/python_files/GAMLSS/fixed_Combined_GAMLSS_Cohort_Faceted.png", 
       p_combined_cohorts, width = 14, height = 8, units = 'in', bg="white", dpi = 300)

ggsave("/home/INT/dienye.h/python_files/GAMLSS/fixed_Combined_GAMLSS_Cohort_Overlay.png", 
       p_combined_overlay, width = 12, height = 9, units = 'in', bg="white", dpi = 300)

ggsave("/home/INT/dienye.h/python_files/GAMLSS/fixed_Combined_GAMLSS_Cohort_Clean.png", 
       p_combined_overlay_clean, width = 12, height = 8, units = 'in', bg="white", dpi = 300)

ggsave("/home/INT/dienye.h/python_files/GAMLSS/fixed_Combined_GAMLSS_Original.png", 
       p_combined_original, width = 12, height = 8, units = 'in', bg="white", dpi = 300)

ggsave("/home/INT/dienye.h/python_files/GAMLSS/fixed_Data_Distribution_by_Cohort.png", 
       p_distribution, width = 10, height = 6, units = 'in', bg="white", dpi = 300)

cat("\n=== SUMMARY ===\n")
cat("All plots with fixed legends saved successfully!\n")
cat("Variables included:", paste(selected_variables, collapse = ", "), "\n")
cat("Cohorts analyzed:", paste(unique(df1$Cohort), collapse = ", "), "\n")
cat("\nFiles created:\n")
cat("1. fixed_Combined_GAMLSS_Cohort_Faceted.png - Side-by-side cohort comparison\n")
cat("2. fixed_Combined_GAMLSS_Cohort_Overlay.png - Overlaid cohort comparison with detailed legends\n") 
cat("3. fixed_Combined_GAMLSS_Cohort_Clean.png - Clean version with clear legends\n")
cat("4. fixed_Combined_GAMLSS_Original.png - Original plot without cohort distinction\n")
cat("5. fixed_Data_Distribution_by_Cohort.png - Data distribution check\n")