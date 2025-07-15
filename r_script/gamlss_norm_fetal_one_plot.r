library(MASS)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)
library(ggplot2)
library(cowplot)
library(patchwork)

# Read and prepare data
df1 <- read.csv("/home/INT/dienye.h/python_files/rough/results_df/both_df_combined_results.csv", header=TRUE, stringsAsFactors = FALSE)

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
fit_gamlss_and_extract_quantiles <- function(x, y, variable_name) {
  # Fit GAMLSS model (same as m4 in your original code)
  m4 <- gam(list(y ~ s(x), 
                 ~ s(x), 
                 ~ 1, 
                 ~ 1), 
            family=shash(), 
            data=data.frame(x=x, y=y))
  
  # Get predictions
  predictions_params <- predict(m4)
  qshash <- m4$family$qf
  
  # Define confidence intervals: 50%, 80%, and 95%
  # For 95% CI: approximately 2.5% and 97.5% quantiles
  # For 80% CI: approximately 10% and 90% quantiles  
  # For 50% CI: approximately 25% and 75% quantiles
  quantiles <- c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
  
  # Convert parameters to quantiles using shash distribution
  predictions_quantiles <- as.data.frame(sapply(quantiles, 
                                               function(q){
                                                 qshash(p=q, mu=predictions_params)
                                               }))
  
  colnames(predictions_quantiles) <- paste0("q", quantiles)
  predictions_quantiles$x <- x
  predictions_quantiles$variable <- variable_name
  predictions_quantiles$y_original <- y
  
  return(predictions_quantiles)
}

# Prepare combined data
combined_quantiles <- data.frame()
combined_points <- data.frame()

for (var in selected_variables) {
  cat("Processing variable:", var, "\n")
  
  x <- df1$gestational_age
  y <- df1[[var]]
  y_norm <- normalize_data(y)  # Normalize for comparison
  
  # Get quantiles for this variable
  var_quantiles <- fit_gamlss_and_extract_quantiles(x, y_norm, var)
  combined_quantiles <- rbind(combined_quantiles, var_quantiles)
  
  # Prepare point data
  cohort_col <- if("cohort" %in% colnames(df1)) df1$cohort else "All Participants"
  var_points <- data.frame(
    x = x,
    y = y_norm,
    variable = var,
    cohort = cohort_col
  )
  combined_points <- rbind(combined_points, var_points)
}


custom_colors <- c(
  "B4 Band Power" = "#00BFC4",  # Teal (original B6 color)
  "B5 Band Power" = "#00BA38",  # Green 
  "B6 Band Power" = "#F8766D"   # Red (original B4 color)
)

# Create the combined plot with 95% confidence intervals (most common choice)
p_combined <- ggplot() +
  # Add points for each variable
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable), 
             size = 1, alpha = 0.6) +
  
  # Add 95% confidence interval ribbons
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
              alpha = 0.2) +
  
  # Add median lines
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable),
            linewidth = 1.2) +
    
  # Add these two lines to control colors:
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +

  # Styling
  labs(title = "GAMLSS Models: Multiple Brain Morphology Variables",
       subtitle = "Normalized data with 95% confidence intervals",
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

# Display the plot
print(p_combined)

# Save the plot
ggsave("/home/INT/dienye.h/python_files/GAMLSS/Combined_GAMLSS_Variables_both_df_power.png", 
       p_combined, width = 12, height = 8, units = 'in', bg="white", dpi = 300)

# Optional: Create a plot with 80% confidence intervals (more conservative)
p_combined_80 <- ggplot() +
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable), 
             size = 1, alpha = 0.6) +
  
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.1, ymax = q0.9, fill = variable),
              alpha = 0.25) +
  
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable),
            linewidth = 1.2) +
  
  # Add these two lines to control colors:
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +

  labs(title = "GAMLSS Models: Multiple Brain Morphology Variables",
       subtitle = "Normalized data with 80% confidence intervals",
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

# Save the 80% CI version
ggsave("/home/INT/dienye.h/python_files/GAMLSS/Combined_GAMLSS_Variables_80CI_both_df_power.png", 
       p_combined_80, width = 12, height = 8, units = 'in', bg="white", dpi = 300)

cat("Combined plots saved successfully!\n")
cat("Variables included:", paste(selected_variables, collapse = ", "), "\n")