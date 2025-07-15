library(MASS)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)
library(ggplot2)
library(cowplot)
library(patchwork)

# Read and prepare data from both datasets
df1 <- read.csv("/home/INT/dienye.h/python_files/dhcp_dataset_info/combined_results.csv", header=TRUE, stringsAsFactors = FALSE)
df2 <- read.csv("/home/INT/dienye.h/python_files/devfetfiles/filtered_qc_3_and_above.csv", header=TRUE, stringsAsFactors = FALSE)  # Replace with your second dataset path

# Add dataset identifier
df1$dataset <- "Dataset 1"
df2$dataset <- "Dataset 2"

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

# Combine datasets
df_combined <- rbind(df1, df2)

# Select key variables for combined plotting
selected_variables <- c("B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

# Function to normalize data to 0-1 scale for comparison (per dataset)
normalize_data_by_dataset <- function(x, dataset) {
  # Normalize within each dataset separately
  datasets <- unique(dataset)
  x_norm <- numeric(length(x))
  
  for (d in datasets) {
    idx <- dataset == d
    x_subset <- x[idx]
    x_norm[idx] <- (x_subset - min(x_subset, na.rm = TRUE)) / (max(x_subset, na.rm = TRUE) - min(x_subset, na.rm = TRUE))
  }
  
  return(x_norm)
}

# Alternative: Global normalization across both datasets
normalize_data_global <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}

# Function to fit GAMLSS model and extract quantiles for each dataset
fit_gamlss_and_extract_quantiles_by_dataset <- function(x, y, dataset, variable_name) {
  quantiles_list <- list()
  
  datasets <- unique(dataset)
  
  for (d in datasets) {
    cat("  Processing dataset:", d, "\n")
    
    # Filter data for this dataset
    idx <- dataset == d
    x_subset <- x[idx]
    y_subset <- y[idx]
    
    # Skip if insufficient data
    if (length(x_subset) < 10) {
      cat("    Warning: Insufficient data for dataset", d, "\n")
      next
    }
    
    # Fit GAMLSS model
    tryCatch({
      m4 <- gam(list(y_subset ~ s(x_subset), 
                     ~ s(x_subset), 
                     ~ 1, 
                     ~ 1), 
                family=shash(), 
                data=data.frame(x=x_subset, y=y_subset))
      
      # Get predictions
      predictions_params <- predict(m4)
      qshash <- m4$family$qf
      
      # Define confidence intervals
      quantiles <- c(0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975)
      
      # Convert parameters to quantiles using shash distribution
      predictions_quantiles <- as.data.frame(sapply(quantiles, 
                                                   function(q){
                                                     qshash(p=q, mu=predictions_params)
                                                   }))
      
      colnames(predictions_quantiles) <- paste0("q", quantiles)
      predictions_quantiles$x <- x_subset
      predictions_quantiles$variable <- variable_name
      predictions_quantiles$dataset <- d
      predictions_quantiles$y_original <- y_subset
      
      quantiles_list[[d]] <- predictions_quantiles
      
    }, error = function(e) {
      cat("    Error fitting model for dataset", d, ":", e$message, "\n")
    })
  }
  
  # Combine results
  if (length(quantiles_list) > 0) {
    return(do.call(rbind, quantiles_list))
  } else {
    return(NULL)
  }
}

# Prepare combined data
combined_quantiles <- data.frame()
combined_points <- data.frame()

# Choose normalization method (change to normalize_data_by_dataset for per-dataset normalization)
normalization_method <- "global"  # or "per_dataset"

for (var in selected_variables) {
  cat("Processing variable:", var, "\n")
  
  x <- df_combined$gestational_age
  y <- df_combined[[var]]
  dataset <- df_combined$dataset
  
  # Apply normalization
  if (normalization_method == "global") {
    y_norm <- normalize_data_global(y)
  } else {
    y_norm <- normalize_data_by_dataset(y, dataset)
  }
  
  # Get quantiles for this variable (by dataset)
  var_quantiles <- fit_gamlss_and_extract_quantiles_by_dataset(x, y_norm, dataset, var)
  if (!is.null(var_quantiles)) {
    combined_quantiles <- rbind(combined_quantiles, var_quantiles)
  }
  
  # Prepare point data
  cohort_col <- if("cohort" %in% colnames(df_combined)) df_combined$cohort else "All Participants"
  var_points <- data.frame(
    x = x,
    y = y_norm,
    variable = var,
    dataset = dataset,
    cohort = cohort_col
  )
  combined_points <- rbind(combined_points, var_points)
}

# Define colors and shapes
custom_colors <- c(
  "B4 Band Relative Power" = "#00BFC4",  # Teal
  "B5 Band Relative Power" = "#00BA38",  # Green 
  "B6 Band Relative Power" = "#F8766D"   # Red
)

# Define shapes for datasets
dataset_shapes <- c(
  "Dataset 1" = 16,  # Circle
  "Dataset 2" = 4    # Cross
)

# Create the combined plot with 95% confidence intervals
p_combined <- ggplot() +
  # Add points for each variable and dataset combination
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable, shape = dataset), 
             size = 1.5, alpha = 0.7) +
  
  # Add 95% confidence interval ribbons (separate for each dataset)
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.025, ymax = q0.975, fill = variable),
              alpha = 0.15) +
  
  # Add median lines (separate for each dataset)
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable, linetype = dataset),
            linewidth = 1.2) +
  
  # Control colors and shapes
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +
  scale_shape_manual(values = dataset_shapes) +
  scale_linetype_manual(values = c("Dataset 1" = "solid", "Dataset 2" = "dashed")) +
  
  # Styling
  labs(title = "GAMLSS Models: Multiple Brain Morphology Variables (Two Datasets)",
       subtitle = paste("Normalized data with 95% confidence intervals -", 
                       ifelse(normalization_method == "global", "Global normalization", "Per-dataset normalization")),
       x = "Gestational Age (weeks)",
       y = "Normalized Values (0-1 scale)",
       color = "Variable",
       fill = "Variable",
       shape = "Dataset",
       linetype = "Dataset") +
  
  scale_x_continuous(breaks = seq(22, 44, by = 2)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  
  theme_cowplot() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.text = element_text(size = 9),
    legend.box = "horizontal"
  ) +
  
  guides(
    color = guide_legend(override.aes = list(alpha = 1, size = 2)),
    fill = guide_legend(override.aes = list(alpha = 0.3)),
    shape = guide_legend(override.aes = list(size = 2)),
    linetype = guide_legend(override.aes = list(size = 1))
  )

# Display the plot
print(p_combined)

# Save the plot
ggsave("/home/INT/dienye.h/python_files/GAMLSS/Combined_GAMLSS_Two_Datasets_95CI.png", 
       p_combined, width = 14, height = 8, units = 'in', bg="white", dpi = 300)

# Optional: Create a version with 80% confidence intervals
p_combined_80 <- ggplot() +
  geom_point(data = combined_points, 
             aes(x = x, y = y, color = variable, shape = dataset), 
             size = 1.5, alpha = 0.7) +
  
  geom_ribbon(data = combined_quantiles,
              aes(x = x, ymin = q0.1, ymax = q0.9, fill = variable),
              alpha = 0.2) +
  
  geom_line(data = combined_quantiles,
            aes(x = x, y = q0.5, color = variable, linetype = dataset),
            linewidth = 1.2) +
  
  scale_color_manual(values = custom_colors) +
  scale_fill_manual(values = custom_colors) +
  scale_shape_manual(values = dataset_shapes) +
  scale_linetype_manual(values = c("Dataset 1" = "solid", "Dataset 2" = "dashed")) +
  
  labs(title = "GAMLSS Models: Multiple Brain Morphology Variables (Two Datasets)",
       subtitle = paste("Normalized data with 80% confidence intervals -", 
                       ifelse(normalization_method == "global", "Global normalization", "Per-dataset normalization")),
       x = "Gestational Age (weeks)",
       y = "Normalized Values (0-1 scale)",
       color = "Variable",
       fill = "Variable",
       shape = "Dataset",
       linetype = "Dataset") +
  
  scale_x_continuous(breaks = seq(22, 44, by = 2)) +
  scale_y_continuous(breaks = seq(0, 1, by = 0.2)) +
  
  theme_cowplot() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14),
    plot.subtitle = element_text(hjust = 0.5, size = 11),
    legend.position = "bottom",
    legend.title = element_text(size = 10),
    legend.text = element_text(size = 9),
    axis.text = element_text(size = 9),
    legend.box = "horizontal"
  ) +
  
  guides(
    color = guide_legend(override.aes = list(alpha = 1, size = 2)),
    fill = guide_legend(override.aes = list(alpha = 0.3)),
    shape = guide_legend(override.aes = list(size = 2)),
    linetype = guide_legend(override.aes = list(size = 1))
  )

# Save the 80% CI version
ggsave("/home/INT/dienye.h/python_files/GAMLSS/Combined_GAMLSS_Two_Datasets_80CI.png", 
       p_combined_80, width = 14, height = 8, units = 'in', bg="white", dpi = 300)

# Print summary information
cat("\n=== SUMMARY ===\n")
cat("Variables included:", paste(selected_variables, collapse = ", "), "\n")
cat("Datasets processed:", paste(unique(df_combined$dataset), collapse = ", "), "\n")
cat("Normalization method:", normalization_method, "\n")
cat("Combined plots saved successfully!\n")

# Print dataset sample sizes
cat("\nDataset sample sizes:\n")
for (d in unique(df_combined$dataset)) {
  n <- sum(df_combined$dataset == d)
  cat("  ", d, ":", n, "observations\n")
}

# Print variable completeness by dataset
cat("\nVariable completeness by dataset:\n")
for (var in selected_variables) {
  cat("  ", var, ":\n")
  for (d in unique(df_combined$dataset)) {
    subset_data <- df_combined[df_combined$dataset == d, ]
    n_complete <- sum(!is.na(subset_data[[var]]))
    n_total <- nrow(subset_data)
    cat("    ", d, ":", n_complete, "/", n_total, "(", round(100*n_complete/n_total, 1), "%)\n")
  }
}