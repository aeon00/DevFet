# Load the ggplot2 library for plotting
library(ggplot2)
library(tidyr)

# Read the data from the string into a data frame
# The 'text' argument allows us to read directly from the string variable
df <- read.csv("band_power_B6_model_family_comparison_results.csv")

# Reshape the data from wide to long format
# This creates two new columns: 'Metric' (which will contain 'BIC' or 'AIC')
# and 'Value' (which will contain the corresponding numerical value).
df_long <- pivot_longer(df, cols = c(BIC, AIC), names_to = "Metric", values_to = "Value")

# Create the bar chart
# 'geom_col' is used because we are providing both x and y values.
# 'position = "dodge"' places the bars for BIC and AIC next to each other.
bar_chart <- ggplot(df_long, aes(x = Family, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  ggtitle("Comparison of BIC and AIC by Family for Band Power B6") +
  xlab("Family") +
  ylab("Value") +
  theme_minimal() # Optional: applies a clean theme to the plot

# Print the plot
print(bar_chart)
ggsave("B6_family_chart.png", bar_chart, width = 16, height = 12, dpi = 300, bg = "white")