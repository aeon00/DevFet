# Load and inspect
final_models <- readRDS("/home/INT/dienye.h/gamlss_normative_paper-main/harmonization/final_fitted_models.rds")

# Check structure
str(final_models, max.level = 1)

# See model names
names(final_models)

# Example model structure
str(final_models[[1]])