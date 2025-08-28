library(splines)
library(MASS)
library(nlme)
library(parallel)
library(gamlss.data)
library(gamlss.dist)
library(gamlss)
library(ggplot2)

evaluate_gamlss_family <- function(family_type, x, y, data) {
  tryCatch({
    # Fit model
    model <- gamlss(y ~ pb(x), sigma.formula = ~pb(x), 
                    nu.formula = ~1, tau.formula = ~1, 
                    family = family_type, data = data)
    
    # Get residuals
    residuals <- resid(model)
    
    # Calculate various metrics
    results <- data.frame(
      Family = family_type,
      AIC = AIC(model),
      BIC = BIC(model),
      LogLik = as.numeric(logLik(model)),
      GlobalDev = deviance(model),
      EDF = model$df.fit,
      
      # Residual diagnostics
      Shapiro_p = ifelse(length(residuals) <= 5000, 
                        shapiro.test(residuals)$p.value, NA),
      KS_p = ks.test(residuals, "pnorm")$p.value,
      
      # Filliben correlation
      Filliben = cor(sort(residuals), qnorm(ppoints(length(residuals)))),
      
      # Model convergence
      Converged = model$converged,
      Iterations = model$iter,
      
      stringsAsFactors = FALSE
    )
    
    return(results)
    
  }, error = function(e) {
    return(data.frame(Family = family_type, 
                     AIC = NA, BIC = NA, LogLik = NA, GlobalDev = NA,
                     EDF = NA, Shapiro_p = NA, KS_p = NA, Filliben = NA,
                     Converged = FALSE, Iterations = NA,
                     Error = e$message, stringsAsFactors = FALSE))
  })
}

rank_models <- function(results_df) {
  # Remove rows with errors or non-converged models
  clean_results <- results_df[results_df$Converged == TRUE & !is.na(results_df$AIC), ]
  
  if(nrow(clean_results) == 0) {
    warning("No successfully fitted models to rank")
    return(results_df)
  }
  
  # Lower is better for AIC, BIC, Global Deviance
  clean_results$AIC_rank <- rank(clean_results$AIC, ties.method = "average")
  clean_results$BIC_rank <- rank(clean_results$BIC, ties.method = "average")
  clean_results$GlobalDev_rank <- rank(clean_results$GlobalDev, ties.method = "average")
  
  # Higher is better for Filliben correlation and p-values
  clean_results$Filliben_rank <- rank(-clean_results$Filliben, ties.method = "average", na.last = TRUE)
  clean_results$Shapiro_rank <- rank(-clean_results$Shapiro_p, ties.method = "average", na.last = TRUE)
  clean_results$KS_rank <- rank(-clean_results$KS_p, ties.method = "average", na.last = TRUE)
  
  # Composite score (adjust weights as needed)
  # Handle NAs in ranking
  clean_results$Composite_Score <- rowMeans(cbind(
    clean_results$AIC_rank * 0.25,
    clean_results$BIC_rank * 0.25,
    clean_results$GlobalDev_rank * 0.15,
    clean_results$Filliben_rank * 0.15,
    clean_results$Shapiro_rank * 0.1,
    clean_results$KS_rank * 0.1
  ), na.rm = TRUE)
  
  # Sort by composite score (lower is better)
  ranked_results <- clean_results[order(clean_results$Composite_Score), ]
  
  # Add overall rank
  ranked_results$Overall_Rank <- 1:nrow(ranked_results)
  
  return(ranked_results)
}

# Load your data
df1 <- read.csv("/home/INT/dienye.h/python_files/combined_dataset/marsfet_qc_filtered.csv", header=TRUE, stringsAsFactors = FALSE)
x <- df1$gestational_age
y <- df1$band_power_B6
data <- na.omit(df1)

# Your family list
family_list <- c('BCCG','ZIBNB','LNO','NOF','BCPE','NET','BNB','BCT', 'GB2', 'GG', 'GT', 'JSU', 'SEP1', 'SEP2', 'SEP3', 'SEP4', 'SHASH', 'SHASHo', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'TF', 'PE', 'PE2')

# Evaluate all families
cat("Evaluating GAMLSS families...\n")
all_results <- do.call(rbind, lapply(family_list, evaluate_gamlss_family, x = x, y = y, data = data))

# Display basic results
print("Basic Results:")
print(all_results[, c("Family", "AIC", "BIC", "Converged")])

# Rank the models
ranked_results <- rank_models(all_results)

# Display ranked results
print("\n=== TOP 10 RANKED MODELS ===")
top_models <- head(ranked_results, 10)
print(top_models[, c("Overall_Rank", "Family", "AIC", "BIC", "Filliben", "Shapiro_p", "Composite_Score")])

# Get top 5 models for detailed analysis
top_5 <- head(ranked_results, 5)

cat("\n=== DETAILED ANALYSIS OF TOP 5 MODELS ===\n")
for(i in 1:nrow(top_5)) {
  family_name <- top_5$Family[i]
  cat("\n", i, ". Family:", family_name, "\n")
  cat("   AIC:", round(top_5$AIC[i], 2), "(rank:", top_5$AIC_rank[i], ")\n")
  cat("   BIC:", round(top_5$BIC[i], 2), "(rank:", top_5$BIC_rank[i], ")\n")
  cat("   Filliben correlation:", round(top_5$Filliben[i], 4), "\n")
  cat("   Shapiro p-value:", round(top_5$Shapiro_p[i], 4), "\n")
  cat("   Composite score:", round(top_5$Composite_Score[i], 2), "\n")
}

# Save comprehensive results
write.csv(ranked_results, file = "band_power_B6_comprehensive_model_ranking.csv", row.names = FALSE)

# Save just the summary
summary_results <- ranked_results[, c("Overall_Rank", "Family", "AIC", "BIC", 
                                     "Filliben", "Shapiro_p", "Composite_Score", "Converged")]
write.csv(summary_results, file = "band_power_B6_model_ranking_summary.csv", row.names = FALSE)

# Create a ranking plot
library(ggplot2)

# Plot AIC vs BIC with family labels
ggplot(ranked_results, aes(x = AIC, y = BIC, label = Family)) +
  geom_point(aes(color = Composite_Score), size = 3) +
  geom_text(nudge_y = 2, size = 3) +
  scale_color_gradient(low = "green", high = "red", name = "Composite\nScore") +
  labs(title = "Model Comparison: AIC vs BIC",
       subtitle = "Lower values are better") +
  theme_minimal()

# Bar plot of top 10 models
top_10 <- head(ranked_results, 10)
ggplot(top_10, aes(x = reorder(Family, -Composite_Score), y = Composite_Score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 Models by Composite Score",
       x = "Family", y = "Composite Score (Lower is Better)") +
  theme_minimal()