library(caret)
library(data.table)
library(MASS)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)

# ============================================================
# DATA LOADING - 4 SITES
# ============================================================
# NOTE: Replace the two placeholder paths/labels below with your
# real third and fourth site CSVs. They must contain the same
# columns as the existing files (gestational_age + the features).

data <- as.data.frame(read.csv('/home/INT/dienye.h/python_files/combined_dataset/dhcp_qc_filtered.csv'))
data$site_id <- "dHCP"

data2 <- as.data.frame(read.csv('/home/INT/dienye.h/python_files/combined_dataset/marsfet_qc_filtered.csv'))
data2$site_id <- "MarsFet"

# ---- NEW SITE 3 (edit path + label) ----
data3 <- as.data.frame(read.csv('/home/INT/dienye.h/Téléchargements/CHUV_Spangy_Analysis/chuv_qc_filtered_data.csv'))
data3$site_id <- "CHUV"

# ---- NEW SITE 4 (edit path + label) ----
data4 <- as.data.frame(read.csv('/home/INT/dienye.h/Téléchargements/BCN_spangy_fet/spangy_fet/qc_filtered_bcn_combined.csv'))
data4$site_id <- "BCN"

# ------------------------------------------------------------
# Combine sites robustly.
# rbind() requires identical columns; sites often differ by a
# stray QC/index column. We align to the columns COMMON to all
# sites and report anything dropped so you can check it.
# ------------------------------------------------------------
site_list <- list(dHCP = data, MarsFet = data2, Site3 = data3, Site4 = data4)

cat("Columns per site:\n")
for (nm in names(site_list)) cat(sprintf("  %-8s: %d cols\n", nm, ncol(site_list[[nm]])))

common_cols <- Reduce(intersect, lapply(site_list, colnames))
if (length(common_cols) == 0) stop("No columns are shared across all site CSVs.")

# Report dropped columns per site
for (nm in names(site_list)) {
  dropped <- setdiff(colnames(site_list[[nm]]), common_cols)
  if (length(dropped) > 0)
    cat(sprintf("  [%s] dropping non-shared columns: %s\n", nm, paste(dropped, collapse = ", ")))
}

# Sanity check: make sure the raw feature columns we model survived
needed_raw <- c("gestational_age", "surface_area_cm2", "analyze_folding_power",
                "B4_vertex_percentage", "B5_vertex_percentage", "B6_vertex_percentage",
                "band_parcels_B4", "band_parcels_B5", "band_parcels_B6", "volume_ml",
                "gyrification_index", "hull_area", "B4_surface_area", "B5_surface_area",
                "B6_surface_area", "B4_surface_area_percentage", "B5_surface_area_percentage",
                "B6_surface_area_percentage", "band_power_B4", "band_power_B5", "band_power_B6",
                "B4_band_relative_power", "B5_band_relative_power", "B6_band_relative_power")
missing_needed <- setdiff(needed_raw, common_cols)
if (length(missing_needed) > 0)
  warning("These modelled columns are NOT shared by all sites and were dropped: ",
          paste(missing_needed, collapse = ", "))

data <- do.call(rbind, lapply(site_list, function(d) d[, common_cols, drop = FALSE]))

# set site as a factor
data$site_id <- as.factor(data$site_id)

# Useful for the (now 4-site) normative-curve plots later
site_levels <- levels(data$site_id)
n_sites <- length(site_levels)
default_site_palette <- c("steelblue", "darkgreen", "darkorange", "purple",
                          "red", "brown", "black", "gray40")
default_site_shapes  <- c(16, 17, 15, 18, 3, 4, 8, 7)
site_colors_plot <- setNames(default_site_palette[seq_len(n_sites)], site_levels)
site_shapes_plot <- setNames(default_site_shapes[seq_len(n_sites)],  site_levels)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================
# (BUG FIX: original used data$gestational, which does not exist.
#  Correct column is gestational_age.)
train_idx_data <- createDataPartition(data$gestational_age, p = 0.7, list = F)
df_train_allsites <- data[train_idx_data, ]
df_test_allsites  <- data[-train_idx_data, ]

df1 <- df_train_allsites

# ------------------------------------------------------------
# Column renaming (training)
# ------------------------------------------------------------
rename_features <- function(df) {
  colnames(df)[colnames(df) == "surface_area_cm2"]            <- "Surface Area cm2"
  colnames(df)[colnames(df) == "analyze_folding_power"]       <- "Folding Power"
  colnames(df)[colnames(df) == "B4_vertex_percentage"]        <- "B4 Vertex Percentage"
  colnames(df)[colnames(df) == "B5_vertex_percentage"]        <- "B5 Vertex Percentage"
  colnames(df)[colnames(df) == "B6_vertex_percentage"]        <- "B6 Vertex Percentage"
  colnames(df)[colnames(df) == "band_parcels_B4"]             <- "Band_parcels B4"
  colnames(df)[colnames(df) == "band_parcels_B5"]             <- "Band Parcels B5"
  colnames(df)[colnames(df) == "band_parcels_B6"]             <- "Band Parcels B6"
  colnames(df)[colnames(df) == "volume_ml"]                   <- "Hemispheric Volume"
  colnames(df)[colnames(df) == "gyrification_index"]          <- "Gyrification Index"
  colnames(df)[colnames(df) == "hull_area"]                   <- "Hull Area"
  colnames(df)[colnames(df) == "B4_surface_area"]             <- "B4 Surface Area"
  colnames(df)[colnames(df) == "B5_surface_area"]             <- "B5 Surface Area"
  colnames(df)[colnames(df) == "B6_surface_area"]             <- "B6 Surface Area"
  colnames(df)[colnames(df) == "B4_surface_area_percentage"]  <- "B4 Surface Area Percentage"
  colnames(df)[colnames(df) == "B5_surface_area_percentage"]  <- "B5 Surface Area Percentage"
  colnames(df)[colnames(df) == "B6_surface_area_percentage"]  <- "B6 Surface Area Percentage"
  colnames(df)[colnames(df) == "band_power_B4"]               <- "B4 Band Power"
  colnames(df)[colnames(df) == "band_power_B5"]               <- "B5 Band Power"
  colnames(df)[colnames(df) == "band_power_B6"]               <- "B6 Band Power"
  colnames(df)[colnames(df) == "B4_band_relative_power"]      <- "B4 Band Relative Power"
  colnames(df)[colnames(df) == "B5_band_relative_power"]      <- "B5 Band Relative Power"
  colnames(df)[colnames(df) == "B6_band_relative_power"]      <- "B6 Band Relative Power"
  df
}
df1 <- rename_features(df1)

y_values <- list("Surface Area cm2", "Folding Power", "B4 Vertex Percentage",
                 "B5 Vertex Percentage", "B6 Vertex Percentage", "Band_parcels B4",
                 "Band Parcels B5", "Band Parcels B6", "Hemispheric Volume",
                 "Gyrification Index", "Hull Area", "B4 Surface Area", "B5 Surface Area",
                 "B6 Surface Area", "B4 Surface Area Percentage", "B5 Surface Area Percentage",
                 "B6 Surface Area Percentage", "B4 Band Power", "B5 Band Power",
                 "B6 Band Power", "B4 Band Relative Power", "B5 Band Relative Power",
                 "B6 Band Relative Power")

results <- data.frame(Model = character(),
                      Y_feature = character(),
                      BIC = double(),
                      AIC = double(), stringsAsFactors = FALSE)

# Table that will hold the site-effect test results
site_effect_results <- data.frame(Feature = character(),
                                  Best_Effect = character(),
                                  p_value = double(),
                                  Significant = character(),
                                  stringsAsFactors = FALSE)

# Store ALL fitted models
all_fitted_models <- list()

# Helper: extract a p-value from an mgcv anova() model comparison robustly
extract_anova_p <- function(aov_obj) {
  pcol <- grep("^Pr", colnames(aov_obj), value = TRUE)
  if (length(pcol) == 0) return(NA_real_)
  pv <- aov_obj[[pcol[1]]]
  pv[length(pv)]   # p-value on the row comparing the larger model
}

# Helper: fallback p-value from summary of the random-effect (mean) term
extract_summary_re_p <- function(model) {
  s <- summary(model)
  st <- s$s.table
  if (is.null(st)) return(NA_real_)
  rn <- rownames(st)
  idx <- which(grepl("site_id", rn))
  if (length(idx) == 0) return(NA_real_)
  st[idx[1], "p-value"]   # first site_id term = mean predictor
}

for (i in y_values) {

    x <- df1$gestational_age
    y <- df1[[i]]

    max_vol <- max(y); min_vol <- min(y)
    max_age <- max(x); min_age <- min(x)

    # ========================================================
    # SET 1 : NO SITE EFFECTS
    # ========================================================
    m1a <- gam(list(y ~ x,
                    ~ 1),
               family = gaulss(), optimizer = 'efs', data = df1)

    m1b <- gam(list(y ~ s(x),
                    ~ 1),
               family = gaulss(), optimizer = 'efs', data = df1)

    m1c <- gam(list(y ~ s(x),
                    ~ s(x)),
               family = gaulss(), optimizer = 'efs', data = df1)

    m1d <- gam(list(y ~ s(x),
                    ~ s(x),
                    ~ 1,
                    ~ 1),
               family = shash(), optimizer = 'efs', data = df1)

    m1e <- gam(list(y ~ s(x),
                    ~ s(x),
                    ~ s(x),
                    ~ s(x)),
               family = shash(), optimizer = 'efs', data = df1)

    # ========================================================
    # SET 2 : RANDOM SITE EFFECTS  (was Set 3; fixed-site set removed)
    # ========================================================
    m2a <- gam(list(y ~ x + s(site_id, bs = "re"),
                    ~ 1),
               family = gaulss(), optimizer = 'efs', data = df1)

    m2b <- gam(list(y ~ s(x) + s(site_id, bs = "re"),
                    ~ 1),
               family = gaulss(), optimizer = 'efs', data = df1)

    m2c <- gam(list(y ~ s(x) + s(site_id, bs = "re"),
                    ~ s(x) + s(site_id, bs = "re")),
               family = gaulss(), optimizer = 'efs', data = df1)

    m2d <- gam(list(y ~ s(x) + s(site_id, bs = "re"),
                    ~ s(x) + s(site_id, bs = "re"),
                    ~ 1,
                    ~ 1),
               family = shash(), optimizer = 'efs', data = df1)

    m2e <- gam(list(y ~ s(x) + s(site_id, bs = "re"),
                    ~ s(x) + s(site_id, bs = "re"),
                    ~ s(x) + s(site_id, bs = "re"),
                    ~ s(x) + s(site_id, bs = "re")),
               family = shash(), optimizer = 'efs', data = df1)

    # --------------------------------------------------------
    # Predictions (linear predictor parameters)
    # --------------------------------------------------------
    predictions_params_m1a <- predict(m1a)
    predictions_params_m1b <- predict(m1b)
    predictions_params_m1c <- predict(m1c)
    predictions_params_m1d <- predict(m1d)
    predictions_params_m1e <- predict(m1e)

    predictions_params_m2a <- predict(m2a)
    predictions_params_m2b <- predict(m2b)
    predictions_params_m2c <- predict(m2c)
    predictions_params_m2d <- predict(m2d)
    predictions_params_m2e <- predict(m2e)

    params_to_quantiles_norm <- function(quantiles, params){
        as.data.frame(sapply(quantiles, function(q){
            qnorm(p = q, mean = params[, 1], sd = exp(params[, 2]))
        }))
    }
    params_to_quantiles_shash <- function(quantiles, params, qshash){
        as.data.frame(sapply(quantiles, function(q){
            qshash(p = q, mu = params)
        }))
    }

    quantiles <- pnorm(c(-2:2))
    qshash <- m1d$family$qf

    # Set 1 quantiles
    predictions_quantiles_m1a <- params_to_quantiles_norm(quantiles, predictions_params_m1a)
    predictions_quantiles_m1b <- params_to_quantiles_norm(quantiles, predictions_params_m1b)
    predictions_quantiles_m1c <- params_to_quantiles_norm(quantiles, predictions_params_m1c)
    predictions_quantiles_m1d <- params_to_quantiles_shash(quantiles, predictions_params_m1d, qshash)
    predictions_quantiles_m1e <- params_to_quantiles_shash(quantiles, predictions_params_m1e, qshash)

    # Set 2 (random) quantiles
    predictions_quantiles_m2a <- params_to_quantiles_norm(quantiles, predictions_params_m2a)
    predictions_quantiles_m2b <- params_to_quantiles_norm(quantiles, predictions_params_m2b)
    predictions_quantiles_m2c <- params_to_quantiles_norm(quantiles, predictions_params_m2c)
    predictions_quantiles_m2d <- params_to_quantiles_shash(quantiles, predictions_params_m2d, qshash)
    predictions_quantiles_m2e <- params_to_quantiles_shash(quantiles, predictions_params_m2e, qshash)

    reshape_quantiles_to_long <- function(quantiles_df, x_var, site_var = NULL){
        quantiles_df$x <- x_var
        if (!is.null(site_var)) {
            quantiles_df$site_id <- site_var
            return(reshape2::melt(quantiles_df, id.vars = c('x', 'site_id')))
        } else {
            return(reshape2::melt(quantiles_df, id.vars = c('x')))
        }
    }

    predictions_quantiles_m1a_long <- reshape_quantiles_to_long(predictions_quantiles_m1a, df1$gestational_age)
    predictions_quantiles_m1b_long <- reshape_quantiles_to_long(predictions_quantiles_m1b, df1$gestational_age)
    predictions_quantiles_m1c_long <- reshape_quantiles_to_long(predictions_quantiles_m1c, df1$gestational_age)
    predictions_quantiles_m1d_long <- reshape_quantiles_to_long(predictions_quantiles_m1d, df1$gestational_age)
    predictions_quantiles_m1e_long <- reshape_quantiles_to_long(predictions_quantiles_m1e, df1$gestational_age)

    predictions_quantiles_m2a_long <- reshape_quantiles_to_long(predictions_quantiles_m2a, df1$gestational_age, df1$site_id)
    predictions_quantiles_m2b_long <- reshape_quantiles_to_long(predictions_quantiles_m2b, df1$gestational_age, df1$site_id)
    predictions_quantiles_m2c_long <- reshape_quantiles_to_long(predictions_quantiles_m2c, df1$gestational_age, df1$site_id)
    predictions_quantiles_m2d_long <- reshape_quantiles_to_long(predictions_quantiles_m2d, df1$gestational_age, df1$site_id)
    predictions_quantiles_m2e_long <- reshape_quantiles_to_long(predictions_quantiles_m2e, df1$gestational_age, df1$site_id)

    # Store fitted models
    all_fitted_models[[paste(i, "m1a", sep = "_")]] <- m1a
    all_fitted_models[[paste(i, "m1b", sep = "_")]] <- m1b
    all_fitted_models[[paste(i, "m1c", sep = "_")]] <- m1c
    all_fitted_models[[paste(i, "m1d", sep = "_")]] <- m1d
    all_fitted_models[[paste(i, "m1e", sep = "_")]] <- m1e
    all_fitted_models[[paste(i, "m2a", sep = "_")]] <- m2a
    all_fitted_models[[paste(i, "m2b", sep = "_")]] <- m2b
    all_fitted_models[[paste(i, "m2c", sep = "_")]] <- m2c
    all_fitted_models[[paste(i, "m2d", sep = "_")]] <- m2d
    all_fitted_models[[paste(i, "m2e", sep = "_")]] <- m2e

    # --------------------------------------------------------
    # AIC / BIC report
    # --------------------------------------------------------
    cat("Results for y =", i, ":\n")
    cat("=== Set 1: No Site Effects ===\n")
    cat("Model 1a (linear):\n");            print(AIC(m1a)); print(BIC(m1a))
    cat("Model 1b (smooth mean):\n");       print(AIC(m1b)); print(BIC(m1b))
    cat("Model 1c (smooth mean+var):\n");   print(AIC(m1c)); print(BIC(m1c))
    cat("Model 1d (SHASH constant shape):\n"); print(AIC(m1d)); print(BIC(m1d))
    cat("Model 1e (SHASH full):\n");        print(AIC(m1e)); print(BIC(m1e))

    cat("\n=== Set 2: Random Site Effects ===\n")
    cat("Model 2a (linear):\n");            print(AIC(m2a)); print(BIC(m2a))
    cat("Model 2b (smooth mean):\n");       print(AIC(m2b)); print(BIC(m2b))
    cat("Model 2c (smooth mean+var):\n");   print(AIC(m2c)); print(BIC(m2c))
    cat("Model 2d (SHASH constant shape):\n"); print(AIC(m2d)); print(BIC(m2d))
    cat("Model 2e (SHASH full):\n");        print(AIC(m2e)); print(BIC(m2e))

    # --------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------
    library(ggplot2)
    library(cowplot)
    library(patchwork)

    plot_data <- data.frame(
        x = x, y = y,
        cohort = if ("cohort" %in% colnames(df1)) df1$cohort else "All Participants",
        site_id = df1$site_id
    )

    quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")

    # ---- Set 1: No site effects ----
    base_no_site <- function(qlong, ttl, sub = NULL) {
        ggplot(plot_data) +
            geom_point(aes(x = x, y = y, color = cohort), size = 2) +
            geom_line(data = qlong,
                      aes(x = x, y = value, group = variable, linetype = as.factor(variable)),
                      linewidth = 0.35) +
            scale_linetype_manual(values = quantile_linetypes) +
            guides(linetype = "none") + geom_rug() +
            labs(title = ttl, subtitle = sub,
                 x = 'Gestational Age in Weeks', y = 'Volume', color = "Participant Class") +
            ylim(c(min_vol - 0.1 * min_vol, max_vol + 0.1 * max_vol)) +
            xlim(c(min_age - 2, max_age + 2)) +
            scale_x_continuous(breaks = seq(21, 45, by = 1)) +
            theme(plot.title = element_text(hjust = 0.5))
    }

    p1a <- base_no_site(predictions_quantiles_m1a_long, paste('Linear (no site):', i),
                        'Mean of y is modeled as a linear function of x')
    p1b <- base_no_site(predictions_quantiles_m1b_long, paste('Smooth mean (no site):', i))
    p1c <- base_no_site(predictions_quantiles_m1c_long, paste('Smooth mean+var (no site):', i))
    p1d <- base_no_site(predictions_quantiles_m1d_long, paste('SHASH constant shape (no site):', i))
    p1e <- base_no_site(predictions_quantiles_m1e_long, paste('SHASH full (no site):', i))

    p_set1 <- p1a + p1b + p1c + p1d + p1e +
        plot_annotation(title = "Set 1: No Site Effects", tag_levels = 'A') &
        theme_cowplot() &
        theme(text = element_text(size = 9),
              axis.text.x = element_text(size = 5), axis.text.y = element_text(size = 8))

    # ---- Set 2: Random site effects ----
    predictions_quantiles_m2a_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2b_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2c_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2d_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2e_long$site_id <- rep(df1$site_id, 5)

    base_site <- function(qlong, ttl) {
        ggplot(plot_data) +
            geom_point(aes(x = x, y = y, color = cohort), size = 2) +
            geom_line(data = qlong,
                      aes(x = x, y = value,
                          group = interaction(variable, site_id),
                          linetype = as.factor(variable)),
                      linewidth = 0.35) +
            scale_linetype_manual(values = quantile_linetypes) +
            guides(linetype = "none") + geom_rug() +
            labs(title = ttl, x = 'Gestational Age in Weeks', y = 'Volume',
                 color = "Participant Class") +
            ylim(c(min_vol - 0.1 * min_vol, max_vol + 0.1 * max_vol)) +
            xlim(c(min_age - 2, max_age + 2)) +
            scale_x_continuous(breaks = seq(21, 45, by = 1)) +
            theme(plot.title = element_text(hjust = 0.5))
    }

    p2a <- base_site(predictions_quantiles_m2a_long, paste('Linear (random site):', i))
    p2b <- base_site(predictions_quantiles_m2b_long, paste('Smooth mean (random site):', i))
    p2c <- base_site(predictions_quantiles_m2c_long, paste('Smooth mean+var (random site):', i))
    p2d <- base_site(predictions_quantiles_m2d_long, paste('SHASH constant shape (random site):', i))
    p2e <- base_site(predictions_quantiles_m2e_long, paste('SHASH full (random site):', i))

    p_set2 <- p2a + p2b + p2c + p2d + p2e +
        plot_annotation(title = "Set 2: Random Site Effects", tag_levels = 'A') &
        theme_cowplot() &
        theme(text = element_text(size = 9),
              axis.text.x = element_text(size = 5), axis.text.y = element_text(size = 8))

    # --------------------------------------------------------
    # BIC / AIC dataframe (10 models: 2 sets x 5 complexities)
    # --------------------------------------------------------
    bic_aic_data <- data.frame(
        Model = c("m1a", "m1b", "m1c", "m1d", "m1e",
                  "m2a", "m2b", "m2c", "m2d", "m2e"),
        Y_feature = rep(i, 10),
        Site_Effect = rep(c("none", "random"), each = 5),
        Complexity = rep(c("linear", "smooth_mean", "smooth_mean_var",
                           "shash_constant", "shash_full"), 2),
        BIC = c(BIC(m1a), BIC(m1b), BIC(m1c), BIC(m1d), BIC(m1e),
                BIC(m2a), BIC(m2b), BIC(m2c), BIC(m2d), BIC(m2e)),
        AIC = c(AIC(m1a), AIC(m1b), AIC(m1c), AIC(m1d), AIC(m1e),
                AIC(m2a), AIC(m2b), AIC(m2c), AIC(m2d), AIC(m2e))
    )
    results <- rbind(results, bic_aic_data)

    # ========================================================
    # SITE-EFFECT TEST
    # --------------------------------------------------------
    # 1) Pick the best complexity within the NO-site set (BIC).
    # 2) Likelihood-ratio test of that no-site model vs the
    #    SAME complexity with a random site effect (nested).
    #    -> p-value for "is there a site effect?".
    # 3) "Best_Effect" = whichever set (none vs random) has the
    #    lower minimum BIC.
    # ========================================================
    none_models <- list(a = m1a, b = m1b, c = m1c, d = m1d, e = m1e)
    rand_models <- list(a = m2a, b = m2b, c = m2c, d = m2d, e = m2e)

    none_bics <- sapply(none_models, BIC)
    rand_bics <- sapply(rand_models, BIC)

    best_letter <- names(none_bics)[which.min(none_bics)]
    none_model  <- none_models[[best_letter]]
    rand_model  <- rand_models[[best_letter]]

    best_effect <- if (min(none_bics) <= min(rand_bics)) "none" else "random"

    p_site <- tryCatch({
        aov_obj <- anova(none_model, rand_model, test = "Chisq")
        pv <- extract_anova_p(aov_obj)
        if (is.na(pv)) extract_summary_re_p(rand_model) else pv
    }, error = function(e) {
        tryCatch(extract_summary_re_p(rand_model),
                 error = function(e2) NA_real_)
    })

    site_effect_results <- rbind(
        site_effect_results,
        data.frame(Feature     = i,
                   Best_Effect = best_effect,
                   p_value     = p_site,
                   Significant = ifelse(!is.na(p_site) & p_site < 0.05, "Yes", "No"),
                   stringsAsFactors = FALSE)
    )

    cat("\n--- Site effect test for", i, "---\n")
    cat("Best complexity (no-site BIC):", best_letter,
        "| Best_Effect:", best_effect,
        "| p =", signif(p_site, 4), "\n\n")

    # --------------------------------------------------------
    # Save model-comparison plots
    # --------------------------------------------------------
    filename_set1 <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/model_plots/", i, "_set1_no_site_Nov.png")
    filename_set2 <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/model_plots/", i, "_set2_random_site_Nov.png")
    ggsave(filename_set1, p_set1, width = 15, height = 8, units = 'in', bg = "white")
    ggsave(filename_set2, p_set2, width = 15, height = 8, units = 'in', bg = "white")
}

# ============================================================
# SAVE SITE-EFFECT TABLE
# ============================================================
dir.create("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site",
           showWarnings = FALSE, recursive = TRUE)

write.csv(site_effect_results,
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/site_effect_results.csv",
          row.names = FALSE)

cat("\n========== SITE EFFECT SUMMARY ==========\n")
print(site_effect_results)

# ============================================================
# BEST MODELS PER FEATURE (by BIC, across both sets)
# ============================================================
library(dplyr)

best_models <- NULL
for (feature in unique(results$Y_feature)) {
  feature_results <- results[results$Y_feature == feature, ]
  best_idx <- which.min(feature_results$BIC)
  best_models <- rbind(best_models, feature_results[best_idx, ])
}

print("Best models for each feature (based on BIC):")
print(best_models)

dir.create("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves",
           showWarnings = FALSE, recursive = TRUE)

write.csv(best_models,
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/best_models_summary.csv",
          row.names = FALSE)

# ============================================================
# TEST SET + NORMATIVE CURVES
# ============================================================
df_test <- rename_features(df_test_allsites)

test_performance <- data.frame(
  Y_feature = character(), Model = character(),
  RMSE = numeric(), MAE = numeric(), R_squared = numeric(),
  stringsAsFactors = FALSE
)

params_to_quantiles_norm <- function(quantiles, params){
  as.data.frame(sapply(quantiles, function(q){
    qnorm(p = q, mean = params[, 1], sd = exp(params[, 2]))
  }))
}
params_to_quantiles_shash <- function(quantiles, params, model){
  qshash <- model$family$qf
  as.data.frame(sapply(quantiles, function(q){
    qshash(p = q, mu = params)
  }))
}

for (i in 1:nrow(best_models)) {

  feature_name <- best_models$Y_feature[i]
  model_name   <- best_models$Model[i]

  cat("\n================================================\n")
  cat("Creating normative curves for:", feature_name, "\n")
  cat("Best model:", model_name, "\n")
  cat("BIC:", best_models$BIC[i], "AIC:", best_models$AIC[i], "\n")

  model_key  <- paste(feature_name, model_name, sep = "_")
  best_model <- all_fitted_models[[model_key]]
  if (is.null(best_model)) { cat("Warning: Model not found for", model_key, "\n"); next }

  train_pred_data <- data.frame(x = df1$gestational_age,      y = df1[[feature_name]],      site_id = df1$site_id)
  test_pred_data  <- data.frame(x = df_test$gestational_age,  y = df_test[[feature_name]],  site_id = df_test$site_id)

  train_predictions_params <- predict(best_model, newdata = train_pred_data)
  test_predictions_params  <- predict(best_model, newdata = test_pred_data)

  quantiles <- pnorm(c(-2, -1, 0, 1, 2))

  # SHASH models are now only m1d/m1e/m2d/m2e (fixed-site set removed)
  if (model_name %in% c("m1d", "m1e", "m2d", "m2e")) {
    train_quantiles <- params_to_quantiles_shash(quantiles, train_predictions_params, best_model)
    test_quantiles  <- params_to_quantiles_shash(quantiles, test_predictions_params,  best_model)
  } else {
    train_quantiles <- params_to_quantiles_norm(quantiles, train_predictions_params)
    test_quantiles  <- params_to_quantiles_norm(quantiles, test_predictions_params)
  }

  y_test_actual <- df_test[[feature_name]]
  y_test_pred   <- test_quantiles[, 3]
  valid_idx     <- !is.na(y_test_actual) & !is.na(y_test_pred)
  y_test_actual_clean <- y_test_actual[valid_idx]
  y_test_pred_clean   <- y_test_pred[valid_idx]

  if (length(y_test_actual_clean) > 0) {
    rmse <- sqrt(mean((y_test_actual_clean - y_test_pred_clean)^2))
    mae  <- mean(abs(y_test_actual_clean - y_test_pred_clean))
    ss_res <- sum((y_test_actual_clean - y_test_pred_clean)^2)
    ss_tot <- sum((y_test_actual_clean - mean(y_test_actual_clean))^2)
    r_squared <- 1 - (ss_res / ss_tot)
  } else { rmse <- NA; mae <- NA; r_squared <- NA }

  test_performance <- rbind(test_performance,
                            data.frame(Y_feature = feature_name, Model = model_name,
                                       RMSE = rmse, MAE = mae, R_squared = r_squared))

  train_plot_data <- data.frame(x = df1$gestational_age,     y = df1[[feature_name]],     site_id = df1$site_id,     dataset = "Training")
  test_plot_data  <- data.frame(x = df_test$gestational_age, y = df_test[[feature_name]], site_id = df_test$site_id, dataset = "Test")
  combined_plot_data <- rbind(train_plot_data, test_plot_data)

  train_quantiles_long <- reshape2::melt(train_quantiles, variable.name = "quantile", value.name = "value")
  train_quantiles_long$x       <- rep(df1$gestational_age, 5)
  train_quantiles_long$site_id <- rep(df1$site_id, 5)

  quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")
  quantile_labels    <- c("2.3%", "15.9%", "50%", "84.1%", "97.7%")

  # PLOT 1: train vs test, site shapes (now adapts to 4 sites)
  p1_color_dataset <- ggplot(combined_plot_data) +
    geom_line(data = train_quantiles_long,
              aes(x = x, y = value, group = interaction(quantile, site_id), linetype = quantile),
              linewidth = 0.5, color = "gray40", alpha = 0.7) +
    geom_point(aes(x = x, y = y, color = dataset, shape = site_id), size = 2, alpha = 0.7) +
    scale_color_manual(values = c("Training" = "blue", "Test" = "red"), name = "Dataset") +
    scale_shape_manual(values = site_shapes_plot, name = "Site") +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels, name = "Percentile") +
    labs(title = paste(feature_name, "- Colored by Dataset"),
         subtitle = paste("Model:", model_name, "| Test R²:", round(r_squared, 3)),
         x = "Gestational Age (weeks)", y = feature_name) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5), legend.position = "right") +
    scale_x_continuous(breaks = seq(20, 45, by = 2))

  # PLOT 2: all data same color
  p2_same_color <- ggplot(combined_plot_data) +
    geom_line(data = train_quantiles_long,
              aes(x = x, y = value, group = interaction(quantile, site_id), linetype = quantile),
              linewidth = 0.5, color = "gray20", alpha = 0.8) +
    geom_point(aes(x = x, y = y, alpha = dataset), color = "coral", size = 2) +
    scale_alpha_manual(values = c("Training" = 0.8, "Test" = 0.5), name = "Dataset") +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels, name = "Percentile") +
    geom_rug(aes(x = x), alpha = 0.3) +
    labs(title = paste(feature_name, "- All Data"),
         subtitle = paste("Model:", model_name, "| Test R²:", round(r_squared, 3)),
         x = "Gestational Age (weeks)", y = feature_name) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5), legend.position = "right") +
    scale_x_continuous(breaks = seq(20, 45, by = 2))

  # PLOT 3: curves only (now adapts to 4 sites)
  p3_curves_only <- ggplot() +
    geom_line(data = train_quantiles_long,
              aes(x = x, y = value, group = interaction(quantile, site_id),
                  linetype = quantile, color = site_id), linewidth = 0.8) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels, name = "Percentile") +
    scale_color_manual(values = site_colors_plot, name = "Site") +
    labs(title = paste(feature_name, "- Normative Curves Only"),
         subtitle = paste("Model:", model_name, "| Test R²:", round(r_squared, 3)),
         x = "Gestational Age (weeks)", y = feature_name) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5), legend.position = "right",
          panel.grid.minor = element_blank()) +
    scale_x_continuous(breaks = seq(20, 45, by = 2))

  base_out <- "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves/"
  filename1 <- paste0(base_out, gsub(" ", "_", feature_name), "_plot1_colored_dataset.png")
  filename2 <- paste0(base_out, gsub(" ", "_", feature_name), "_plot2_same_color.png")
  filename3 <- paste0(base_out, gsub(" ", "_", feature_name), "_plot3_curves_only.png")

  tryCatch({
    ggsave(filename1, p1_color_dataset, width = 11, height = 7, dpi = 300)
    ggsave(filename2, p2_same_color,    width = 11, height = 7, dpi = 300)
    ggsave(filename3, p3_curves_only,   width = 11, height = 7, dpi = 300)
    cat("Saved plots for", feature_name, "\n")
  }, error = function(e) cat("Error saving plots:", e$message, "\n"))

  library(patchwork)
  p_combined <- (p1_color_dataset | p2_same_color) / p3_curves_only +
    plot_annotation(title = paste("Normative Modeling -", feature_name),
                    subtitle = paste("Best Model:", model_name, "| Test Performance: R² =",
                                     round(r_squared, 3), ", RMSE =", round(rmse, 3)))
  filename_combined <- paste0(base_out, gsub(" ", "_", feature_name), "_combined_plots.png")
  ggsave(filename_combined, p_combined, width = 16, height = 12, dpi = 300)
}

# Save test performance too
write.csv(test_performance,
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/test_performance.csv",
          row.names = FALSE)