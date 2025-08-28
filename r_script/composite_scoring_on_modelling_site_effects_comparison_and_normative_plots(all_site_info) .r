
library(caret)
library(data.table)
library(MASS)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)

data <- as.data.frame(read.csv('/home/INT/dienye.h/python_files/combined_dataset/dhcp_qc_filtered.csv'))
data$site_id <- "dHCP"
data2 <- as.data.frame(read.csv('/home/INT/dienye.h/python_files/combined_dataset/marsfet_qc_filtered.csv'))
data2$site_id <- "MarsFet"
data <- rbind(data, data2)

# gset site as a factor
data$site_id <- as.factor(data$site_id)

train_idx_data2 <- createDataPartition(data2$gestational_age, p = 0.7, list = F)
df_train_data2 <- data2[train_idx_data2,]
df_test_data2 <- data2[-train_idx_data2,]

train_idx_data <- createDataPartition(data$gestational, p = 0.7, list = F)
df_train_allsites <- data[train_idx_data,]
df_test_allsites <- data[-train_idx_data,]


df1 <- df_train_allsites

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

y_values <- list("Surface Area cm2", "Folding Power", "B4 Vertex Percentage", "B5 Vertex Percentage","B6 Vertex Percentage", "Band_parcels B4", "Band Parcels B5", "Band Parcels B6", "Hemispheric Volume", "Gyrification Index", "Hull Area", "B4 Surface Area", "B5 Surface Area", "B6 Surface Area", "B4 Surface Area Percentage", "B5 Surface Area Percentage", "B6 Surface Area Percentage", "B4 Band Power", "B5 Band Power", "B6 Band Power","B4 Band Relative Power", "B5 Band Relative Power", "B6 Band Relative Power")

results <- data.frame(Model = character(),
                      Y_feature = character(),
                      BIC = double(),
                      AIC = double(), stringsAsFactors = FALSE)

# create a list to store ALL fitted models
all_fitted_models <- list()

for (i in y_values) {
    #colnames(df1)[12] <- "volume"
    
    
    x <- df1$gestational_age
    y <- df1[[i]]
    
    max_vol <- max(y)
    min_vol <- min(y)
    
    max_age <- max(x)
    min_age <- min(x)
    
    #Starting GAMLSS:

     # Set 1: No site effects

    m1a <- gam(list(y ~ x, # basically the same model as lm(y~x, data=df1)
                    ~ 1 ),
                    family=gaulss(),
                    optimizer = 'efs',
                    data=df1)

    m1b <- gam(list(y ~ s(x) , # fit mu as a smooth function of x
                    ~ 1 ), # fit sigma only as an intercept
                    family=gaulss(),
                    optimizer = 'efs',
                    data=df1)

    m1c <- gam(list(y ~ s(x) , # fit mu as a smooth function of x
                    ~ s(x) ), # fit sigma as a smooth function of x
                    family=gaulss(),
                    optimizer = 'efs',
                    data=df1)

    m1d <- gam(list(y ~ s(x), # fit mu as a smooth function of x
                    ~ s(x) , # fit sigma as a smooth function of x
                    ~ 1 , # fit nu (skewness) as an intercept
                    ~ 1 ), # fit tau (kurtosis) as an intercept
                    family=shash(), # shash distribution instead of gaussian
                    optimizer = 'efs',
                    data=df1)
            
    m1e <- gam(list(y ~ s(x), # fit mu as a smooth function of x
                    ~ s(x) , # fit sigma as a smooth function of x
                    ~ s(x) , # fit nu (skewness) as a smooth function of x
                    ~ s(x) ), # fit tau (kurtosis) as a smooth function of x
                    family=shash(), # shash distribution instead of gaussian
                    optimizer = 'efs',
                    data=df1)


    # Set 2: Fixed site effects

    m2a <- gam(list(y ~ x + site_id, # linear age + fixed site effects
               ~ 1),             # constant variance
          family = gaulss(),
          optimizer = 'efs',
          data = df1)

    m2b <- gam(list(y ~ s(x) + site_id, # smooth age + fixed site effects
                  ~ 1),               # constant variance
              family = gaulss(),
              optimizer = 'efs',
              data = df1)

    m2c <- gam(list(y ~ s(x) + site_id, # smooth age + fixed site effects
                  ~ s(x) + site_id),  # smooth age + fixed site effects on variance
              family = gaulss(),
              optimizer = 'efs',
              data = df1)

    m2d <- gam(list(y ~ s(x) + site_id, # smooth age + fixed site effects
                  ~ s(x) + site_id,   # smooth age + fixed site effects on variance
                  ~ 1,                # constant skewness
                  ~ 1),               # constant kurtosis
              family = shash(),
              optimizer = 'efs',
              data = df1)

    m2e <- gam(list(y ~ s(x) + site_id, # smooth age + fixed site effects
                  ~ s(x) + site_id,   # smooth age + fixed site effects on variance
                  ~ s(x) + site_id,                # smooth age + fixed site effects on  skewness
                  ~ s(x) + site_id),               # smooth age + fixed site effects on  kurtosis
              family = shash(),
              optimizer = 'efs',
              data = df1)
    


    # Set 3: Random site effects

    m3a <- gam(list(y ~ x + s(site_id, bs="re"), # linear age + random site effects
                ~ 1),                        # constant variance
            family = gaulss(),
            optimizer = 'efs',
            data = df1)

    m3b <- gam(list(y ~ s(x) + s(site_id, bs="re"), # smooth age + random site effects
                  ~ 1),                            # constant variance
              family = gaulss(),
              optimizer = 'efs',
              data = df1)

    m3c <- gam(list(y ~ s(x) + s(site_id, bs="re"), # smooth age + random site effects
                  ~ s(x) + s(site_id, bs="re")),   # smooth age + random site effects on variance
              family = gaulss(),
              optimizer = 'efs',
              data = df1)

    m3d <- gam(list(y ~ s(x) + s(site_id, bs="re"), # smooth age + random site effects
                  ~ s(x) + s(site_id, bs="re"),    # smooth age + random site effects on variance
                  ~ 1,                             # constant skewness
                  ~ 1),                            # constant kurtosis
              family = shash(),
              optimizer = 'efs',
              data = df1)

    m3e <- gam(list(y ~ s(x) + s(site_id, bs="re"), # smooth age + random site effects
                  ~ s(x) + s(site_id, bs="re"),    # smooth age + random site effects on variance
                  ~ s(x) + s(site_id, bs="re"),     # smooth age + random site effects on  skewness
                  ~ s(x) + s(site_id, bs="re")),    # smooth age + random site effects on kurtosis
              family = shash(),
              optimizer = 'efs',
              data = df1)
    
    
    # Begin plotting parameters - updated for all models
    
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

    predictions_params_m3a <- predict(m3a)
    predictions_params_m3b <- predict(m3b)
    predictions_params_m3c <- predict(m3c)
    predictions_params_m3d <- predict(m3d)
    predictions_params_m3e <- predict(m3e)

    params_to_quantiles_norm <- function(quantiles, params){
        as.data.frame(sapply(quantiles, 
                            function(q){
                                qnorm(p=q, mean=params[,1], sd = exp(params[,2]))
                            }))
    }

    params_to_quantiles_shash <- function(quantiles, params, qshash){
        as.data.frame(sapply(quantiles, 
                            function(q){
                                qshash(p=q, 
                                        # param is called mu, but it expects 
                                        # a vector of all 4 shash parameters
                                        mu=params)
                            }))
    }

    quantiles <- pnorm(c(-2:2)) # definition of quantile lines to plot

    # Get qshash function from SHASH models
    qshash <- m1d$family$qf  # or m2d$family$qf or m3d$family$qf

    # Generate quantiles for all models
    # Set 1: No site effects
    predictions_quantiles_m1a <- params_to_quantiles_norm(quantiles, predictions_params_m1a)
    predictions_quantiles_m1b <- params_to_quantiles_norm(quantiles, predictions_params_m1b)
    predictions_quantiles_m1c <- params_to_quantiles_norm(quantiles, predictions_params_m1c)
    predictions_quantiles_m1d <- params_to_quantiles_shash(quantiles, predictions_params_m1d, qshash)
    predictions_quantiles_m1e <- params_to_quantiles_shash(quantiles, predictions_params_m1e, qshash)

    # Set 2: Fixed site effects
    predictions_quantiles_m2a <- params_to_quantiles_norm(quantiles, predictions_params_m2a)
    predictions_quantiles_m2b <- params_to_quantiles_norm(quantiles, predictions_params_m2b)
    predictions_quantiles_m2c <- params_to_quantiles_norm(quantiles, predictions_params_m2c)
    predictions_quantiles_m2d <- params_to_quantiles_shash(quantiles, predictions_params_m2d, qshash)
    predictions_quantiles_m2e <- params_to_quantiles_shash(quantiles, predictions_params_m2e, qshash)

    # Set 3: Random site effects
    predictions_quantiles_m3a <- params_to_quantiles_norm(quantiles, predictions_params_m3a)
    predictions_quantiles_m3b <- params_to_quantiles_norm(quantiles, predictions_params_m3b)
    predictions_quantiles_m3c <- params_to_quantiles_norm(quantiles, predictions_params_m3c)
    predictions_quantiles_m3d <- params_to_quantiles_shash(quantiles, predictions_params_m3d, qshash)
    predictions_quantiles_m3e <- params_to_quantiles_shash(quantiles, predictions_params_m3e, qshash)

    # Enhanced reshape function that includes site information
    reshape_quantiles_to_long <- function(quantiles_df, x_var, site_var = NULL){
        quantiles_df$x <- x_var
        if(!is.null(site_var)) {
            quantiles_df$site_id <- site_var
            return(reshape2::melt(quantiles_df, id.vars = c('x', 'site_id')))
        } else {
            return(reshape2::melt(quantiles_df, id.vars = c('x')))
        }
    }

    # Convert to long format for all models
    # Set 1: No site effects
    predictions_quantiles_m1a_long <- reshape_quantiles_to_long(predictions_quantiles_m1a, df1$gestational_age)
    predictions_quantiles_m1b_long <- reshape_quantiles_to_long(predictions_quantiles_m1b, df1$gestational_age)
    predictions_quantiles_m1c_long <- reshape_quantiles_to_long(predictions_quantiles_m1c, df1$gestational_age)
    predictions_quantiles_m1d_long <- reshape_quantiles_to_long(predictions_quantiles_m1d, df1$gestational_age)
    predictions_quantiles_m1e_long <- reshape_quantiles_to_long(predictions_quantiles_m1e, df1$gestational_age)

    # Set 2: Fixed site effects
    predictions_quantiles_m2a_long <- reshape_quantiles_to_long(predictions_quantiles_m2a, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m2b_long <- reshape_quantiles_to_long(predictions_quantiles_m2b, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m2c_long <- reshape_quantiles_to_long(predictions_quantiles_m2c, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m2d_long <- reshape_quantiles_to_long(predictions_quantiles_m2d, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m2e_long <- reshape_quantiles_to_long(predictions_quantiles_m2e, 
                                                            df1$gestational_age, 
                                                            df1$site_id)

    # Set 3: Random site effects
    predictions_quantiles_m3a_long <- reshape_quantiles_to_long(predictions_quantiles_m3a, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m3b_long <- reshape_quantiles_to_long(predictions_quantiles_m3b, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m3c_long <- reshape_quantiles_to_long(predictions_quantiles_m3c, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m3d_long <- reshape_quantiles_to_long(predictions_quantiles_m3d, 
                                                            df1$gestational_age, 
                                                            df1$site_id)
    predictions_quantiles_m3e_long <- reshape_quantiles_to_long(predictions_quantiles_m3e, 
                                                            df1$gestational_age, 
                                                            df1$site_id)

    # Store each fitted model with a unique key
    all_fitted_models[[paste(i, "m1a", sep="_")]] <- m1a
    all_fitted_models[[paste(i, "m1b", sep="_")]] <- m1b
    all_fitted_models[[paste(i, "m1c", sep="_")]] <- m1c
    all_fitted_models[[paste(i, "m1d", sep="_")]] <- m1d
    all_fitted_models[[paste(i, "m1e", sep="_")]] <- m1e

    all_fitted_models[[paste(i, "m2a", sep="_")]] <- m2a
    all_fitted_models[[paste(i, "m2b", sep="_")]] <- m2b
    all_fitted_models[[paste(i, "m2c", sep="_")]] <- m2c
    all_fitted_models[[paste(i, "m2d", sep="_")]] <- m2d
    all_fitted_models[[paste(i, "m2e", sep="_")]] <- m2e

    all_fitted_models[[paste(i, "m3a", sep="_")]] <- m3a
    all_fitted_models[[paste(i, "m3b", sep="_")]] <- m3b
    all_fitted_models[[paste(i, "m3c", sep="_")]] <- m3c
    all_fitted_models[[paste(i, "m3d", sep="_")]] <- m3d
    all_fitted_models[[paste(i, "m3e", sep="_")]] <- m3e

    # Update the results printing section
    cat("Results for y =", i, ":\n")
    cat("=== Set 1: No Site Effects ===\n")
    cat("Model 1a (linear):\n"); print(AIC(m1a)); print(BIC(m1a))
    cat("Model 1b (smooth mean):\n"); print(AIC(m1b)); print(BIC(m1b))
    cat("Model 1c (smooth mean+var):\n"); print(AIC(m1c)); print(BIC(m1c))
    cat("Model 1d (SHASH constant shape):\n"); print(AIC(m1d)); print(BIC(m1d))
    cat("Model 1e (SHASH full):\n"); print(AIC(m1e)); print(BIC(m1e))

    cat("\n=== Set 2: Fixed Site Effects ===\n")
    cat("Model 2a (linear):\n"); print(AIC(m2a)); print(BIC(m2a))
    cat("Model 2b (smooth mean):\n"); print(AIC(m2b)); print(BIC(m2b))
    cat("Model 2c (smooth mean+var):\n"); print(AIC(m2c)); print(BIC(m2c))
    cat("Model 2d (SHASH constant shape):\n"); print(AIC(m2d)); print(BIC(m2d))
    cat("Model 2e (SHASH full):\n"); print(AIC(m2e)); print(BIC(m2e))

    cat("\n=== Set 3: Random Site Effects ===\n")
    cat("Model 3a (linear):\n"); print(AIC(m3a)); print(BIC(m3a))
    cat("Model 3b (smooth mean):\n"); print(AIC(m3b)); print(BIC(m3b))
    cat("Model 3c (smooth mean+var):\n"); print(AIC(m3c)); print(BIC(m3c))
    cat("Model 3d (SHASH constant shape):\n"); print(AIC(m3d)); print(BIC(m3d))
    cat("Model 3e (SHASH full):\n"); print(AIC(m3e)); print(BIC(m3e))
        
        
library(ggplot2)
    library(cowplot)
    library(patchwork)

    # Create plotting data frame with proper variable handling
    plot_data <- data.frame(
        x = x,
        y = y,
        cohort = if("cohort" %in% colnames(df1)) df1$cohort else "All Participants",
        site_id = df1$site_id  # Add site_id to plot_data
    )

    quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")

    # Plot Set 1: No site effects
    p1a <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m1a_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Linear (no site):', i),
            subtitle='Mean of y is modeled as a linear function of x',
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p1b <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m1b_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Smooth mean (no site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p1c <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m1c_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Smooth mean+var (no site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p1d <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m1d_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title=paste('SHASH constant shape (no site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Cohort") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p1e <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m1e_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title=paste('SHASH full (no site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Cohort") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    # Create combined plots
    p_set1 <- p1a + p1b + p1c + p1d + p1e + 
        plot_annotation(title = "Set 1: No Site Effects", tag_levels = 'A') & 
        theme_cowplot() &
        theme(text=element_text(size=9),
              axis.text.x = element_text(size=5),axis.text.y = element_text(size=8))

    # Plot Set 2: Fixed site effects - CORRECTED with interaction grouping
    # Add site_id to the long format data for Set 2
    predictions_quantiles_m2a_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2b_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2c_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2d_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m2e_long$site_id <- rep(df1$site_id, 5)
    
    p2a <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m2a_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Linear (fixed site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p2b <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m2b_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Smooth mean (fixed site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p2c <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m2c_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Smooth mean+var (fixed site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p2d <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m2d_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title=paste('SHASH constant shape (fixed site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Cohort") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p2e <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m2e_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title=paste('SHASH full (fixed site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Cohort") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    # Create combined plot for Set 2
    p_set2 <- p2a + p2b + p2c + p2d + p2e + 
        plot_annotation(title = "Set 2: Fixed Site Effects", tag_levels = 'A') & 
        theme_cowplot() &
        theme(text=element_text(size=9),
            axis.text.x = element_text(size=5),axis.text.y = element_text(size=8)) 


    # Plot Set 3: Random site effects - CORRECTED with interaction grouping
    # Add site_id to the long format data for Set 3
    predictions_quantiles_m3a_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m3b_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m3c_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m3d_long$site_id <- rep(df1$site_id, 5)
    predictions_quantiles_m3e_long$site_id <- rep(df1$site_id, 5)
    
    p3a <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m3a_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Linear (random site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p3b <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m3b_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Smooth mean (random site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p3c <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m3c_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Smooth mean+var (random site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p3d <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m3d_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title=paste('SHASH constant shape (random site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Cohort") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    p3e <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m3e_long, 
                  aes(x=x, y=value, 
                      group=interaction(variable, site_id), 
                      linetype = as.factor(variable)),
                  linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title=paste('SHASH full (random site):', i),
            x = 'Gestational Age in Weeks', 
            y='Volume')  +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Cohort") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))

    # Create combined plot for Set 3
    p_set3 <- p3a + p3b + p3c + p3d + p3e + 
        plot_annotation(title = "Set 3: Random Site Effects", tag_levels = 'A') & 
        theme_cowplot() &
        theme(text=element_text(size=9),
            axis.text.x = element_text(size=5),axis.text.y = element_text(size=8))

    # Updated BIC/AIC comparison for all models
    cat("=== Model Comparison Results for", i, "===\n")

    # Set 1: No site effects
    cat("\n--- Set 1: No Site Effects ---\n")
    print(paste("BIC m1a:", round(BIC(m1a), 2)))
    print(paste("BIC m1b:", round(BIC(m1b), 2)))
    print(paste("BIC m1c:", round(BIC(m1c), 2)))
    print(paste("BIC m1d:", round(BIC(m1d), 2)))
    print(paste("BIC m1e:", round(BIC(m1e), 2)))

    print(paste("AIC m1a:", round(AIC(m1a), 2)))
    print(paste("AIC m1b:", round(AIC(m1b), 2)))
    print(paste("AIC m1c:", round(AIC(m1c), 2)))
    print(paste("AIC m1d:", round(AIC(m1d), 2)))
    print(paste("AIC m1e:", round(AIC(m1e), 2)))

    # Set 2: Fixed site effects
    cat("\n--- Set 2: Fixed Site Effects ---\n")
    print(paste("BIC m2a:", round(BIC(m2a), 2)))
    print(paste("BIC m2b:", round(BIC(m2b), 2)))
    print(paste("BIC m2c:", round(BIC(m2c), 2)))
    print(paste("BIC m2d:", round(BIC(m2d), 2)))
    print(paste("BIC m2e:", round(BIC(m2e), 2)))

    print(paste("AIC m2a:", round(AIC(m2a), 2)))
    print(paste("AIC m2b:", round(AIC(m2b), 2)))
    print(paste("AIC m2c:", round(AIC(m2c), 2)))
    print(paste("AIC m2d:", round(AIC(m2d), 2)))
    print(paste("AIC m2e:", round(AIC(m2e), 2)))

    # Set 3: Random site effects
    cat("\n--- Set 3: Random Site Effects ---\n")
    print(paste("BIC m3a:", round(BIC(m3a), 2)))
    print(paste("BIC m3b:", round(BIC(m3b), 2)))
    print(paste("BIC m3c:", round(BIC(m3c), 2)))
    print(paste("BIC m3d:", round(BIC(m3d), 2)))
    print(paste("BIC m3e:", round(BIC(m3e), 2)))

    print(paste("AIC m3a:", round(AIC(m3a), 2)))
    print(paste("AIC m3b:", round(AIC(m3b), 2)))
    print(paste("AIC m3c:", round(AIC(m3c), 2)))
    print(paste("AIC m3d:", round(AIC(m3d), 2)))
    print(paste("AIC m3e:", round(AIC(m3e), 2)))

    # Create comprehensive results dataframe
    bic_aic_data <- data.frame(
        Model = c("m1a", "m1b", "m1c", "m1d", "m1e",
                  "m2a", "m2b", "m2c", "m2d", "m2e", 
                  "m3a", "m3b", "m3c", "m3d", "m3e"),
        Y_feature = rep(i, 15),
        Site_Effect = rep(c("none", "fixed", "random"), each = 5),
        Complexity = rep(c("linear", "smooth_mean", "smooth_mean_var", "shash_constant", "shash_full"), 3),
        BIC = c(BIC(m1a), BIC(m1b), BIC(m1c), BIC(m1d), BIC(m1e),
                BIC(m2a), BIC(m2b), BIC(m2c), BIC(m2d), BIC(m2e),
                BIC(m3a), BIC(m3b), BIC(m3c), BIC(m3d), BIC(m3e)),
        AIC = c(AIC(m1a), AIC(m1b), AIC(m1c), AIC(m1d), AIC(m1e),
                AIC(m2a), AIC(m2b), AIC(m2c), AIC(m2d), AIC(m2e),
                AIC(m3a), AIC(m3b), AIC(m3c), AIC(m3d), AIC(m3e))
    )

    results <- rbind(results, bic_aic_data)

    # Save plots
    filename_set1 <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/model_plots/", i, "_set1_no_site_Nov.png")
    filename_set2 <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/model_plots/", i, "_set2_fixed_site_Nov.png")
    filename_set3 <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/model_plots/", i, "_set3_random_site_Nov.png")

    ggsave(filename_set1, p_set1, width = 15, height = 8, units = 'in', bg="white")
    ggsave(filename_set2, p_set2, width = 15, height = 8, units = 'in', bg="white")
    ggsave(filename_set3, p_set3, width = 15, height = 8, units = 'in', bg="white")
    # Uncomment to save other sets as well

    }

    # Save comprehensive results
    # write.csv(results, "comprehensive_model_comparison_results.csv", row.names = FALSE)



# ============================================
# INTEGRATED PIPELINE: COMPREHENSIVE EVALUATION + PLOTTING
# ============================================
# This code should be added AFTER your main loop where models are fitted and stored

library(gamlss.dist)
library(dplyr)
library(ggplot2)
library(reshape2)
library(patchwork)

# ============================================
# STEP 1: HELPER FUNCTIONS
# ============================================

params_to_scores <- function(y, params){
  if(ncol(params) == 2){
    params <- cbind(params, rep(0, nrow(params)))
    params <- cbind(params, rep(log(1), nrow(params)))
  }
  
  log_densities <- gamlss.dist::dSHASHo2(y,
                                         mu=params[,1],
                                         sigma=exp(params[,2]),
                                         nu=params[,3],
                                         tau=exp(params[,4]),
                                         log=T)
  
  quantiles <- gamlss.dist::pSHASHo2(y, mu=params[,1],
                                     sigma=exp(params[,2]),
                                     nu=params[,3],
                                     tau=exp(params[,4]))
  
  z_randomized <- qnorm(quantiles, mean = 0, sd = 1)
  
  log_densities_censored <- log_densities
  log_densities_censored[abs(z_randomized) > 2.326348] <- log(0.02)
  
  data.frame('log_densities' = log_densities,
             'log_densities_censored' = log_densities_censored,
             'quantiles' = quantiles,
             'z_randomized' = z_randomized)
}

calibration_descriptives <- function(x) {
  x_clean <- x[!is.na(x) & !is.infinite(x)]
  n_clean <- length(x_clean)
  
  if (n_clean < 10) {
    return(data.frame(t(c('mean'=NA, 'sd'=NA, 'skew'=NA, 'kurtosis'=NA, 'W'=NA))))
  }
  
  m1 <- mean(x_clean)
  s1 <- sd(x_clean)
  m2 <- sum((x_clean-m1)^2)
  m3 <- sum((x_clean-m1)^3)
  m4 <- sum((x_clean-m1)^4)
  
  if(s1 > 0 && n_clean > 3) {
    skew <- n_clean*m3/(n_clean-1)/(n_clean-2)/s1^3
    kurtosis <- (n_clean*(n_clean+1)*m4 - 3*m2^2*(n_clean-1)) / 
                ((n_clean-1)*(n_clean-2)*(n_clean-3)*s1^4)
  } else {
    skew <- NA
    kurtosis <- NA
  }
  
  if (length(x_clean) >= 3 && length(x_clean) < 5000){
    W = tryCatch(shapiro.test(x_clean)$statistic, error = function(e) NA)
  } else if (length(x_clean) >= 5000) {
    W = tryCatch(shapiro.test(sample(x_clean, 5000))$statistic, error = function(e) NA)
  } else {
    W = NA
  }
  
  return(data.frame(t(c('mean'=m1, 'sd'=s1, 'skew'=skew, 'kurtosis'=kurtosis, 'W'=W))))
}

normalize_metric <- function(x, higher_better = TRUE, ideal_value = NULL) {
  x_clean <- x[!is.na(x) & !is.infinite(x)]
  if (length(x_clean) == 0) return(rep(NA, length(x)))
  
  if (!is.null(ideal_value)) {
    deviation <- abs(x - ideal_value)
    max_dev <- max(deviation, na.rm = TRUE)
    if (max_dev == 0) return(rep(1, length(x)))
    return(1 - deviation / max_dev)
  } else if (higher_better) {
    min_val <- min(x_clean)
    max_val <- max(x_clean)
    if (max_val == min_val) return(rep(0.5, length(x)))
    return((x - min_val) / (max_val - min_val))
  } else {
    min_val <- min(x_clean)
    max_val <- max(x_clean)
    if (max_val == min_val) return(rep(0.5, length(x)))
    return(1 - (x - min_val) / (max_val - min_val))
  }
}

# ============================================
# STEP 2: COMPREHENSIVE EVALUATION
# ============================================

cat("Starting comprehensive model evaluation...\n")

# Prepare test set with renamed columns
df_test <- df_test_allsites
colnames(df_test)[colnames(df_test) == "surface_area_cm2"] <- "Surface Area cm2"
colnames(df_test)[colnames(df_test) == "analyze_folding_power"] <- "Folding Power"
colnames(df_test)[colnames(df_test) == "B4_vertex_percentage"] <- "B4 Vertex Percentage"
colnames(df_test)[colnames(df_test) == "B5_vertex_percentage"] <- "B5 Vertex Percentage"
colnames(df_test)[colnames(df_test) == "B6_vertex_percentage"] <- "B6 Vertex Percentage"
colnames(df_test)[colnames(df_test) == "band_parcels_B4"] <- "Band_parcels B4"
colnames(df_test)[colnames(df_test) == "band_parcels_B5"] <- "Band Parcels B5"
colnames(df_test)[colnames(df_test) == "band_parcels_B6"] <- "Band Parcels B6"
colnames(df_test)[colnames(df_test) == "volume_ml"] <- "Hemispheric Volume"
colnames(df_test)[colnames(df_test) == "gyrification_index"] <- "Gyrification Index"
colnames(df_test)[colnames(df_test) == "hull_area"] <- "Hull Area"
colnames(df_test)[colnames(df_test) == "B4_surface_area"] <- "B4 Surface Area"
colnames(df_test)[colnames(df_test) == "B5_surface_area"] <- "B5 Surface Area"
colnames(df_test)[colnames(df_test) == "B6_surface_area"] <- "B6 Surface Area"
colnames(df_test)[colnames(df_test) == "B4_surface_area_percentage"] <- "B4 Surface Area Percentage"
colnames(df_test)[colnames(df_test) == "B5_surface_area_percentage"] <- "B5 Surface Area Percentage"
colnames(df_test)[colnames(df_test) == "B6_surface_area_percentage"] <- "B6 Surface Area Percentage"
colnames(df_test)[colnames(df_test) == "band_power_B4"] <- "B4 Band Power"
colnames(df_test)[colnames(df_test) == "band_power_B5"] <- "B5 Band Power"
colnames(df_test)[colnames(df_test) == "band_power_B6"] <- "B6 Band Power"
colnames(df_test)[colnames(df_test) == "B4_band_relative_power"] <- "B4 Band Relative Power"
colnames(df_test)[colnames(df_test) == "B5_band_relative_power"] <- "B5 Band Relative Power"
colnames(df_test)[colnames(df_test) == "B6_band_relative_power"] <- "B6 Band Relative Power"

# Start with existing results that have BIC/AIC
comprehensive_results <- results
comprehensive_results$LogScore <- NA
comprehensive_results$LogScore_Censored <- NA
comprehensive_results$W_statistic <- NA
comprehensive_results$Skewness <- NA
comprehensive_results$Kurtosis <- NA
comprehensive_results$Mean_Z <- NA
comprehensive_results$SD_Z <- NA

# Evaluate all models on test set
cat("Evaluating", nrow(comprehensive_results), "models on test set...\n")

for (i in 1:nrow(comprehensive_results)) {
  if (i %% 15 == 0) {
    cat("Progress:", round(100*i/nrow(comprehensive_results), 1), "%\n")
  }
  
  feature_name <- comprehensive_results$Y_feature[i]
  model_name <- comprehensive_results$Model[i]
  model_key <- paste(feature_name, model_name, sep="_")
  model <- all_fitted_models[[model_key]]
  
  if (!is.null(model)) {
    test_pred_data <- data.frame(
      x = df_test$gestational_age,
      y = df_test[[feature_name]],
      site_id = df_test$site_id
    )
    
    tryCatch({
      test_predictions <- predict(model, newdata = test_pred_data)
      test_scores <- params_to_scores(test_pred_data$y, test_predictions)
      calib_stats <- calibration_descriptives(test_scores$z_randomized)
      
      comprehensive_results$LogScore[i] <- mean(test_scores$log_densities, na.rm = TRUE)
      comprehensive_results$LogScore_Censored[i] <- mean(test_scores$log_densities_censored, na.rm = TRUE)
      comprehensive_results$W_statistic[i] <- calib_stats$W
      comprehensive_results$Skewness[i] <- calib_stats$skew
      comprehensive_results$Kurtosis[i] <- calib_stats$kurtosis
      comprehensive_results$Mean_Z[i] <- calib_stats$mean
      comprehensive_results$SD_Z[i] <- calib_stats$sd
    }, error = function(e) {
      # Silent error handling
    })
  }
}

# ============================================
# STEP 3: COMPOSITE SCORING
# ============================================

cat("\nCalculating composite scores...\n")

# Normalize metrics by feature
comprehensive_results <- comprehensive_results %>%
  group_by(Y_feature) %>%
  mutate(
    norm_BIC = normalize_metric(BIC, higher_better = FALSE),
    norm_AIC = normalize_metric(AIC, higher_better = FALSE),
    norm_LogScore = normalize_metric(LogScore, higher_better = TRUE),
    norm_W = normalize_metric(W_statistic, higher_better = TRUE),
    norm_Skew = normalize_metric(abs(Skewness), higher_better = FALSE),
    norm_Kurt = normalize_metric(abs(Kurtosis), higher_better = FALSE),
    norm_MeanZ = normalize_metric(Mean_Z, ideal_value = 0),
    norm_SDZ = normalize_metric(SD_Z, ideal_value = 1)
  ) %>%
  ungroup()

# Calculate balanced composite score
weights <- c(BIC = 0.15, AIC = 0.1, LogScore = 0.2, W = 0.2,
            Skew = 0.1, Kurt = 0.1, MeanZ = 0.075, SDZ = 0.075)

comprehensive_results$composite_score <- 
  comprehensive_results$norm_BIC * weights["BIC"] +
  comprehensive_results$norm_AIC * weights["AIC"] +
  comprehensive_results$norm_LogScore * weights["LogScore"] +
  comprehensive_results$norm_W * weights["W"] +
  comprehensive_results$norm_Skew * weights["Skew"] +
  comprehensive_results$norm_Kurt * weights["Kurt"] +
  comprehensive_results$norm_MeanZ * weights["MeanZ"] +
  comprehensive_results$norm_SDZ * weights["SDZ"]

# Select best models based on composite score
best_models <- comprehensive_results %>%
  group_by(Y_feature) %>%
  slice_max(composite_score, n = 1, with_ties = FALSE) %>%
  ungroup()

cat("\nBest models selected using composite scoring:\n")
print(table(best_models$Model))

# Save results
write.csv(comprehensive_results, 
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/comprehensive_evaluation.csv",
          row.names = FALSE)
write.csv(best_models,
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/best_models_composite.csv",
          row.names = FALSE)


# ============================================
# STEP 4: GENERATE PLOTS WITH BEST MODELS
# ============================================

cat("\n\nGenerating plots with best composite models...\n")

dir.create("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves_composite", 
           showWarnings = FALSE, recursive = TRUE)

# Store final performance metrics
final_performance <- data.frame()

for (i in 1:nrow(best_models)) {
  
  feature_name <- best_models$Y_feature[i]
  model_name <- best_models$Model[i]
  composite_score <- best_models$composite_score[i]
  
  cat("\n================================================\n")
  cat("Creating plots for:", feature_name, "\n")
  cat("Best model:", model_name, "\n")
  cat("Composite score:", round(composite_score, 3), "\n")
  cat("Metrics - BIC:", round(best_models$BIC[i], 1), 
      "| LogScore:", round(best_models$LogScore[i], 3),
      "| W:", round(best_models$W_statistic[i], 3), "\n")
  
  # Retrieve the best model
  model_key <- paste(feature_name, model_name, sep="_")
  best_model <- all_fitted_models[[model_key]]
  
  if (is.null(best_model)) {
    cat("Warning: Model not found\n")
    next
  }
  
  # Prepare data for prediction
  train_pred_data <- data.frame(
    x = df1$gestational_age,
    y = df1[[feature_name]],
    site_id = df1$site_id
  )
  
  test_pred_data <- data.frame(
    x = df_test$gestational_age,
    y = df_test[[feature_name]],
    site_id = df_test$site_id
  )
  
  # Get predictions
  train_predictions_params <- predict(best_model, newdata = train_pred_data)
  test_predictions_params <- predict(best_model, newdata = test_pred_data)
  
  # Get test scores for this model
  test_scores <- params_to_scores(test_pred_data$y, test_predictions_params)
  
  # Store performance
  final_performance <- rbind(final_performance, 
                            data.frame(
                              Y_feature = feature_name,
                              Model = model_name,
                              Composite_Score = composite_score,
                              BIC = best_models$BIC[i],
                              LogScore = best_models$LogScore[i],
                              W_statistic = best_models$W_statistic[i],
                              Skewness = best_models$Skewness[i],
                              Kurtosis = best_models$Kurtosis[i]
                            ))
  
  # Generate quantiles for plotting
  quantiles <- pnorm(c(-2, -1, 0, 1, 2))
  
  params_to_quantiles_plot <- function(params, quantiles){
    if(ncol(params) == 2){
      params <- cbind(params, matrix(0, nrow = nrow(params), ncol = 2))
    }
    return(as.data.frame(sapply(quantiles,
                                function(q){gamlss.dist::qSHASHo2(q, params[,1],
                                                                  exp(params[,2]),
                                                                  params[,3],
                                                                  exp(params[,4]))})))
  }
  
  train_quantiles <- params_to_quantiles_plot(train_predictions_params, quantiles)
  
  # Prepare plotting data
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
  
  combined_plot_data <- rbind(train_plot_data, test_plot_data)
  
  # Prepare quantile data
  train_quantiles_long <- reshape2::melt(train_quantiles, variable.name = "quantile", value.name = "value")
  train_quantiles_long$x <- rep(df1$gestational_age, 5)
  train_quantiles_long$site_id <- rep(df1$site_id, 5)
  
  quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")
  quantile_labels <- c("2.3%", "15.9%", "50%", "84.1%", "97.7%")
  
  # Create three plot types as requested
  
  # PLOT 1: Training (blue) vs Test (red) with site shapes
  p1 <- ggplot(combined_plot_data) +
    geom_line(data = train_quantiles_long,
              aes(x = x, y = value, 
                  group = interaction(quantile, site_id),
                  linetype = quantile),
              linewidth = 0.5, color = "gray40", alpha = 0.7) +
    geom_point(aes(x = x, y = y, color = dataset, shape = site_id), 
               size = 2, alpha = 0.7) +
    scale_color_manual(values = c("Training" = "blue", "Test" = "red")) +
    scale_shape_manual(values = c("dHCP" = 17, "MarsFet" = 16)) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    labs(title = paste(feature_name, "- Colored by Dataset"),
         subtitle = paste("Model:", model_name, "| Composite Score:", round(composite_score, 3)),
         x = "Gestational Age (weeks)", y = feature_name) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5))
  
  # PLOT 2: All data same color
  p2 <- ggplot(combined_plot_data) +
    geom_line(data = train_quantiles_long,
              aes(x = x, y = value, 
                  group = interaction(quantile, site_id),
                  linetype = quantile),
              linewidth = 0.5, color = "gray20", alpha = 0.8) +
    geom_point(aes(x = x, y = y, alpha = dataset), 
               color = "coral", size = 2) +
    scale_alpha_manual(values = c("Training" = 0.8, "Test" = 0.5)) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    geom_rug(aes(x = x), alpha = 0.3) +
    labs(title = paste(feature_name, "- All Data"),
         subtitle = paste("W:", round(best_models$W_statistic[i], 3), 
                         "| LogScore:", round(best_models$LogScore[i], 3)),
         x = "Gestational Age (weeks)", y = feature_name) +
    theme_bw()
  
  # PLOT 3: Curves only
  p3 <- ggplot() +
    geom_line(data = train_quantiles_long,
              aes(x = x, y = value, 
                  group = interaction(quantile, site_id),
                  linetype = quantile, color = site_id),
              linewidth = 0.8) +
    scale_linetype_manual(values = quantile_linetypes, labels = quantile_labels) +
    scale_color_manual(values = c("dHCP" = "steelblue", "MarsFet" = "darkgreen")) +
    labs(title = paste(feature_name, "- Normative Curves"),
         subtitle = paste("Best Model:", model_name),
         x = "Gestational Age (weeks)", y = feature_name) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Save individual plots
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves_composite/",
                gsub(" ", "_", feature_name), "_dataset_colored.png"),
         p1, width = 10, height = 6, dpi = 300)
  
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves_composite/",
                gsub(" ", "_", feature_name), "_uniform_color.png"),
         p2, width = 10, height = 6, dpi = 300)
  
  ggsave(paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves_composite/",
                gsub(" ", "_", feature_name), "_curves_only.png"),
         p3, width = 10, height = 6, dpi = 300)
  
  cat("Saved plots for", feature_name, "\n")

    # Optional: Create a combined plot with all three
    library(patchwork)
    p_combined <- (p1 | p2) / p3 +
    plot_annotation(
        title = paste("Normative Modeling -", feature_name),
        subtitle = paste("Best Model:", model_name, 
                        "| Composite Score:", round(composite_score, 3),
                        "| W:", round(best_models$W_statistic[i], 3),
                        "| LogScore:", round(best_models$LogScore[i], 3))
    )

    filename_combined <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves_composite/",
                            gsub(" ", "_", feature_name), "_combined_plots.png")

    ggsave(filename_combined, p_combined, width = 16, height = 12, dpi = 300)
}

# Save final performance summary
write.csv(final_performance,
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/final_performance_composite.csv",
          row.names = FALSE)

cat("\n================================================\n")
cat("COMPLETE! Generated plots using composite scoring.\n")
cat("Files saved in: /normative_curves_composite/\n")
cat("================================================\n")