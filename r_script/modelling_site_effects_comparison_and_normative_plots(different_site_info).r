
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
# FIXED CODE - CORRECT VARIABLE NAMES FOR PREDICTION
# ============================================

library(dplyr)

# Find best models
best_models <- NULL
for (feature in unique(results$Y_feature)) {
  feature_results <- results[results$Y_feature == feature, ]
  best_idx <- which.min(feature_results$BIC)
  best_models <- rbind(best_models, feature_results[best_idx, ])
}

print("Best models for each feature (based on BIC):")
print(best_models)

# Create directory if it doesn't exist
dir.create("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves", 
           showWarnings = FALSE, recursive = TRUE)

# Save best models summary
write.csv(best_models, 
          "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/best_models_summary.csv", 
          row.names = FALSE)

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

# Store test set performance metrics
test_performance <- data.frame(
  Y_feature = character(),
  Model = character(),
  RMSE = numeric(),
  MAE = numeric(),
  R_squared = numeric(),
  stringsAsFactors = FALSE
)

# Helper functions for quantile conversion
params_to_quantiles_norm <- function(quantiles, params){
  as.data.frame(sapply(quantiles, 
                      function(q){
                        qnorm(p=q, mean=params[,1], sd = exp(params[,2]))
                      }))
}

params_to_quantiles_shash <- function(quantiles, params, model){
  qshash <- model$family$qf
  as.data.frame(sapply(quantiles, 
                      function(q){
                        qshash(p=q, mu=params)
                      }))
}

# Generate normative curves for each feature using its best model
for (i in 1:nrow(best_models)) {
  
  feature_name <- best_models$Y_feature[i]
  model_name <- best_models$Model[i]
  
  cat("\n================================================\n")
  cat("Creating normative curves for:", feature_name, "\n")
  cat("Best model:", model_name, "\n")
  cat("BIC:", best_models$BIC[i], "AIC:", best_models$AIC[i], "\n")
  
  # Retrieve the already fitted model
  model_key <- paste(feature_name, model_name, sep="_")
  best_model <- all_fitted_models[[model_key]]
  
  if (is.null(best_model)) {
    cat("Warning: Model not found for", model_key, "\n")
    next
  }
  
  # IMPORTANT: Create prediction data frames with the SAME variable names used during fitting
  # The models were fitted with 'x' and 'y' as variable names
  
  # Prepare training data for prediction
  train_pred_data <- data.frame(
    x = df1$gestational_age,
    y = df1[[feature_name]],  # Include y even though it's not used for prediction
    site_id = df1$site_id
  )
  
  # Prepare test data for prediction
  test_pred_data <- data.frame(
    x = df_test$gestational_age,
    y = df_test[[feature_name]],  # Include y even though it's not used for prediction
    site_id = df_test$site_id
  )
  
  # Get predictions using the correctly named data
  train_predictions_params <- predict(best_model, newdata = train_pred_data)
  test_predictions_params <- predict(best_model, newdata = test_pred_data)
  
  # Define quantiles for normative curves
  quantiles <- pnorm(c(-2, -1, 0, 1, 2))  # 2.3%, 15.9%, 50%, 84.1%, 97.7% percentiles
  
  # Generate quantiles based on model type
  if (model_name %in% c("m1d", "m1e", "m2d", "m2e", "m3d", "m3e")) {
    # SHASH models
    train_quantiles <- params_to_quantiles_shash(quantiles, train_predictions_params, best_model)
    test_quantiles <- params_to_quantiles_shash(quantiles, test_predictions_params, best_model)
  } else {
    # Gaussian models
    train_quantiles <- params_to_quantiles_norm(quantiles, train_predictions_params)
    test_quantiles <- params_to_quantiles_norm(quantiles, test_predictions_params)
  }
  
  # Calculate test set performance metrics
  y_test_actual <- df_test[[feature_name]]
  y_test_pred <- test_quantiles[, 3]  # Use median (50th percentile) as point prediction
  
  # Remove NA values for calculation
  valid_idx <- !is.na(y_test_actual) & !is.na(y_test_pred)
  y_test_actual_clean <- y_test_actual[valid_idx]
  y_test_pred_clean <- y_test_pred[valid_idx]
  
  if (length(y_test_actual_clean) > 0) {
    rmse <- sqrt(mean((y_test_actual_clean - y_test_pred_clean)^2))
    mae <- mean(abs(y_test_actual_clean - y_test_pred_clean))
    ss_res <- sum((y_test_actual_clean - y_test_pred_clean)^2)
    ss_tot <- sum((y_test_actual_clean - mean(y_test_actual_clean))^2)
    r_squared <- 1 - (ss_res/ss_tot)
  } else {
    rmse <- NA
    mae <- NA
    r_squared <- NA
  }
  
  # Store performance metrics
  test_performance <- rbind(test_performance, 
                           data.frame(Y_feature = feature_name,
                                    Model = model_name,
                                    RMSE = rmse,
                                    MAE = mae,
                                    R_squared = r_squared))
  
  # Prepare data for plotting
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
  
  # Combine train and test data
  combined_plot_data <- rbind(train_plot_data, test_plot_data)
  
  # Prepare quantile data for plotting
  train_quantiles_long <- reshape2::melt(train_quantiles, variable.name = "quantile", value.name = "value")
  train_quantiles_long$x <- rep(df1$gestational_age, 5)
  train_quantiles_long$site_id <- rep(df1$site_id, 5)
  train_quantiles_long$dataset <- "Training"
  
  test_quantiles_long <- reshape2::melt(test_quantiles, variable.name = "quantile", value.name = "value")
  test_quantiles_long$x <- rep(df_test$gestational_age, 5)
  test_quantiles_long$site_id <- rep(df_test$site_id, 5)
  test_quantiles_long$dataset <- "Test"
  
  # Combine all quantile data
  all_quantiles_long <- rbind(train_quantiles_long, test_quantiles_long)
  
  # Define line types for quantiles
  quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")
  quantile_labels <- c("2.3%", "15.9%", "50%", "84.1%", "97.7%")
  
  # Create the normative curve plot
  p_normative <- ggplot(combined_plot_data) +
    geom_point(aes(x = x, y = y, color = interaction(site_id, dataset)), 
               size = 1.5, alpha = 0.6) +
    geom_line(data = all_quantiles_long[all_quantiles_long$dataset == "Training", ],
              aes(x = x, y = value, 
                  group = interaction(quantile, site_id),
                  linetype = quantile),
              linewidth = 0.6, color = "blue", alpha = 0.8) +
    geom_line(data = all_quantiles_long[all_quantiles_long$dataset == "Test", ],
              aes(x = x, y = value, 
                  group = interaction(quantile, site_id),
                  linetype = quantile),
              linewidth = 0.6, color = "red", alpha = 0.6) +
    scale_linetype_manual(values = quantile_linetypes,
                         labels = quantile_labels,
                         name = "Percentile") +
    scale_color_manual(values = c("dHCP.Training" = "blue",
                                 "MarsFet.Training" = "darkblue",
                                 "dHCP.Test" = "red",
                                 "MarsFet.Test" = "darkred"),
                      name = "Site.Dataset",
                      labels = c("dHCP Training", "MarsFet Training", 
                                "dHCP Test", "MarsFet Test")) +
    labs(title = paste("Normative Curves -", feature_name),
         subtitle = paste("Best Model:", model_name, 
                         "| Test RMSE:", round(rmse, 3), 
                         "| Test R²:", round(r_squared, 3)),
         x = "Gestational Age (weeks)",
         y = feature_name) +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5),
          legend.position = "right") +
    scale_x_continuous(breaks = seq(20, 45, by = 2))
  
  # Save the normative curve plot
  filename <- paste0("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/normative_curves/",
                    gsub(" ", "_", feature_name), "_normative_curves.png")
  
  tryCatch({
    ggsave(filename, p_normative, width = 12, height = 8, dpi = 300)
    cat("Saved plot:", filename, "\n")
  }, error = function(e) {
    cat("Error saving plot:", e$message, "\n")
  })
}

# Print test performance summary
cat("\n\n================================================\n")
cat("TEST SET PERFORMANCE SUMMARY\n")
cat("================================================\n")
print(test_performance)

# Save test performance
write.csv(test_performance, 
         "/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/test_performance.csv",
         row.names = FALSE)

# Create a performance summary plot if we have valid results
if (nrow(test_performance) > 0 && any(!is.na(test_performance$R_squared))) {
  p_performance <- ggplot(test_performance[!is.na(test_performance$R_squared), ], 
                          aes(x = reorder(Y_feature, R_squared), y = R_squared)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
    geom_text(aes(label = round(R_squared, 3)), 
              hjust = -0.1, size = 3) +
    coord_flip() +
    labs(title = "Model Performance on Test Set",
         subtitle = "R² values for best models",
         x = "Feature",
         y = "R-squared") +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5)) +
    ylim(0, 1)
  
  ggsave("/home/INT/dienye.h/gamlss_normative_paper-main/test/single_site/model_performance_summary.png",
         p_performance, width = 10, height = 8, dpi = 300)
}

cat("\n================================================\n")
cat("Analysis complete! Check the outputs:\n")
cat("1. Best models summary: best_models_summary.csv\n")
cat("2. Test performance: test_performance.csv\n")
cat("3. Normative curves: /normative_curves/ folder\n")
cat("4. Performance summary plot: model_performance_summary.png\n")
cat("================================================\n")