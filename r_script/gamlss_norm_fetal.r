library(MASS)
library(gamlss.dist)
library(nlme)
library(mgcv)
library(reshape2)


df1 <- read.csv("/home/INT/dienye.h/python_files/dhcp_dataset_info/combined_results.csv", header=TRUE, stringsAsFactors = FALSE)

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


y_values <- list("Surface Area cm2", "Folding Power", "B4 Vertex Percentage", "B5 Vertex Percentage","B6 Vertex Percentage", "Band_parcels B4", "Band Parcels B5", "Band Parcels B6", "Hemispheric Volume", "Gyrification Index", "Hull Area", "B4 Surface Area", "B5 Surface Area", "B6 Surface Area", "B4 Surface Area Percentage", "B5 Surface Area Percentage", "B6 Surface Area Percentage", "B4 Band Power", "B5 Band Power", "B6 Band Power")

results <- data.frame(Model = character(),
                      Y_feature = character(),
                      BIC = double(),
                      AIC = double(), stringsAsFactors = FALSE)

for (i in y_values) {
    #colnames(df1)[12] <- "volume"
    
    
    x <- df1$gestational_age
    y <- df1[[i]]
    
    max_vol <- max(y)
    min_vol <- min(y)
    
    max_age <- max(x)
    min_age <- min(x)
    
    #Starting GAMLSS:
    
    m1 <- gam(list(y ~ x,  # basically the same model as lm(y~x, data=df1)
                   ~ 1 ), 
              family=gaulss(),
              data=df1)
    m2 <- gam(list(y ~ s(x) , # fit mu as a smooth function of x
                   ~ 1 ), # fit sigma only as an intercept
              family=gaulss(),
              data=df1)
    m3 <- gam(list(y ~ s(x) , # fit mu as a smooth function of x
                   ~ s(x) ), # fit sigma as a smooth function of x
              family=gaulss(),
              data=df1)
    m4 <- gam(list(y ~ s(x), # fit mu as a smooth function of x
                   ~ s(x) , # fit sigma as a smooth function of x
                   ~ 1 , # fit nu (skewness) as an intercept
                   ~ 1 ), # fit tau (kurtosis) as an intercept
              family=shash(), # shash distribution instead of gaussian 
              data=df1)
    
    
    #In order to statistically compare groups (ex. add sex or scanner to m4 model):
    cat("Results for y =", i, ":\n")
    print(summary(m4))
    
    
    
    
    #Begin plotting parameters
    predictions_params_m1 <- predict(m1)
    predictions_params_m2 <- predict(m2)
    predictions_params_m3 <- predict(m3)
    predictions_params_m4 <- predict(m4)
    
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
    quantiles <- pnorm(c(-2:2)) # definition of quantile lines to plot (can cahnge to -1:1 for ex)
    qshash <- m4$family$qf
    predictions_quantiles_m1 <- params_to_quantiles_norm(quantiles, 
                                                         predictions_params_m1)
    predictions_quantiles_m2 <- params_to_quantiles_norm(quantiles,
                                                         predictions_params_m2)
    predictions_quantiles_m3 <- params_to_quantiles_norm(quantiles, 
                                                         predictions_params_m3)
    predictions_quantiles_m4 <- params_to_quantiles_shash(quantiles, 
                                                          predictions_params_m4,
                                                          qshash)
    reshape_quantiles_to_long <- function(quantiles_df, x_var){
        quantiles_df$x <- x_var
        return(reshape2::melt(quantiles_df, id.vars = c('x')))
    }
    
    predictions_quantiles_m1_long <- reshape_quantiles_to_long(predictions_quantiles_m1, df1$gestational_age)
    predictions_quantiles_m2_long <- reshape_quantiles_to_long(predictions_quantiles_m2, df1$gestational_age)
    predictions_quantiles_m3_long <- reshape_quantiles_to_long(predictions_quantiles_m3, df1$gestational_age)
    predictions_quantiles_m4_long <- reshape_quantiles_to_long(predictions_quantiles_m4, df1$gestational_age)
    
    
    
library(ggplot2)
    library(cowplot)
    library(patchwork)
    
    # Create plotting data frame with proper variable handling
    plot_data <- data.frame(
        x = x,
        y = y,
        cohort = if("cohort" %in% colnames(df1)) df1$cohort else "All Participants"
    )
    
    quantile_linetypes <- c("dashed", "dashed", "solid", "dashed", "dashed")
    p1 <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m1_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('Linear model:', i),
             subtitle='Mean of y is modeled as a linear function of x',
             x = 'Gestational Age in Weeks', 
             y='Volume')  +
        annotate("text", x = 35, y = 30, size=3,
                 label = "") +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))
    
    p2 <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m2_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('GAM - homoskedastic:', i),
             # subtitle='Mean of y is modeled as a smooth function of x',
             x = 'Gestational Age in Weeks', 
             y='Volume')  +
        annotate("text", x = 35, y = 30, size=3,
                 label = "") +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol))+ xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))
    
    p3 <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m3_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title= paste('GAM - heteroskedastic:', i),
             # subtitle='Mean and variance of y are modeled\nas smooth functions of x',
             x = 'Gestational Age in Weeks', 
             y='Volume')  +
        annotate("text", x = 35, y = 30, size=3,
                 label = "") +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Participant Class") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))
    
    p4 <- ggplot(plot_data) +
        geom_point(aes(x=x, y=y, color = cohort), size=2) +
        geom_line(data=predictions_quantiles_m4_long, 
                  aes(x=x, y=value, group=variable, linetype = as.factor(variable)),linewidth = 0.35) +
        scale_linetype_manual(values = quantile_linetypes) +
        guides(linetype = "none") +
        geom_rug() +
        labs(title=paste('GAMLSS Model:', i),
             # subtitle='Location, scale, and shape\nare modeled as functions of x',
             x = 'Gestational Age in Weeks', 
             y='Volume')  + theme(plot.title = element_text(size = 15)) +
        annotate("text", x = 35, y = 30, size=3,
                 label = "") +
        ylim(c(min_vol - 0.1*min_vol , max_vol + 0.1*max_vol)) + xlim(c(min_age-2,max_age+2))+
        labs(color = "Cohort") +
        scale_x_continuous(breaks = seq(21, 45, by = 1)) + 
        theme(plot.title = element_text(hjust = 0.5))
    
    p_all <- p1 + p2 + p3 + p4 + 
        plot_annotation(tag_levels = 'A') & 
        theme_cowplot() &
        theme(text=element_text(size=9),
              axis.text.x = element_text(size=5),axis.text.y = element_text(size=8)); p_all
    #Computed BIC and AIC for every model during model selection
    print(paste("BIC (m1) for", i, ":", BIC(m1)))
    print(paste("BIC (m2) for", i, ":", BIC(m2)))
    print(paste("BIC (m3) for", i, ":", BIC(m3)))
    print(paste("BIC (m4) for", i, ":", BIC(m4)))
    
    print(paste("AIC (m1) for", i, ":", AIC(m1)))
    print(paste("AIC (m2) for", i, ":", AIC(m2)))
    print(paste("AIC (m3) for", i, ":", AIC(m3)))
    print(paste("AIC (m4) for", i, ":", AIC(m4)))
    
    bic_aic_data <- data.frame(Model = c("m1", "m2", "m3", "m4"),
                               Y_feature = i,
                               BIC = c(BIC(m1), BIC(m2), BIC(m3), BIC(m4)),
                               AIC = c(AIC(m1), AIC(m2), AIC(m3), AIC(m4)))
    results <- rbind(results, bic_aic_data)
    
    
    #filename <- paste0("~/4_models_", i, "_month.png")
    #ggsave(filename, p_all, width = 10, height = 5, units = 'in', bg="white")
    filename2 <- paste0("/home/INT/dienye.h/python_files/GAMLSS/", i, "_GAMLSS_Nov.png")
    ggsave(filename2, p4, width = 10, height = 5, units = 'in', bg="white")
    
}