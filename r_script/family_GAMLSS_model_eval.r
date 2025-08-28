library(splines)
library(MASS)
library(nlme)
library(parallel)
library(gamlss.data)
library(gamlss.dist)
library(gamlss)
library(ggplot2)

df1 <- read.csv("/home/INT/dienye.h/python_files/combined_dataset/marsfet_qc_filtered.csv", header=TRUE, stringsAsFactors = FALSE) 
x <- df1$gestational_age
y <- df1$band_power_B6

data <- na.omit(df1)

#All models:
#family_list <- c('BE', 'BB', 'BNB', 'BEOI', 'BEZI', 'BEINF', 'BI', 'BCCG', 'BCPE', 'BCT', 'DEL', 'DBURR12', 'DPO', 'DBI', 'EXP', 'exGAUS', 'EGB2', 'GA', 'GB1', 'GB2', 'GG', 'GIG', 'GT', 'GEOM', 'GEOMo', 'GU', 'IGAMMA', 'IG', 'JSU', 'LG', 'LO', 'LOGITNO', 'LOGNO', 'LNO', 'NBI', 'NBII', 'NBF', 'NET', 'NO', 'NOF', 'LQNO', 'PARETO2', 'PARETO2o', 'PE', 'PE2', 'PO', 'PIG', 'RGE', 'RG', 'SEP1', 'SEP2', 'SEP3', 'SEP4', 'SHASH', 'SHASHo', 'SHASH', 'SI', 'SICHEL', 'SIMPLEX', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'TF', 'WARING', 'WEI', 'WEI2', 'WEI3', 'YULE', 'ZABI', 'ZABNB', 'ZAIG', 'ZALG', 'ZANBI', 'ZAP', 'ZASICHEL', 'ZAZIPF', 'ZIBI', 'ZIBNB', 'ZINBI', 'ZIP', 'ZIP2', 'ZIPIG', 'ZISICHEL', 'ZIPF')
#All models with at least 3 parameters:
#family_list <- c('BNB','BEOI', 'BEZI', 'BCCG','DEL','DBURR12','exGAUS','GIG','LNO', 'NBF','NOF','RGE','SI','SICHEL','ZANBI','ZINBI','ZIPIG','BEINF','BCPE', 'EGB2', 'GB1','NET','ZABNB','ZASICHEL','ZIBNB','ZISICHEL','BCT', 'GB2', 'GG', 'GT', 'JSU', 'SEP1', 'SEP2', 'SEP3', 'SEP4', 'SHASH', 'SHASHo', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'TF', 'PE', 'PE2')

#Models tested in brainchart paper (except removed 'DBURR12' and 'exGAUS' and a few others that didnt work). All models with min 3 paramters that worked:
family_list <- c('BCCG','ZIBNB','LNO','NOF','BCPE','NET','ZIBNB','BNB','BCT', 'GB2', 'GG', 'GT', 'JSU', 'SEP1', 'SEP2', 'SEP3', 'SEP4', 'SHASH', 'SHASHo', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'TF', 'PE', 'PE2')

results <- data.frame(Family = character(0), BIC = numeric(0), AIC = numeric(0))


for (family_type in family_list) {
  m5 <- gamlss(y ~ pb(x), sigma.formula = ~pb(x), nu.formula = ~1, tau.formula = ~1, family = family_type, data = data)
  
  bic <- BIC(m5)
  aic <- AIC(m5)
  
  results <- rbind(results, data.frame(Family = family_type, BIC = bic, AIC = aic))
}
print(results)

write.csv(results, file = "band_power_B6_model_family_comparison_results.csv", row.names = FALSE)


# ##Loop thru y:

# df1 <- read.csv("/~/latest_table_to_use_Oct2023_updated_final_subj_list.csv", header=TRUE, stringsAsFactors = FALSE) 
# df1$csf_vent = df1$volume_CSF + df1$volume_VENTRICLES

# x <- df1$scan_age

# y_list <- list(
#      volume_CSF = df1$volume_CSF,
#      volume_cGM = df1$volume_cGM,
#      volume_WM = df1$volume_WM,
#      volume_VENTRICLES = df1$volume_VENTRICLES, 
#      volume_CEREBELLUM = df1$volume_CEREBELLUM, 
#      volume_dGM = df1$volume_dGM, 
#      volume_BRAINSTEM = df1$volume_BRAINSTEM, 
#      volume_HIPPOCAMPI = df1$volume_HIPPOCAMPI,
#      csf_vent  = df1$csf_vent ,
#      white_surface_area = df1$white_surface_area, 
# #     white_gyrification_index = df1$white_gyrification_index
# )

# data <- na.omit(df1)

# family_list <- c('BCCG','ZIBNB','LNO','NOF','BCPE','NET','ZIBNB','BNB','BCT', 'GB2', 'GG', 'GT', 'JSU', 'SEP1', 'SEP2', 'SEP3', 'SEP4', 'SHASH', 'SHASHo', 'ST1', 'ST2', 'ST3', 'ST4', 'ST5', 'TF', 'PE', 'PE2')
# results <- data.frame(Y_Variable = character(0), Family = character(0), BIC = numeric(0), AIC = numeric(0))

# for (y_var_name in names(y_list)) {
#     y <- y_list[[y_var_name]]
    
#     for (family_type in family_list) {
#         m5 <- gamlss(y ~ pb(x), sigma.formula = ~pb(x), nu.formula = ~1, tau.formula = ~1, family = family_type, data = data)
        
#         bic <- BIC(m5)
#         aic <- AIC(m5)
        
#         results <- rbind(results, data.frame(Y_Variable = y_var_name, Family = family_type, BIC = bic, AIC = aic))
#     }
# }

# print(results)

# write.csv(results, file = "/~/gamlss_eval/distribution_families_results_BIC_AIC.csv", row.names = FALSE)



