#' author: "Kangjoo Lee, Grega Repovš"
#' output:
#'   html_document:
#'     toc: true
#'     theme: default
#'     highlight: haddock
#'   pdf_document:
#'     toc: true
#' ---
#' <style type='text/css'>
#' .table {
#'   width: auto;
#'   font-size: 9pt;
#'   margin-bottom: 20px
#' }
#' </style>

rm(list = ls())


library(fmsb)
library(ggplot2)
library(ggpubr)
library(scales)
library(stringr)
library(asbio)
library(tidyverse)
library(car)
library(broom)

# -- Set ggplot basic theme
theme1<- theme(panel.border = element_rect(size = 2, colour = 'NA', fill='NA'), 
               axis.line = element_line(colour = "black", size = 2),
               panel.grid.major = element_blank(), 
               panel.grid.minor = element_blank(), 
               plot.background = element_rect(fill = "transparent", colour = NA),
               panel.background = element_rect(fill = "white", colour = NA),
               legend.position="none",
               axis.ticks = element_line(colour = "black", size=1),
               axis.ticks.length = unit(.4, "cm"),
               axis.text.x = element_text(size = 30, color="black"), 
               axis.title.x = element_text(size =30,margin=margin(10,0,0,0)),
               axis.text.y = element_text(size = 30, color="black",angle=90, hjust=0.5), 
               axis.title.y = element_text(size =35,margin=margin(0,-20,0,0)))

theme2<- theme(panel.border = element_rect(size = 2, colour = 'NA', fill='NA'), 
               axis.line = element_line(colour = "black", size = 2),
               axis.text.x = element_text(size = 40, color="black",margin=margin(10,20,0,0)),
               axis.title.x = element_blank(),
               axis.text.y = element_blank(),
               axis.title.y = element_text(size = 40,margin=margin(0,5,0,0)),
               panel.grid.major = element_blank(), 
               panel.grid.minor = element_blank(),
               plot.background = element_rect(fill = "transparent", colour = NA),
               panel.background = element_rect(fill = "transparent", colour = NA),
               legend.position="none",
               axis.ticks = element_line(colour = "black", size=2),
               axis.ticks.length = unit(.4, "cm"),
               axis.ticks.y = element_blank())   





#' ## Definitions


#-----------------------------------------------
# Get data
#-----------------------------------------------
# read in dataframe - BehaviorPCALoadings.tsv
df_bPC <- read.table(file = '/results/Connectome_All_All_flip_dropAS_30nf_n337/analysis/results/NBRIDGE_Connectome_All_All_flip_dropAS_30nf_n337_BehaviorPCAScores.tsv', sep = '\t', header = TRUE)

# read in dataframe - Neural PCs
df_nPC <- read.table(file = '/results/gsr_seedfree/P100.0_FO_mDT_vDT_neuralPCs_30nf.csv',sep=",",header= TRUE)

# Find the behavioral raw data to get age and sex info
df_behav <- read.table(file = 'CAP_behavior_prep_RESTRICTED_hcp_n337_All_All_flip_30nf.tsv', sep = '\t', header = TRUE)
df_behav_list <- read.table(file='/results/Connectome_All_All_flip_30nf_n337/analysis/figures/NBRIDGE_Connectome_All_All_flip_30nf_n337_prep_CD_BehaviorPCALoadings_Legend.tsv')
behav_col_names <- df_behav_list[, 3]
df_behav_selected <- df_behav[, behav_col_names]

# final dataframe
d <- data.frame(paste0("s",seq(1:337)))
colnames(d)[1]="subject"
d$bPC1 <- df_bPC$PC1
d$nPC1 <- df_nPC$neuralPC1
d$nPC2 <- df_nPC$neuralPC2
d$nPC3 <- df_nPC$neuralPC3
d$age <- df_behav_selected$Age_in_Yrs
d$sex <- df_behav_selected$Gender
d$group <- df_bPC$Group

# save data
library(readr)
write_csv(d, "/softwares/others/bPC_nPC.csv")



#-----------------------------------------------
#' Compute mixed linear model -- regular
#-----------------------------------------------

# Full model 
lm_noi <- lm(bPC1 ~ nPC1 + nPC2 + nPC3 + age + sex, data=d)

# Reduced models 
lm_r1  <- lm(bPC1 ~        nPC2 + nPC3 + age + sex, data=d)
lm_r2  <- lm(bPC1 ~ nPC1        + nPC3 + age + sex, data=d)
lm_r3  <- lm(bPC1 ~ nPC1 + nPC2        + age + sex, data=d)
lm_r4  <- lm(bPC1 ~ nPC1 + nPC2 + nPC3       + sex, data=d)
lm_r5  <- lm(bPC1 ~ nPC1 + nPC2 + nPC3 + age      , data=d)

# Results
summary(lm_noi)
partial.R2(lm_r1,lm_noi)
partial.R2(lm_r5,lm_noi)





pdf("/results/Connectome_All_All_flip_dropAS_30nf_n337/analysis/results/Figures/nPC1_bPC1_mlr.pdf", width=7,height=7)
slope <- lm(bPC1 ~ nPC1, data = d)$coefficients[2]
d$group <- as.factor(d$group)
color_palette <- c("#FDE725FF", "#238A8DFF", "#481567FF")
d %>%
  ggplot(aes(x = nPC1, y = bPC1)) +
  geom_point(size = 4, alpha = 0.9, aes(fill = group, color = group), shape = 21, stroke = 1) +
  geom_smooth(method = 'lm', color = "black", size = 2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  theme1 +
  xlim(-6, 12) +
  ylim(-25, 15) +
  scale_fill_manual(values = color_palette) +
  scale_color_manual(values = rep("black", length(color_palette)))
dev.off()




pdf("/results/Connectome_All_All_flip_dropAS_30nf_n337/analysis/results/Figures/nPC2_bPC1_mlr.pdf", width=7,height=7)
slope <- lm(bPC1 ~ nPC2, data = d)$coefficients[2]
d$group <- as.factor(d$group)
color_palette <- c("#FDE725FF", "#238A8DFF", "#481567FF")
d %>%
  ggplot(aes(x = nPC2, y = bPC1)) +
  geom_point(size = 4, alpha = 0.9, aes(fill = group, color = group), shape = 21, stroke = 1) +
  geom_smooth(method = 'lm', color = "black", size = 2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  theme1 +
  xlim(-6, 12) +
  ylim(-25, 15) +
  scale_fill_manual(values = color_palette) +
  scale_color_manual(values = rep("black", length(color_palette)))
dev.off()





pdf("/results/Connectome_All_All_flip_dropAS_30nf_n337/analysis/results/Figures/nPC3_bPC1_mlr.pdf", width=7,height=7)
slope <- lm(bPC1 ~ nPC3, data = d)$coefficients[2]
d$group <- as.factor(d$group)
color_palette <- c("#FDE725FF", "#238A8DFF", "#481567FF")
d %>%
  ggplot(aes(x = nPC3, y = bPC1)) +
  geom_point(size = 4, alpha = 0.9, aes(fill = group, color = group), shape = 21, stroke = 1) +
  geom_smooth(method = 'lm', color = "black", size = 2) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  theme1 +
  xlim(-6, 12) +
  ylim(-25, 15) +
  scale_fill_manual(values = color_palette) +
  scale_color_manual(values = rep("black", length(color_palette)))
dev.off()








