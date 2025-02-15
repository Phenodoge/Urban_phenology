# Fig.2b decision tree significant_out----------------------
rm(list=ls())
library(readxl)
library("rpart")
library("partykit")
library(ggpubr)
library(eoffice)
TT_GOSIF1 <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/3 Manuscript/20240730 NC/6 Submit/Final/2 DATA/Global/RF_data/Final_RF_Outward.xlsx", sheet = "Sheet1")


urbanization <- rpart(abs(ST_Diff_ABS) ~ abs(Bary)+abs(Ta_Diff)+abs(Temperature), data = TT_GOSIF1,
                      control = rpart.control(minsplit = 27))
print(urbanization$cptable)
opt <- which.min(urbanization$cptable[,"xerror"])

fig.1=plot(as.party(urbanization), tp_args = list(id = FALSE))

# Fig.S13b decision tree significant_in #############
rm(list=ls())

library(readxl)
library("rpart")
library("partykit")

TT_GOSIF1 <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/6 Submit/Final/2 DATA/Global/RF_data/Final_RF_Inward.xlsx", sheet = "Sheet1")

urbanization <- rpart(abs(ST_Diff_ABS) ~ abs(Bary)+abs(Ta_Diff)+abs(Temperature), data = TT_GOSIF1,
                      control = rpart.control(minsplit = 23))
print(urbanization$cptable)
opt <- which.min(urbanization$cptable[,"xerror"])
fig.1=plot(as.party(urbanization), tp_args = list(id = FALSE))




