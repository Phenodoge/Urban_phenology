### Fig.3f urban-rural variation in SOS ####################################
rm(list=ls())
library(ggplot2)
library(ggpubr)
library(Rmisc)
library(readxl)
library(Rmisc)
library(tidyr)
library(dplyr)
library(ggdist)
library(data.table)
library(raincloudplots)


data_treat <- fread("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Satellite_species_level.csv") 

data_treat$site<-factor(data_treat$site,levels=c("Rural", "Urban"))
data_treat$SP<-factor(data_treat$SP,levels=c("Ginkgo", "Acer", "Fraxinus", "Platanus"))

##### 1 separate species boxplot #############
mycolors<-c("#BF9000", "#789440", "#31859B", "#7E649E")
g1<-ggplot(data_treat,aes(x=site, y=mean_sos))+
  geom_boxplot(aes(colour=SP, fill = SP), outlier.shape = NA)+
  coord_cartesian(ylim = c(90,150))+
  scale_y_continuous(breaks = seq(90,150, 10))+
  labs(x=NULL,y="Leaf-out date (d)")+
  theme_bw()+
  ggtitle("(a)")+
  theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=18,family="sans"),
        axis.title.x=element_text(size=18,face = "bold",family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        legend.text =element_text(size=16,family="sans",face = "italic"),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = c(0.78, 0.85),
        legend.title = element_blank()) +
  scale_color_manual(values = mycolors)+ 
  scale_fill_manual(values = alpha(mycolors, 0.2)) 

##### 2 all mean #############
mycolors<-c("#5D8DCB", "#E6603D")
Data_summary <- summarySE(data_treat, measurevar="mean_sos", groupvars=c("site"))
g2 <- ggplot(data_treat,aes(x=site, y=mean_sos,fill=site))+
  coord_cartesian(ylim = c(90,150))+
  scale_y_continuous(breaks = seq(90,150, 10))+
  labs(x=NULL,y="Leaf-out date (d)")+
  geom_half_violin(position=position_nudge(x=0,y=0),
                   side="l",adjust=1.2,trim=F,color=NA)+
  geom_point(aes(x = as.numeric(site)+0.1,
                 y = mean_sos,color = site),position = position_jitter(width =0.04),size =4, shape = 20)+
  geom_errorbar(data = Data_summary,aes(ymin = mean_sos-sd, ymax=mean_sos+sd),
                width=0.1,
                position=position_dodge(0),color="black",size=0.8)+
  geom_point(data = Data_summary,aes(x=site, y=mean_sos),pch=19,position=position_dodge(0.9),size=5,col="black")+
  theme_bw()+
  ggtitle("(b)")+
  theme(panel.grid=element_blank())+#去掉灰底和背景网格线
  theme(axis.text   =element_text(size=18,family="sans"),
        axis.title.x=element_text(size=18,face = "bold",family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = "none",
        legend.text =element_text(size=16,family="sans"))+
  scale_color_manual(values = mycolors) +
  scale_fill_manual(values = mycolors)



### Fig.3e urban-rural variation in ST ####################################
rm(list=ls())
library(ggplot2)
library(ggpubr)
library(Rmisc)
library(readxl)
library(Rmisc)
library(dplyr)
library(data.table)


data_treat <- fread("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Satellite_species_level.csv") 

data_treat_all <- data_treat %>%
  mutate(SP = "All")
data_treat <- bind_rows(data_treat, data_treat_all)

data_treat$site<-factor(data_treat$site,levels=c("Rural", "Urban"))
data_treat$SP<-factor(data_treat$SP,levels=c("All", "Ginkgo", "Acer", "Fraxinus", "Platanus"))
tgc <- summarySE(data_treat, measurevar="ST_C", groupvars=c("site", "SP"))

mycolors<-c("#6991C7", "#E97456")
Fig.3e <- ggplot(tgc, aes(x = SP, y = ST_C, fill = site)) +
  geom_bar(stat = "identity",  position = position_dodge(), width = 0.6) +
  geom_errorbar(aes(ymin = ST_C - se, ymax = ST_C + se),
                width = 0.17, position = position_dodge(width =0.6)) +
  coord_cartesian(ylim = c(-0.38, -8)) +
  scale_y_continuous(breaks = seq(0, -8, -2)) + 
  labs(x="", y="Temperature sensitivity (d/°C)", fill = "SITE") +
  theme_bw()+
  theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=18,family="sans"), ## 字体字号
        axis.title.x=element_text(size=18,family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        legend.text =element_text(size=14,family="sans"),
        legend.title =element_blank(),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = c(0.14, 0.91),
        legend.key.size = unit(0.5, "cm"), 
        axis.text.x = element_text(margin = margin(t = 8),face = "italic"),  
        axis.text.y = element_text(margin = margin(r = 8)))+ 
  scale_fill_manual(values = mycolors)+ #, labels = c("Rural", "Urban")
  ggtitle("(b)")
