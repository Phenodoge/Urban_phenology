# 1 Fig.3d urban-rural variation in SOS #############
rm(list=ls())
library(ggplot2)
library(ggpubr)
library(Rmisc)
library(readxl)
library(Rmisc)
library(dplyr)

data_treat <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Manipulative experiment.xlsx", sheet = "Sheet1")
unique(data_treat$SP)

data_treat$TEM1<-factor(data_treat$TEM1,levels=c("T05", "T10", "T15", "T20"))
data_treat$PHO1<-factor(data_treat$PHO1,levels=c("P08", "P16"))
data_treat$CHI1<-factor(data_treat$CHI1,levels=c("C1", "C2"))
data_treat$SITE1<-factor(data_treat$SITE1,levels=c("Rural", "Urban"))
data_treat$SP<-factor(data_treat$SP,levels=c("Ginkgo", "Acer", "Fraxinus", "Platanus"))

### 1.1 separate species  #############
tgc <- summarySE(data_treat, measurevar="LEAF_S2", groupvars=c("SITE1", "SP")); tgc
mycolors<-c("#BF9000", "#789440", "#31859B", "#7E649E")
g1 <- ggplot(tgc, aes(x = SITE1, y = LEAF_S2, color = SP)) + 
  geom_errorbar(aes(ymin = LEAF_S2, ymax = LEAF_S2),
                width = 0.2, position = position_dodge(0), size = 0.75) +
  geom_line(position = position_dodge(0), size = 0.75) +
  geom_point(shape = 16, position = position_dodge(0), size = 4) +
  coord_cartesian(ylim = c(40, 80)) +
  ggtitle("(a)") +
  labs(x = "", y = "Time to budburst (d)") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(size = 18, family = "sans"),
    axis.title.x = element_text(size = 18, family = "sans"),
    axis.title.y = element_text(size = 18, family = "sans"),
    legend.text = element_text(size = 10, family = "sans", face = "italic"),
    legend.title = element_blank(),
    plot.title = element_text(family = "sans", size = 18, face = "bold"),
    legend.position = c(0.8, 0.18),
    axis.text.x = element_text(margin = margin(t = 8)),
    axis.text.y = element_text(margin = margin(r = 8))
  ) +
  scale_color_manual(values = mycolors); g1

### 1.2 variation in SOS #############
mycolors<-c("#5D8DCB", "#E6603D")
tgc1 <- summarySE(tgc, measurevar="LEAF_S2", groupvars=c("SITE1"))
g2 <- ggplot(tgc1, aes(x = SITE1, y = LEAF_S2, color = SITE1)) + 
  geom_errorbar(aes(ymin = LEAF_S2-sd, ymax = LEAF_S2+sd),
                width = 0.2, position = position_dodge(0), size = 0.75) +
  geom_line(position = position_dodge(0), size = 0.75) +
  geom_point(shape = 15, position = position_dodge(0), size = 6) +
  coord_cartesian(ylim = c(40, 80)) +
  ggtitle("(b)") +
  labs(x = "", y = "Time to budburst (d)") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(size = 18, family = "sans"),
    axis.title.x = element_text(size = 18, family = "sans"),
    axis.title.y = element_text(size = 18, family = "sans"),
    legend.text = element_text(size = 18, family = "sans"),
    legend.title = element_blank(),
    legend.position = "none",
    plot.title = element_text(family = "sans", size = 18, face = "bold"),
    axis.text.x = element_text(margin = margin(t = 8)),
    axis.text.y = element_text(margin = margin(r = 8))
  ) +
  scale_color_manual(values = mycolors);g2


# 2 Fig.3e urban-rural variation in ST ##########################
rm(list=ls())
library(ggplot2)
library(ggpubr)
library(Rmisc)
library(readxl)
library(Rmisc)
library(dplyr)

### 2.1 difference in temperature sensitivity #############
data_treat <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Manipulative experiment.xlsx", sheet = "Sheet2")


data_treat_all <- data_treat %>%
  mutate(SP = "All")
data_treat <- bind_rows(data_treat, data_treat_all)

data_treat$PHO1<-factor(data_treat$PHO1,levels=c("P08", "P16"))
data_treat$CHI1<-factor(data_treat$CHI1,levels=c("C1", "C2"))
data_treat$SITE1<-factor(data_treat$SITE1,levels=c("Rural", "Urban"))
data_treat$SP<-factor(data_treat$SP,levels=c("All", "Ginkgo", "Acer", "Fraxinus", "Platanus"))

tgc <- summarySE(data_treat, measurevar="ST_C", groupvars=c("SITE1", "SP"))
mycolors<-c("#6991C7", "#E97456")

Fig.3e_1 <- ggplot(tgc, aes(x = SP, y = ST_C, fill = SITE1)) +
  geom_bar(stat = "identity",  position = position_dodge(), width = 0.6) +
  geom_errorbar(aes(ymin = ST_C - se, ymax = ST_C + se),
                width = 0.17, position = position_dodge(width =0.60)) +
  coord_cartesian(ylim = c(-0.38, -8)) +
  scale_y_continuous(breaks = seq(0, -8, -2)) + 
  labs(x="", y="Temperature sensitivity (d/°C)", fill = "SITE") +
  theme_bw()+
  theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=18,family="sans"), 
        axis.title.x=element_text(size=18,family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        legend.text =element_text(size=14,family="sans"),
        legend.title =element_blank(),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = c(0.15, 0.91),
        legend.key.size = unit(0.5, "cm"), 
        axis.text.x = element_text(margin = margin(t = 8), face = "italic"),  
        axis.text.y = element_text(margin = margin(r = 8)))+ 
  scale_fill_manual(values = mycolors)+
  ggtitle("(a)")

### 2.2 interactive effect #################
data_treat <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Manipulative experiment.xlsx", sheet = "Sheet2")

M1=aov(ST_C ~SITE1*SP+CHI1+PHO1, data = data_treat)
M2=aov(ST_C ~SITE1+SP+CHI1+PHO1, data = data_treat)

summary(M1)
summary(M2)

AIC(M1)
AIC(M2)

anova(M2,M1)



# 3 Fig.S16 effect of chilling and photoperiod on urban-rural variation in SOS #############
library(ggplot2)
library(ggpubr)
library(Rmisc)
library(readxl)
library(Rmisc)
library(dplyr)

### 3.1 chilling  #############
rm(list=ls())
tgc <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Manipulative experiment.xlsx", sheet = "Sheet3")

tgc$CHI1<-factor(tgc$CHI1,levels=c("C1", "C2"))
tgc$SITE1<-factor(tgc$SITE1,levels=c("Rural", "Urban"))
tgc$SP<-factor(tgc$SP,levels=c("Ginkgo", "Acer", "Fraxinus", "Platanus"))

mycolors<-c("#6991C7", "#E97456")
g1 <- ggplot(tgc, aes(x = SP, y = LEAF_S2, color = SITE1, shape = CHI1)) + 
  geom_errorbar(aes(ymin = LEAF_S2 - se, ymax = LEAF_S2 + se),
                width = 0.25, position = position_dodge(0.85), size = 0.75) +
  geom_line(position = position_dodge(0.85), size = 0.75) +
  geom_point(position = position_dodge(0.85), size = 4) +
  coord_cartesian(ylim = c(35, 90)) +
  scale_y_continuous(breaks = seq(40,90, 10))+
  ggtitle("(a)") +
  labs(x = "", y = "Time to budburst (d)") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(size = 18, family = "sans"),
    axis.title.x = element_text(size = 18, family = "sans"),
    axis.title.y = element_text(size = 18, family = "sans"),
    legend.text = element_text(size = 14, family = "sans"),
    legend.title = element_blank(),
    plot.title = element_text(family = "sans", size = 18, face = "bold"),
    legend.position = c(0.17, 0.8),
    axis.text.x = element_text(margin = margin(t = 8),face = "italic"),
    axis.text.y = element_text(margin = margin(r = 8))
  ) +
  scale_color_manual(values = mycolors) + # , labels = c("Rural", "Urban")
  scale_shape_manual(values = c(15, 0)) +
  guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1)) ;g1

### 3.2 photoperiod  #############
rm(list=ls())
tgc <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Manipulative experiment.xlsx", sheet = "Sheet4")

tgc$PHO1<-factor(tgc$PHO1,levels=c("P08", "P16"))
tgc$SITE1<-factor(tgc$SITE1,levels=c("Rural", "Urban"))
tgc$SP<-factor(tgc$SP,levels=c("Ginkgo", "Acer", "Fraxinus", "Platanus"))

mycolors<-c("#6991C7", "#E97456")
g2 <- ggplot(tgc, aes(x = SP, y = LEAF_S2, color = SITE1, shape = PHO1)) + 
  geom_errorbar(aes(ymin = LEAF_S2 - se, ymax = LEAF_S2 + se),
                width = 0.25, position = position_dodge(0.85), size = 0.75) +
  geom_line(position = position_dodge(0.85), size = 0.75) +
  geom_point(position = position_dodge(0.85), size = 4) +
  coord_cartesian(ylim = c(35, 90)) +
  scale_y_continuous(breaks = seq(40,90, 10))+
  ggtitle("(b)") +
  labs(x = "", y = "Time to budburst (d)") +
  theme_bw() +
  theme(
    panel.grid = element_blank(),
    axis.text = element_text(size = 18, family = "sans"),
    axis.title.x = element_text(size = 18, family = "sans"),
    axis.title.y = element_text(size = 18, family = "sans"),
    legend.text = element_text(size = 14, family = "sans"),
    legend.title = element_blank(),
    plot.title = element_text(family = "sans", size = 18, face = "bold"),
    legend.position = c(0.17, 0.8),
    axis.text.x = element_text(margin = margin(t = 8),face = "italic"),
    axis.text.y = element_text(margin = margin(r = 8))
  ) +
  scale_color_manual(values = mycolors, labels = c("Rural", "Urban")) +
  scale_shape_manual(values = c(16, 1)) +
  guides(color = guide_legend(nrow = 1), shape = guide_legend(nrow = 1));g2 






#4 Fig.S17 effect of chilling and photoperiod on urban-rural variation in SOS #############
rm(list=ls())
library(ggplot2)
library(ggpubr)
library(Rmisc)
library(readxl)
library(Rmisc)
library(dplyr)

data_treat <- read_excel("D:/0 Work/Work/3 BNU/Lab/2 Manuscripts/MS6 城市热岛效应/3 Manuscript/20240730 NC/4 Major revision/2 MS/20241127 MS_Submit/data/Manipulative experiment.xlsx", sheet = "Sheet5")
data_treat$TEM1<-factor(data_treat$TEM1,levels=c("T05", "T10", "T15", "T20"))
data_treat$PHO1<-factor(data_treat$PHO1,levels=c("P08", "P16"))
data_treat$CHI1<-factor(data_treat$CHI1,levels=c("C1", "C2"))
data_treat$SITE1<-factor(data_treat$SITE1,levels=c("Rural", "Urban"))
data_treat$SP<-factor(data_treat$SP,levels=c("Ginkgo", "Acer", "Fraxinus", "Platanus"))

### 4.1 urban-rural  #############
tgc <- summarySE(data_treat, measurevar="GDD", groupvars=c("SITE1", "SP"))
mycolors<-c("#6991C7", "#E97456")
g1 <- ggplot(tgc, aes(x = SP, y = GDD, fill = SITE1)) +
  geom_bar(stat = "identity",  position = position_dodge(), width = 0.4) +
  geom_errorbar(aes(ymin = GDD - se, ymax = GDD + se),
                width = 0.17, position = position_dodge(width =0.40)) +
  coord_cartesian(ylim = c(200, 800)) +
  scale_y_continuous(breaks = seq(200, 800, 200)) + 
  labs(x="", y="Growing degree days (°C)", fill = "SITE") +
  theme_bw()+
  theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=18,family="sans"), ## 字体字号
        axis.title.x=element_text(size=18,family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        legend.text =element_text(size=18,family="sans"),
        legend.title =element_blank(),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = c(0.15, 0.92),
        axis.text.x = element_text(margin = margin(t = 8),face = "italic"),  
        axis.text.y = element_text(margin = margin(r = 8)))+ 
  scale_fill_manual(values = mycolors, labels = c("Rural", "Urban"))+
  ggtitle("(a)")

### 4.2 chilling  #############
tgc <- summarySE(data_treat, measurevar="GDD", groupvars=c("CHI1", "SP"))
mycolors<-c("#92CDDC", "#31859B")
g2 <- ggplot(tgc, aes(x = SP, y = GDD, fill = CHI1)) +
  geom_bar(stat = "identity",  position = position_dodge(), width = 0.4) +
  geom_errorbar(aes(ymin = GDD - se, ymax = GDD + se),
                width = 0.17, position = position_dodge(width =0.40)) +
  coord_cartesian(ylim = c(200, 800)) +
  scale_y_continuous(breaks = seq(200, 800, 200)) + 
  labs(x="", y="Growing degree days (°C)", fill = "CHI1") +
  theme_bw()+
  theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=18,family="sans"), ## 字体字号
        axis.title.x=element_text(size=18,family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        legend.text =element_text(size=18,family="sans"),
        legend.title =element_blank(),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = c(0.15, 0.92),
        axis.text.x = element_text(margin = margin(t = 8),face = "italic"),  
        axis.text.y = element_text(margin = margin(r = 8)))+ 
  scale_fill_manual(values = mycolors, labels = c("Low chilling", "High chilling"))+
  ggtitle("(b)")

### 4.3 photoperiod  #############
tgc <- summarySE(data_treat, measurevar="GDD", groupvars=c("PHO1", "SP"))
mycolors<-c("#F5E3E2", "#DB9A95")
g3 <- ggplot(tgc, aes(x = SP, y = GDD, fill = PHO1)) +
  geom_bar(stat = "identity",  position = position_dodge(), width = 0.4) +
  geom_errorbar(aes(ymin = GDD - se, ymax = GDD + se),
                width = 0.17, position = position_dodge(width =0.40)) +
  coord_cartesian(ylim = c(200, 800)) +
  scale_y_continuous(breaks = seq(200, 800, 200)) + 
  labs(x="", y="Growing degree days (°C)", fill = "PHO1") +
  theme_bw()+
  theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=18,family="sans"), ## 字体字号
        axis.title.x=element_text(size=18,family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        legend.text =element_text(size=18,family="sans"),
        legend.title =element_blank(),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = c(0.15, 0.92),
        axis.text.x = element_text(margin = margin(t = 8),face = "italic"),  
        axis.text.y = element_text(margin = margin(r = 8)))+ 
  scale_fill_manual(values = mycolors, labels = c("8-hour", "16-hour"))+
  ggtitle("(c)")

### 4.4 temperature  #############
tgc <- summarySE(data_treat, measurevar="GDD", groupvars=c("TEM1", "SP"))
mycolors<-c("#FBE5D6", "#F8CBAD", "#F4B183", "#C55A11")
g4 <- ggplot(tgc, aes(x = SP, y = GDD, fill = TEM1)) +
  geom_bar(stat = "identity",  position = position_dodge(), width = 0.6) +
  geom_errorbar(aes(ymin = GDD - se, ymax = GDD + se),
                width = 0.2, position = position_dodge(width =0.6)) +
  coord_cartesian(ylim = c(200, 800)) +
  scale_y_continuous(breaks = seq(200, 800, 200)) + 
  labs(x="", y="Growing degree days (°C)", fill = "TEM1") +
  theme_bw()+
  theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=18,family="sans"), ## 字体字号
        axis.title.x=element_text(size=18,family="sans"),
        axis.title.y=element_text(size=18,family="sans"),
        legend.text =element_text(size=18,family="sans"),
        legend.title =element_blank(),
        plot.title = element_text(family = "sans", size = 18, face="bold"),
        legend.position = c(0.15, 0.85),
        axis.text.x = element_text(margin = margin(t = 8),face = "italic"),  
        axis.text.y = element_text(margin = margin(r = 8)))+ 
  scale_fill_manual(values = mycolors, labels = c("T5", "T10", "T15", "T20"))+
  ggtitle("(d)")

