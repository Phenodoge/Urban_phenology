### demo_code ----------------------
rm(list=ls())
library(data.table)
library(tidyverse)
library(ggplot2)
library(dplyr)
library(broom)

### read demo data ----------------
TT_GOSIF1 <- fread("D:/0 Work/2 Manuscripts_urban/demo_data.csv")

### Statistics for different beta diversity ----------------
TT_GOSIF2 <- TT_GOSIF1 %>%
  group_by(Bary01) %>%
  dplyr::summarise(
    N   =n(),
    ST  =mean(ST_Diff_ABS),
    ST_SD  =sd(ST_Diff_ABS),
    ST_SE  =sd(ST_Diff_ABS)/sqrt(N),
    SOS  =mean(SOS_Diff_ABS),
    SOS_SD  =sd(SOS_Diff_ABS),
    SOS_SE  =sd(SOS_Diff_ABS)/sqrt(N)) %>%
  mutate(id=1:5)%>%
  ungroup()

g1 <- ggplot(TT_GOSIF2, aes(x = Bary01, y=SOS, fill = factor(Bary01))) +
  geom_bar(position = position_dodge(0.8), stat="identity",width = 0.7)+
  geom_errorbar(aes(ymin=SOS-SOS_SE, ymax=SOS+SOS_SE), width=.2)+
  labs(x="Beta diversity",y="SOS")+
  coord_cartesian(ylim = c(0.65,14))+
  scale_y_continuous(breaks = seq(0, 14, 2))+
  theme_bw()+theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=20,family="sans"),
        axis.title.x=element_text(size=20,family="sans"),
        axis.title.y=element_text(size=20,family="sans"),
        legend.text =element_text(size=20,family="sans"))+
  theme(legend.position="none")

g2 <- ggplot(TT_GOSIF2, aes(x = Bary01, y=ST, fill = factor(Bary01))) +
  geom_bar(position = position_dodge(0.8), stat="identity",width = 0.7)+
  geom_errorbar(aes(ymin=ST-ST_SE, ymax=ST+ST_SE), width=.2)+
  labs(x="Beta diversity",y="ST")+
  coord_cartesian(ylim = c(0.17,3.7))+
  scale_y_continuous(breaks = seq(0, 3.5, 1))+
  theme_bw()+theme(panel.grid=element_blank())+
  theme(axis.text   =element_text(size=20,family="sans"),
        axis.title.x=element_text(size=20,family="sans"),
        axis.title.y=element_text(size=20,family="sans"),
        legend.text =element_text(size=20,family="sans"))+
  theme(legend.position="none") 