
# MIT License
#
# Copyright (c) 2024 Hongzhou Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# ...




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
config={'font.family':'Times New Roman','font.size': 15,'mathtext.fontset':'stix'}
rcParams.update(config)
import pandas as pd
import numpy as np
import scipy
from scipy import stats
def SST(Y):
    sst = sum(np.power(Y - np.mean(Y), 2))
    return sst
def SSA(data, x_name, y_name):
    total_avg = np.mean(data[y_name])
    df = data.groupby([x_name]).agg(['mean', 'count'])
    df = df[y_name]
    ssa = sum(df["count"] * (np.power(df["mean"] - total_avg, 2)))
    return ssa
def SSE(data, x_name, y_name):
    df = data.groupby([x_name]).agg(['mean'])
    df = df[y_name]
    dict_ = dict(df["mean"])
    data_ = data[[x_name, y_name]]
    data_["add_mean"] = data_[x_name].map(lambda x: dict_[x])
    sse = sum(np.power(data_[y_name] - data_["add_mean"], 2))
    return sse
palette=["#FF5B9B","#459DFF"]
def BOX(df,ax,xlim_1,xlim_2,ylabel,
        xmajor,xminor,
        texta,textb,texta1,textb1,
        textfont,jjj,aa):
    df.columns=["Beta diversity","Δ Richness","Δ Temperature",
                "MAT","MAP",
                "Δ Elevation","Area","Population density","Δ EVI",
                ]
    print(df)
    palette = ["#FF5B9B", "#459DFF"]
    col_mean = df.mean(axis=0)
    col_sem = df.sem(axis=0)
    indices = np.argsort(col_mean)[::-1]
    print(indices)
    feat_labels=df.columns
    feat_labels=feat_labels[indices][::-1]
    col_mean=col_mean[indices][::-1]
    col_sem=col_sem[indices][::-1]
    print(feat_labels)
    box =ax.barh([1, 2, 3, 4, 5, 6, 7, 8, 9], col_mean, color="lightgreen",
                 xerr=col_sem,  # 水平误差
                 error_kw={
                     "capsize": 3,  # 设置端点帽为0，即移除上下两根线
                     "elinewidth": 1,  # 设置误差棒的线宽，加粗
                     "ecolor": "black"  # 设置误差棒的颜色
                 })
    #               ax=ax)
    text_font = {"family": "Times New Roman", "size": "14", "weight": "bold", "color": "black"}
    from matplotlib import pyplot as plt
    dev_x = [0, 0, 1, 1,]
    dev_y = [1.05, 1.1, 1.1,1.05,]
    ax.set_xlim((xlim_1,xlim_2))
    fontdict={"size":20,"color":"k","family":"Times New Roman"}
    ax.set(facecolor=(1,1,1))
    ax.set_xlabel("Variable importance",fontdict=fontdict)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8,9])
    ax.set_yticklabels(feat_labels, fontdict=fontdict)
    ax.set_ylabel(ylabel,fontdict=fontdict)
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    xmajorLocator = MultipleLocator(xmajor) # 将y轴主刻度标签设置为0.5的倍数
    xmajorFormatter = FormatStrFormatter('%.2f') # 设置y轴标签文本的格式
    xminorLocator   = MultipleLocator(xminor) # 将此y轴次刻度标签设置为0.1的倍数
    ax.xaxis.set_major_locator(xmajorLocator)  # 设置y轴主刻度
    ax.xaxis.set_major_formatter(xmajorFormatter)  # 设置y轴标签文本格式
    ax.xaxis.set_minor_locator(xminorLocator)  # 设置y轴次刻度
    ax.tick_params(which='major',direction='out', length=4, width=0.5,colors='k',labelsize=15)

    ax.tick_params(which='minor',direction='out', length=2, width=0.5,colors='k',labelsize=15)
    plt.rcParams['ytick.direction'] = 'out'
    labels=ax.get_xticklabels()+ax.get_yticklabels()
    [label.set_fontname("Times New Roman") for label in labels]
    text_font01 = {"family": "Times New Roman", "size": "15", "weight": "bold", "color": "black"}
    text_font02 = {"family": "Times New Roman", "size": "20", "weight": "bold", "color": "black"}
    fontdict = {"size": 20, "color": "Black", "family": "Arial", "weight": "bold"}
    ax.text(-0.08, 1.02, jjj,
             fontdict=fontdict,transform=ax.transAxes )
    ax.text(0.75, 0.1, textfont, transform=ax.transAxes, fontdict=text_font01, zorder=1)
    return box
fig=plt.figure(figsize=(20,5), dpi= 100,)
BOX(pd.read_csv(r"..\DATA\In_ST_impor.csv",header=0,index_col=0,encoding='gbk'),
    fig.add_subplot(1,2,1),0,0.35,"",
    0.1,0.02,
    0.7,0.9,0.7,0.8," Inward_sig","C",aa="TRUE")
BOX(pd.read_csv(r"..\DATA\Out_ST_impor.csv",header=0,index_col=0,encoding='gbk'),
    fig.add_subplot(1,2,2),0,0.35,"",
    0.1,0.02,
    0.7,0.9,0.7,0.8," Outward_sig","D",aa="TRUE")
# position = ax.get_position()
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 9, }
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.subplots_adjust(wspace=0.4,hspace=0.2)
plt.savefig(r"..\Figure\RF_ST.pdf",dpi=600,bbox_inches='tight')
