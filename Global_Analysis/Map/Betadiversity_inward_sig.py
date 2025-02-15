
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







import matplotlib.transforms as mtransforms
from matplotlib import rcParams
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
config={'font.family':'Arial','font.size': 13,'mathtext.fontset':'stix'}
rcParams.update(config)
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import rcParams
config={'font.family':'Arial','font.size': 10,'mathtext.fontset':'stix'}
rcParams.update(config)
proj = ccrs.PlateCarree()
subplot_kw = {'projection': proj}
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
def add_right_cax(ax, pad, width):
    axpos = ax.get_position()
    caxpos = mtransforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)
    return cax
def judge(XARRAY):
    COUNT=len(XARRAY)
    Quadrant1=0
    Quadrant2=0
    for i in range(COUNT):
        x=np.array(XARRAY)[i]
        if x < 0:
            Quadrant2=(Quadrant2+1)
        else:
            Quadrant1=(Quadrant1+1)
    Quadrant1="%.2f" % (Quadrant1/COUNT*100)
    Quadrant2="%.2f" % (Quadrant2/COUNT*100)
    return Quadrant1,Quadrant2#(zheng,fu)
def make_ticks(ax):
    ax.patch.set_visible(False)
    ax.tick_params(direction='in')
    ax.tick_params(axis='x',which='major',direction='in',length=0,labelsize=4)
    ax.tick_params(axis='y',which='major',direction='in',length=0,labelsize=4)
    ax.set_yticks(np.arange(0,35,10))
    ax.set_xticks(np.arange(0,21,5))
    ax.set_ylim(0,30)
def MAIN_AXES(DATA_PLOT,lon,lat,extents_main
              ,ax1,NAME
              ,cmaps_ncl
              ,bins
              ,bins2,bins_in,rotation
              ,form
                ,Unit,GG,HH,DD,FF
              ):
    lonmin, lonmax, latmin, latmax = extents_main
    xticks = np.arange(lonmin, lonmax + 1, 50)
    yticks = np.arange(30, latmax + 1, 30)
    ax1.set_extent(extents_main, crs=proj)
    if GG=="True":
        ax1.set_xticks(xticks, crs=proj)
    elif GG=="":
        ax1.set_xticks(xticks, crs=proj)
        ax1.set_xticklabels([])
    if HH=="True":
        ax1.set_yticks(yticks, crs=proj)
    elif HH=="":
        ax1.set_yticks(yticks, crs=proj)
    ax1.tick_params(labelsize=11)
    ax1.set_ylabel("Latitude", size=11)
    fontdict = {"size": 16, "color": "Black", "family": "Arial", "weight": "bold"}#
    fontdict01 = {"size": 12.5, "color": "Black", "family": "Arial", }
    data1_mean=np.mean(DATA_PLOT)
    ax1.add_feature(cfeature.LAND.with_scale('50m'),facecolor=(225/255,225/255,225/255))
    cmap1 = cmaps_ncl
    from matplotlib.colors import ListedColormap, BoundaryNorm
    bins = bins
    bounds = bins
    norms1 = BoundaryNorm(bounds, cmap1.N)
    box = {
        'facecolor': 'White',
        'edgecolor': None,
        'boxstyle': 'round'
    }
    ax1.text(-0.08, 1.08, NAME,
             fontdict=fontdict,transform=ax1.transAxes )
    ax1.text(0.01, 0.85, "Mean= " + str('%.2f' % round(np.mean(np.array(pd.Series(DATA_PLOT).dropna())), 2)) ,
             fontdict=fontdict01, transform=ax1.transAxes)
    ax1.set_title(DD,fontdict={"size": 18, "color": "Black", "family": "Arial"})
    dataQ = ax1.scatter(lon, lat, s=30, c=DATA_PLOT, edgecolors='white', linewidths=.2,
                        cmap=cmap1, norm=norms1,
                        transform=ccrs.PlateCarree())
    if FF=="True":
        cbar = fig.colorbar(dataQ, ax=ax1, shrink=1, aspect=30, fraction=.03, pad=0.02,
                            extend='both')
        ticks = bins
        import matplotlib as mpl
        formatter = mpl.ticker.StrMethodFormatter(form)
        cbar.formatter = formatter
        ticklabels = [formatter(tick) for tick in ticks]
        print(ticklabels )
        cbar.set_ticks(ticks)
        cbar.update_ticks()
        fontdict = {"size": 15, "color": "Black", "family": "Arial"}
        cbar.ax.tick_params(length=1.1, width=1, colors='k', labelsize=10)
    ax3 = ax1.inset_axes(bounds=[0.77, 0.55, 0.217, 0.4], transform=ax1.transAxes)

    bins1=bins2
    data=DATA_PLOT
    data1=np.array(data.dropna())
    data1_all=len(data1)
    hist, bin_edges = np.histogram(data1, bins1)  # make the histogram
    ratio_list=[]
    for i in hist:
        ratio=i/data1_all
        ratio_list.append(ratio)
    print(ratio_list)
    newcolors = cmap1(norms1(bins))
    ax3.bar(range(len(hist)), np.array(ratio_list)*100, width=1,color=newcolors,edgecolor='white')
    ax3.patch.set_visible(False)
    ax3.set_xticks(np.arange(len(hist) - 1) + 0.5)
    ax3.set_xticklabels(bins_in, rotation=rotation)
    ax3.tick_params(axis='x', which='major', direction='out', length=3, labelsize=8)
    ax3.set_ylabel("Frequency (%)",fontsize =8)
    # ax3.set_xticks([])
    ax3.tick_params(axis='y', which='major', direction='in', length=3, labelsize=8)
    ax3.tick_params(axis='y', which='minor', direction='in', length=2, labelsize=8)
    import math
    ax3.set_yticks(np.arange(0, math.ceil(np.max(np.array(ratio_list)*100))+1, 8))
    from matplotlib.ticker import MultipleLocator
    xminorLocator = MultipleLocator(30)
    ax3.set(facecolor=(1,1,1))
    return ax1,dataQ
def MAIN_AXES01(DATA_PLOT,lon,lat,extents_main
              ,i,NAME
              ,cmaps_ncl
              ,bins
              ,bins2,bins_in,rotation
              ,form
                ,Unit,GG,HH,DD,FF,color01,color02
              ):
    lonmin, lonmax, latmin, latmax = extents_main
    xticks = np.arange(lonmin, lonmax + 1, 50)
    yticks = np.arange(30, latmax + 1, 30)
    ax1 = fig.add_subplot(i, projection=proj)
    ax1.set_extent(extents_main, crs=proj)
    if GG=="True":
        ax1.set_xticks(xticks, crs=proj)
    elif GG=="":
        ax1.set_xticks(xticks, crs=proj)
        ax1.set_xticklabels([])
    if HH=="True":
        ax1.set_yticks(yticks, crs=proj)
    elif HH=="":
        ax1.set_yticks(yticks, crs=proj)
        ax1.set_yticklabels([])
    ax1.tick_params(labelsize=11)
    ax1.set_ylabel("Latitude", size=11)
    fontdict = {"size": 16, "color": "Black", "family": "Arial", "weight": "bold"}
    fontdict01 = {"size": 12.5, "color": "Black", "family": "Arial",}
    data1_mean=np.mean(DATA_PLOT)
    ax1.add_feature(cfeature.LAND.with_scale('50m'),facecolor=(225/255,225/255,225/255))
    cmap1 = cmaps_ncl
    from matplotlib.colors import ListedColormap, BoundaryNorm
    bins = bins
    bounds = bins
    norms1 = BoundaryNorm(bounds, cmap1.N)
    box = {
        'facecolor': 'White',
        'edgecolor': None,
        'boxstyle': 'round'
    }
    ax1.text(-0.08, 1.08, NAME,
             fontdict=fontdict,transform=ax1.transAxes )
    ax1.text(0.01, 0.85, "Mean = " + str('%.2f' % round(np.mean(np.array(pd.Series(DATA_PLOT).dropna())), 2)) ,
             fontdict=fontdict01, transform=ax1.transAxes)
    ax1.set_title(DD,fontdict={"size": 18, "color": "Black", "family": "Arial"})
    dataQ = ax1.scatter(lon, lat, s=30, c=DATA_PLOT, edgecolors='white', linewidths=0.2,
                        cmap=cmap1, norm=norms1,
                        transform=ccrs.PlateCarree())
    if FF=="True":
        cbar = fig.colorbar(dataQ, ax=ax1, shrink=1, aspect=30, fraction=.03, pad=0.02,
                            extend='both')
        ticks = bins
        import matplotlib as mpl
        formatter = mpl.ticker.StrMethodFormatter(form)
        cbar.formatter = formatter
        ticklabels = [formatter(tick) for tick in ticks]
        print(ticklabels )
        cbar.set_ticks(ticks)
        cbar.update_ticks()
        fontdict = {"size": 15, "color": "Black", "family": "Arial"}
        cbar.ax.tick_params(length=1.1, width=1, colors='k', labelsize=10)
    axx = ax1.inset_axes(bounds=[0.77, 0.55, 0.217, 0.4], transform=ax1.transAxes)
    bins1=bins2
    data=DATA_PLOT
    data1=np.array(data.dropna())
    data1_all=len(data1)
    hist, bin_edges = np.histogram(data1, bins1)  # make the histogram
    ratio_list=[]
    for i in hist:
        ratio=i/data1_all
        ratio_list.append(ratio)
    print(ratio_list)
    newcolors = cmap1(norms1(bins))
    import math
    axx.set_ylim((0, math.ceil(np.max(np.array(ratio_list)*100))+10))
    axx.bar(range(len(hist)), np.array(ratio_list)*100, width=1,color=newcolors,edgecolor='white')
    axx.plot(np.ones(100)*4.5, np.arange(0,100,1), linewidth=0.8, color='grey',linestyle="--")
    fontdict = {"size": 9.5, "color": "Black", "family": "Arial"}
    axx.text(0.54, 0.8, "(P) " + str(judge(DATA_PLOT)[0]) + "%", fontdict=fontdict, color=color01,
             transform=axx.transAxes)
    axx.text(0.04, 0.8, "(N) " + str(judge(DATA_PLOT)[1]) + "%", fontdict=fontdict, color=color02,
             transform=axx.transAxes)
    axx.patch.set_visible(False)
    axx.set_ylabel("Frequency (%)",fontsize =8)
    axx.set_xticks(np.arange(len(hist) - 1) + 0.5)
    axx.set_xticklabels(bins_in, rotation=rotation)
    axx.tick_params(axis='x', which='major', direction='out', length=3, labelsize=8)
    axx.tick_params(axis='y', which='major', direction='in', length=3, labelsize=8)
    import math
    axx.set_yticks(np.arange(0, math.ceil(np.max(np.array(ratio_list)*100))+10, 8))
    axx.set(facecolor=(1,1,1))

    return ax1,dataQ
def scatter(
            n,esti_n,jj,
             ax, hhh, a,b,c,d,h,j,m,**kwargs):
    x = n
    y = esti_n
    C = round(r_score(x, y), 5)
    x2 = np.linspace(0,1000)
    y2 = x2
    def f_1(x, A, B):
        return A * x + B
    A1, B1 = optimize.curve_fit(f_1, x, y)[0]
    y3= A1 * x + B1
    import statsmodels.api as sm
    X =pd.concat([n,jj],axis=1)
    print(X)
    Y = esti_n
    X = sm.add_constant(X)  # adding a constant
    model = sm.OLS(Y, X).fit()
    print(model.summary())
    y4fit=model.fittedvalues
    y4=model.params[1]*x+model.params[2]*np.mean(jj)+model.params[0]
    sns.regplot(x=n, y=esti_n, ax=ax,ci=95,marker="o",scatter_kws={'s':5,'color':"black"} ,line_kws={'color':(214 / 255, 47 / 255, 39 / 255),"linewidth":0} )  # ci是置信区间，此处设置为95%
    ax.plot(x, y4, color=sns.xkcd_rgb['red'], linewidth=2.5, linestyle="-")
    fontdict1 = {"size": 28, "color": "k", "family": "Arial"}
    ax.set(facecolor=(1, 1, 1))
    ax.set_xlabel(b, fontdict=fontdict1)
    ax.set_ylabel(a, fontdict=fontdict1)
    ax.grid(False)
    ax.set_xlim((-0.02,1.1))
    ax.set_ylim((c,d))
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    xmajorLocator = MultipleLocator(0.2)
    xmajorFormatter = FormatStrFormatter('%.1f')
    ymajorLocator = MultipleLocator(h)
    ymajorFormatter = FormatStrFormatter(m)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.tick_params(which='major', direction='in', length=3, colors='k', labelsize=10)#width=1
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname("Arial") for label in labels]
    titlefontdict = {"size": 20, "color": "k", "family": "Arial"}
    fontdict = {"size": 8, "color": "black", "family": "Arial"}
    import scipy
    UU=scipy.stats.linregress(x, y)
    if 0.05<UU.pvalue<=0.1:
        p="*"
    elif 0.01<UU.pvalue<0.05:
        p="**"
    elif UU.pvalue<=0.01:
        p="***"
    else:
        p=""
    UU=scipy.stats.linregress(y, y4fit)
    if 0.05<UU.pvalue<=0.1:
        p="*"
    elif 0.01<UU.pvalue<0.05:
        p="**"
    elif UU.pvalue<=0.01:
        p="***"
    else:
        p=""
    ax.text(0.05, 0.8,
             "R" +"$^{"+"2"+"}$"+"= "+ str('%.3f' % round(r_score(y, y4fit)**2, 3)), fontdict=fontdict,transform=ax.transAxes,color=(214 / 255, 47 / 255, 39 / 255))
    text_font = {"family": "Arial", "size": "10", "color": "black"}
    ax.text(0.05, 1.03, hhh, transform=ax.transAxes, fontdict=text_font, zorder=1)
    return ax
def barplot(dis01,ax,MEAN01,SE,ylabel,xlabels,Text,ylima,ylimb,ymajor,AA0='SOS_Diff_ABS'):
    data1=ax.bar([0.1,0.2,0.3,0.4,0.5,0.6],MEAN01,yerr=SE, width=0.06,
            color=cmaps.cmocean_deep(np.linspace(0,1,9))[2:8],error_kw={"capsize":5,"elinewidth":0.5} )
    ax.set_xticks([0.1,0.2,0.3,0.4,0.5,0.6])
    ax.set_xticklabels( ['0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5' ,'0.5-0.7', '0.7-1.0'],rotation=45)
    ax.set_xlabel(xlabels,size=11)
    ax.set_ylabel(ylabel,size=11)
    ax.tick_params(labelsize=10)
    text_font = {"family":"Arial", "size": "16", "weight": "bold", "color": "black"}
    ax.text(-0.16, 1.08, Text, transform=ax.transAxes, fontdict=text_font, zorder=1)
    ax.set_ylim((ylima,ylimb))
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    ymajorLocator = MultipleLocator(ymajor)
    ymajorFormatter = FormatStrFormatter('%.0f')
    ax.yaxis.set_major_locator(ymajorLocator)
    ax.yaxis.set_major_formatter(ymajorFormatter)
    ax.tick_params(which='major', direction='out', length=3, colors='k', labelsize=11)
    ax.tick_params(axis='x', direction='out', length=4, width=1, colors='k')
    AA =  ['0.1~0.2', '0.2~0.3', '0.3~0.4', '0.4~0.5' ,'0.5~0.7', '0.7~1.0']
    print(AA)
    MEAN=[0.15,0.25,0.35,0.45,0.6,0.85]
    ST_MEAN=MEAN01
    print(ST_MEAN)
    def f_1(x, A, B,C):
        return A * x ** B+C
    def f_2(x, A, B):
        return A * x+ B
    return ax
def r_score(y,y1):
    r_score=(y.corr(y1))
    return r_score
import matplotlib as mpl
fig = plt.figure(figsize=(9.5,10),dpi=100,facecolor="white")
ax1=plt.subplot2grid((4,2),(0,0),colspan=2, projection=proj)
dis01=pd.read_excel(r"..\DATA\Final_Betadiversity_inward_sig.xlsx",header=0,index_col=0)
import matplotlib
list_cmap1=cmaps.GreenMagenta16_r(np.linspace(0,1,14))[1:6]
list_cmap2=cmaps.GreenMagenta16_r(np.linspace(0,1,14))[8:13]
new_color_list=np.vstack((list_cmap1,list_cmap2))
new_cmap=matplotlib.colors.ListedColormap(new_color_list,name='new_cmap ')
cmap = mpl.colors.ListedColormap([(69/255, 117/255, 181/255),
                                  (110/255, 143/255, 184/255),
                                  (153/255, 174/255, 189/255),
                                  (192/255, 204/255,190/255),
                                  (233 / 255,237 / 255, 190/ 255),
                                  (255 / 255, 233 / 255,173 / 255),
                                  (250 / 255, 185 / 255, 132 / 255),
                                  (242 / 255, 141 / 255, 97 / 255),
                                  (230 / 255, 96 / 255,61/ 255),
                                  (214 / 255, 47 / 255, 39 / 255),
                                  ])
AA01=MAIN_AXES01(dis01["SOS_Diff"],
       dis01["Lon"],
         dis01["Lat"],[-150, 150, 18, 90],ax1, 'a',
             cmap
             , [-5,-4,-3,-2,-1,0,1,2,3,4,5,],
             [-9999, -4,-3,-2,-1,0,1,2,3,4,9999],[-4,-3,-2,-1,0,1,2,3,4,] ,0,'{x:.0f}', "","","True","",'',
             (214 / 255, 47 / 255, 39 / 255),(69/255, 117/255, 181/255))
position = ax1.inset_axes(bounds=[1.01, 0.0, 0.02, 1], transform=ax1.transAxes)
cbar = fig.colorbar(AA01[1], ax=AA01[0], cax=position,shrink=0.8, aspect=30, fraction=.03,
                    pad=0.02,
                     )
ticks = [-4,-3,-2,-1,0,1,2,3,4,]
formatter = mpl.ticker.StrMethodFormatter('{x:.0f}')
cbar.formatter = formatter
ticklabels = [formatter(tick)+" " for tick in ticks]
print(ticklabels)
cbar.set_ticks(ticks)
cbar.update_ticks()
fontdict = {"size": 15, "color": "Black", "family": "Arial"}
cbar.ax.tick_params(length=0.5, width=1, colors='k', labelsize=10)
fontdict = {"size": 11, "color": "Black", "family": "Arial"}
cbar.ax.text(3.5, -3.4,"ΔSOS (d)",
              fontdict=fontdict,rotation=270)
ax2=plt.subplot2grid((4,2),(1,0),colspan=2, projection=proj)
new_cmap =cmaps.cmocean_deep
cmap = mpl.colors.ListedColormap([(69/255, 117/255, 181/255),
                                  (110/255, 143/255, 184/255),
                                  (153/255, 174/255, 189/255),
                                  (192/255, 204/255,190/255),
                                  (233 / 255,237 / 255, 190/ 255),
                                  (255 / 255, 233 / 255,173 / 255),
                                  (250 / 255, 185 / 255, 132 / 255),
                                  (242 / 255, 141 / 255, 97 / 255),
                                  (230 / 255, 96 / 255,61/ 255),
                                  (214 / 255, 47 / 255, 39 / 255),
                                  ])
AA03=MAIN_AXES01(dis01["ST_Diff"],
       dis01["Lon"],
         dis01["Lat"],[-150, 150, 18, 90],ax2, 'b',
             cmap
             ,  [-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5],
             [-9999, -0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4, 9999], [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,] ,45,'{x:.0f}', "","","True","","",
             (214 / 255, 47 / 255, 39 / 255),(69/255, 117/255, 181/255))
position = ax2.inset_axes(bounds=[1.01, 0.0, 0.02, 1], transform=ax2.transAxes)
cbar = fig.colorbar(AA03[1], cax=position, ax=[ AA03[0]], shrink=0.8, aspect=30, fraction=.03,
                    pad=0.02,
                    )
ticks = [-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,]
import matplotlib as mpl
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
cbar.formatter = formatter
ticklabels = [formatter(tick) for tick in ticks]
print(ticklabels)
cbar.set_ticks(ticks)
cbar.update_ticks()
fontdict = {"size": 15, "color": "Black", "family": "Arial"}
cbar.ax.tick_params(length=0.5, width=1, colors='k', labelsize=10)
fontdict = {"size": 11, "color": "Black", "family": "Arial"}
cbar.ax.text(3.5, -0.42,"ΔST (d"+" "+"$^\mathrm{o}$"+"C$^{-1}$"+")",
              fontdict=fontdict,rotation=270)
ax33=plt.subplot2grid((4,2),(2,0),colspan=2, projection=proj)
AA=MAIN_AXES(dis01["Bary"],
       dis01["Lon"],
         dis01["Lat"],[-150, 150, 18, 90],ax33, 'c',
             cmaps.MPL_YlGn
             ,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
             [-9999, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,9999], [ 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,] ,45,'{x:.1f}', "","True","True","",''
             )
position = ax33.inset_axes(bounds=[1.01, 0.0, 0.02, 1], transform=ax33.transAxes)
cbar = fig.colorbar(AA[1], cax=position, ax=[AA[0]], shrink=0.8, aspect=30, fraction=.01,
                    pad=0.02,
                    )
ticks =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]
import matplotlib as mpl

formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
cbar.formatter = formatter
ticklabels = [formatter(tick) for tick in ticks]
print(ticklabels)
cbar.set_ticks(ticks)
cbar.update_ticks()
fontdict = {"size": 15, "color": "Black", "family": "Arial"}
cbar.ax.tick_params(length=0.5, width=1, colors='k', labelsize=10)
fontdict = {"size": 11, "color": "Black", "family": "Arial"}
cbar.ax.text(3.5, 0.29,"Beta diversity",
              fontdict=fontdict,rotation=270)
ax4=ax33.inset_axes(bounds=[0.0, -1.42, 0.45, 1.1], transform=ax33.transAxes)
def MeanSe(dataframe):
    AA =  ['0.1~0.2', '0.2~0.3', '0.3~0.4', '0.4~0.5' ,'0.5~0.7', '0.7~1.0']
    print(AA)
    MEAN=[]
    SE=[]
    MEAN01=[]
    SE01=[]
    for i in AA:
        df=dataframe[dataframe['Bary01']==i]
        MEAN.append(df['SOS_Diff'].mean(axis=0))
        SE.append(df['SOS_Diff'].sem(axis=0))
        MEAN01.append(df['SOS_Diff_ABS'].mean(axis=0))
        SE01.append(df['SOS_Diff_ABS'].sem(axis=0))
    return MEAN,SE,MEAN01,SE01
DD=barplot(dis01,ax4, MeanSe(dis01)[2],MeanSe(dis01)[3],"ΔSOS"+" (d)","Beta diversity","d",0,10.5,3,AA0='SOS_Diff_ABS')
ax5=ax33.inset_axes(bounds=[0.55, -1.42, 0.45, 1.1], transform=ax33.transAxes)
def MeanSe(dataframe):
    AA =  ['0.1~0.2', '0.2~0.3', '0.3~0.4', '0.4~0.5' ,'0.5~0.7', '0.7~1.0']
    MEAN=[]
    SE=[]
    MEAN01=[]
    SE01=[]
    for i in AA:
        df = dataframe[dataframe['Bary01'] == i]
        MEAN.append(df['ST_Diff'].mean(axis=0))
        SE.append(df['ST_Diff'].sem(axis=0))
        MEAN01.append(df['ST_Diff_ABS'].mean(axis=0))
        SE01.append(df['ST_Diff_ABS'].sem(axis=0))
    return MEAN, SE, MEAN01, SE01
DD=barplot(dis01,ax5, MeanSe(dis01)[2], MeanSe(dis01)[3],  "ΔST (d"+" "+"$^\mathrm{o}$"+"C$^{-1}$"+")","Beta diversity","e", 0, 3.5, 1,AA0='ST_Diff_ABS')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.subplots_adjust(wspace=0.05,hspace=0.2)
plt.savefig(r"..\Figure\Betadiversity_inward_sig.pdf",dpi=600,bbox_inches='tight')
plt.show()


