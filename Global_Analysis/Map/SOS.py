
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
from turtle import *
import pandas as pd
import numpy as np
from matplotlib import rcParams
config={'font.family':'Arial','font.size': 10,'mathtext.fontset':'stix'}
rcParams.update(config)
from matplotlib import font_manager
for font in font_manager.fontManager.ttflist:
    print(font)
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
    return Quadrant1,Quadrant2


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
                ,Unit,GG,HH,DD,FF,ylabel,textdis
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
        ax1.set_yticklabels([])
    ax1.tick_params(labelsize=11)
    ax1.set_ylabel(ylabel, size=11)

    fontdict = {"size": 20, "color": "Black", "family": "Arial", "weight": "bold"}
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
    ax1.text(textdis, 1.08, NAME,
             fontdict=fontdict,transform=ax1.transAxes )
    ax1.text(0.01, 0.85, "Mean= " + str('%.1f' % round(np.mean(np.array(pd.Series(DATA_PLOT).dropna())), 2)) ,
             fontdict=fontdict01, transform=ax1.transAxes)
    ax1.set_title(DD,fontdict={"size": 18, "color": "Black", "family": "Arial"})
    dataQ = ax1.scatter(lon, lat, s=30, c=DATA_PLOT, edgecolors='white', linewidths=.2,
                        cmap=cmap1, norm=norms1,
                        transform=ccrs.PlateCarree())
    if FF=="True":
        cbar = fig.colorbar(dataQ, ax=ax1, shrink=1, aspect=30, fraction=.03, pad=0.02,
                            extend='both')  # orientation='horizontal'位置参数
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
    ax3 = ax1.inset_axes(bounds=[0.77, 0.5, 0.217, 0.45], transform=ax1.transAxes)

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
    ax3.set_xticks(np.arange(len(hist)-1)+0.5)
    ax3.set_xticklabels(bins_in, rotation=rotation)
    ax3.tick_params(axis='x', which='major', direction='out', length=3, labelsize=8)
    ax3.set_ylabel("Frequency (%)",fontsize =8)
    ax3.tick_params(axis='y', which='major', direction='in', length=3, labelsize=8)
    ax3.tick_params(axis='y', which='minor', direction='in', length=2, labelsize=8)
    import math
    ax3.set_yticks(np.arange(0, math.ceil(np.max(np.array(ratio_list)*100))+1, 4))
    from matplotlib.ticker import MultipleLocator
    xminorLocator = MultipleLocator(30)
    ax3.set(facecolor=(1,1,1))
    return ax1,dataQ

fig = plt.figure(figsize=(9,4.25),dpi=100,facecolor="white")
dis01 = pd.read_excel(
        r"..\DATA\Final_Betadiversity_inward_all.xlsx",
        header=0, index_col=0)
dis03= pd.read_excel(
        r"..\DATA\Final_Betadiversity_outward_sig.xlsx",
        header=0, index_col=0)


ax2=plt.subplot2grid((2,1),(0,0),colspan=1, projection=proj)
AA=MAIN_AXES(dis03["SOS_U"],
       dis03["Lon"],
         dis03["Lat"],[-150, 150, 18, 90],ax2, 'a',
             cmaps.MPL_YlGn
             , [90,95,100,105,110,115,120,125,130,135,140,145],
             [-9999, 95, 100, 105, 110, 115, 120, 125, 130, 140, 9999], [95, 100, 105, 110, 115, 120, 125, 130, 140,],55,'{x:.0f}', "","","True","",""
             , ylabel="Latitude"
             , textdis=-0.08)
import matplotlib as mpl
position = ax2.inset_axes(bounds=[1.01, 0.0, 0.02, 1], transform=ax2.transAxes)
cbar = fig.colorbar(AA[1], ax=AA[0], cax=position,shrink=0.8, aspect=30, fraction=.03,
                    pad=0.02,
                     )
ticks = [95,100,105,110,115,120,125,130,135,140,]
formatter = mpl.ticker.StrMethodFormatter('{x:.0f}')
cbar.formatter = formatter
ticklabels = [formatter(tick)+" " for tick in ticks]
print(ticklabels)
cbar.set_ticks(ticks)
cbar.update_ticks()
fontdict = {"size": 15, "color": "Black", "family": "Arial"}
cbar.ax.tick_params(length=0.5, width=1, colors='k', labelsize=10)
# cbar.ax.set_title('Day', size=10)
fontdict = {"size": 11, "color": "Black", "family": "Arial"}
cbar.ax.text(3.4, 111,"SOS"+" (d)",
              fontdict=fontdict,rotation=270)
ax4=plt.subplot2grid((2,1),(1,0),colspan=1, projection=proj)
AA=MAIN_AXES(dis03["SOS_R"],
       dis03["Lon"],
         dis03["Lat"],[-150, 150, 18, 90],ax4, 'b',
             cmaps.MPL_YlGn
             , [90,95,100,105,110,115,120,125,130,135,140,145],
             [-9999, 95,100,105,110,115,120,125,130,140, 9999],[95,100,105,110,115,120,125,130,140,] ,55,'{x:.0f}', "","True","True","",""
             , ylabel="Latitude"
             , textdis=-0.08)
import matplotlib as mpl
position = ax4.inset_axes(bounds=[1.01, 0.0, 0.02, 1], transform=ax4.transAxes)
cbar = fig.colorbar(AA[1], ax=AA[0], cax=position,shrink=0.8, aspect=30, fraction=.03,
                    pad=0.02,
                     )
ticks = [95,100,105,110,115,120,125,130,135,140,]
formatter = mpl.ticker.StrMethodFormatter('{x:.0f}')
cbar.formatter = formatter
ticklabels = [formatter(tick)+" " for tick in ticks]
print(ticklabels)
cbar.set_ticks(ticks)
cbar.update_ticks()
fontdict = {"size": 15, "color": "Black", "family": "Arial"}
cbar.ax.tick_params(length=0.5, width=1, colors='k', labelsize=10)
fontdict = {"size": 11, "color": "Black", "family": "Arial"}
cbar.ax.text(3.4, 111,"SOS"+" (d)",
              fontdict=fontdict,rotation=270)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.subplots_adjust(wspace=0.04,hspace=0.25)
plt.savefig(r"..\Figure\Phnology.pdf",dpi=800,bbox_inches='tight')
plt.show()
