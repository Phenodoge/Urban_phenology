
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

import numpy as np
import pandas as pd
import pandas as pd
import xarray as xr
import geopandas as gp
# from osgeo import gdal
import numpy as np
# import regionmask
# import shapefile
from matplotlib.path import Path
import os
import multiprocessing
import pandas as pd
import os
# import fiona
import matplotlib.pyplot as plt
import numpy as np
from pyproj import CRS
import matplotlib.transforms as mtransforms
from matplotlib import rcParams
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature
# from cartopy.io.shapereader import Reader
# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cmaps
import pandas as pd

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import os
# import sys
# os.environ['PROJ_LIB'] = os.path.dirname(sys.argv[0])


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.transforms as mtransforms
from matplotlib import rcParams
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cmaps
import pandas as pd
# 设置字体.
config={'font.family':'Arial','font.size': 13,'mathtext.fontset':'stix'}
rcParams.update(config)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from turtle import *
import pandas as pd
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from matplotlib import rcParams
config={'font.family':'Arial','font.size': 10,'mathtext.fontset':'stix'}
rcParams.update(config)
from matplotlib import font_manager
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
    import cmaps
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
    ax3 = ax1.inset_axes(bounds=[0.77, 0.5, 0.217, 0.45], transform=ax1.transAxes)

    bins1=bins2
    data=DATA_PLOT
    data1=np.array(data.dropna())
    data1_all=len(data1)
    hist, bin_edges = np.histogram(data1, bins1)
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
                ,Unit,GG,HH,DD,FF,color01,color02,ylabel,textdis
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
    ax1.set_ylabel(ylabel, size=11)
    fontdict = {"size": 20, "color": "Black", "family": "Arial", "weight": "bold"}
    fontdict01 = {"size": 12, "color": "Black", "family": "Arial",}
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
    ax1.text(textdis,1.08, NAME,
             fontdict=fontdict,transform=ax1.transAxes )
    ax1.text(0.01, 0.85, "Mean= " + str('%.2f' % round(np.mean(np.array(pd.Series(DATA_PLOT).dropna())), 2)) ,
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
    axx = ax1.inset_axes(bounds=[0.77, 0.5, 0.217, 0.45], transform=ax1.transAxes)
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
    fontdict = {"size": 8, "color": "Black", "family": "Arial",}
    axx.patch.set_visible(False)
    axx.set_xticks(np.arange(len(hist)-1)+0.5)
    axx.set_xticklabels(bins_in, rotation=rotation)
    axx.tick_params(axis='x', which='major', direction='out', length=3, labelsize=8)
    axx.set_ylabel("Frequency (%)",fontsize =8)
    # axx.set_xticks([])
    axx.tick_params(axis='y', which='major', direction='in', length=3, labelsize=8)

    import math
    axx.set_yticks(np.arange(0, math.ceil(np.max(np.array(ratio_list)*100))+10, 8))
    from matplotlib.ticker import MultipleLocator
    axx.set(facecolor=(1,1,1))
    return ax1,dataQ

def CBAR(ax,ticks=[95, 100, 105, 110, 115, 120, 125, 130, 135, 140, ],axtext=107,
         axtext01="SOS" + "$_\mathrm{" + "urban" + "}$" + " (d)",aa='{x:.1f}'):
    import matplotlib as mpl
    position = ax.inset_axes(bounds=[1.01, 0.0, 0.02, 1], transform=ax.transAxes)
    cbar = fig.colorbar(AA[1], ax=AA[0], cax=position, shrink=0.8, aspect=30, fraction=.03,
                        pad=0.02,
                        )  # orientation='horizontal'位置参数cax=position,extend='both',
    ticks = ticks
    formatter = mpl.ticker.StrMethodFormatter(aa)
    cbar.formatter = formatter
    ticklabels = [formatter(tick) + " " for tick in ticks]
    print(ticklabels)
    cbar.set_ticks(ticks)
    cbar.update_ticks()
    fontdict = {"size": 15, "color": "Black", "family": "Times New Roman"}
    cbar.ax.tick_params(length=0.5, width=1, colors='k', labelsize=11)
    # cbar.ax.set_title('Day', size=10)
    fontdict = {"size": 11, "color": "Black", "family": "Times New Roman"}
    cbar.ax.text(4.6 ,axtext, axtext01,
                 fontdict=fontdict, rotation=270)
file_path = r"..\DATA\Final_Betadiversity_outward_all.xlsx"
dis01= pd.read_excel(file_path, sheet_name="Final_Betadiversity_outward_all")
import matplotlib
list_cmap1=cmaps.cmocean_ice(np.linspace(0,1,6))
list_cmap2=cmaps.MPL_YlOrRd(np.linspace(0,1,5))
new_color_list=np.vstack((list_cmap1,list_cmap2))
new_cmap01=matplotlib.colors.ListedColormap(new_color_list,name='new_cmap ')
fig = plt.figure(figsize=(8,6),dpi=100,facecolor="white")
new_cmap=cmaps.MPL_hot_r
import matplotlib as mpl
new_cmap = mpl.colors.ListedColormap([(252/255, 236/255, 177/255),
                                  (247/255, 213/255, 158/255),
                                  (242/255, 194/255, 143/255),
                                  (237/255, 175/255,130/255),
                                  (232 / 255,157 / 255, 116/ 255),
                                  (224 / 255, 136 / 255,101/ 255),
                                  (219/ 255, 120/ 255, 90/ 255),
                                  (212 / 255, 103 / 255, 78/ 255),
                                  (204 / 255, 86 / 255,67/ 255),
                                  (196 / 255, 69 / 255, 57 / 255),
                                  ])
ax1=plt.subplot2grid((3,1),(0,0),colspan=1, projection=proj)
AA=MAIN_AXES(dis01["SPringT_U"],
       dis01["Lon"],
         dis01["Lat"],[-150, 150, 18, 90],ax1, 'a',###############
             new_cmap
             ,[4, 6,8,10,12,14,16,18,],
             [-9999,6,8,10,12,14,16,9999], [ 6,8,10,12,14,16,],0,'{x:.0f}', "","","True","",'',
            ylabel = ""
            , textdis = -0.05)
CBAR(ax1,ticks=[6,8,10,12,14,16, ],axtext=12,axtext01="AMT (℃)",aa='{x:.0f}')
ax2=plt.subplot2grid((3,1),(1,0),colspan=1, projection=proj)

AA=MAIN_AXES(dis01["SPringT_R"],
       dis01["Lon"],
         dis01["Lat"],[-150, 150, 18, 90],ax2, 'b',###############
             new_cmap
             ,[4, 6,8,10,12,14,16,18,],
             [-9999,6,8,10,12,14,16,9999], [6,8,10,12,14,16,],0,'{x:.0f}', "","","True","","",
ylabel = ""
, textdis = -0.05)
import matplotlib as mpl
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

CBAR(ax2,ticks=[6,8,10,12,14,16, ],axtext=12,axtext01="AMT (℃)",aa='{x:.0f}')

ax3=plt.subplot2grid((3,1),(2,0),colspan=1, projection=proj)
AA=MAIN_AXES01(dis01["SPringT_Diff"],
       dis01["Lon"],
         dis01["Lat"],[-150, 150, 18, 90],ax3, 'c',
             new_cmap
             ,[0.1,0.15,0.2,0.25,0.3,0.35,0.4],
             [-9999,0.15,0.2,0.25,0.3,0.35,9999],[0.15,0.2,0.25,0.3,0.35,],45,'{x:.1f}', "","True","True","","",
               (214 / 255, 47 / 255, 39 / 255), (69 / 255, 117 / 255, 181 / 255),ylabel=""
             ,textdis=-0.05)
CBAR(ax3,ticks=[0.15,0.2,0.25,0.3,0.35],axtext=0.2,axtext01="AMT"+"$_\mathrm{{urban}}$"+"-"+"Ta"+"$_\mathrm{{rural}}$"+" (℃)",aa='{x:.2f}')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
plt.subplots_adjust(wspace=0.04,hspace=0.25)
plt.savefig(r"..\Figure\\Spring T.pdf",dpi=800,bbox_inches='tight')
plt.show()

