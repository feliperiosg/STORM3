import dask
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr
from gc import collect
from tqdm import tqdm
from osgeo import gdal
from pyproj import CRS
from fitter import Fitter
from skimage import morphology
from rasterstats import zonal_stats
from lmfit import Model, Parameters
from warnings import filterwarnings, warn, simplefilter
from scipy import optimize, stats, special
from geopandas import GeoDataFrame, GeoSeries, points_from_xy, read_file, sjoin
from os.path import abspath, dirname, exists, join
from statsmodels.distributions.copula.api import GaussianCopula
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from rasterio.enums import Resampling  # IF doing BILINEAR or NEAREST too?
from rasterio.features import shapes
from rasterio import open as openr
from matplotlib import pyplot as plt
from parameters import SHP_FILE, DEM_FILE, WKT_OGC, BUFFER, X_RES, Y_RES, ZON_FILE
from parameters import PDF_FILE, RAIN_MAP, NREGIONS, SEASON_TAG, Z_CUTS, Z_STAT
# from dask.distributed import Client, LocalCluster

# https://stackoverflow.com/a/9134842/5885810  (supress warning by message)
filterwarnings('ignore', message='You will likely lose important projection '
               'information when converting to a PROJ string from another format')
# because the "EPSG_CODE = 42106" is not a standard proj?
filterwarnings('ignore', message="GeoDataFrame's CRS is not "
               "representable in URN OGC format")
# GDAL misleading warning  [https://stackoverflow.com/a/9134842/5885810]
filterwarnings('ignore', message="^.*not successfully written. Possibly due to "
               "too larger number with respect to field width.*$")
# https://stackoverflow.com/a/41126025/5885810  because 'decode_cf=True' (xarray)
simplefilter('ignore', category=UserWarning)

# https://gdal.org/api/python_gotchas.html#gotchas-that-are-by-design-or-per-history
# https://github.com/OSGeo/gdal/blob/master/NEWS.md#ogr-370---overview-of-changes
# enable exceptions for GDAL <= 4.0
if gdal.__version__.__getitem__(0) == '3':
    gdal.UseExceptions()
    # gdal.DontUseExceptions()
    # gdal.__version__ # wos_ '3.6.2' # linux_ '3.7.0'

parent_d = dirname(__file__)  # otherwise, will append the path.of.the.tests
# parent_d = './'  # to be used in IPython

# # https://stackoverflow.com/a/20627316/5885810
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
tqdm.pandas(ncols=50)  # , desc="progress-bar")


# %% parameters

# EVENT_DATA = './model_input/0326_collect_MAM_track_hadIMERG_.nc'
EVENT_DATA = './model_input/0326_collect_OND_track_hadIMERG_.nc'
ALTERNATIV = 1  # 1-for.simple.totals; 2-simple.totals+copula; 3-pf-based

# # parameters imported from "parameters.py"

# PDF_FILE = './model_input/ProbabilityDensityFunctions.csv'  # pdf.pars file
# ZON_FILE = './model_input/regions.shp'  # nK regions file
# SHP_FILE = './model_input/HAD_basin.shp'  # catchment shape-file in WGS84
# DEM_FILE = './model_input/HAD_wgs84.tif'  # aoi raster-file (optional**)
# # RAIN_MAP = './model_input/rainfall_MAM.nc'
# RAIN_MAP = './model_input/rainfall_OND.nc'
# NREGIONS = 4
# SEASON_TAG = 'OND'

# # Z_CUTS = [ 400, 1000]  # [34.2, 67.5]%
# # Z_CUTS = [1000, 2000, 3000]  # [67.53, 97.15, 99.87]%
# Z_CUTS = [300,  600, 1200]  # [28.13, 48.75, 78.57]%
# # Z_CUTS = []
# # Z_CUTS = None

# BUFFER    =  7000.  # in meters! -> buffer distance (out of the HAD)
# X_RES     =  5000.  # in meters! (pxl.resolution for the 'regular/local' CRS)
# Y_RES     =  5000.  # in meters! (pxl.resolution for the 'regular/local' CRS)

# # OGC-WKT for HAD [taken from https://epsg.io/42106]
# WKT_OGC = 'PROJCS["WGS84_/_Lambert_Azim_Mozambique",'\
#     'GEOGCS["unknown",'\
#         'DATUM["unknown",'\
#             'SPHEROID["Normal Sphere (r=6370997)",6370997,0]],'\
#         'PRIMEM["Greenwich",0,'\
#             'AUTHORITY["EPSG","8901"]],'\
#         'UNIT["degree",0.0174532925199433,'\
#             'AUTHORITY["EPSG","9122"]]],'\
#     'PROJECTION["Lambert_Azimuthal_Equal_Area"],'\
#     'PARAMETER["latitude_of_center",5],'\
#     'PARAMETER["longitude_of_center",20],'\
#     'PARAMETER["false_easting",0],'\
#     'PARAMETER["false_northing",0],'\
#     'UNIT["metre",1,'\
#         'AUTHORITY["EPSG","9001"]],'\
#     'AXIS["Easting",EAST],'\
#     'AXIS["Northing",NORTH],'\
#     'AUTHORITY["EPSG","42106"]]'


# %% region masking

class masking:

    def __init__(self, **kwargs):
        """
        reads catchment-SHP, buffers it, and burns them into 2D-numpy.\n
        Input: none.\n
        **kwargs\n
        catchment: shp; shapefile.
        wkt_prj: char; WKT (well-known text) of the local CRS.
        buffer: float; buffer (in meters) to expand the shapefile.
        x_res: float; spatial resolution (in meters) along the X-dimension.
        y_res: float; spatial resolution (in meters) along the Y-dimension.
        initValues: int/float; default value to pre-empt the raster (with).
        burn_tag: char; geoPandas.GeoDataFrame column used as burning values.\n
        Output -> a class having the buffer & catchment rasters.
        """

        self.catch_shp = kwargs.get('catchment', SHP_FILE)
        self.wkt_prj = kwargs.get('wkt_prj', WKT_OGC)
        self.buffer = kwargs.get('buffer', BUFFER)
        self.x_res = kwargs.get('x_res', X_RES)
        self.y_res = kwargs.get('y_res', Y_RES)
        self.seed_val = kwargs.get('initValues', 0)
        self.burn_tag = kwargs.get('burn_tag', 'burn')
        self.bbbox = None
        self.buffer_mask, self.catchment_mask = self.masks()
        self.xs, self.ys = self.coords(self.bbbox)

    @staticmethod
    def shapsterize(jeometry, x_res, y_res, zero, burn_tag, limits, out_crs,
                    **kwargs):
        """
        burn a shp/geopandas.GeoDataFrame into a raster/2D-numpy.\n
        Input ->
        *jeometry* : str; representing a JSON (from geopandas.GeoDataFrame).
        *x_res* : float; spatial resolution (in meters) along the X-dimension.
        *y_res* : float; spatial resolution (in meters) along the Y-dimension.
        *zero* : int/float; default value to pre-empt the raster (with).
        *burn_tag* : char; geoPandas.GeoDataFrame column used as burning values.
        *limits* : list; float limits in the order: 'l', 'b', 'r', 't'
        *out_crs* : char; PROJ4 string of output's CRS.\n
        **kwargs ->
        name : char; the path to raster's location (ending in ".tif").
        format : char; output format ('MEM' for inMemory; 'GTiff'...).
        out_type : int?; gdal.object specifying the output bit-size.
        add : bool; aggregate geometry.values in case of overlapping?.\n
        Output -> numpy; 2D-numpy (or '.tif' file).
        """
        r_name = kwargs.get('name', '')
        f_rmat = kwargs.get('format', 'MEM')
        out_type = kwargs.get('outputType', gdal.GDT_Int16)
        add_bool = kwargs.get('add', True)

        tmp = gdal.Rasterize(
            # 'tmp-raster_mask-buff.tif', jeometry, format='GTiff',  # (in TIFF)
            r_name, jeometry, format=f_rmat,  # (in MEMORY)
            xRes=x_res, yRes=y_res,
            # initValues=zero, burnValues=1,  # noData=0,
            initValues=zero, attribute=burn_tag, allTouched=True, add=add_bool,
            outputType=out_type, outputBounds=limits, targetAlignedPixels=True,
            # https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap
            # targetAlignedPixels=False,
            # UPDATE needed for outputSRS [in WKT instead of PROJ4]
            outputSRS=out_crs,
            # width=(abs(limits[2] - limits[0]) / x_res).astype('u2'),
            # height=(abs(limits[3] - limits[1]) / y_res).astype('u2'),
            )
        burnt = tmp.ReadAsArray()  # .astype('i1')
        tmp = None
        return burnt

    def masks(self,):
        """
        define coordinates of the X and Y axes.\n
        Input: none.\n
        Output -> tuple; 2D-numpy(s) having the buffer & catchment rasters.
        """
        # read WG-catchment shapefile (assumed to be in WGS84)
        wtrwgs = read_file(abspath(join(parent_d, self.catch_shp)))
        # transform it into EPSG:42106 & make the buffer
        # https://gis.stackexchange.com/a/328276/127894  (geo series into gpd)
        # wtrshd = wtrwgs.to_crs(epsg=42106)  # this code does NOT work!
        wtrshd = wtrwgs.to_crs(crs=self.wkt_prj)  # //epsg.io/42106.wkt
        # doing buffer in splitted.conjoined pols is complicated; so avoid such
        self.buffer = 0. if len(wtrshd) > 1 else self.buffer
        BUFFRX = GeoDataFrame(geometry=wtrshd.buffer(self.buffer))
        BUFFRX[self.burn_tag] = np.arange(1, len(BUFFRX) + 1)
        # infering (and rounding) the limits of the buffer-zone
        self.bbbox = {
            'l': np.floor(BUFFRX.bounds.minx.min() / self.x_res) * self.x_res,
            'r': np.ceil(BUFFRX.bounds.maxx.max() / self.x_res) * self.x_res,
            'b': np.floor(BUFFRX.bounds.miny.min() / self.y_res) * self.y_res,
            't': np.ceil(BUFFRX.bounds.maxy.max() / self.y_res) * self.y_res,
            }
        # re-arranging the BBOX differently
        sort_bbox = [self.bbbox[i] for i in ['l', 'b', 'r', 't']]
        out_proj4 = CRS.from_wkt(self.wkt_prj).to_proj4()

        # BURN THE BUFFER SHP INTO RASTER
        BUFFRX_MASK = masking.shapsterize(
            BUFFRX.to_json(), self.x_res, self.y_res, self.seed_val,
            self.burn_tag, sort_bbox, out_proj4,
            )

        # # xport it as NUMPY
        # np.save('tmp-raster_mask-buff', BUFFRX_MASK.astype('u1'),
        #         allow_pickle=True, fix_imports=True)
        # # xport it as PICKLE [but don't use it for NUMPYs!]
        # # https://stackoverflow.com/a/62883390/5885810
        # import pickle
        # with open('tmp-raster_mask-buff.pkl', 'wb') as f:
        #     pickle.dump(BUFFRX_MASK.astype('u1'), f)
        # # read it back as numpy (through pandas)
        # with open('tmp-raster_mask-buff.pkl', 'rb') as db_file:
        #     db_pkl = pickle.load(db_file)

        # # some (none matplotlib) PLOTTING
        # from rasterio.plot import show
        # from rasterio import open as ropen
        # tmp_file = 'tmp-raster.tif'
        # srcras = ropen(tmp_file)
        # fig, ax = plt.subplots()
        # ax = show(srcras, ax=ax, cmap='viridis',
        #           extent=[srcras.bounds[0], srcras.bounds[2],
        #                   srcras.bounds[1], srcras.bounds[3]])
        # srcras.close()

        # # # some xtra.TESTING
        # # # https://gis.stackexchange.com/q/344942/127894   (flipped raster)
        # # ds = gdal.Open('tmp-raster.tif')
        # # gt = ds.GetGeoTransform()
        # # if gt[2] != 0.0 or gt[4] != 0.0:
        # #     print('file is not stored with north up')

        # BURN THE CATCHMENT SHP INTO RASTER (WITHOUT BUFFER EXTENSION)
        CATCHMENT_MASK = masking.shapsterize(
            wtrshd.to_json(), self.x_res, self.y_res, self.seed_val,
            'BASIN_ID', sort_bbox, out_proj4,
            )
        # https://stackoverflow.com/a/47551616/5885810  (idx polygons intersect)
        # https://gdal.org/programs/gdal_rasterize.html
        # https://lists.osgeo.org/pipermail/gdal-dev/2009-March/019899.html (xport ASCII)
        # https://gis.stackexchange.com/a/373848/127894 (outputing NODATA)
        # https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap (targetAlignedPixels==True)

        # # if xporting the.mask as ASCII
        # # CORRECT.way (as it registers what is NODATA)
        # tmp = gdal.Rasterize(
        #     '', wtrshd.to_json(), format='MEM',
        #     xRes=self.x_res, yRes=self.y_res,  # add=0,
        #     allTouched=True, initValues=-9999., burnValues=1., noData=-9999.,
        #     outputType=gdal.GDT_Float32, targetAlignedPixels=True,
        #     outputBounds=sort_bbox, out_proj4,
        #     )
        # # CATCHMENT_MASK = tmp.ReadAsArray()
        # tmv_file = 'tmp-raster_mask.asc'
        # tmv = gdal.GetDriverByName('AAIGrid').CreateCopy(tmv_file, tmp)
        # tmv = None  # flushing!
        # tmp = None  # flushing!
        # import os
        # os.unlink(f"./{tmv_file.replace('.asc', '.prj')}")
        # # xport it as NUMPY
        # np.save('tmp-raster_mask', CATCHMENT_MASK.astype('u1'),
        #         allow_pickle=True, fix_imports=True)
        # # xport it as PICKLE [but don't use it for NUMPYs!]
        # # https://stackoverflow.com/a/62883390/5885810
        # import pickle
        # with open('tmp-raster_mask.pkl', 'wb') as f:
        #     pickle.dump(CATCHMENT_MASK.astype('u1'), f)
        # # read it back as numpy (through pandas)
        # with open('tmp-raster_mask-buff.pkl', 'rb') as db_file:
        #     db_pkl = pickle.load(db_file)

        return BUFFRX_MASK, CATCHMENT_MASK

    def coords(self, bbox):
        """
        define coordinates of the X and Y axes.\n
        Input ->
        *bbox* : dict; dictionary wiht the bounding.box limits {l, r, b, t}.\n
        Output -> tuple; 1D-numpy with local coordinates.
        """
        # define coords of the XY.AXES
        XS, YS = list(map(lambda a, b, c: np.arange(a + c / 2, b + c / 2, c),
                          [bbox['l'], bbox['b']], [bbox['r'], bbox['t']],
                          [self.x_res, self.y_res]))
        # flip YS??: so rasters are compatible with numpys
        YS = np.flipud(YS)
        return XS, YS

    def plot(self, **kwargs):
        prnt = kwargs.get('file', None)
        texc = kwargs.get('text_color', 'xkcd:acid green')
    # plot
        fig = plt.figure(figsize=(8.1, 4), dpi=150)
        # fig.subplots_adjust(left=.1, right=.1, bottom=.1, top=.1, wspace=.3)
    # plot 1: buffer
        ax = fig.add_subplot(121)
        plt.imshow(self.buffer_mask, interpolation='none', aspect='equal',
                   origin='upper', cmap='inferno',)
        ax.text(0.04, 0.96, 'buffer', ha='left', va='top', color=texc,
                transform=ax.transAxes,)
        # ax.set_xlabel('grid_cells')
        # ax.set_ylabel('grid_cells')
    # plot 1: catchment
        ax = fig.add_subplot(122)
        plt.imshow(
            self.catchment_mask, interpolation='none', aspect='equal',
            origin='upper', cmap='cividis', extent=tuple(self.bbbox.values()),
            )
        ax.text(0.04, 0.96, 'catchment', ha='left', va='top', color=texc,
                transform=ax.transAxes,)
    # show
        plt.show() if not prnt else\
            plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
                        facecolor=fig.get_facecolor())


class field:

    def __init__(self, rain_map, x_prj, y_prj, **kwargs):

        self.field = rain_map
        self.x_prj = x_prj
        self.y_prj = y_prj
        self.wkt_prj = kwargs.get('wkt_prj', WKT_OGC)
        self.resampling = kwargs.get('resampling', Resampling.nearest)
        self._void = field.empty_map(self.x_prj, self.y_prj, self.wkt_prj)
        self.field_prj = field.reproject(self.field, self._void, self.resampling)

    # void RIO.XARRAY in some local system
    @staticmethod
    def empty_map(xs, ys, WKT_OGC):
        # # coordinates from HAD.grid ??
        # ys = np.linspace(1167500., -1177500., 470, endpoint=True)
        # xs = np.linspace(1342500.,  3377500., 408, endpoint=True)
        # void numpy
        void = np.empty((len(ys), len(xs)))
        void.fill(np.nan)
        xr_void = xr.DataArray(
            data=void, dims=['y', 'x'],
            # name='void',
            coords=dict(y=(['y'], ys), x=(['x'], xs),),
            attrs=dict(_FillValue=np.nan, units='mm',),
            )
        # xr_void = xr.DataArray(
        #     data=void, name='rain', dims=['time', 'lat', 'lon'],
        #     coords=dict(
        #         time=(['time'], np.r_[1000, 2000, 3000, 4000, 5000]),
        #         lat=(['lat'], np.r_[1, 2, 3, 4, 5, 6, 7]),
        #         lon=(['lon'], np.r_[1, 2, 3]),
        #         ),
        #     attrs=dict(_FillValue=np.nan, units='mm',),
        #     )
        # assign CRS
        # xr_void.rio.write_crs(rio.crs.CRS(WKT_OGC),
        #                       grid_mapping_name='spatial_ref', inplace=True)
        xr_void.rio.write_crs(WKT_OGC, grid_mapping_name='spatial_ref', inplace=True)
        # # IF xported
        # xr_void.to_netcdf('./void.nc', mode='w',
        #     # encoding={'void': {
        #     #     'dtype': 'f8', 'zlib': True, 'complevel': 9,
        #     #     'grid_mapping': 'spatial_ref'}, }
        #     encoding={'__xarray_dataarray_variable__': {
        #         'dtype': 'f8', 'zlib': True, 'complevel': 9,
        #         'grid_mapping': 'spatial_ref'}, }
        #     )
        return xr_void

    @staticmethod
    def reproject(src_xr, dst_xr, resampling_method):
        # xile = src_xr.rio.reproject(self.wkt_prj)
        # xile = xile.rio.reproject_match(dst_xr, resampling=resampling_method)
        pile = src_xr.rio.reproject_match(dst_xr, resampling=resampling_method)

        # # some.VISUALISATION (assuming RAIN is the variable!!)
        # import cmaps
        # # from cmcrameri import cm as cmc
        # cmaps.precip2_17lev
        # # cmaps.wh_bl_gr_ye_re
        # # cmaps.WhiteBlueGreenYellowRed
        # seas.rain.plot(cmap='precip2_17lev', levels=10, vmin=100, vmax=1000,
        #                add_colorbar=True,  # robust=True,
        #                )
        # pile.rain.plot(cmap='precip2_17lev', levels=10, vmin=100, vmax=1000,
        #                add_colorbar=True,  # robust=True,
        #                )
        # # xport.it as NUMPY
        # np.save('./realisation', pile.rain.data,
        #         allow_pickle=True, fix_imports=True)
        # # xport it as PICKLE [but don't use it for NUMPYs!]
        # # https://stackoverflow.com/a/62883390/5885810
        # import pickle
        # with open('./realisation.pkl', 'wb') as f:
        #     pickle.dump(pile.rain.data, f)

        # pile.to_netcdf(
        #     'realisation_had.nc', mode='w', engine='netcdf4', encoding={
        #         'rain': {'dtype': 'f4', 'zlib': True, 'complevel': 9,
        #                  'grid_mapping': pile.rio.grid_mapping},
        #         'mask': {'dtype': 'u1', 'grid_mapping': pile.rio.grid_mapping,
        #                  # '_FillValue': 0
        #                  },
        #         })

        return pile


# %% region k-means

class regional:

    def __init__(self, map_prj, mask_bfr, mask_cat, **kwargs):

        self.up_dic = None
        self.wkt_prj = kwargs.get('wkt_prj', WKT_OGC)
        self.n_ = kwargs.get('nr', 1)
        self.statistic = kwargs.get('stat', 'mean')
        self.realization = map_prj
        self.buffer = mask_bfr
        self.catchment = mask_cat
        if np.isin(self.buffer, np.array([0, 1])).all():
            # this means there are NO.partitions/regions in buffer!
            self.regions, self.nr_dic = self.k_regions(
                self.realization, self.catchment, self.n_)
            self.morph = self.morphopen(self.regions)
            # self.morph = self.regions
            self.gshape = self.to_shp(self.morph, self.realization, self.buffer,
                                      self.catchment, self.wkt_prj)
        else:
            # this means that k-means should NOT be computed!
            self.n_ = np.unique(self.buffer).size - 1  # 0 doesn't count
            self.regions, self.nr_dic = self.n_regions(
                self.realization, self.catchment, self.buffer, self.statistic)
            self.morph = None
            self.gshape = self.to_shp(self.regions, self.realization, self.buffer,
                                      self.catchment, self.wkt_prj)

    def n_regions(self, r_eal, catch, buff, statistic):
        """
        computes the n-statistic over the field.\n
        Input ->
        *r_eal* : 2D.numpy; rainfall.field [realization].
        *catch* : 2D.numpy; catchment/region.
        *buff* : 2D.numpy; catchment/region [previously regionalized].
        *statistic*: str; nan.statistic to compute (default 'mean').\n
        **kwargs\n
        Output -> 2D.numpy with splitted regions; and dic with n_means.per.region.
        """

        # # FOR A MASK IN THE WHOLE [RECTANGULAR] DOMAIN
        # reg = np.ma.MaskedArray(r_eal.rain.data.copy(), False)
        # FOR A MASK IN THE WHOLE OF THE CATCHMENT
        reg = np.ma.MaskedArray(r_eal.rain.data.copy(), ~catch.astype('bool'))
        # # ... if BAND was NOT removed!
        # reg = np.ma.MaskedArray(
        #     r_eal.rain.rain['band'==1, :].data.copy(), ~catch.astype('bool'))

        # nans outside mask
        reg[reg.mask] = np.nan

        LAB = buff.astype('f4')
        LAB[LAB == 0] = np.nan
        # # test (to check if it is! the sought region)
        # rog = reg.copy()
        # rog[LAB != 1] = np.nan

        reg_tag = np.unique(LAB)
        reg_tag = reg_tag[~np.isnan(reg_tag)]
        kcentr = []
        for x in reg_tag:
        # eval don't work inside class:  https://stackoverflow.com/a/61498252/5885810
            kcentr.append(np.array([eval(f'np.nan{statistic}(reg[LAB == {x}])')]))
        KAT = list(map(str, reg_tag.astype('u2')))

        return LAB, dict(zip(KAT, kcentr))


    def sort_kmeans(self, k_means):
        # ONLY LABELS & CENTERS are SORTED!
        old_klabel = k_means.labels_
        old_kcentr = k_means.cluster_centers_
        # https://stackoverflow.com/a/35464758/5885810
        # from_ = np.unique(old_klabel)
        # to_ = np.argsort(old_kcentr.ravel())
        to_ = np.unique(old_klabel)
        from_ = np.argsort(old_kcentr.ravel())
        # d = dict(zip(from_, to_))
        sort_idx = np.argsort(from_)
        idx = np.searchsorted(from_, old_klabel, sorter=sort_idx)
        new_klabel = to_[sort_idx][idx]
        # sort centers
        new_kcentr = old_kcentr[from_]
        return new_klabel, new_kcentr

    # IMAGE SEGMENTATION via SCIKIT-LEARN
    # following: https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb
    # -----------------------------------
    def k_regions(self, r_eal, catch, n_c):
        # r_eal = rain_.field_prj; catch = space.catchment_mask; n_c = 4
        """
        split the field into *n_c* k-means.\n
        Input:\n
        r_eal : 2D.np; rainfall.field [realization].
        catch : 2D.np; catchment/region.
        n_c : int; number of clusters.\n
        Output -> 2D.numpy with splitted regions; and dic with k_means.per.region.
        """

        # # FOR A MASK IN THE WHOLE [RECTANGULAR] DOMAIN
        # reg = np.ma.MaskedArray(r_eal.rain.data.copy(), False)
        # FOR A MASK IN THE WHOLE OF THE CATCHMENT
        reg = np.ma.MaskedArray(r_eal.rain.data.copy(), ~catch.astype('bool'))
        # # ... if BAND was NOT removed!
        # reg = np.ma.MaskedArray(
        #     r_eal.rain.rain['band'==1, :].data.copy(), ~catch.astype('bool'))

        # nans outside mask
        reg[reg.mask] = np.nan
        # ravel and indexing
        ravl = reg.ravel()
        idrs = np.arange(len(ravl))[~np.isnan(ravl)]
        # transform the non-void (RGB?) field into 1D.numpy
        X = ravl[idrs].data.reshape(-1, 1)
        # kmeans = KMeans(n_clusters=3, init=np.array([[70],[220],[800]]), n_init='auto').fit(X)
        # kmeans = KMeans(n_clusters=n_c, n_init=11, random_state=None).fit(X)
        kmeans = KMeans(n_clusters=n_c, n_init=11, random_state=42).fit(X)

        klabel = kmeans.labels_
        kcentr = kmeans.cluster_centers_
        # IF SORTING THE CENTERS (from low to high)
        klabel, kcentr = self.sort_kmeans(kmeans)

        # expand the result into void-array
        ravl[idrs] = klabel
        LAB = ravl.reshape(reg.shape).data
        # # some plotting
        # import matplotlib.pyplot as plt
        # plt.imshow(LAB, origin='upper', cmap='turbo', interpolation='none')
        # # plt.savefig('realization.pdf', bbox_inches='tight', pad_inches=0.02)
        # # plt.close()
        # # plt.clf()

        # https://stackoverflow.com/a/25715954/5885810  # np.object to np.string
        # https://www.w3resource.com/numpy/string-operations/strip.php  # strip np.string.arrays
        KAT = np.char.strip(kmeans.get_feature_names_out().astype('U'), 'kmeans')

        return LAB, dict(zip(KAT, kcentr))

        # # TESTING IMAGE.SEGMENTATION (from SCIKIT-LEARN)
        # # from: https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb
        # import urllib.request
        # from matplotlib.image import imread
        # from cmcrameri import cm as cmc
        # # testing LADYBUG
        # images_path = "."
        # DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
        # filename = "ladybug.png"
        # print("Downloading", filename)
        # url = DOWNLOAD_ROOT + "images/unsupervised_learning/" + filename
        # urllib.request.urlretrieve(url, os.path.join(images_path, filename))

        # image = imread(join(images_path, filename))
        # image.shape

        # X = image.reshape(-1, 3)
        # kmeans = KMeans(n_clusters=3, n_init=11, random_state=42).fit(X)
        # segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        # segmented_img = segmented_img.reshape(image.shape)
        # plt.imshow(segmented_img)

        # meansea = np.flip(rain_.field_prj.rain, axis=0)
        # X = meansea.data.reshape(-1, 1)
        # kmeans = KMeans(n_clusters=4, n_init=11).fit(X)
        # segmented_img = kmeans.cluster_centers_[kmeans.labels_]
        # segmented_img = segmented_img.reshape(meansea.data.shape)
        # plt.imshow(segmented_img, origin='lower', cmap=cmc.hawaii_r)

    # MORPHOLOGICAL FILTERING via SCIKIT-IMAGE
    # https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html#opening
    def morphopen(self, LAB):
        # [2023.09.04]: self.morphopen does NOT deal (for now) with NAs, so:
        # LABalt = LAB
        num = 100
        LABalt = np.nan_to_num(LAB, nan=num)
        # # "morphology.ellipse(2, 3)" does almost the same job as
        # # "cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 7))"
        new = morphology.opening(LABalt,
                                 # morphology.ellipse(2, 3)
                                 morphology.ellipse(4, 5)
                                 )
        new[new == num] = np.nan
        # plt.imshow(new, cmap='turbo', interpolation='none'); plt.colorbar()
        # # plt.savefig('realization_opening.pdf',
        # #             bbox_inches='tight', pad_inches=0.02)
        # # plt.close()
        # # plt.clf()
        # return new.astype('u1')
        return new

        # mopen = test.regions; realization = rain_.field_prj; wkt_prj = WKT_OGC
        # buffer_mask = space.buffer_mask; catchm_mask = space.catchment_mask
    def to_shp(self, mopen, realization, buffer_mask, catchm_mask, wkt_prj):
        # np2shp [.rio.transform() IS QUITE OF THE ESSENCE HERE!]
        lopen = list(shapes(mopen, mask=buffer_mask, connectivity=4,
                            transform=realization.rio.transform()))
        # lopen = list(shapes(mopen, mask=space.buffer_mask, connectivity=4,
        #                     transform=rain_.field_prj.rio.transform()))
        # remove NAN.regions??
        # https://stackoverflow.com/a/25050572/5885810
        # https://stackoverflow.com/a/3179137/5885810
        lopen = [x for x, y in zip(lopen, ~np.isnan(list(zip(*lopen))[-1])) if y]
        lopen = list(map(
            lambda x: dict(geometry=x[0], properties={
                'region': f'{int(x[-1])}', }), lopen))
        # into GEOPANDAS
        feats = GeoDataFrame.from_features(
            {'type': 'FeatureCollection', 'features': lopen})
        # # do we obtain HAD if merging all masks?
        # feats.union_all()

        # grouping to retrieve just the CLUSTER.masks (the output is a Series)
    # nasks = feats.groupby(by='region').apply(lambda x: x.unary_union, include_groups=False)
        nasks = feats.groupby(by='region').apply(
            lambda x: x.union_all(), include_groups=False)
        # nasks[0] ; nasks[1] ; nasks[2] ; nasks[3]

        # turn-back them into GeoPandas
        masks = GeoDataFrame(geometry=nasks, crs=wkt_prj)
        # masks.geometry.iloc[0]
        # masks.geometry.loc[0]
        # masks.loc[0].geometry
        # masks.geometry.xs(0)
        masks['area_m'] = masks.area

        # project to WGS84
        wasks = masks.to_crs(crs='EPSG:4326')
        # turn index into column
        wasks.reset_index(inplace=True)
        # compute updated seasonal rain
        self.up_dic = self._update_means(realization.rain, mopen, wasks.region)
        wasks['u_rain'] = np.asarray(list(self.up_dic.values())).ravel()
        wasks['k_rain'] = np.asarray(list(self.nr_dic.values())).ravel()
        return wasks

    def _update_means(self, mapa, up_reg, reg_lab):
        # zone = list(reg_lab.keys())
        zone = reg_lab
        new_ = [mapa.where(up_reg == i, other=np.nan).mean().data for i in
                np.array(zone, dtype='u2')]
        return dict(zip(zone, new_))

    def xport_shp(self, **kwargs):
        drvr = kwargs.get('driver', 'ESRI Shapefile')
        lyr = kwargs.get('layer', None)
        prnt = kwargs.get('file', ZON_FILE)
        self.gshape.to_file(prnt, driver=drvr, layer=lyr)
        return


# %% mask  clipping

class region:
    def __init__(self, shp=None):

        assertdir = 'Wrong argument passed!\n'\
            'You are trying to pass shapefile that may not exist '\
            'or it is not correctly addressed (i.e. wrong path).'
        assertshp = 'Wrong argument passed!\n'\
            'You are trying to pass as shapefile something that is '\
            'neither a proper path nor a geopandas.dataframe.'

        try:
            if isinstance(shp, GeoDataFrame):
                self.path = None
                self.region = shp
            else:
                if isinstance(shp, str):
                    self.path = abspath(shp)
                elif shp is None:
                    self.path = abspath(SHP_FILE)
                if not exists(self.path):
                    raise AssertionError(assertdir)
                self.region = read_file(self.path)
        except AttributeError:
            raise AssertionError(assertshp)


class clipping:
    # ??having done: from parameters import *
    def __init__(self, xset, area, **kwargs):

        self.full_set = xset
        self.clip_area = area
        self.land_frac = kwargs.get('lf', 0.9)
        self.drop_bool = kwargs.get('drop', True)
        self.clip_set, self._inset, self._outset = clipping.clip_trac(
            self.full_set, self.clip_area, self.land_frac, self.drop_bool)

    # xset=xr.open_dataset(EVENT_DATA, chunked_array_type='dask', chunks='auto')
    # area=region(shp=areas.gshape.iloc[[0]]).region  ; kut = 0.9
    @staticmethod
    def clip_trac(xset, area, kut, drop):
        # remove tracks not quite (90% or less) on land
        lf = xset.pf_landfrac.load()
        li = lf.where(lf == 1, np.nan).count(dim='times') / lf.count(dim='times')
        li = li[li >= kut].tracks
        xset = xset.sel(tracks=li)
        # find which lat.lon are not void [do NOT use NP.WHERE]
        # https://stackoverflow.com/a/70266407/5885810 (not used anymore)
        xx = xset.pf_lon[:, :, 0].stack(x=('tracks', 'times',)).load()
        yy = xset.pf_lat[:, :, 0].stack(y=('tracks', 'times',)).load()
        ii = np.intersect1d(np.arange(len(xx))[xx.notnull()],
                            np.arange(len(yy))[yy.notnull()])
        # intersect the points with the shape
        mcs_p = GeoDataFrame(
            geometry=points_from_xy(x=xx[ii], y=yy[ii]),
            crs=f'EPSG:{xset.rio.crs.to_epsg()}')  # 'EPSG:4326'
        in_ix = sjoin(mcs_p, area, how='inner', predicate='within').index
        no_ix = np.setdiff1d(mcs_p.index, in_ix)
        # removing data outside area
        xx[ii[no_ix]] = np.nan
        yy[ii[no_ix]] = np.nan
        tx = xx.unstack(dim='x').dropna(dim='tracks', how='all').tracks
        ty = yy.unstack(dim='y').dropna(dim='tracks', how='all').tracks
        tt = np.intersect1d(tx, ty)
        del xx, yy, ii, tx, ty
        if drop is True:
            xset = xset.sel(tracks=tt)
            # clipping indices
            tid_ = xset.pf_lon.dropna(dim=('times'), how='all').times
            xset = xset.sel(times=tid_)
            pfs_ = xset.pf_lon.dropna(dim=('nmaxpf'), how='all').nmaxpf
            xset = xset.sel(nmaxpf=pfs_)
        collect()
        return xset, mcs_p.loc[in_ix], mcs_p.loc[no_ix]

        # # map the nans to all other vars
        # var2d = [
        #     'area',
        #     # 'pf_npf',
        #     'total_rain',
        #     # 'pf_landfrac',
        #     'movement_speed',
        #     'movement_theta'
        #     ]
        # var3d = [
        #       'pf_area',
        #       'pf_rainrate',
        #       'pf_accumrain',
        #       'pf_maxrainrate',
        #       # 'pf_lon_centroid',
        #       # 'pf_lat_centroid',
        #     ]
        # xset.update(xset[var2d].where(xset['pf_lon_centroid'].notnull(), np.nan))
        # xset.update(xset[var3d].where(xset['pf_lon_centroid'].notnull(), np.nan))
        # # dosd = xset['pf_lon_centroid']
        # # xset.update(xset[var2d] * dosd / dosd)
        # # xset.update(xset[var3d] * dosd / dosd)

    def plot(self, **kwargs):
        prnt = kwargs.get('file', None)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
        self.clip_area.plot(edgecolor='darkgreen', facecolor='none',
                            lw=.7, zorder=1, ax=ax)
        self._inset.plot(marker='.', markersize=.1, color='orange',
                         zorder=2, ax=ax)
        self._outset.plot(marker='x', markersize=.1, color='lightblue',
                          zorder=0, ax=ax)
        plt.show() if not prnt else\
            plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
                        facecolor=fig.get_facecolor())


# %% circular class

def mode_(vec):
# https://stackoverflow.com/a/16331189/5885810
    l_vec = len(vec.movement_theta)
    u_vec = stats.mode(vec.movement_theta)
    # u_vec = stats.mode(vec.movement_theta.sample(frac=1))
    return pd.Series({'s':l_vec, 'mode':u_vec.mode, 'per':u_vec.count / l_vec})


def one_vm(trac):  # trac = mdir[2222,:].compute()
    trac = trac[np.isfinite(trac)]  # https://www.statology.org/numpy-remove-nan/
    # trac = np.concatenate((trac[np.nonzero(np.nan_to_num(trac))], trac[trac==0]))
    case = np.unique(trac)
    # if len(case) >= 3:
    if len(case) >= 2 and len(trac) >= 3:
        nu = stats.vonmises.fit(trac)[1]
        # # DO.NOT USE the function below!! (it's too slow for large arrays)
        # parvm = stats.fit(stats.vonmises, one_trac,
        #     bounds={'kappa':(0,500), 'loc':(-np.pi,np.pi)}).params.loc
    else:
        nu = np.nan
    return nu


class circular:

    def __init__(self, data, **kwargs):
        """
        fitting, random sampling, plotting and whatnots? for circular stats.\n
        Input ->
        *data* : dict/numpy; either a dict with parameters or numpy with data.\n
        **kwargs ->
        data_type : str; 'tod', 'doy','dir' or 'rad' (only for *data*==numpy).
        max_mix : int; maximum number of vMF distros to fit.
        met_cap : float; capping percentage of the lowest/total criterion.
        criterion : char; selection of optimal fit ('BIC' or 'AIC').\n
        Output -> a class random sampled (spatial) points.
        """

        self.data = data
        self.transform_pars = None
        self._transform_dic = {
            'tod': {'add': 0, 'scl':  24, 'non': 0},
            'doy': {'add': 1, 'scl': 366, 'non': 0},
            'dir': {'add': 0, 'scl': 360, 'non': 0},
            'rad': {'add': 0, 'scl': 2*np.pi, 'non': np.pi},
            }

        if type(self.data) is dict:
            self.data_type = 'rad'
            self.transform_pars = self._transform_dic.get(self.data_type)
            self.phi_ = self.data['phi']
            self.kappa_ = self.data['kappa']
            self.alpha_ = self.data['alpha']
        elif isinstance(self.data, np.ndarray):
            self.data_type = kwargs.get('data_type', None)
            if self.data_type is None:
                raise TypeError("Data structure not passed!\nPass 'tod' or 'doy'"
                                " or 'dir' or 'rad' to the 'data_type' argument.")
            self.transform_pars = self._transform_dic.get(self.data_type)
            self.data = self._to_rad(self.data, self.transform_pars)
            self.max_mix = kwargs.get('max_mix', 9)
            self.met_cap = kwargs.get('met_cap', 0.9)
            self.criterion = kwargs.get('criterion', 'BIC')
            self._criteria, self.opt_mix, self.model_0 = None, None, None
            # fit n models & pick up the optimum
            self._fit_n_models(self.data, self.max_mix)
            # parameters initial set up
            self.phi_ = self.model_0.means_.ravel()
            self.kappa_ = self._estimate_kappa(self.model_0, self.data)
            self.alpha_ = self.model_0.weights_
            self.mix_str = None
            self._vmtab = self.fit()
        else:
            raise TypeError("Invalid data type!\nErroneous data type passed. "
                            "Only DICTIONARY or NUMPY accepted.")

    def _from_rad(self, x, _pars):
        xt = _pars['add'] +\
            ((x + np.pi - _pars['non']) / (2 * np.pi) * _pars['scl'])
        return xt

    def _to_rad(self, xt, _pars):
        x = (xt - _pars['add']) * (2 * np.pi) / _pars['scl'] +\
            _pars['non'] - np.pi
        return x

    def samples(self, N_=629, **kwargs):
        """
        generates a random sample for the mixture of von Mises distributions.\n
        Input ->
        *N_* : int; number of points in the random sample.\n
        **kwargs ->
        phi_ : np.float; parameter phi (or mu) of each von Mises.
        kappa_ : np.float; parameter kappa of each von Mises.
        alpha_ : np.float [0-1]; relative probability of each von Mises.
        data_type : char; type of output (i.e., 'tod', 'doy' or 'dir').\n
        Output -> array of size 'N_'.
        """
    # set up & update **kwargs
    # https://stackoverflow.com/a/1098556/5885810
    # https://thepythoncodingbook.com/2022/11/30/what-are-args-and-kwargs-in-python/
        phi_ = kwargs.get('phi_', self.phi_)
        kappa_ = kwargs.get('kappa_', self.kappa_)
        alpha_ = kwargs.get('alpha_', self.alpha_)
        data_type = kwargs.get('data_type', self.data_type)
        if data_type not in ['tod', 'doy', 'dir', 'rad']:
            raise TypeError("Data structure not passed!\nPass 'tod' or 'doy'"
                            " or 'dir' or 'rad' to the 'data_type' argument.")
        transform_pars = self._transform_dic.get(data_type)
    # https://framagit.org/fraschelle/mixture-of-von-mises-distributions
    # taken from: ...\site-packages\vonMisesMixtures\tools.py
        rng = np.random.default_rng()
        s = rng.choice(range(len(phi_)), size=N_, p=alpha_, replace=True)
        v, c = np.unique(s, return_counts=True)
        sample = np.concatenate(list(map(
            lambda a1, a2, n: stats.vonmises.rvs(loc=a1, kappa=a2, size=n),
            phi_, kappa_, c)))
        sample_transform = self._from_rad(sample, transform_pars)
        return sample_transform

    def plot_samples(self, **kwargs):
        """
        plot a histogram with the optimal von Mises-mix model.\n
        **kwargs ->
        n : int; number of samples.
        data : np.array; vector of random samples.
        data_type : str; 'tod' or 'doy' or 'dir' or 'rad'.
        bins : int; bins for histogram.
        file : str; name of the output-plot.\n
        Output -> plot (matplotlib.figure or exported file).
        """
    # reading/assigning kwargs
        n_ = kwargs.get('n', 1000)
        hist_data = kwargs.get('data', None)
        data_type = kwargs.get('data_type', self.data_type)
        transform_pars = self._transform_dic.get(data_type)
        bins = kwargs.get('bins', 80)
        prnt = kwargs.get('file', None)

        import matplotlib.pyplot as plt
        plt.rc('font', size=11)
    # define model
        mod_n = 'vmix'
        _, _, _, str_0 = circular.model_str(len(self.phi_), mod_n)
        exec(str_0)
        str_1 = ', '.join(np.concatenate(
            (self.phi_, self.kappa_, self.alpha_)).astype('str'))
    # estimate values
        # # ADD + ((_rad + np.pi - NON) / (2 * np.pi) * SCL) = _data
        # # ((_data - ADD) / SCL * (2 * np.pi) + NON - np.pi) = _rad
        # xs_rad = self._to_rad(
        #     np.sort(self.samples(N_=n_, data_type=data_type)), transform_pars)
        xs_rad = np.linspace(-np.pi, np.pi, num=n_)  # , endpoint=False)
        ys = eval(f'{mod_n}(xs_rad, {str_1})')
    # plot (update 'hist_data' first)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.vonmises.html
        hist_data = self.data if isinstance(self.data, np.ndarray) else\
            self._to_rad(hist_data, transform_pars)
        # bins = int(np.sqrt(len(hist_data)))  # a better approach of BINS?
        x_ticks = np.linspace(-np.pi, np.pi, 9)
        x_label = self._from_rad(x_ticks, transform_pars)
        fig = plt.figure(figsize=(12, 5), dpi=200)
        left = plt.subplot(121)
        rigt = plt.subplot(122, projection='polar')
    # The left image contains the Cartesian plot
        left.hist(hist_data, bins=bins, color='xkcd:apple',
                  alpha=.7, label='samples', density=True)
        left.plot(xs_rad, ys, color='xkcd:blood orange',
                  lw=1.7, label='von Mises\nmixture')
        left.legend(loc='best')
        left.set_xticks(x_ticks, labels=list(map(lambda x: '{:.0f}'.format(x), x_label)))
        left.set(xlim=[-np.pi, np.pi], xlabel=data_type, ylabel='$p(x)$')
        # left.grid(True)
        left.grid(which='major', alpha=0.5)
        # left.set_title('Cartesian plot')
    # The right image contains the polar plot
        rigt.hist(hist_data + np.pi, bins=bins, color='xkcd:neon blue',
                  alpha=.5, label='histogram', density=True)
        rigt.plot(xs_rad + np.pi, ys, color='xkcd:electric pink',
                  lw=1.9, label='von Mises\nmixture')
        rigt.legend(bbox_to_anchor=(0.18, 0.08))
    # https://stackoverflow.com/q/67282865/5885810
        rigt.set_ylim(-.1, left.get_ylim()[-1])
    # https://stackoverflow.com/a/20416681/5885810
    # https://stackoverflow.com/a/24953575/5885810
        rigt.set_yticklabels([])
        rigt.grid(which='major', alpha=0.5)
    # https://www.tutorialspoint.com/how-to-make-the-angles-in-a-matplotlib-polar-plot-go-clockwise-with-0-at-the-top
        rigt.set_theta_direction(-1)
        rigt.set_theta_offset(np.pi * 1 / 2.)
        # rigt.set_xticks(x_ticks + np.pi, labels=x_label.round(2))
        polar_ticks = np.linspace(-np.pi, np.pi, 12, endpoint=False)
        rigt.set_xticks(polar_ticks + np.pi, labels=list(map(
            lambda x: '{:.0f}'.format(x), self._from_rad(polar_ticks, transform_pars))))
        # rigt.set_title('Polar plot')
        plt.show() if not prnt else\
            plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
                        facecolor=fig.get_facecolor())

    def _min_bic(self, bic, cap):
    # cap = 0.95  # percentage of the rel.dif for which the min.bic is ok
    # transform it first into numpy
        bic = np.array(list(bic.values()))
    # starting from left (we want models with the least pars) cut everything above bic[0]
        bic[bic > bic[0]] = bic[0]
    # make it relative
        bic = (bic - bic.min())
    # https://stackoverflow.com/a/51432592/5885810
        if np.all(np.isclose(bic, 0.)):
            bpos = -1
            bpos = 0
        else:
            bic = bic / bic.max()
            bpos = (np.diff(bic).cumsum() < -cap).nonzero()[0][0]
        # return int(bpos + 1)
        return bpos

    def _fit_n_models(self, data, N):
        X = data.reshape(-1, 1)
    # fit models with 1-max_mix components
        models = list(map(lambda i:
                          GaussianMixture(i).fit(X), np.arange(1, N + 1)))
    # compute the AIC and the BIC
        AIC = dict(map(dict.popitem, [{m.n_components: m.aic(X)} for m in models]))
        BIC = dict(map(dict.popitem, [{m.n_components: m.bic(X)} for m in models]))
    # filling some variables
        criteria = dict(BIC=BIC, AIC=AIC)
        tmp_n = self._min_bic(criteria[self.criterion], self.met_cap)
        self.opt_mix = list(criteria[self.criterion].keys())[tmp_n]
        self.model_0 = models[tmp_n]
        self._criteria = criteria

    def _estimate_kappa(self, M_best, X):
        Xp = X.reshape(-1, 1)
        responsibilities, pdf = circular.compute_probs(M_best, Xp)
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        mu = M_best.means_.ravel()
        cs = np.cos(X)@(pdf_individual*np.cos(mu)) + np.sin(X)@(pdf_individual*np.sin(mu))
        k0 = lambda kappa: (cs - special.iv(1, kappa) / special.iv(0, kappa) *
                            np.sum(pdf_individual, axis=0)).reshape(len(cs))
        kappas = optimize.fsolve(k0, np.zeros(len(cs)))
        return kappas

    @staticmethod
    def compute_probs(M_best, Xp):
        logprob = M_best.score_samples(Xp)
        responsibilities = M_best.predict_proba(Xp)
        pdf = np.exp(logprob)
        return responsibilities, pdf

    @staticmethod
    def model_str(n_mix, mod_n):
        nu = list(map(lambda x: f'phi_{x + 1}', range(n_mix)))
        ka = list(map(lambda x: f'kappa_{x + 1}', range(n_mix)))
        w_ = list(map(lambda x: f'alpha_{x + 1}', range(n_mix)))
        str_1 = ', '.join(np.concatenate([['x'], nu, ka, w_]))
        str_2 = ' + '.join(list(map(lambda n, k, w:
            f'{w} * stats.vonmises.pdf(x, loc={n}, kappa={k})', nu, ka, w_)))
        return nu, ka, w_, f'def {mod_n}({str_1}): return {str_2}'

# https://www.lancaster.ac.uk/staff/drummonn/PHYS281/demo-classes/
    @staticmethod
    def find_best_parameters(data, **kwargs):
        """
        estimates the optima parameters for a von Mises model of n-mixtures.\n
        Input: np.array.\n
        **kwargs ->
        best_model : *sklearn.mixture._gaussian_mixture.GaussianMixture*.
        n_mix      : int; must be provided if *best_model* is not.
        phi_       : np.float; parameter phi (or mu) of each von Mises.
        kappa_     : np.float; parameter kappa of each von Mises.
        alpha_     : np.float [0-1]; relative probability of each von Mises.\n
        Output -> list with pd.DataFrame (optima pars) and str.model
        """
    # initial reading
        data_p = data.reshape(-1, 1)  # one may use this caopy later
        best_model = kwargs.get('best_model', None)
        n_mix = kwargs.get('n_mix', None)
    # how many mixtures?
        if best_model is None:
            n_mix = n_mix
            if n_mix is None:
                raise TypeError("you must supply the number of mixtures 'n_mix'")
        else:
            # variation = False
            n_mix = best_model.n_components
        # # i didn't like this approach (taking all data... doesn't help??)
        #     _, probs = circular.compute_probs(best_model, data_p)
        #     data = data_p
    # continue reading/assigning kwargs
        phi_ = kwargs.get('phi_', np.linspace(-np.pi, np.pi, n_mix + 2)[1:-1])
        kappa_ = kwargs.get('kappa_', np.ones(n_mix))
        alpha_ = kwargs.get('alpha_', np.repeat(1 / n_mix, n_mix))

    # compute the probs from a histogram (and modify data accordingly)
        probs, data = np.histogram(data, bins=24*4*3*1, density=True)
        data = (data[1:] + data[:-1]) / 2
        variation = True

    # define variables and strings to build up the model from
        eps = 1e-5
        mod_n = 'vm_mixture'
        nu, ka, w_, str_0 = circular.model_str(n_mix, 'vm_mixture')
        exec(str_0)
        fmodel = eval(f'Model({mod_n})')
    # define model parameters
        params = Parameters()
        list(map(lambda w, v: params.add(w, value=v, min=-np.pi, max=np.pi,
                                         vary=variation), nu, phi_.ravel()))
        list(map(lambda w, v: params.add(w, value=v, min=0, vary=variation),
                 ka, kappa_.ravel()))
        params.add('epsilon', value=eps, min=0, max=1e-9, vary=True)
        list(map(lambda w, v: params.add(w, value=v, min=0, max=1, vary=True),
                 w_[:-1], alpha_[:-1].ravel()))
        params.add(w_[-1], expr=f'1-{"-".join(w_[:-1])}+epsilon')
    # fit the model
        result = fmodel.fit(probs, params, x=data)
        # print(result.fit_report(show_correl=False))
        # # sum([result.values[x] for x in w_])
    # parameteres stored in a pandas
        fit_params = pd.DataFrame(
            index=[f'm{str(i+1)}' for i in range(n_mix)],
            data={'alpha': [result.values[x] for x in w_],
                  'phi': [result.values[x] for x in nu],
                  'kappa': [result.values[x] for x in ka]})
    # return model.string and pars
        return fit_params, str_0

    def fit(self,):
        parameters, self.mix_str = circular.find_best_parameters(
            self.data, best_model=self.model_0, phi_=self.phi_,
            kappa_=self.kappa_, alpha_=self.alpha_)
    # updating the parameters in the class
        self.phi_ = parameters.phi.values
        self.kappa_ = parameters.kappa.values
        self.alpha_ = parameters.alpha.values
        return parameters

    def save(self, file, region, tag):
        with open(file, 'a') as f:
            [f.write(f"R{region}+{tag}_VMF+m{x+1},{','.join(map(str, [*xtem]))}\n")
             for x, xtem in enumerate(self._vmtab.to_numpy())]

    def plot_bic(self, **kwargs):
        """
        3-plot panel with the optimum von Mises-mix and AIC/BIC analysis.\n
        **kwargs ->
        file : str; name of the output-plot.
        """
        from sys import platform
        import matplotlib.pyplot as plt
        plt.rc('font', size=11)
        from matplotlib import colormaps as mcl
    # https://stackoverflow.com/a/62837117/5885810
        try:
            from astroML.plotting import setup_text_plots
            setup_text_plots(fontsize=11,
                             usetex=True if platform == 'linux' else False)
        except ModuleNotFoundError:
            pass
    # reading/assigning kwargs
        prnt = kwargs.get('file', None)
        M_best = self.model_0
        XD = self.data
        N = np.arange(1, self.max_mix + 1)
        AIC = list(self._criteria['AIC'].values())
        BIC = list(self._criteria['BIC'].values())
    # plot
        fig = plt.figure(figsize=(11.6, 4), dpi=200)
        fig.subplots_adjust(left=0.1, right=0.8, bottom=0.2, top=0.8, wspace=0.3)
    # plot 1: data + best-fit mixture
        ax = fig.add_subplot(131)
        xlim = np.abs(np.r_[np.floor(XD.min()), np.ceil(XD.max())]).max()
        x = np.linspace(-xlim, xlim, 1000)
        responsibilities, pdf = circular.compute_probs(M_best, x.reshape(-1, 1))
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        ax.hist(XD, 30, density=True, histtype='stepfilled', alpha=0.4)
        ax.plot(x, pdf, '-k')
        ax.plot(x, pdf_individual, '--k')
        ax.text(0.04, 0.96, 'Best-fit\nmixture', ha='left', va='top',
                transform=ax.transAxes)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')
    # plot 2: AIC and BIC
        ax = fig.add_subplot(132)
        ax.plot(N, AIC, ls='dashed', label='AIC', color='xkcd:electric pink')
        ax.plot(N, BIC, ls='solid', label='BIC', color='xkcd:navy blue')
        ax.set_xlabel('n. components')
        ax.set_ylabel('information criterion')
        ax.legend(loc='best')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # plot 3: posterior probabilities for each component
        ax = fig.add_subplot(133)
        p = responsibilities
        # rearrange order so the plot looks better??
        p = p[:, np.argsort(M_best.means_.ravel())]
        p = p.cumsum(1).T
        cm = mcl['tab20']
        gray = 1 / M_best.n_components
        for i in range(M_best.n_components - 1):
            if i == 0:
                ax.fill_between(x, 0, p[0], color=cm(i),)
            elif i < M_best.n_components - 1:
                ax.fill_between(x, p[i-1], p[i], color=cm(i),)
        ax.fill_between(x, p[i], 1, color=cm(i+1),)
        # ax.set_xlim(-xlim, xlim)
        # ax.set_ylim(0, 1)
        # ax.set_xlabel('$x$')
        # ax.set_ylabel(r'$p({\rm class}|x)$')
        ax.set(xlim=[-xlim, xlim], ylim=[0, 1],
               xlabel='$x$', ylabel=r'$p({\rm class}|x)$')
        for i, x in enumerate(np.sort(M_best.means_.ravel())):
            ax.text(x, 0.3, f'mix {i+1}', rotation='vertical', ha='center', va='center')
        plt.show() if not prnt else\
            plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
                        facecolor=fig.get_facecolor())


# %% ztratification

class elevation:

    def __init__(self, x, y, **kwargs):  # lon=c_x; lat=c_y; buffer=c_r
        """
        creates greometrie(s) from coordainates and radii.\n
        Input ->
        *x* : xarray/numpy of xs/lons.
        *y* : xarray/numpy of ys/lats.\n
        **kwargs\n
        index         : index/numpy (custom index).\n
        buffer        : xarray/numpy of radii (in km) to buffer centers.\n
        units         : units of buffer (default: 'km').\n
        projected_crs : WKT of local projection (default: 'WKT_OGC').\n
        dem_path      : str; path-to-DEM.\n
        Output -> gpd.Series of buffered geometries (points or circles).
        """

        self._x = x
        self._y = y
        self._index = kwargs.get('index', range(len(self._x)))
        self._buffer = kwargs.get('buffer', None)
        self._units = kwargs.get('units', 'km')
        if isinstance(self._buffer, xr.DataArray):
            try:
                self._units = self._buffer.attrs['units']
            except AttributeError:
                pass
                self._units = None
        # compute radii
        self.radii = np.sqrt(self._buffer / np.pi).values if\
            self._units == 'km^2' else self._buffer
        self.projected_crs = kwargs.get('loc_crs', WKT_OGC)
        self._input_crs = None
        # guess input.proj & create buffers
        self._guess_crs(self._x, self._y)
        self.geom_ = self.rings(self._x, self._y, self.radii, index=self._index)

    # # https://stackoverflow.com/a/248066/5885810
    #     # parent_d = dirname(__file__)  # otherwise, will append the tests-path
    #     parent_d = './'
        self._parent_dir = parent_d
        self.dem_path = kwargs.get('dem_path',
                                   abspath(join(self._parent_dir, DEM_FILE)))

    def _guess_crs(self, x, y):
        assertwgs = f"No projected system found!\n"\
            "STORM3 will assign WGS84 as default (and resume)."
        try:
            self._input_crs = x.spatial_ref.crs_wkt
        except AttributeError:
            pass
            warn(assertwgs)
            if (x.min() >= -180 and x.max() <= 180) and\
                (y.min() >= -90 and y.max() <= 90):
                self._input_crs = 'GEOGCS["WGS 84",'\
                    'DATUM["WGS_1984",'\
                        'SPHEROID["WGS 84",6378137,298.257223563,'\
                            'AUTHORITY["EPSG","7030"]],'\
                        'AUTHORITY["EPSG","6326"]],'\
                    'PRIMEM["Greenwich",0,'\
                        'AUTHORITY["EPSG","8901"]],'\
                    'UNIT["degree",0.0174532925199433,'\
                        'AUTHORITY["EPSG","9122"]],'\
                    'AUTHORITY["EPSG","4326"]]'
            else:
                self._input_crs = self.projected_crs

    def rings(self, xs, ys, rs, **kwargs):
        r_idx = kwargs.get('index', range(len(xs)))
        cents = GeoSeries(points_from_xy(x=xs, y=ys), index=r_idx,
                          crs=self._input_crs).to_crs(crs=self.projected_crs)
        # is ok if we don't want circles but points
        if rs is None:
            rings = cents
        else:
            # *1e3 to go from km to m
            rings = cents.buffer(rs * 1e3, resolution=8)
        # reproject back to wgs84
        rings = rings.to_crs(crs=self._input_crs)
        return rings

    def reproj(self, **kwargs):
        crs_out = kwargs.get('crs_out', self.projected_crs)
        return self.geom_.to_crs(crs_out)

    @staticmethod
    def retrieve_z(geo_, raster, **kwargs):
        """
        retrieves the elevation (from *raster*) for a set of geo_features.\n
        Input ->
        *geo_*   : gpd.Geoseries.
        *raster* : path-to-TIFF.\n
        **kwargs\n
        z_stat  : str (from *rasterstats*) to compute metrics on raster (count, \
            min, max, mean, sum, std, median, majority, minority, unique, \
            range, nodata, nan).\n
        zones   : list; specifying the limits of the elevation bands.\n
        dem_crs : DEM's WKT.\n
        batch_size : int; amount of centers to process at a time.\n
        Output -> tuple; pd.DataFrame containing summary.stats (first), and \
            elevations tied to an elevation band (last).
        """
        # reading input & defaulting
        ztat = kwargs.get('z_stat', Z_STAT)
        quts = kwargs.get('batch_size', 2000)
        dem_crs = kwargs.get('dem_crs', None)
        zones = kwargs.get('zones', None)
        zones_lab, zones_bin = [''], 1
        if zones:
            zones_lab = [f'Z{x+1}' for x in range(len(zones) + 1)]
            zones_bin = np.union1d(zones, [0, 9999])
        # reprojecting
        # https://gis.stackexchange.com/a/441328/127894
        # https://gis.stackexchange.com/a/410891/127894
        # with openr(raster, 'r+') as rstr:
        with openr(raster, 'r') as rstr:
            rarr = rstr.read(1)
            raff = rstr.transform
            rcrs = rstr.crs
        # plt.colorbar(plt.imshow(rarr, cmap='jet', norm=mpl.colors.SymLogNorm(linthresh=0.3, vmin=-200, vmax=5000)))
        # plt.colorbar(plt.imshow(rarr[60:160,450:550], cmap='rainbow', norm=mpl.colors.SymLogNorm(linthresh=0.3, vmin=-200, vmax=5000)))
        if rcrs is None:
            rcrs = dem_crs
        try:
            geo_p = geo_.to_crs(rcrs)
        except ValueError:
            raise Warning(
                "No CRS found anywhere!\n"\
                "You must provide a raster with defined CRS or a valid CRS via 'dem_crs'."
                )
        # calculate zonal statistics:
        # ... if the GeoSeries is too large, split it into chunks (saves mem??)
        if geo_p.index.size > quts:
            zplit = np.array_split(geo_p.index, int(geo_p.index.size / quts))
            ztats = list(map(
                lambda x: zonal_stats(vectors=geo_p.loc[x], raster=rarr,
                                      affine=raff, stats=ztat, nodata=-9999)
                , zplit))
            ztats = pd.concat(list(map(pd.DataFrame, ztats)), ignore_index=True)
        else:
            ztats = zonal_stats(vectors=geo_p, raster=rarr, affine=raff,
                                stats=ztat, nodata=-9999)
            # # line below works for raster==path-to-dem (twice!! as slow, apparently)
            # ztats = zonal_stats(vectors=geo_.geometry, raster=rarr, stats=ztat, nodata=-9999)
            ztats = pd.DataFrame(ztats)
        # assign index
        ztats.reset_index(names='in_id', inplace=True)
        ztats.set_index(keys=geo_p.index, inplace=True)
        ztats.where(ztats>=0, other=0., inplace=True)  # other=np.nan
        # column 'E' classifies all Z's according to the CUTS
        ztats['E'] = pd.cut(ztats[ztat], bins=zones_bin,
                            labels=zones_lab, include_lowest=True)
        ztats.sort_values(by='E', inplace=True)
        # storm centres/counts grouped by BAND
        # https://stackoverflow.com/a/20461206/5885810  (index to column)
        qants = ztats.groupby(by='E', observed=False).count().reset_index(level=0)
        return qants, ztats


# %% copulas

class copulas:

    def __init__(self, cset, pair, method, **kwargs):
        """
        computes bi-variate copulas.\n
        Input ->
        *cset* : xarray; clipped dataset; or pandas.
        *pair* : tuple; either ('intensity', 'duration') or ('volume', 'duration').\n
        *method* : str; one of 'pf', 'total', or None
        **kwargs\n
        xy_b: xarray (or pandas?) containing lon, lat, and buffer.\n
        copula: statsmodels copula; e.g., GaussianCopula().\n
        z_stat: str; metric to average a DEM-selection.\n
        zones:  list; elevation bands to cut the region (Z_CUTS).\n
        Output -> a class where the copulas rhos are in *.rhos*.
        """

        self.copula = kwargs.get('copula', GaussianCopula())
        self.ztat = kwargs.get('z_stat', Z_STAT)
        self.zones = kwargs.get('zones', Z_CUTS)
        self._cset = cset
        self._xy_b = kwargs.get('xy_b', self._cset)
        self._pdic = {
            'duration': 'track_duration',
            'duration_alt': 'pf_lifetime',
            'volume': 'total_rain',
            'volume_alt': 'pf_accumrain',
            'max_intensity': 'pf_maxrainrate',
            'intensity': 'pf_rainrate',
            'mm_ratio': 'rratio',
            }
        self.v, self.u = self.pairing(pair)
        # self.rhos_alt = None
        self.z = self.z_frame(self._xy_b, pair, method)
        self.rhos = self.z.groupby(by=['E'], observed=False).apply(
            lambda x: self.copula.fit_corr_param(
                x[list(pair)]), include_groups=False)

    def pairing(self, pair):
        vvar = self._cset[self._pdic[pair[0]]]
        uvar = self._cset[self._pdic[pair[1]]]
        if isinstance(self._cset, xr.Dataset):
            vvar = vvar.load()
            uvar = uvar.load()
        if pair[0] == 'volume':
            vvar = vvar.sum(dim='times', keep_attrs=True)
            # vvar = vvar.mean(dim='times', keep_attrs=True)
        elif pair[0] == 'volume_alt':
            vvar = vvar.sum(dim=('nmaxpf', 'times'), keep_attrs=True)
            # vvar = vvar.mean(dim=('nmaxpf', 'times'), keep_attrs=True)
        # elif pair[0] == 'max_intensity' and isinstance(self._cset, xr.Dataset):
        else:
            # NOT defined
            pass
        if pair[1] == 'duration':
            uvar = uvar * self._cset.attrs['time_resolution_hour']
        else:
            pass
        # var = vvar * self._cset.attrs['pixel_radius_km']**2  # when VOLUME!
        return vvar, uvar

    def z_frame(self, xyz, pair, method):
        # all PFs!
        if isinstance(xyz, xr.Dataset):
            base = xyz[['pf_lon', 'pf_lat', 'pf_area',]]
            base = base.drop_vars('spatial_ref').to_dataframe()
            if method == 'total':
                base = base.groupby(level=['tracks', 'times']).agg(
                    {'pf_lon': 'mean', 'pf_lat': 'mean', 'pf_area': 'max'})
            base['idx'] = np.arange(len(base))
            base.dropna(axis='index', how='any', inplace=True)
            # base.set_index(keys=['idx'], drop=True, inplace=True)
        # IF a pd.DataFrame is passed...
        # THE indices of XYZ must similar.in.origin as those of ._CSET
        zeta = elevation(base['pf_lon'], base['pf_lat'], buffer=base['pf_area'],
                         units='km^2', index=base.index)
        qant, zlev = elevation.retrieve_z(zeta.geom_, zeta.dem_path,
                                          z_stat=self.ztat, zones=self.zones
                                          # batch_size=500, # dem_crs=,
                                          )
        zlev['Z'] = zlev['E'].str.removeprefix('Z').astype('f4')\
            if self.zones else 0.
        # collect()
        if pair[0] in ('volume', 'volume_alt'):
            r_c = xyz['pf_area'].stack(
                u=('tracks', 'times', 'nmaxpf')).compute()
            c_r = r_c[~r_c.isnull()]
            m_r = c_r.copy(deep=True)
            # "c_r[zlev.index]" not tested if entirely accurate!!
            c_r[zlev.index] = zlev['Z']
            m_r[zlev.index] = zlev[self.ztat]
            c_r = c_r.unstack(dim='u')
            m_r = m_r.unstack(dim='u')
            # compressed modes & stat
            upz = stats.mode(c_r, axis=(2,1), nan_policy='omit')
            sta_ = xr.DataArray(data=upz[0], coords={'tracks':c_r.coords['tracks']},
                                name='mode')
            tmp = m_r.where(c_r==sta_, np.nan)
            tmp = eval(f"tmp.{self.ztat}(dim=('times', 'nmaxpf'))")
            # update zlev (to tracks-dim)
            tlev = pd.DataFrame({
                'stat': tmp,
                'zone': np.char.add('Z', upz[0].astype('u2').astype('str')),
                }).sort_values(by='zone', ascending=True)
            tlev.columns = zlev.columns[1:3]
            df_tmp = pd.DataFrame(data={pair[0]: self.v, pair[1]: self.u})
        else:
            if isinstance(self._cset, xr.Dataset):
                self.v = copulas.compress(self.v)
                self.u = copulas.compress(self.u)
                df_tmp = self.v.join(self.u, how='inner')
                df_tmp.columns = [pair[0], pair[1]]
            else:
                # self.v = vvar
                # self.u = uvar
                df_tmp = pd.DataFrame(data={pair[0]: self.v, pair[1]: self.u})
            # adjust tlev
            tlev = zlev.drop(columns='Z')
            # NO.NEED when using COPULA XYZ for TRACKS.&.TIMES!
            # tlev.set_index(keys=base.index[tlev.index], inplace=True)
        # append the pandas
        copula_frame = tlev.merge(df_tmp, left_index=True, right_index=True,)
        return copula_frame

    @staticmethod
    def compress(xset_var):
        x_ = xset_var.drop_vars('spatial_ref').to_dataframe()
        x_['idx'] = np.arange(len(x_))
        x_.set_index(keys=['idx'], drop=True, inplace=True)
        x_ = x_.dropna(axis='index', how='any')
        return x_

    def save(self, file, region, tag):
        # pdf-agnostic xporting ?
        with open(file, 'a') as f:
            [f.write(f'R{region}+{tag}+{x},{self.rhos[x]}\n')
             for x in self.rhos.index]

    def plot(self, **kwargs):
        prnt = kwargs.get('file', None)
        mark = kwargs.get('marker', '+')
        colr = kwargs.get('color', 'xkcd:berry')
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
        self.z.iloc[:, -2:].plot(kind='scatter', x=-1, y=-2, s=42, ax=ax,
            marker=mark, color=colr, logx=True, logy=True,)
        plt.show() if not prnt else\
            plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
                        facecolor=fig.get_facecolor())


# %% d e c a y

class betas:

    def __init__(self, cset, method, **kwargs):
        """
        computes (among other key variables) the rainfall radial decay rate.\n
        Input ->
        *cset* : xarray - clipped dataset; or pandas - sampled values.
        *method* : str; one of 'pf', 'total', or None.\n
        **kwargs ->
        t_res: float; seed data temporal resolution (in hours).
        seed: float; initial/guess value for optimize.fsolve (of betas).
        flag: bool; whether to only use (or not) "converging" betas.\n
        Output -> a class with an updated pd.DataFrame in *.df*.
        """
        # exponential decay model (alt. 1)
        self.model = lambda x, v, i, r: v -\
            1 / (2 * x**2) * i * np.pi * (1 - np.exp(-2 * x**2 * r**2))
        self.tres = kwargs.get('t_res', None)
        self.flag = kwargs.get('flag', True)
        self.seed = kwargs.get('seed', 0.5)
        # reading data into df
        if isinstance(cset, xr.Dataset) and method == 'pf':
            self.df = self.pf_base(cset)
        if isinstance(cset, xr.Dataset) and method == 'total':
            self.df = self.tt_base(cset)
        elif isinstance(cset, pd.DataFrame):
            self.df = self.pd_base(cset,)
        else:
            raise AssertionError(
                f'WRONG INPUT TYPE!\n'
                f'cset must be either xr.Dataset or pd.DataFrame.'
                )
        # updating df
        self.df = betas.find_beta(self.df, self.model, self.seed, self.flag)

    def pf_base(self, cset):
        """
        computes variables based on 'precipitation-feature' tracking.\n
        Input ->
        *cset* : xarray; storm-tracking data.\n
        Output -> pd.DataFrame with computed/augmented/more variables.
        """
        base = cset[['pf_maxrainrate', 'pf_rainrate', 'pf_area',]]
        base = base.drop_vars('spatial_ref').to_dataframe()
        base['idx'] = np.arange(len(base))
        base.dropna(axis='index', how='any', inplace=True)
        # computing xtra-variables
        base['pf_radii'] = np.sqrt(base['pf_area'] / np.pi)
        base['pf_vol'] = base['pf_rainrate'] * base['pf_area']\
            * cset.attrs['time_resolution_hour']
        base['rratio'] = base['pf_maxrainrate'] / base['pf_rainrate']
        base.set_index(keys=['idx'], drop=True, inplace=True)
        return base

    def tt_base(self, cset):
        """
        computes variables based on 'totals-feature' tracking.\n
        Input ->
        *cset* : xarray; storm-tracking data.\n
        Output -> pd.DataFrame with computed/augmented/more variables.
        """
        max_ = cset[['pf_maxrainrate',]].max(dim=('nmaxpf'), keep_attrs=True)
        base = xr.merge([cset[['total_rain', 'area', 'movement_speed',]], max_],)
        base = base.drop_vars('spatial_ref').to_dataframe()
        base['idx'] = np.arange(len(base))
        base.dropna(axis='index', how='any', inplace=True)
        base.rename(columns={'total_rain': 'volume', 'movement_speed': 'velocity',
                             # 'pf_maxrainrate': 'maxrainrate',
                             }, inplace=True)
        base['volume'] = base['volume'] * cset.attrs['pixel_radius_km']**2\
            * cset.attrs['time_resolution_hour']
        # computing xtra-variables
        base['radii'] = np.sqrt(base['area'] / np.pi)
        # ratio between maxrainrate & meanrainrate
        base['rratio'] = base['pf_maxrainrate'] /\
            (base['volume'] / (base['area'] * cset.attrs['time_resolution_hour']))
        # base.set_index(keys=['idx'], drop=True, inplace=True)
        return base

    def pd_base(self, cset):
        """
        add estra variables to enable model fitting/optimization/solving.\n
        Input ->
        *cset* : pd.DataFrame; with fields\
            'pf_maxrainrate' & 'radii'-or-'area' & 'avgrainrate'-or-'rratio'.\n
        Output -> pd.DataFrame with extra variables (suited for model-fitting).
        """
        if 'avgrainrate' not in cset.columns:
            cset['avgrainrate'] = cset['pf_maxrainrate'] / cset['rratio']
        if 'area' not in cset.columns:
            cset['area'] = np.pi * cset['radii']**2
        if 'radii' not in cset.columns:
            cset['radii'] = np.sqrt(cset['area'] / np.pi)
        cset['volume'] = cset['avgrainrate'] * cset['area'] * self.tres
        return cset

    @staticmethod
    def find_beta(base, decay_model, seed, use_flag):
        """
        appends the retrieved betas to the passed data.frame.\n
        Input ->
        *base* : pd.DataFrame; storm-tracking data.
        *decay_model* : function; lambda-function representing the model to fit.
        *seed* : float; starting estimate for the roots.
        *use_flag* : bool; use only 'converging'[1] values (True); or all (F).\n
        Output -> updated and trimmed pd.DataFrame with 'beta' & 'flag' fields.
        """
        beta, flag = betas.beta_calc(base, decay_model, seed)
        tmp_df = pd.DataFrame({
            'beta': abs(np.asarray(beta).ravel()),
            'flag': flag,
            }, index=base.index)
        if use_flag:
            # 1 means 'The solution converged.'
            tmp_df = tmp_df.loc[tmp_df['flag'] == 1, ['beta']]
        # tmp_df.plot(y='beta', kind='hist', bins=42); plt.show()
        base = base.join(tmp_df, how='inner')
        return base

    @staticmethod
    def beta_calc(base_df, equation, iseed):
        """
        finds roots for the passed *equation*.\n
        Input ->
        *base_df* : pd.DataFrame; with 'pf_maxrainrate' & 'radii'& 'volume'.
        *equation* : function; lambda-function representing the model to fit.
        *iseed* : float; starting estimate for the roots.\n
        Output -> tuple with the found roots (left) and their flags (right).
        """
        # https://stackoverflow.com/a/7015366/5885810
        l_, s_, i_, m_ = zip(*map(
            lambda v, i, r: optimize.fsolve(equation, x0=iseed, args=(v, i, r),
                                            full_output=True),
            base_df['volume'], base_df['pf_maxrainrate'], base_df['radii']))
        return l_, i_


# %% pdfitting

class fit_pdf:

    def __init__(self, data, family, **kwargs):
        """
        tools for optimun-pdf-fit to raw-data via FITTER.\n
        Input ->
        *data* : numpy; vector of measurements to fit.
        *family* : list; list of scipy-pdfs to try and fit.
        **kwargs\n
        method:  str; selection method for optimal fit\
            ('bic', 'aic', or 'sumsquare_error').\n
        Output -> a class where the optimal fit is in *.pdf*.
        """

        self.method = kwargs.get('method', 'sumsquare_error')
        self.ecol = kwargs.get('e_col', None)

        # right/positive-skewed distros
        # maxintensity & radii & rratio
        self._f_rskm = [
            'burr', 'fisk', 'genlogistic', 'genextreme', 'gengamma',
            'genhyperbolic', 'gumbel_r', 'invgauss', 'invweibull', 'johnsonsb',
            'ksone', 'kstwobign', 'mielke', 'ncx2', 'ncf', 'norminvgauss',
            # 'foldcauchy',
            ]
        # seasonal/totals [more-less]
        self._f_lskm = [
            'anglit', 'argus', 'burr12', 'chi', 'cosine', 'crystalball',
            'genlogistic', 'gumbel_l', 'johnsonsu', 'levy_l', 'levy_stable',
            'loggamma', 'powernorm', 'weibull_max',
            # 'genhyperbolic', 'foldnorm', 'exponpow',
            # 'laplace_asymmetric', 'gausshyper', 'genhalflogistic',
            ]
        # duration & velocity & area (more not-symmetric, less right-skewed)
        self._f_nsym = [
            'alpha', 'betaprime', 'fisk', 'f', 'gamma', 'genextreme',
            'geninvgauss', 'gibrat', 'invgamma', 'ksone', 'levy', 'lognorm',
            'ncf', 'powerlognorm', 'rayleigh', 'rice', 'recipinvgauss',
            'truncweibull_min', 'wald', 'weibull_min',
            # 'studentized_range',
            ]
        # heavily gaussian/symmetric distros
        # betas
        self._f_norm = [
            'anglit', 'burr', 'chi', 'chi2', 'cosine', 'exponweib', 'hypsecant',
            'jf_skew_t', 'johnsonsb', 'logistic', 'maxwell', 'ncx2',
            'nct', 'norm', 'powerlognorm', 'powernorm', 't', 'weibull_max',
            # 'nakagami', 'norminvgauss',
            ]
        # exponential-like (decrease) distros
        self._f_expn = [
            'bradford', 'expon', 'genpareto', 'genexpon', 'gompertz',
            'halfcauchy', 'halfnorm', 'halflogistic', 'loguniform', 'lomax',
            'powerlaw', 'truncexpon', 'truncpareto',
            ]

        if isinstance(family, list) and\
            isinstance(np.random.choice(family), str):
            self.family = family
        elif isinstance(family, str) and family in\
            ['rskm', 'lskm', 'nsym', 'norm', 'expn']:
            self.family = eval(f'self._f_{family}')
        else:
            raise AssertionError(
                f'WRONG INPUT TYPE!\n'
                f'family must be a list of scipy-pdf families or one of the '
                f'following strings "rskm", "lskm", "nsym", "norm" or "expn".'
                )
        self.data = data
        self.group = self.e_group(self.data)
        self.pdf = dict(zip(self.group.index, list(zip(*self.group.tolist()))[1]))

    def e_group(self, data):
        # data = decay.df.beta
        # data = cone.z[['E', 'max_intensity']]
        # ecol = 'E'
        if isinstance(data, pd.DataFrame):
            if self.ecol is not None:
                assert self.ecol in data.columns, 'wrong e_col (kwargs) passed.\n'
                egroup = data.groupby(by=[self.ecol], observed=False).apply(
                    lambda x: fit_pdf.pdf_fit(
                        x, family=self.family, method_sel=self.method),
                    include_groups=False)
        elif isinstance(data, pd.Series) or isinstance(data, np.ndarray):
            egroup = []
            egroup.append(fit_pdf.pdf_fit(data,
                family=self.family, method_sel=self.method))
            egroup = pd.Series(egroup)
        else:
            # what if it's something else?
            pass
        return egroup

    @staticmethod
    def pdf_fit(data, family, method_sel):
        best_fit = Fitter(data, distributions=family)
        best_fit.fit()  # storing the fits
        # best_fit.summary()
        # SELECT THE PARAMETERS FOR THE BEST.FIT ('BIC' is the preference)
        pdfit = best_fit.get_best(method=method_sel)
        return best_fit, pdfit

    def save(self, file, region, tag):
        # pdf-agnostic xporting (i don't think is the case anymore)
        # https://stackoverflow.com/a/27638751/5885810
        # https://stackoverflow.com/a/56736691/5885810
        # https://stackoverflow.com/a/55481809/5885810
        # https://stackoverflow.com/a/3590175/5885810
        with open(file, 'a') as f:
            for k in self.pdf.keys():
                q = '' if k == 0 else f'+{k}'
                f.write(f'R{region}+{tag}{q}+{next(iter(self.pdf.get(k).keys()))},')
                # 'unbreakable' string
                f.write(f"{','.join(map(str, [*next(iter(self.pdf.get(k).values())).values()]))}\n")

    def plot(self, **kwargs):
        prnt = kwargs.get('file', None)
        N = kwargs.get('N', 4)
        Z = kwargs.get('zone', 0)
        meth = kwargs.get('method', self.method)
        xlim = kwargs.get('xlim', None)
        nbin = kwargs.get('bins', 23)
        # color scale
        kolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
        kolor = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(33)]
        np.random.shuffle(kolor)
        # sort by metric
        olast = list(self.group.iloc[Z][0].df_errors.dropna(how='any').sort_values(
            by=meth).index)[0:N]
        ilist = list(self.group.iloc[Z][0].fitted_param.keys())
        pos = [ilist.index(x) for x in olast]
        p_data = self.data if Z == 0 else\
            self.data[self.data['E'] == f'Z{Z}'].drop(columns=self.ecol)
        # plot
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        plt.hist(p_data, bins=nbin, histtype='bar', density=True, label=None,
                 color='xkcd:silver', alpha=5/7, rwidth=.93)
        for i, n in enumerate(pos):
            distro = eval(f'stats.{ilist[n]}{list(self.group.iloc[Z][0].fitted_param.values())[n]}')
            # # the line below doesn't work in some scipy.functions
            # # e.g.: distro = stats.powerlognorm(0.001499, 0.021, -6.1284, 7.191)
            # xs = np.linspace(distro.ppf(0.001), distro.ppf(0.999), 101)
            # so use this heavy.stuff instead
            xs = np.linspace(
                optimize.fsolve(lambda x: 0.001 - distro.cdf(x), self.data.min())[0],
                optimize.fsolve(lambda x: 0.999 - distro.cdf(x), self.data.max())[0],
                201
                )
            ys = distro.pdf(xs)
            ax.plot(xs, ys, color=kolor[n], lw=4 if i == 0 else 2,
                    label=ilist[n], zorder=20-i)
            ax.legend()
        if xlim is not None:
            ax.set_xlim(xlim)
        plt.show() if not prnt else\
            plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
                        facecolor=fig.get_facecolor())
        # plt.clf()


        # # PLOT.TESTING
        # # color scale
        # kolor = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # kolor = ['#%06X' % np.random.randint(0, 0xFFFFFF) for _ in range(33)]
        # np.random.shuffle(kolor)
        # # sort by metric
        # Z = 0
        # olast = list(fit_dur.group.iloc[Z][0].df_errors.dropna(how='any').sort_values(
        #     by=fit_dur.method).index)[0:4]
        # ilist = list(fit_dur.group.iloc[Z][0].fitted_param.keys())
        # pos = [ilist.index(x) for x in olast]
        # # plot
        # fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        # plt.hist(fit_dur.data, bins=23, histtype='bar', density=True, label=None,
        #          color='xkcd:silver', alpha=5/7, rwidth=.93)
        # # for i, n in enumerate(pos):
        # # i=0; n=13
        # distro = eval(f'stats.{ilist[n]}{list(fit_dur.group.iloc[Z][0].fitted_param.values())[n]}')
        # # xs = np.linspace(distro.ppf(0.001), distro.ppf(0.999), 100)
        # xs = np.linspace(
        #     optimize.fsolve(lambda x: 0.001 - distro.cdf(x), flim[0]).item(),
        #     optimize.fsolve(lambda x: 0.999 - distro.cdf(x), flim[1]).item(),
        #     201
        #     )
        # ys = distro.pdf(xs)
        # ax.plot(xs, ys, color=kolor[n], lw=4 if i == 0 else 2, label=ilist[n], zorder=20-i)
        # ax.legend()
        # plt.show()


# %% discrete xtras

class discrete:

    def __init__(self, data, **kwargs):
        """
        fits the best PMF.\n
        Input ->
        *data* : numpy; vector of measurements to fit.
        **kwargs\n
        method:  str; selection method for optimal fit ('bic' or 'nllf').\n
        max_dur: float; maximum value of duration.\n
        location_guess: float; guess-timate where the mean might be.\n
        Output -> a class where the optimal fit is in *.pdf*.
        """

        self.method = kwargs.get('method', 'bic')  # (or 'nllf')
        # self.ecol = kwargs.get('e_col', None)
        self.max_dur = kwargs.get('max_dur', 200)  # in hours (max in OND == 109)
        self._lguess = kwargs.get('location_guess', 9)
        self._xfactr = 100
        # potential discrete distros (& parameter-boundaries) for storm-duration
        self.distros = {
            'betabinom': ((0, self.max_dur), (0, self._xfactr), (0, self._xfactr)),
            'binom': ((0, self._lguess), (0, self._xfactr),),
            'dlaplace': ((0, self.max_dur),),
            'nbinom': ((0, self.max_dur), (0, self._xfactr),),
            'nhypergeom': ((0, self.max_dur), (0, self._xfactr), (0, self._xfactr)),
            # 'nchypergeom_fisher': ((0, self._lguess), (0, self._xfactr),
            #                        (0, self._xfactr), (0, self._xfactr)),
            # 'nchypergeom_wallenius': ((0, self._lguess), (0, self._xfactr),
            #                           (0, self._xfactr), (0, self._xfactr)),
            'poisson': ((0, self.max_dur),),
            }
        self.data = data
        self._brief = list(map(lambda f, b: duration.pmf_fit(self.data, f, b),
            list(self.distros.keys()), list(self.distros.values())))
        # self.pdf = self.best(self._brief, self.method)
        # following FIT_PDF class
        self.group = self.e_group()
        self.pdf = dict(zip(self.group.index, self.group.tolist()))

    @staticmethod
    def pmf_fit(data, distro, bounds):
        fit_pmf = eval(f'stats.fit(stats.{distro}, data, {bounds})')
        # https://stanfordphd.com/BIC.html
        bic = np.nan if (np.isinf(fit_pmf.nllf()) or np.isnan(fit_pmf.nllf()))\
            else -2 * fit_pmf.nllf() + len(bounds) * np.log(len(data))
        return (distro, fit_pmf.nllf(), bic, *fit_pmf.params)

    def best(self, brief_list, method):
        method = 2 if method == 'bic' else 1
        selector = list(zip(*brief_list))[method]
        best_fit = brief_list[selector.index(np.nanmin(selector))]
        # https://stackoverflow.com/a/20540948/5885810
        par_dict = dict(zip(list(map(
            lambda x: f'p{x}', range(len(best_fit[3:])))), best_fit[3:]))
        # create the whole 'distro'.dict
        dis_dict = eval(f'dict({best_fit[0]}={par_dict})')
        return dis_dict

    def e_group(self, ):
        egroup = [self.best(self._brief, self.method)]
        egroup = pd.Series(egroup)
        return egroup

    def save(self, file, region, tag):
        # # FOR NO-eGROUP CASES
        # with open(file, 'a') as f:
        #     f.write(f'R{region}+{tag}+{next(iter(self.pdf.keys()))},')
        #     f.write(f"{','.join(map(str, [*next(iter(self.pdf.values())).values()]))}\n")
        with open(file, 'a') as f:
            for k in self.pdf.keys():
                q = '' if k == 0 else f'+{k}'
                f.write(f'R{region}+{tag}{q}+{next(iter(self.pdf.get(k).keys()))},')
                f.write(f"{','.join(map(str, [*next(iter(self.pdf.get(k).values())).values()]))}\n")

    # def plot(self, **kwargs):
    #     prnt = kwargs.get('file', None)
    #     N = kwargs.get('N', 4)
    #     Z = kwargs.get('zone', 0)
    #     meth = kwargs.get('method', self.method)
    #     xlim = kwargs.get('xlim', None)
    #     distro = eval(f'stats.{list(self.pdf[Z].keys())[0]}{tuple([*next(iter(self.pdf.get(Z).values())).values()])}')
    #     # distro = eval(f'stats.{list(fit_dur.pdf[Z].keys())[0]}{tuple([*next(iter(fit_dur.pdf.get(Z).values())).values()])}')
    #     xs = np.linspace(distro.ppf(0.001), distro.ppf(0.999), 100)
    #     ys = distro.pmf(xs)

    #     # fot_pmf = stats.fit(stats.nhypergeom, dur.data*2, ((0, 200), (100, 150), (0, 20)))
    #     # # *fot_pmf.params
    #     # distro = eval(f'stats.nhypergeom{tuple(fot_pmf.params)}')
    #     # xs = np.linspace(distro.ppf(0.001), distro.ppf(0.999), 100)
    #     # ys = distro.pmf(xs)
    #     # fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    #     # ax.plot(xs, ys, marker='.', mfc='none', mew=1., ms=5, ls='none', color='xkcd:marine')#, lw=0
    #     # ax.vlines(xs, 0, ys, lw=0.45, color='xkcd:marine blue')
    #     # plt.show()

    #     # distro = stats.binom(9.0, 0.5763003605380603, 0.0)
    #     # distro = stats.dlaplace(0.11034853599074149, 0.0)
    #     # distro = stats.nbinom(3.0, 0.24909104523268863, 0.0)
    #     # distro = stats.poisson(9.043831229348728, 0.0)
    #     # distro = stats.betabinom(187.0, 2.6969348156804664, 54.506416659346655, 0.0)
    #     # distro = stats.nhypergeom(131.0, 99.0, 3.0, 0.0)
    #     # xs = np.linspace(distro.ppf(0.001), distro.ppf(0.999), 100)
    #     # ys = distro.pmf(xs)
    #     # fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    #     # ax.plot(xs, ys, marker='.', mfc='none', mew=1., ms=5, ls='none', color='xkcd:marine')#, lw=0
    #     # ax.vlines(xs, 0, ys, lw=0.45, color='xkcd:marine blue')
    #     # plt.show()

    #     p_data = self.data if Z == 0 else\
    #         self.data[self.data['E'] == f'Z{Z}'].drop(columns=self.ecol)
    #     # plot
    #     fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    #     plt.hist(p_data, bins=23, histtype='bar', density=True, label=None,
    #              color='xkcd:silver', alpha=5/7, rwidth=.93)
    #     ax.plot(xs, ys, marker='.', mfc='none', mew=1., ms=5, ls='none', color='xkcd:marine')#, lw=0
    #     ax.vlines(xs, 0, ys, lw=0.45, color='xkcd:marine blue')
    #     # ax.plot(xs, ys, color=kolor[n], lw=4 if i == 0 else 2, label=ilist[n], zorder=20-i)
    #     ax.legend()

    #     if xlim is not None:
    #         ax.set_xlim(xlim)
    #     plt.show() if not prnt else\
    #         plt.savefig(prnt, bbox_inches='tight', pad_inches=0.01,
    #                     facecolor=fig.get_facecolor())


def totals(set_tot, area_km):
    # set_tot=fset.clip_set
    # area_km=(one_area.region.area_m / 1e6).values; area_km=npix
    stot = betas(set_tot, method='total')
    # now compute seasonal.totals
    ttmp = stot.df[['area', 'pf_maxrainrate', 'rratio']]
    ttmp['year'] = ttmp.index.get_level_values('tracks').astype(
        str).str[:4].values.astype('u4')
    ttmp['avg_r'] = ttmp.pf_maxrainrate / ttmp.rratio * set_tot.attrs['time_resolution_hour']
    # ttmp['total'] = ttmp.avg_r * ttmp.area / area_km  # in km2
    ttmp['total'] = ttmp.avg_r * ttmp.area / (X_RES * Y_RES / 1e6) / area_km  # in pxls
    return ttmp.groupby(by='year').agg('sum')


# # reading MONTHLY.IMERG (for a given season)
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# from glob import glob

# sdic = {'MAM': [3, 4, 5], 'OND': [10, 11, 12]}

# def imorg(year):  # year=2000; year=2021
#     flist = glob(f'../SETS/imerg_m/3B-MO*{year}*' )
#     ix = np.isin(list(map(lambda x: int(x.split('E235959')[1][1:3]), flist)), sdic[SEASON_TAG])
#     flist = [value for bool_, value in zip(ix, flist) if bool_]
#     if len(flist)!=0:
#         imerg_mo = xr.open_mfdataset(
#             flist, combine='nested', concat_dim='time', decode_times=True,
#             use_cftime=True, decode_cf=True,  # data_vars=['precipitation'],
#             mask_and_scale=True , parallel=False).transpose('time', 'lat', 'lon')
#         daysim = imerg_mo.time.dt.days_in_month
#         seas_rain = (imerg_mo * daysim * 24).sum(dim='time').compute()
#         seas_rain['precipitationQualityIndex'] =\
#             imerg_mo['precipitationQualityIndex'].mean(dim='time').compute()
#         seas_rain.rio.write_crs(CRS('EPSG:4326'),
#                                 grid_mapping_name='spatial_ref', inplace=True)
#         # seas_rain.precipitation.plot(robust=True, cmap='gist_ncar')
#         # seas_rain.precipitationQualityIndex.plot(robust=True, cmap='turbo')
#     else:
#         seas_rain = []
#     return seas_rain

# ilis = list(map(imorg, np.r_[2000:2022]))
# # remove.voids.in.a.list (no tracking of voids!)
# # https://stackoverflow.com/a/40598286/5885810
# seasonal_monthly_imerg = xr.concat(
#     list(filter(None, ilis)), 'time', compat='override', coords='minimal')


# %% call all

# def compute(region=None):
#     if region=None:
def compute():

#  1. SPLITING SHP INTO AOI
    # if reading SEASONAL.MAP with XARRAY
    seas = xr.open_dataset(
        abspath(join(parent_d, RAIN_MAP)),
        chunks='auto', decode_cf=True, use_cftime=True,
        decode_coords='all',  # "decode_coords" helps to interpret CRS
        )
    seas = seas.drop_vars('spatial_ref').rename({'lat': 'y', 'lon': 'x'})
    # seas.rain.plot(cmap='gist_ncar', robust=True)

    # space = masking()
    space = masking(catchment=SHP_FILE)  # space.plot()
    rain_ = field(seas, space.xs, space.ys)

    areas = regional(rain_.field_prj, space.buffer_mask.astype('u1'),
                     space.catchment_mask.astype('u1'), nr=NREGIONS)
    # update xport.shp.file name & xport it
    FILE_ZON = abspath(join(parent_d,
        ZON_FILE.replace('.shp', f'_{SEASON_TAG}_{areas.n_}r.shp')
        ))
    areas.xport_shp(file=FILE_ZON)
    # areas.xport_shp(file=FILE_ZON.replace('.shp', '.gpkg'), driver='GPKG', layer='rrain')

#  2. CREATE XPORTING FILE (for csv & shp)
    FILE_PDF = abspath(join(parent_d,
        PDF_FILE.replace('.csv', f'_{SEASON_TAG}_{areas.n_}r.csv')
        ))
    # https://www.geeksforgeeks.org/create-an-empty-file-using-python/
    try:
        with open(FILE_PDF, 'w'):
            pass
    except FileNotFoundError:
        print(f"for some reason the path '{dirname(FILE_PDF)}' does not "\
              'exist!!.\nplease, create such a folder... and come back.')

#  3. READ TRACK.DATA
    dzet = xr.open_dataset(abspath(join(parent_d, EVENT_DATA)),
                           chunked_array_type='dask', chunks='auto')

    # for i, r in enumerate(areas.gshape.region):
    for i, r in enumerate(tqdm(areas.gshape.region, ncols=50)):  # i=0

#  4. DEFINE THE AREA OF ANALYSIS
        # one_area = region()  # if.doing.the.whole.area (also?)
        one_area = region(shp=areas.gshape.iloc[[i]])
        # areas.gshape.iloc[[i]].geometry.xs(i)     # plotting
        # one_area.region.geometry.xs(i)            # plotting
        print(f'\n{one_area.region}')

#  5. CAPTURING TOTALS
        fset = clipping(dzet, one_area.region, lf=.0)
        npix = areas.regions[areas.regions == int(list(areas.nr_dic.keys())[i])].size
        # tots = totals(fset.clip_set, (one_area.region.area_m / 1e6).values)
        tots = totals(fset.clip_set, npix)
        # tots.total.mean()  # tots.total.values

        # # checking against IMERG_monthly
        # # first run "seasonal_monthly_imerg" in [discrete xtras]
        # mo_clip = seasonal_monthly_imerg.rio.clip(one_area.region.geometry,)
        # # mo_clip.precipitation[0,:].plot(cmap='turbo')
        # avgs = mo_clip.precipitation.sum(dim=('lat', 'lon')) / mo_clip.precipitation.count(dim=('lat', 'lon'))
        # avgs.mean().data  # avgs.data

        fit_tot = fit_pdf(tots.total, family='lskm',)
        # fit_tot = fit_pdf(tots.total, family='norm',)
        fit_tot.save(file=FILE_PDF, tag='TOTALP_PDF', region=r)
        # fit_tot.plot()

#  6. CLIP (again) TRACK.DATA to A.O.I (with STORMS having at.least 80% of LF)
        dset = clipping(dzet, one_area.region, lf=.8)
        # dset.plot()  # dset.plot(file='p01.png')

#  7. FIT DURATION
        dur = (dset.clip_set.track_duration *
               dset.clip_set.attrs['time_resolution_hour']).load()
        # plt.hist(dur, bins=29)
        # [1.5, 81.] -> MAM; [1.5, 62.5] -> OND;

        # # DOES NOT WORK ON DISCRETE (because of 0.5 res
        # # ... but it works for 1.0 res -> if you find the right pars/bounds)
        # fit_dur = discrete(dur.data, method='nllf')

        fit_dur = fit_pdf(dur.data, family='nsym',)
        fit_dur.save(file=FILE_PDF, tag='AVGDUR_PDF', region=r)  # 'AVGDUR_PMF'
        # fit_dur.pdf
        # fit_dur.plot()

#  8. PANDAS (containing all the rest vars)
        # ALTERNATIVE 1: stats based on TOTALS (plus RRATIO)
        all_vars = betas(dset.clip_set, method='total')
        # all_vars.df

        if ALTERNATIV == 1:
    # 11. FIT MAXINTENSITY
            fit_int = fit_pdf(all_vars.df.pf_maxrainrate, family='rskm',)
            fit_int.save(file=FILE_PDF, tag='MAXINT_PDF', region=r)
            # fit_int.pdf
            # fit_int.plot()

            # # if doing log-transform
            # fit_int = fit_pdf(np.log(all_vars.df.pf_maxrainrate), family='norm',)

    # 12. FIT DECAY (very optional in ALTERNATIV 2)
            fit_dec = fit_pdf(all_vars.df.beta, family='norm',)
            fit_dec.save(file=FILE_PDF, tag='BETPAR_PDF', region=r)
            # fit_dec.plot()
            # fit_dec.plot(file='zome.jpg', xlim=(0.005, 0.09), method='sumsquare_error')
            # fit_dec.group.iloc[0][0].summary()

        elif ALTERNATIV == 2:
    #  9. FIT SOME COPULA ('max_intensity', 'mm_ratio')
            cop_one = copulas(all_vars.df, ('max_intensity', 'mm_ratio'),
                              method='total', xy_b=dset.clip_set)
            # cop_one.rhos  # cop_one.z
            # cop_one.plot()
            # cop_one.plot(marker='x', color='xkcd:cool green')
            cop_one.save(file=FILE_PDF, tag='COPONE_RHO', region=r)

    # 10. NEW E FRAME
            all_vars_e = all_vars.df.merge(cop_one.z, left_index=True, right_index=True,)

    # 11. FIT MAXINTENSITY
            fit_int = fit_pdf(all_vars_e[['max_intensity', 'E']], e_col='E', family='rskm',)
            fit_int.save(file=FILE_PDF, tag='MAXINT_PDF', region=r)
            # fit_int.pdf
            # fit_int.plot()

    # 13. FIT RRATIO (very optional in ALTERNATIV 1)
            fit_rat = fit_pdf(all_vars_e[['rratio', 'E']], e_col='E', family='rskm',)
            fit_rat.save(file=FILE_PDF, tag='INTRAT_PDF', region=r)
            # fit_rat.plot(bins=51, xlim=(-1,41))

        else:
            raise TypeError("wrong ALTERNATIV parameter passed!.")

# 14. FIT RADII
        fit_rad = fit_pdf(all_vars.df.radii, family='rskm',)
        fit_rad.save(file=FILE_PDF, tag='RADIUS_PDF', region=r)
        # fit_rad.plot()
        # [18.7, 434.6] -> MAM; [17.8, 435.6] -> OND;

        # # if working with areas
        # fit_are = fit_pdf(all_vars.df.area, family='nsym',)
        # fit_are.plot()

# 15. FIT VELOCITY (in m/s)
        fit_vel = fit_pdf(all_vars.df.velocity, family='norm',)
        fit_vel.save(file=FILE_PDF, tag='VELMOV_PDF', region=r)
        # fit_vel.plot(bins=51, xlim=(-11, 41), N=5)

        # # in not minding ZEROS, we're removing ~1/3 of the data!!
        # vel = all_vars.df.velocity[all_vars.df.velocity > 0.01]
        # fit_vel = fit_pdf(vel, family='nsym',)

# 16. DIRECTION [CIRCULAR]
        # track-based
        mdir = dset.clip_set.movement_theta / 360 * 2 * np.pi - np.pi
        mdir_rad = xr.apply_ufunc(one_vm, mdir, dask='parallelized'#'allowed'
            ,vectorize=True, input_core_dims=[['times',]]).compute().data
        dir_c = circular(mdir_rad[~np.isnan(mdir_rad)], data_type='rad', met_cap=.91)
        # tod_c.plot_samples(file='zome_file.jpg', data_type='dir', bins=50)
        # dir_c.plot_samples(data_type='dir')
        # dir_c.plot_bic()
        dir_c.save(file=FILE_PDF, tag='DIRMOV', region=r)

        # # for all tracks+timesteps
        # mdir = dset.clip_set.movement_theta / 360 * 2 * np.pi - np.pi
        # mdir = mdir.stack({'t_n':['tracks', 'times',]}).drop_vars('spatial_ref')
        # mdir_rad = mdir.dropna(dim='t_n', how='all').compute().data
        # dir_c = circular(mdir_rad, data_type='rad', met_cap=.7)

# 17. TIME of DAY [CIRCULAR]
        tod = dset.clip_set.start_basetime.load()
        tod = (tod.dt.hour + tod.dt.minute / 60 + tod.dt.second / 3600).data
        tod_c = circular(tod, data_type='tod', met_cap=.93)
        # tod_c.plot_samples(data_type='tod', bins=50)
        # tod_c.plot_bic()
        tod_c.save(file=FILE_PDF, tag='DATIME', region=r)

# 18. DAY of YEAR [CIRCULAR]
        doy = (dset.clip_set.start_basetime.dt.dayofyear + tod / 24).compute().data
        doy_c = circular(doy, data_type='doy', met_cap=.83)
        # doy_c.plot_samples(data_type='doy', bins=20)
        # doy_c.plot_bic()
        doy_c.save(file=FILE_PDF, tag='DOYEAR', region=r)


# %% main

if __name__ == '__main__':
    compute()
