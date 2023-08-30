import numpy as np
import xarray as xr
import rioxarray as rio
import geopandas as gpd
import pyproj as pp
from osgeo import gdal
from rasterio.enums import Resampling # IF you're doing BILINEAR or NEAREST too?
from rasterio.features import shapes
from pandas import RangeIndex

from sklearn.cluster import KMeans
from skimage import morphology#, color
# # import matplotlib.pyplot as plt


#%% SOME.DEFAULTS (for testing outside STORM3?)


# ## RAINFALL_MAP stored in some NC.like.these...
# ## --------------------------------------------
# RAIN_MAP = '../CHIMES/3B-HHR.MS.MRG.3IMERG.20101010-S100000-E102959.0600.V06B.HDF5'     # no.CRS at all!
# RAIN_MAP = './realisation_MAM_crs-wrong.nc'                         # no..interpretable CRS
# RAIN_MAP = './model_output/SOME_test.nc'                            # HAD.projected CRS
# RAIN_MAP = './model_output/SOME_test-for.nc'                        # HAD.projected CRS

# RAIN_MAP = './realisation_MAM_crs-OK.nc'                            # yes.interpretable CRS
# SUBGROUP = ''
# CLUSTERS = 4


#%% DEFINING FUNCTIONS


#~ COORDINATE SYSTEMS & GLOBAL.VARS TO USE ONLY IN THIS SCRIPT ~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def CRSS():
    global WKT_OGC, WGS84_WKT
    # HAD projection
    WKT_OGC = 'PROJCS["WGS84_/_Lambert_Azim_Mozambique",'\
        'GEOGCS["unknown",'\
            'DATUM["unknown",'\
                'SPHEROID["Normal Sphere (r=6370997)",6370997,0]],'\
            'PRIMEM["Greenwich",0,'\
                'AUTHORITY["EPSG","8901"]],'\
            'UNIT["degree",0.0174532925199433,'\
                'AUTHORITY["EPSG","9122"]]],'\
        'PROJECTION["Lambert_Azimuthal_Equal_Area"],'\
        'PARAMETER["latitude_of_center",5],'\
        'PARAMETER["longitude_of_center",20],'\
        'PARAMETER["false_easting",0],'\
        'PARAMETER["false_northing",0],'\
        'UNIT["metre",1,'\
            'AUTHORITY["EPSG","9001"]],'\
        'AXIS["Easting",EAST],'\
        'AXIS["Northing",NORTH],'\
        'AUTHORITY["EPSG","42106"]]'
    # WGS84 projection
    WGS84_WKT = 'GEOGCS["WGS 84",'\
        'DATUM["WGS_1984",'\
            'SPHEROID["WGS 84",6378137,298.257223563,'\
                'AUTHORITY["EPSG","7030"]],'\
            'AUTHORITY["EPSG","6326"]],'\
        'PRIMEM["Greenwich",0,'\
            'AUTHORITY["EPSG","8901"]],'\
        'UNIT["degree",0.0174532925199433,'\
            'AUTHORITY["EPSG","9122"]],'\
        'AUTHORITY["EPSG","4326"]]'


#~ BUFFER & CATCHMENT MASKS INTO NUMPY ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def NUMPY_MASKS( SHP_FILE, BUFFER ):# SHP_FILE='./model_input/HAD_basin.shp' ; BUFFER=8000.
    global XS, YS
# some already defaults
    X_RES     =  5000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
    Y_RES     =  5000.                      # in meters! (pxl.resolution for the 'regular/local' CRS)
# read WG-catchment shapefile (assumed to be in WGS84)
    wtrwgs = gpd.read_file( SHP_FILE )
# transform it into EPSG:42106 & make the buffer    # this code does NOT work!
# https://gis.stackexchange.com/a/328276/127894     (geo series into gpd)
    # wtrshd = wtrwgs.to_crs( epsg=EPSG_CODE )
    # wtrshd = wtrwgs.to_crs( crs =PROJ4_STR )
    wtrshd = wtrwgs.to_crs( crs =WKT_OGC )          # //epsg.io/42106.wkt
    BUFFRX = gpd.GeoDataFrame(geometry=wtrshd.buffer( BUFFER ))#.to_crs(epsg=4326)
# infering (and rounding) the limits of the buffer-zone
    llim = np.floor( BUFFRX.bounds.minx[0] /X_RES ) *X_RES #+X_RES/2
    rlim = np.ceil(  BUFFRX.bounds.maxx[0] /X_RES ) *X_RES #-X_RES/2
    blim = np.floor( BUFFRX.bounds.miny[0] /Y_RES ) *Y_RES #+Y_RES/2
    tlim = np.ceil(  BUFFRX.bounds.maxy[0] /Y_RES ) *Y_RES #-Y_RES/2

#~IN.CASE.YOU.WANNA.XPORT.(OR.USE).THE.MASK+BUFFER.as.geoTIFF~~~~~~~~~~~~~~~~~~#
    # # ACTIVATE if IN.TIFF
    # tmp_file = 'tmp-raster_mask-buff.tif'
    # tmp = gdal.Rasterize(tmp_file, BUFFRX.to_json(), xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True, format='GTiff'
    # ACTIVATE if IN.MEMORY
    tmp = gdal.Rasterize('', BUFFRX.to_json(), xRes=X_RES, yRes=Y_RES, noData=0, burnValues=1, allTouched=True, format='MEM'
        , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
        , targetAlignedPixels=True
        # , targetAlignedPixels=False # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
#!!
# UPDATE HERE -> outputSRS [in WKT instead of PROJ4]
#!!
    # OR! USE THE PROPER WKT!
        #  outputSRS=f'EPSG:{EPSG_CODE}'#f'{PROJ4_STR}'
        , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
        # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
        )
    BUFFRX_MASK = tmp.ReadAsArray().astype('u1')
    tmp = None

#~BURN THE CATCHMENT SHP INTO RASTER (WITHOUT BUFFER EXTENSION)~~~~~~~~~~~~~~~~#
# https://stackoverflow.com/a/47551616/5885810  (idx polygons intersect)
# https://gdal.org/programs/gdal_rasterize.html
# https://lists.osgeo.org/pipermail/gdal-dev/2009-March/019899.html (xport ASCII)
# https://gis.stackexchange.com/a/373848/127894 (outputing NODATA)
# https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap (targetAlignedPixels==True)
    tmp = gdal.Rasterize(''
        # , gagnet.to_json() if MODE.lower() == 'validation' else wtrshd.to_json()
        # , add=(1 if MODE.lower() == 'validation' else 0)
        , wtrshd.to_json()
        , add=0
        , xRes=X_RES, yRes=Y_RES, allTouched=True, noData=0, burnValues=1, format='MEM'
        , outputType=gdal.GDT_Int16, outputBounds=[llim, blim, rlim, tlim]
        , targetAlignedPixels=True
        # ,targetAlignedPixels=False, # (check: https://gdal.org/programs/gdal_rasterize.html#cmdoption-gdal_rasterize-tap)
        # , outputSRS=f'EPSG:{EPSG_CODE}'#f'{PROJ4_STR}'
#!!
# UPDATE HERE -> outputSRS [in WKT instead of PROJ4]
#!!
    # OR! USE THE PROPER WKT!
        , outputSRS=pp.CRS.from_wkt(WKT_OGC).to_proj4()
        # # , width=(abs(rlim-llim)/X_RES).astype('u2'), height=(abs(tlim-blim)/X_RES).astype('u2')
        )
    CATCHMENT_MASK = tmp.ReadAsArray().astype('u1')
    tmp = None           # flushing!

# DEFINE THE COORDINATES OF THE XY.AXES
    XS, YS = list(map( lambda a,b,c: np.arange(a +c/2, b +c/2, c),
                      [llim,blim], [rlim,tlim], [X_RES,Y_RES] ))
# flip YS??
    YS = np.flipud( YS )      # -> important...so rasters are compatible with numpys

    return BUFFRX_MASK, CATCHMENT_MASK


#~ GENERATES VOID RIO.XARRAY IN THE LOCAL SYSTEM ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def EMPTY_MAP( yyss, xxss, WKT_OGC ):# yyss=YS ; xxss=XS
    # # coordinates from HAD.grid
    # yyss = np.linspace(1167500., -1177500., 470, endpoint=True)
    # xxss = np.linspace(1342500.,  3377500., 408, endpoint=True)
    # void numpy
    void = np.empty( (len(yyss),len(xxss)) )
    void.fill(np.nan)
    # create xarray
    xr_void = xr.DataArray(data=void, dims=['y', 'x'],
        # coords=dict(y=(['y'], YS), x=(['x'], XS), ),
        coords=dict(y=(['y'], yyss), x=(['x'], xxss), ),
        attrs=dict(_FillValue=np.nan, units='mm', ),
        )
    # xr_void = xr.DataArray(data=void, name='rain', dims=['time','lat', 'lon'],
    #     coords=dict(time=(['time'],np.r_[1000,2000,3000,4000,5000]),
    #     lat=(['lat'], np.r_[1,2,3,4,5,6,7]), lon=(['lon'], np.r_[1,2,3]), ),
    #     attrs=dict(_FillValue=np.nan, units='mm', ))
    # assign CRS
    xr_void.rio.write_crs(rio.crs.CRS( WKT_OGC ),
                          grid_mapping_name='spatial_ref', inplace=True)
    # xr_void.to_netcdf('00.nc')
    return xr_void


#~ READING & REPROJECTING a REALIZATION RAINFALL.FIELD ~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
def READ_REALIZATION( RAIN_MAP, SUBGROUP, YS, XS, WKT_OGC ):
    global xoid
# create empty xarray
    xoid = EMPTY_MAP( YS, XS, WKT_OGC )

    xile = rio.open_rasterio( RAIN_MAP, group=SUBGROUP )
    # xile = rio.open_rasterio( RAIN_MAP, group='Grid')[0]
    # xile = rio.open_rasterio( RAIN_MAP, group='somethn_01')#[0]

# REMOVING the annoying BAND dimension (assuming we only have ONE band!)
    if 'band' in list(xile.dims):
        for x in list(xile.data_vars):
            # https://stackoverflow.com/a/41836191/5885810
            xile[ x ] = xile[ x ].sel(band=1, drop=True)
        xile = xile.drop_dims(drop_dims='band')

    # try:
    #     xile[ x ] =
    # except IndexError:
    #     PASS??

    # where is stored?
    xvar = xile.rio.grid_mapping
    # actual crs
    xcrs = xile.rio.crs
    # # trasform4fun
    # xtra = xile.rio.transform()
    xile.close()

    if xcrs==None:
    ## WARNING!: FISHING for CRS can BE.TRICKY
    ## ---------------------------------------
        print(f'\nFishing CRS in "{RAIN_MAP}"... ', end='', flush=True)

        assert xvar!=None, f"CRS NOT FOUND!\nSTORM3 couldn't discern any CRS in "\
            f'{RAIN_MAP}. Please, some CRS is properly stored in the aformentioned file.'

    # here we assume the CRS.exist & can be retrieved
    # ...we also assume that it's in "crs_wkt" but it can also be in "spatial_ref"
        alter = xr.open_dataset( RAIN_MAP )#, decode_coords='all')
        alter_wkt = rio.crs.CRS( alter[ xvar ].attrs['crs_wkt'] )
        # alter_wkt = rio.crs.CRS( alter[ xvar ].attrs['spatial_ref'] )
        alter.rio.write_crs(alter_wkt, grid_mapping_name='spatial_ref', inplace=True)
        # alter.rio.crs
    # update "xile" now with 'readable' CRS
        xile = alter
        xvar = xile.rio.grid_mapping
        xcrs = xile.rio.crs
        print(f'CRS found! [{xcrs}]')

# here we assume that any two different CRSs can be reprojected unto each other
    '''
FAILS with GOOD CRS
    '''
    if xoid.rio.crs.to_string() != xcrs.to_string():
        # xcrs.is_geographic
    # renaming coordinates for 'easy' reprojection?
    # https://www.geeksforgeeks.org/python-get-dictionary-keys-as-a-list/
        c_xoid = list( xoid.coords.dims )
        # ['y', 'x']
        # ['lat', 'lon']
        c_xile = list( xile.coords.dims )
        # ['lat', 'lon']
        # ['band', 'x', 'y']

        # https://stackoverflow.com/a/176921/5885810
        c_ids = list(map(lambda i:c_xile.index(i), c_xoid))

    # assuming LAT goes first
    # https://www.geeksforgeeks.org/python-convert-two-lists-into-a-dictionary/
    # https://stackoverflow.com/a/56163051/5885810  -> rename coordinates
    # https://stackoverflow.com/a/51988240/5885810 -> slicing lists
        # # the line below gives WARNING
        # xile = xile.rename( dict(map(lambda i,j : (i,j) ,
        #     list(map(c_xile.__getitem__, c_ids)), c_xoid)) )
        xile = xile.set_index(indexes=dict(zip( list(map(c_xile.__getitem__, c_ids)), c_xoid )),)

    # reprojection happens here
        pile = xile.rio.reproject_match(xoid, resampling=Resampling.nearest )

    # # some.VISUALISATION (assuming RAIN is the variable!!)
    # # ----------------------------------------------------
    # import cmaps
    # # from cmcrameri import cm as cmc
    # cmaps.precip2_17lev
    # # cmaps.wh_bl_gr_ye_re
    # # cmaps.WhiteBlueGreenYellowRed
    # xile.rain.plot(cmap='precip2_17lev', levels=10,vmin=100,vmax=1000, add_colorbar=True)#, robust=True)#, ax=ax)
    # pile.rain.plot(cmap='precip2_17lev', levels=10,vmin=100,vmax=1000, add_colorbar=True)#, robust=True)#, ax=ax)

        xile = pile
        xvar = xile.rio.grid_mapping
        # actual crs
        xcrs = xile.rio.crs

    return xile
    # class RAINFIELD:
    #     rain = xile
    #     void = xoid

    # return RAINFIELD

    # pile.to_netcdf('realisation_had.nc', mode='w', engine='netcdf4',
    #     encoding={'rain':{'dtype':'f4','zlib':True,'complevel':9, 'grid_mapping':pile.rio.grid_mapping},
    #               'mask':{'dtype':'u1', 'grid_mapping':pile.rio.grid_mapping}, })#,'_FillValue':0

    # # puta = rio.open_rasterio( './realisation_had.nc' )
    # # puta.rio.grid_mapping
    # # puta.rio.crs
    # # puta.rio.transform()
    # # puta.close()


#~ IMAGE SEGMENTATION via SCIKIT-LEARN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# following: https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb

# def KREGIONS( REG, N_C=4 ):# REG=np.flip(real.rain[0,:].data, axis=0) # REG=meal
# # REG: 2D.numpy representing the rainfall.field [realization]
# # N_C: number of clusters
#     # transform the (RGB?) field into 1D.numpy
#     X = REG.reshape(-1, 1)
#     # kmeans = KMeans(n_clusters=3, init=np.array([[70],[220],[800]]), n_init='auto').fit( X )
#     kmeans = KMeans(n_clusters=N_C, n_init=11).fit( X )

# # # we're interested in the classes.. not in the actual means
# #     segmented_img = kmeans.cluster_centers_[ kmeans.labels_ ]
# #     # bsctransform to 2D.numpy
# #     segmented_img = segmented_img.reshape( REG.shape )

#     LAB = kmeans.labels_.reshape( REG.shape )
#     # plt.imshow(LAB, origin='lower', cmap='turbo', interpolation='none')#.resampled(3))
#     # # plt.savefig('realization.pdf', bbox_inches='tight',pad_inches=0.02)
#     # # plt.close() ; plt.clf()

# # https://stackoverflow.com/a/25715954/5885810  # np.object to np.string
# # https://www.w3resource.com/numpy/string-operations/strip.php  # strip np.string.arrays
#     KAT = np.char.strip(kmeans.get_feature_names_out().astype('U'), 'kmeans')

#     return LAB, dict(zip(KAT, kmeans.cluster_centers_))#, np.unique(kmeans.labels_)
def KREGIONS( REG, N_C=4 ):
# REG=real.rain.data#np.flip(real.rain[0,:].data, axis=0)
# REG=np.ma.MaskedArray( real.rain.data, ~CATCHMENT_MASK.astype('bool') )

# REG: 2D.numpy representing the rainfall.field [realization]
# N_C: number of clusters

    # nans outside mask
    REG[ REG.mask ] = np.nan
    # ravel and indexing
    ravl = REG.ravel()
    idrs = np.arange(len(ravl))[ ~np.isnan( ravl ) ]
    # transform the non-void (RGB?) field into 1D.numpy
    X = ravl[ idrs ].data.reshape(-1, 1)
    # kmeans = KMeans(n_clusters=3, init=np.array([[70],[220],[800]]), n_init='auto').fit( X )
    kmeans = KMeans(n_clusters=N_C, n_init=11).fit( X )

# # we're interested in the classes.. not in the actual means
#     segmented_img = kmeans.cluster_centers_[ kmeans.labels_ ]
#     # bsctransform to 2D.numpy
#     segmented_img = segmented_img.reshape( REG.shape )

    # expand the result into void-array
    ravl[ idrs ] = kmeans.labels_
    LAB = ravl.reshape( REG.shape ).data
    # plt.imshow(LAB, origin='lower', cmap='turbo', interpolation='none')#.resampled(3))
    # # plt.savefig('realization.pdf', bbox_inches='tight',pad_inches=0.02)
    # # plt.close() ; plt.clf()

# https://stackoverflow.com/a/25715954/5885810  # np.object to np.string
# https://www.w3resource.com/numpy/string-operations/strip.php  # strip np.string.arrays
    KAT = np.char.strip(kmeans.get_feature_names_out().astype('U'), 'kmeans')

    return LAB, dict(zip(KAT, kmeans.cluster_centers_))#, np.unique(kmeans.labels_)


    # #-TESTING IMAGE.SEGMENTATION (from SCIKIT-LEARN)--------------------------
    # # from: https://github.com/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb
    # import os
    # import urllib.request
    # from matplotlib.image import imread
    # from sklearn.cluster import KMeans

    # # testing LADYBUG
    # images_path = "."
    # DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    # filename = "ladybug.png"
    # print("Downloading", filename)
    # url = DOWNLOAD_ROOT + "images/unsupervised_learning/" + filename
    # urllib.request.urlretrieve(url, os.path.join(images_path, filename))

    # image = imread(os.path.join(images_path, filename))
    # image.shape

    # X = image.reshape(-1, 3)
    # kmeans = KMeans(n_clusters=3, n_init=11, random_state=42).fit(X)
    # segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    # segmented_img = segmented_img.reshape(image.shape)

    # plt.imshow(segmented_img)

    # meansea = np.flip(real.rain[0,:], axis=0)
    # # testing rainfall MEANSEA
    # X = meansea.data.reshape(-1, 1)
    # # kmeans = KMeans(n_clusters=3, init=np.array([[70],[220],[800]]), n_init='auto').fit(X)
    # kmeans = KMeans(n_clusters=3, n_init=11).fit(X)
    # # kmeans = KMeans(n_clusters=4, n_init=11).fit(X)
    # segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    # segmented_img = segmented_img.reshape(meansea.data.shape)

    # plt.imshow(segmented_img, origin='lower', cmap=cmc.hawaii_r)#.resampled(3))
    # #-------------------------------------------------------------------------


#~ MORPHOLOGICAL FILTERING via SCIKIT-IMAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html#opening

def MORPHOPEN( LAB ):
    # new = morphology.opening(LAB, morphology.ellipse(4,4))
    new = morphology.opening(LAB, morphology.ellipse(2,3))
    # plt.imshow(new, origin='lower', cmap='turbo', interpolation='none')
    # # plt.savefig('realization_opening.pdf', bbox_inches='tight',pad_inches=0.02)
    # # plt.close() ; plt.clf()
# "morphology.ellipse(2,3)" does almost the same job as "cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,7)"
    return new.astype('u1')

    # #-ALTERNATIVE (to SCIKIT-IMAGE MORPHOLOGICAL FILTERING) via OPEN.CV-------
    # import cv2

    # # define some RGB space of your zoned.data
    # rgbcol = np.array([[255,255,  0],     # -> 0
    #                     [255,  0,100],     # -> 1
    #                     [100,  0,  0],     # -> 2
    #                     [101,101,101]])    # -> 3
    # BAL = rgbcol.T[:,LAB]
    # # np.unique(BAL)
    # plt.imshow(BAL.swapaxes(0,2).swapaxes(0,1), origin='lower', interpolation='none')

    # # DENOISE works nice... but generates more categorical data/colors
    # new = cv2.fastNlMeansDenoisingColored((BAL.T).astype(np.uint8),None,30,30,7,21)
    # plt.imshow(new.T.swapaxes(0,2).swapaxes(0,1), origin='lower', interpolation='none')

    # # MORPHOLOGY seems to do the trick... but then one might as well use SKIMAGE!!
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,11))
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,7))
    # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4,3))
    # # new = cv2.morphologyEx((BAL.T).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # new = cv2.morphologyEx((BAL.T).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # plt.imshow(new.T.swapaxes(0,2).swapaxes(0,1), origin='lower', interpolation='none')

    # # # if the exercise is to be done in GRAY
    # # LOB = np.array(LAB/LAB.max(), dtype='f4')
    # # vis2 = cv2.cvtColor(LOB, cv2.COLOR_GRAY2BGR)
    # # plt.imshow(vis2)
    # # dst = cv2.fastNlMeansDenoisingColored((vis2*255).astype(np.uint8),None,10,10,7,21)
    # # dst = cv2.fastNlMeansDenoisingColored((vis2*255).astype(np.uint8),None,15,15,14,43)
    # # plt.imshow(dst, origin='lower', interpolation='none')
    # #-------------------------------------------------------------------------


#%% CALL & COMPUTE...


# #-READING NUMPYS& PICKLES-#-------------------
# #-----------------------------------------------------------------------------
# # reading the xported.numpy
# real = np.load('realisation_MAM.npy')
# # # plotting/testing
# # plt.imshow(real, origin='lower', vmin=0,vmax=1000)#, cmap=cmc.tokyo_r)#, interpolation='nearest')

# # reading the xported.mask -> [NEED TO.BE FLIPPED HERE! **XARRAY/NUMPY**]
# dhad = np.flip( np.load('tmp-raster_mask-buff.npy'), axis=0 )
# # # or reading PICKLEs
# # import pickle
# # with open('tmp-raster_mask-buff.pkl','rb') as f:
# #     dhod = pickle.load(f)
# # # or...
# # dhod = np.load('tmp-raster_mask-buff.pkl', allow_pickle=True)
# # # plotting/testing
# plt.imshow(dhad, origin='lower', interpolation='none')
# #-----------------------------------------------------------------------------


def REGIONALISATION( real, CLUSTERS, BUFFRX_MASK, CATCHMENT_MASK ):#, xoid ):

#-WORK UNDER PROGRESS... TO MAKE COMPATIBLE MORE OPTIONS!!----------------------
    # npreg, cdic = KREGIONS( real.rain[0,:].data, CLUSTERS )
    # npreg, cdic = KREGIONS( np.flip(real.rain[0,:].data, axis=0), CLUSTERS )
    # npreg, cdic = KREGIONS( np.flip(real.rain['band'==1,:].data, axis=0), CLUSTERS )
#-------------------------------------------------------------------------------
# # FOR A MASK IN THE WHOLE [RECTANGULAR] DOMAIN USE:
#     mask_regn = np.ma.MaskedArray( real.rain.data.copy(), False )
# FOR A MASK IN THE WHOLE OF THE CATCHMENT USE:
    mask_regn = np.ma.MaskedArray( real.rain.data.copy(), ~CATCHMENT_MASK.astype('bool') )
    # ... in the line below (previous cases) BAND was removed!
    # mask_regn = np.ma.MaskedArray( real.rain['band'==1,:].data.copy(), ~CATCHMENT_MASK.astype('bool') )

    npreg, cdic = KREGIONS( mask_regn, CLUSTERS )

# # reduce the vertices of a shape -> MORPHOLOGY
#     mopen = MORPHOPEN( npreg )
# ... or NOT!! :D
    mopen = npreg

# NUMPY to SHAPE
# .rio.transform() IS QUITE OF THE ESSENCE HERE!
    lopen = list( shapes(mopen, mask=BUFFRX_MASK, connectivity=4, transform=real.rio.transform()) )
# remove NAN.regions??
    # https://stackoverflow.com/a/25050572/5885810
    # https://stackoverflow.com/a/3179137/5885810
    lopen = [x for x, y in zip(lopen, ~np.isnan(list(zip(*lopen))[-1])) if y]
    lopen = list(map(lambda x:dict(geometry=x[0], properties={'label':f'region{int(x[-1])}', }), lopen))
# into GEOPANDAS
    feats = gpd.GeoDataFrame.from_features( {'type':'FeatureCollection','features':lopen} )
    # # line below is an alternative... BUT you mig have troubles grouping it
    # feats = gpd.GeoDataFrame.from_dict( list( shapes(mopen, mask=BUFFRX_MASK, connectivity=4, transform=real.rio.transform()) ),)

# grouping to retrieve just the CLUSTER.masks (the output is a Series)
    nasks = feats.groupby(by='label').apply(lambda x: x.unary_union)
    # nasks[0]
    # nasks[1]
    # nasks[2]
    # nasks[3]

    # # do we obtain HAD if merging all masks?
    # feast.unary_union

# turn-back them into GeoPandas
    masks = gpd.GeoDataFrame(geometry=nasks)
    # masks.geometry.iloc[0]
    # masks.geometry.loc['region0']
    # masks.loc['region0'].geometry
    # masks.geometry.xs('region0')

# update the REAL -> realization.map
    # real['catchm'] = xr.DataArray(CATCHMENT_MASK, coords=[YS, XS], dims=['y', 'x'])
    # real['buffer'] = xr.DataArray(BUFFRX_MASK, coords=[YS, XS], dims=['y', 'x'])
    # real['kmeans'] = xr.DataArray(npreg, coords=[YS, XS], dims=['y', 'x'])
    # real['region'] = xr.DataArray(mopen, coords=[YS, XS], dims=['y', 'x'])
# https://realpython.com/iterate-through-dictionary-python/
    for keys, values in\
        dict(zip(['catchm'      ,'buffer'   ,'kmeans','region'],
                 [CATCHMENT_MASK,BUFFRX_MASK,npreg   ,mopen   ])).items():
        real[keys] = xr.DataArray(values, coords=xoid.coords, dims=xoid.coords.dims)
        # real[keys] = xr.DataArray(values, coords=real.coords, dims=real.coords.dims)
# trims "real['region']"
    # trimming around the CATCHMENT_MASK is what we want; as we compute PTOT within "catchm"
    real['region'] = xr.where(real.catchm==1, real.region, -1)
    # real['region'] = xr.where(real.buffer==1, real.region, -1) #-trims around the BUFFER -> (this we want NOT!)

#-THESE 3 vars ARE IN THE ORDER OF cdic.keys()
# new means (regions inside the HAD)
    # old_ks = list(map(lambda x:real.rain.where(real.kmeans==int(x)).mean().data, cdic.keys()))
    new_ks = list(map(lambda x:real.rain.where(real.catchm==1, np.nan).where(
        real.kmeans==int(x)).mean().data, cdic.keys()))
# numpy masks
    # ... 1st transform K-mean into 1s (because of the 0 K-mean); and the assign 0 everywhere else
    reg_np = list(map(lambda x:
        real.region.where(real.region!=int(x), 1).where(real.region==int(x), 0).data.astype('u1'), cdic.keys()))
# shapes
    # # the line below are "pandas.core.series.Series"
    # zhapez = list(map(lambda x:masks.loc[f'region{int(x)}'], cdic.keys()))
    # ...in case "pandas.core.series.Series" try converting them into "geopandas.geodataframe.GeoDataFrame"
    zhapez = list(map(lambda x:gpd.GeoDataFrame(
        geometry=masks.loc[f'region{int(x)}'], crs=xoid.rio.crs).set_index(RangeIndex(0,1,1)), cdic.keys()))
    # zhapez = list(map(lambda x:gpd.GeoDataFrame(geometry=masks.loc[f'region{int(x)}']).set_index(RangeIndex(0,1,1)), cdic.keys()))
# grouping the output into a dict
    output = dict(zip(('mask','npma','rain'),(zhapez, reg_np, new_ks)))
    output['kmeans'] = xr.where(real.catchm==1, real.kmeans, -1).data.astype('i1')

    return output

    # # CHECK THE CORRECT ORIENTATION of NNUMPYs
    # plt.imshow(BUFFRX_MASK, origin='lower', cmap='turbo', interpolation='none')#.resampled(3)))
    # plt.imshow(reg_np[0],   origin='lower', cmap='turbo', interpolation='none')
    # plt.imshow(reg_np[1],   origin='lower', cmap='turbo', interpolation='none')
    # plt.imshow(reg_np[-1],  origin='lower', cmap='turbo', interpolation='none')

    # # some.VISUALISATION (assuming RAIN is the variable!!)
    # # ----------------------------------------------------
    # import cmaps
    # from cmcrameri import cm as cmc
    # cmaps.precip2_17lev
    # # cmaps.wh_bl_gr_ye_re
    # # cmaps.WhiteBlueGreenYellowRed

    # fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    # real.rain.plot(cmap='precip2_17lev', levels=10,vmin=100,vmax=1000, add_colorbar=True, ax=ax,# 'orientation':'vertical',
    #     cbar_kwargs={'shrink':0.5, 'aspect':30, 'label':'seasonal rainfall [mm]', 'pad':+.02,})#, robust=True)
    # #-HAVING COMPUTED "wtrshd" & "BUFFRX" STANDALONE
    # wtrshd.plot(ax=ax, fc='none', ec='xkcd:electric pink', lw=.9)
    # BUFFRX.plot(ax=ax, fc='none', ec='xkcd:blood orange', lw=.3)
    # # plt.show()
    # plt.savefig(f'realization_test00.png', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
    # plt.close()
    # plt.clf()

    # fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    # real.kmeans.plot(cmap=cmc.batlowW_r, levels=np.r_[-1:5], add_colorbar=True, ax=ax,# 'orientation':'vertical',
    #     cbar_kwargs={'shrink':0.5, 'aspect':30, 'label':'K-means [-]', 'pad':+.040,})#, robust=True)
    # #-HAVING COMPUTED "wtrshd" & "BUFFRX" STANDALONE
    # wtrshd.plot(ax=ax, fc='none', ec='xkcd:electric pink', lw=.9)
    # BUFFRX.plot(ax=ax, fc='none', ec='xkcd:blood orange', lw=.3)
    # # plt.show()
    # plt.savefig(f'realization_test01.png', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
    # plt.close()
    # plt.clf()

    # fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    # real.region.plot(cmap=cmc.batlowW_r, levels=np.r_[-1:5], add_colorbar=True, ax=ax,# 'orientation':'vertical',
    #     cbar_kwargs={'shrink':0.5, 'aspect':30, 'label':'K-means + morphOpen [-]', 'pad':+.040,})#, robust=True)
    # # masks.plot(ax=ax, cmap='viridis', ec='none',)
    # #-HAVING COMPUTED "wtrshd" & "BUFFRX" STANDALONE
    # wtrshd.plot(ax=ax, fc='none', ec='xkcd:electric pink', lw=.9)
    # BUFFRX.plot(ax=ax, fc='none', ec='xkcd:blood orange', lw=.3)
    # # plt.show()
    # plt.savefig(f'realization_test02.png', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
    # plt.close()
    # plt.clf()

    # fig, ax = plt.subplots(figsize=(10,10), dpi=300)
    # real.region.plot(cmap=cmc.batlowW_r, levels=np.r_[-1:5], add_colorbar=True, ax=ax,# 'orientation':'vertical',
    #     cbar_kwargs={'shrink':0.5, 'aspect':30, 'label':'K-means + morphOpen [-]', 'pad':+.040,})#, robust=True)
    # # masks.plot(ax=ax, cmap='viridis', ec='none',)
    # #-HAVING COMPUTED "wtrshd" & "BUFFRX" STANDALONE
    # masks.plot(ax=ax, fc='none', ec='xkcd:bright magenta', lw=.9)
    # wtrshd.plot(ax=ax, fc='none', ec='xkcd:chartreuse', lw=.7)
    # BUFFRX.plot(ax=ax, fc='none', ec='xkcd:blood orange', lw=.2)
    # # plt.show()
    # plt.savefig(f'realization_test03.png', bbox_inches='tight',pad_inches=0.02, facecolor=fig.get_facecolor())
    # plt.close()
    # plt.clf()


#%%

if __name__ == '__main__':
    ## RAINFALL_MAP stored in some NC.like.these...
    ## --------------------------------------------
    RAIN_MAP = '../CHIMES/3B-HHR.MS.MRG.3IMERG.20101010-S100000-E102959.0600.V06B.HDF5'     # no.CRS at all!
    RAIN_MAP = './realisation_MAM_crs-wrong.nc'                         # no..interpretable CRS
    RAIN_MAP = './model_output/SOME_test.nc'                            # HAD.projected CRS
    RAIN_MAP = './model_output/SOME_test-for.nc'                        # HAD.projected CRS

    RAIN_MAP = './realisation_MAM_crs-OK.nc'                            # yes.interpretable CRS
    SUBGROUP = ''
    CLUSTERS = 4

    CRSS()
    # BUFFER in meters! -> buffer distance (out of the HAD)
    BUFFRX_MASK, CATCHMENT_MASK = NUMPY_MASKS( './model_input/HAD_basin.shp', 8000. )
    # read the REALIZATION
    real = READ_REALIZATION( RAIN_MAP, SUBGROUP, YS, XS, WKT_OGC )
    regs = REGIONALISATION( real, CLUSTERS, BUFFRX_MASK, CATCHMENT_MASK )
    # regs = REGIONALISATION( real.rain, CLUSTERS, BUFFRX_MASK, CATCHMENT_MASK, real.void )


#%% IMPROVE READING with RASTERIO??

#     ncrx = xr.open_dataset('realisation_MAM.nc', engine='rasterio', decode_coords="all",
#                             decode_times=True, decode_cf=True)#, use_cftime=True)

#     cuca = xr.open_dataset('realisation_MAM.nc', engine="rasterio")
#                             decode_times=True, decode_cf=True, use_cftime=True)

#     ncxr = xr.open_dataset('realisation_MAM.nc', decode_coords='all')


#     ncrio = rio.open_rasterio('realisation_MAM.nc')

#     perra = rio.open_rasterio('realisation_MAMz.nc')
#     perra.rio.crs
#     perra.rio.transform()
#     perra.rio.grid_mapping
#     perra.close()

#     puta = rio.open_rasterio('../CHIMES/3B-HHR.MS.MRG.3IMERG.20101010-S100000-E102959.0600.V06B.HDF5')
#     puta = xr.open_dataset('../CHIMES/3B-HHR.MS.MRG.3IMERG.20101010-S100000-E102959.0600.V06B.HDF5', group='Grid', decode_coords='all')

# """
# C:\Users\manuel\miniconda3\envs\py39\lib\site-packages\rasterio\__init__.py:304: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
#   dataset = DatasetReader(path, driver=driver, sharing=sharing, **kwargs)
# C:\Users\manuel\miniconda3\envs\py39\lib\site-packages\rioxarray\_io.py:1132: NotGeoreferencedWarning: Dataset has no geotransform, gcps, or rpcs. The identity matrix will be returned.
# """

#     culo = rio.open_rasterio('./model_output/SOME_test.nc', group='somethn_01', decode_coords='all')
#     culo.rio.crs
#     culo.rio.set_crs(rasterio.crs.CRS.from_wkt(culo.crs.attrs['crs_wkt']), inplace=True)
#     culo.rio.crs

#     teta = culo.rio.crs
#     teta.is_geographic
#     teta.is_projected


# #%% TOY NETCDF4

# import netCDF4 as nc4
# import numpy as np


# void = np.round(np.random.rand(4,6,3) *10, 1)

# sub_grp = nc4.Dataset('sax2.nc', 'w', format='NETCDF4')

# sub_grp.createDimension('y', 6)
# sub_grp.createDimension('x', 3)
# sub_grp.createDimension('z', None)
# vlen_t = sub_grp.createVLType(np.uint64, 'z_t')
# # vlen_t = sub_grp.createVLType(np.uint64, 'phony_z')

# # timexx = sub_grp.createVariable('time', 'u8', ('z'), fill_value=0)#, chunksizes=chunkt)
# timexx = sub_grp.createVariable('time', vlen_t, ('z'))#, fill_value=0)

# yy = sub_grp.createVariable('lat', 'i4', dimensions=('y'))
# xx = sub_grp.createVariable('lon', 'i4', dimensions=('x'))

# # timexx[:]= np.r_[1000,2000,3000,4000].astype('u8')
# timexx[:]= np.array([np.array(1000,dtype='u8'),np.array(2000,dtype='u8'),
#                       np.array(3000,dtype='u8'),np.array(4000,dtype='u8')],dtype='object')

# yy[:] = np.r_[1,2,3,4,5,6]
# xx[:] = np.r_[1,2,3]

# ncvarx = sub_grp.createVariable('rain', datatype='f8'
#     ,dimensions=('z','y','x'), zlib=True, complevel=9, fill_value=np.nan)#,least_significant_digit=3)
# ncvarx[:] = void

# # ass = np.empty( (4) )
# # ass[0:3] = np.r_[1000,2000,3000]
# # timexx[:] = ass.astype('u8')
# # # timexx[:]= np.r_[1000,2000,3000].astype('u8')

# ncvarx[-1,:,:] = np.nan

# timexx[:]= np.array([np.array(1000,dtype='u8'),np.array(2000,dtype='u8'),
#                       np.array(3000,dtype='u8') ],dtype='object')

# sub_grp.close()
