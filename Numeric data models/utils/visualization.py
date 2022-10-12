import leafmap.foliumap as leafmapfol
import leafmap
import ast
import geopandas as gpd
from shapely.geometry import Polygon
import datacube
from odc.ui import with_ui_cbk
from deafrica_tools.plotting import rgb, display_map
%matplotlib inline
import matplotlib.pyplot as plt
import geopandas as gpd
from datacube.utils import geometry
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.plotting import rgb, map_shapefile
from deafrica_tools.spatial import xr_rasterize
from deafrica_tools.classification import HiddenPrints
from odc.io.cgroups import get_cpu_quota
from deafrica_tools.classification import collect_training_data
import numpy as np
import xarray as xr
from odc.algo import xr_reproject
from pyproj import Proj, transform
from datacube.utils.geometry import assign_crs
from datacube.testutils.io import rio_slurp_xarray
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.dask import create_local_dask_cluster
pd.set_option('display.max_colwidth', 500)



def plot_shapefile(crop_vectors):
    center = (crop_vectors.total_bounds[0] + crop_vectors.total_bounds[2])*0.5, (crop_vectors.total_bounds[1] + crop_vectors.total_bounds[3])*0.5
    map_select = leafmapfol.Map(center= center, draw_control=True,layer_control=True)
    map_select.add_basemap("Esri.WorldImagery")
    map_select.add_gdf(crop_vectors, layer_name = 'polygons', fill_colors=["red", "green", "blue", "yellow", "black"])
    map_select.add_gdf(gpd.read_file('Supplementary_data/maroc_regions.geojson'), layer_name = 'regions')
    return map_select