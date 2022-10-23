import numpy as np
import datacube
from datacube.utils import geometry
import odc.ui
from odc.ui import to_png_data
from odc.ui import mk_data_uri
from odc.ui import to_jpeg_data
from odc.algo import to_rgba, is_rgb
import xarray as xr
import cv2
import geopandas as gpd
from typing import Tuple
from typing import Union
from typing import Optional
from PIL import Image
from PIL import ImageEnhance
from shapely.geometry.polygon import Polygon
from shapely.geometry import LineString, MultiPolygon
from shapely.ops import polygonize, unary_union
import leafmap
from ipyleaflet import Map, basemaps
from ipywidgets import widgets as w


def simplify_crop(polygon):
    lineString = LineString(polygon)
    lr = LineString(lineString.coords[:] + lineString.coords[0:1])
    mls = unary_union(lr)
    mp = MultiPolygon(list(polygonize(mls)))
    mp = mp.buffer(0)
    if mp.geom_type == 'MultiPolygon':
        new_polygon = mp[0].exterior.coords[:]
    elif mp.geom_type == 'Polygon':
        new_polygon = mp.exterior.coords[:]
    new_polygon = [(int(x), int(y)) for (x, y) in new_polygon]
    return new_polygon