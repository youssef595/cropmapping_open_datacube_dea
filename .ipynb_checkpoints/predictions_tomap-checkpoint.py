import datacube
from datacube.utils import geometry
import odc.ui
from odc.ui import to_png_data
from odc.ui import mk_data_uri
from odc.ui import to_jpeg_data
from odc.algo import to_rgba, is_rgb
import xarray as xr

#dea tools
from deafrica_tools.plotting import display_map, rgb
from deafrica_tools.bandindices import calculate_indices
from deafrica_tools.datahandling import load_ard
from deafrica_tools.plotting import map_shapefile
from deafrica_tools.dask import create_local_dask_cluster

#geodata plotting
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

#map
import folium
import leafmap.foliumap as leafmapfol
import leafmap
from ipyleaflet import Map, basemaps
from ipywidgets import widgets as w
import ipyleaflet

#typing
from typing import Tuple
from typing import Union
from typing import Optional

#streamlit
#from streamlit_folium import folium_static
#import streamlit as st
#from streamlit_option_menu import option_menu
from datacube.utils import geometry
from deafrica_tools.dask import create_local_dask_cluster
#from st_btn_select import st_btn_select

#text
#from annotated_text import annotated_text

#time
import datetime
import time

#image
from PIL import Image
import numpy as np

#modeling
import itertools
import joblib
from deafrica_tools.classification import predict_xr
from deafrica_tools.spatial import xr_rasterize
from datacube.utils.geometry import box
from datacube.testutils.geom import epsg4326


def xr_bounds(x, crs=None) -> Tuple[Tuple[float, float], Tuple[float, float]]:

    def get_range(a: np.ndarray) -> Tuple[float, float]:
        b = (a[1] - a[0]) * 0.5
        return a[0] - b, a[-1] + b

    if "latitude" in x.coords:
        r1, r2 = (get_range(a.values) for a in (x.latitude, x.longitude))
        p1, p2 = ((r1[i], r2[i]) for i in (0, 1))
        return p1, p2

    if crs is None:
        geobox = getattr(x, "geobox", None)
        if geobox:
            crs = geobox.crs

    if crs is None:
        raise ValueError("Need to supply CRS or use latitude/longitude coords")

    if not all(d in x.coords for d in crs.dimensions):
        raise ValueError("Incompatible CRS supplied")

    (t, b), (l, r) = (get_range(x.coords[dim].values) for dim in crs.dimensions)

    l, b, r, t = box(l, b, r, t, crs).to_crs(epsg4326).boundingbox
    return ((t, r), (b, l))



def mk_image_overlay(
    xx: Union[xr.Dataset, xr.DataArray],
    clamp: Optional[float] = None,
    bands: Optional[Tuple[str, str, str]] = None,
    layer_name="Image",
    fmt="png",
    **opts
):
    """Create ipyleaflet.ImageLayer from raster data.
    xx - xarray.Dataset that will be converted to RGBA or
         xarray.DataArray that is already in RGB(A) format
    clamp, bands -- passed on to to_rgba(..), only used when xx is xarray.Dataset
    Returns
    =======
    ipyleaflet.ImageOverlay or a list of them one per time slice
    """

    comp, mime = dict(
        png=(to_png_data, "image/png"),
        jpg=(to_jpeg_data, "image/jpeg"),
        jpeg=(to_jpeg_data, "image/jpeg"),
    ).get(fmt.lower(), (None, None))

    if comp is None or mime is None:
        raise ValueError("Only support png an jpeg formats")

    if "time" in xx.dims:
        nt = xx.time.shape[0]
        if nt == 1:
            xx = xx.isel(time=0)
        else:
            return [
                mk_image_overlay(
                    xx.isel(time=t),
                    clamp=clamp,
                    bands=bands,
                    layer_name="{}-{}".format(layer_name, t),
                    fmt=fmt,
                    **opts
                )
                for t in range(nt)
            ]

    if isinstance(xx, xr.Dataset):
        rgba = to_rgba(xx, clamp=clamp, bands=bands)
    else:
        if not is_rgb(xx):
            raise ValueError("Expect RGB xr.DataArray")
        rgba = xx

    im_url = mk_data_uri(comp(rgba.values, **opts), mime)
    return im_url, rgba