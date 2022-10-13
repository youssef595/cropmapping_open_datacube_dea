import sys
import datacube
import numpy as np
import xarray as xr
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from odc.algo import xr_geomedian


def feature_layer_gm_1(ds, era):
    
    # Normalize bands 
    for band in ds.data_vars:
        if band not in ["sdev", "bcdev"]:
            ds[band] = ds[band] / 10000

    # Add indices
    feature_data = calculate_indices(
        ds,
        index=["NDVI", "LAI", "EVI", "SAVI", "NDMI"],
        drop=False,
        normalise=False,
        collection="s2")
    
    # Normalize geomads using log
    feature_data["sdev"] = -np.log(feature_data["sdev"])
    feature_data["bcdev"] = -np.log(feature_data["bcdev"])
    feature_data["edev"] = -np.log(feature_data["edev"])
    
    for band in feature_data.data_vars:
        feature_data = feature_data.rename({band: band + era})
    
    return feature_data


def gm_1(query):
    
    # Connect to the datacube
    dc = datacube.Datacube(app='feature_layers')
    
    # load S2 geomedian
    ds = dc.load(product='gm_s2_semiannual', **query)
    
    # load the data
    dss = {"S1_n": ds.isel(time=0),
           "S2_n": ds.isel(time=1), }
    
    # Create features
    epoch1 = feature_layer_gm_1(dss["S1"], era="_S1")
    epoch2 = feature_layer_gm_1(dss["S2"], era="_S2")
    
    #merge features
    result = xr.merge([epoch1, epoch2], compat="override")
    
    return result.astype(np.float32).squeeze()


####################################################################

def feature_layer_gm1_1(ds, era):
    
    # Normalize bands 
    for band in ds.data_vars:
        if band not in ["sdev", "bcdev"]:
            ds[band] = ds[band] / 10000

    # Add indices
    feature_data = calculate_indices(
        ds,
        index=["NDVI", "LAI", "EVI", "SAVI", "NDMI"],
        drop=False,
        normalise=False,
        collection="s2",
    )
    
    # Normalize geomads using log
    feature_data["sdev"] = -np.log(feature_data["sdev"])
    feature_data["bcdev"] = -np.log(feature_data["bcdev"])
    feature_data["edev"] = -np.log(feature_data["edev"])
    
    for band in feature_data.data_vars:
        feature_data = feature_data.rename({band: band + era})
    
    return feature_data


def gm1_1(query):
    
    # Connect to the datacube
    dc = datacube.Datacube(app='feature_layers')
    
    # load S2 geomedian
    ds = dc.load(product='gm_s2_semiannual', **query)
    
    # load the data
    dss = {"S2_n_1": ds.isel(time=1),
           "S1_n": ds.isel(time=2)}
    
    # Create features
    epoch1 = feature_layer_gm1_1(dss["S2_n_1"], era="_S2_n_1")
    epoch2 = feature_layer_gm1_1(dss["S1_n"], era="_S1_n")

    result = xr.merge([epoch1, epoch2], compat="override")

    return result.astype(np.float32).squeeze()


####################################################################

