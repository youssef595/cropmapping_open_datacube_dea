import sys
import datacube
import numpy as np
import xarray as xr
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from odc.algo import xr_geomedian
from feature_utils import *


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
    epoch1 = feature_layer_gm_1(dss["S1_n"], era="_S1_n")
    epoch2 = feature_layer_gm_1(dss["S2_n"], era="_S2_n")
    
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

def cus1_gm_1(query):
    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")
    
    ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        verbose=False,
        **query)
    
    time_ranges = {
        "Q_n_1": slice(f"{get_years(ds)[0]}-10-01", f"{get_years(ds)[0]}-12-31"),
        "Q1_n": slice(f"{get_years(ds)[1]}-01-01", f"{get_years(ds)[1]}-03-31"),
        "Q2_n": slice(f"{get_years(ds)[1]}-04-01", f"{get_years(ds)[1]}-07-31")}

    # Apply geomedian over time ranges and calculate band indices
    s2_geomad_list = apply_function_over_custom_times(ds, geomedian_with_indices_wrapper, "s2", time_ranges)
    ds_list = []
    ds_list.extend(s2_geomad_list)
    ds_final = xr.merge(ds_list)

    return ds_final


####################################################################

def cus_gm_3(query):
    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")
    
    ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        verbose=False,
        **query)
    
    time_ranges = {
        "Q0_n_1": slice(f"{get_years(ds)[0]}-10-01", f"{get_years(ds)[0]}-11-15"),
        "Q1_n_1": slice(f"{get_years(ds)[0]}-11-16", f"{get_years(ds)[0]}-12-31")}

    # Apply geomedian over time ranges and calculate band indices
    s2_geomad_list = apply_function_over_custom_times(ds, geomedian_with_indices_wrapper, "s2", time_ranges)
    ds_list = []
    ds_list.extend(s2_geomad_list)
    ds_final = xr.merge(ds_list)

    return ds_final


####################################################################

def cus1_gm_3(query):
    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")
    
    ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        verbose=False,
        **query)
    
    time_ranges = {
        "Q0_n_1": slice(f"{get_years(ds)[0]}-10-01", f"{get_years(ds)[0]}-11-15"),
        "Q1_n_1": slice(f"{get_years(ds)[0]}-11-16", f"{get_years(ds)[0]}-12-31"),
        "Q0_n": slice(f"{get_years(ds)[1]}-01-01", f"{get_years(ds)[1]}-02-15"),
        "Q1_n": slice(f"{get_years(ds)[1]}-02-16", f"{get_years(ds)[1]}-03-31")}

    # Apply geomedian over time ranges and calculate band indices
    s2_geomad_list = apply_function_over_custom_times(ds, geomedian_with_indices_wrapper, "s2", time_ranges)
    ds_list = []
    ds_list.extend(s2_geomad_list)
    ds_final = xr.merge(ds_list)

    return ds_final


####################################################################

def cus2_gm_3(query):
    # Connnect to datacube
    dc = datacube.Datacube(app="crop_type_ml")
    
    ds = load_ard(
        dc=dc,
        products=["s2_l2a"],
        verbose=False,
        **query)
    
    time_ranges = {
        "Q0_n_1": slice(f"{get_years(ds)[0]}-10-01", f"{get_years(ds)[0]}-11-15"),
        "Q1_n_1": slice(f"{get_years(ds)[0]}-11-16", f"{get_years(ds)[0]}-12-31"),
        "Q0_n": slice(f"{get_years(ds)[1]}-01-01", f"{get_years(ds)[1]}-02-15"),
        "Q1_n": slice(f"{get_years(ds)[1]}-02-16", f"{get_years(ds)[1]}-03-31"),
        "Q2_n": slice(f"{get_years(ds)[1]}-04-01", f"{get_years(ds)[1]}-05-15"),
        "Q3_n": slice(f"{get_years(ds)[1]}-05-16", f"{get_years(ds)[1]}-07-31")}

    # Apply geomedian over time ranges and calculate band indices
    s2_geomad_list = apply_function_over_custom_times(ds, geomedian_with_indices_wrapper, "s2", time_ranges)
    ds_list = []
    ds_list.extend(s2_geomad_list)
    ds_final = xr.merge(ds_list)

    return ds_final


####################################################################

def feature_layer_ts_grpw_(ds, era):
    #add indices
    feature_data = calculate_indices(
        ds,
        index=["NDVI", "EVI", "MSAVI", "NDMI"],
        drop=False,
        normalise=True,
        collection="s2")
    
    for band in feature_data.data_vars:
        feature_data = feature_data.rename({band: band + era})
    
    return feature_data


def ts_grpw_(query):
    
    #connect to the datacube
    dc = datacube.Datacube(app='feature_layers')
    #load S2
    ds = dc.load(product='s2_l2a', **query, resampling={"mask": "nearest", "*": "bilinear"})
    #ds = load_ard(dc=dc, products=['s2_l2a'], **query)
    ds = ds.groupby('time.week').median(dim='time')
    keys = [ds.week.values[i].astype(str) for i in range(ds.week.values.shape[0])]
    values = [ds.isel(week=i) for i in range(ds.week.values.shape[0])]
    dss = dict(zip(keys, values))
    # load the data
    epochs = [feature_layer_ts_grpw_(dss[k], era="_{}".format(k)) for k in keys]
    result = xr.merge(epochs, compat="override")

    return result.astype(np.float32).squeeze()
