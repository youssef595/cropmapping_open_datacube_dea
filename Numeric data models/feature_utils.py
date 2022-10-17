import sys
import datacube
import numpy as np
import xarray as xr
from deafrica_tools.datahandling import load_ard
from deafrica_tools.bandindices import calculate_indices
from odc.algo import xr_geomedian
import numpy as np
    
def apply_function_over_custom_times(ds, func, func_name, time_ranges):
    output_list = []

    for timelabel, timeslice in time_ranges.items():

        if isinstance(timeslice, slice):
            ds_timeslice = ds.sel(time=timeslice)
        else:
            ds_timeslice = ds.sel(time=timeslice, method="nearest")

        ds_modified = func(ds_timeslice)

        rename_dict = {
            key: f"{key}_{func_name}_{timelabel}" for key in list(ds_modified.keys())
        }

        ds_modified = ds_modified.rename(name_dict=rename_dict)

        if "time" in list(ds_modified.coords):
            ds_modified = ds_modified.reset_coords().drop_vars(["time", "spatial_ref"])

        output_list.append(ds_modified)

    return output_list


def geomedian_with_indices_wrapper(ds):
    indices = ["NDVI", "EVI", "SAVI", "NDMI"]
    satellite_mission = "s2"

    ds_geomedian = xr_geomedian(ds)

    ds_geomedian = calculate_indices(
        ds_geomedian,
        index=indices,
        drop=False,
        satellite_mission=satellite_mission)

    return ds_geomedian


def get_years(ds):
    return np.unique(ds.time.dt.year.values)