import urllib.request
import shutil
import os
import xarray as xr
import xgcm

def download_MOM6_example_data():
    # download the data
    url = 'https://zenodo.org/record/15085089/files/'
    file_name = 'MOM6_global_example_diagnostics.nc'
    destination_path = f"../data/{file_name}"
    if not os.path.exists(destination_path):
        with urllib.request.urlopen(url + file_name) as response, open(destination_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f"File '{file_name}' [1.1 GB] has been downloaded to {destination_path}.")
    else:
        print(f"File '{file_name}' already exists at {destination_path}. Skipping download.")

    return destination_path

def load_MOM6_example_grid():
    destination_path = download_MOM6_example_data()
    ds = xr.open_dataset(destination_path)
    coords={
        'X': {'center': 'xh', 'outer': 'xq'},
        'Y': {'center': 'yh', 'outer': 'yq'},
    }
    grid = xgcm.Grid(ds, coords=coords, boundary={"X":"periodic", "Y":"extend"}, autoparse_metadata=False)
    return grid