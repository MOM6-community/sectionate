def get_geo_corners(grid):
    """
    Find longitude and latitude coordinates from grid dataset, assuming the coordinate
    names contain the sub-strings "lon" and "lat", respectively.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    dict
        Dictionary containing names of longitude and latitude coordinates.
    """
    dims = {}
    for axis in ["X", "Y"]:
        if "outer" in grid.axes[axis].coords:
            dims[axis] = grid.axes[axis].coords["outer"]
        elif "right" in grid.axes[axis].coords:
            dims[axis] = grid.axes[axis].coords["right"]
        else:
            raise ValueError("Only 'symmetric' and 'non-symmetric' grids\
            are currently supported. They require C-grid topology, i.e. with\
            vorticity coordinates at 'outer' and 'right' positions, respectively.")

    coords = grid._ds.coords
    return {
        axis: [
            coords[c] for c in coords
            if (
                (geoc in c) and
                (dims["X"] in coords[c].dims) and
                (dims["Y"] in coords[c].dims)
            )
        ][0]
        for axis, geoc in zip(["X", "Y"], ["lon", "lat"])
    }
    
def coord_dict(grid):
    """
    Find names of "X" and "Y" dimension variables from grid dataset.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    dict
        Dictionary containing names of "X" and "Y" dimension variables, at both cell 'center' position
        and either 'outer' or 'right' position. ('left' position not yet supported.)
    """
    if check_symmetric(grid):
        corner_pos = "outer"
    else:
        corner_pos = "right"
        
    return {
        "X": {
            "center": grid.axes["X"].coords["center"],
            "corner": grid.axes["X"].coords[corner_pos]},
        "Y": {
            "center": grid.axes["Y"].coords["center"],
            "corner": grid.axes["Y"].coords[corner_pos]},
    }
    
def check_symmetric(grid):
    """
    Check whether the horizontal ocean model grid is symmetric or not, according to MOM6 conventions.
    Symmetric C-grids have tracers on (M,N) 'center' positions and vorticity on (M+1, N+1) 'outer' positions.
    Non-symmetric C-grids instead have vorticity on (M,N) 'right' positions.

    Parameters
    ----------
    grid: xgcm.Grid
        Contains information about ocean model grid discretization, e.g. coordinates and metrics.
        
    Returns
    -------
    symmetric : bool
        True if symmetric; False if non-symmetric.
        
    """
    pos_dict = {
        p : ((p in grid.axes["X"].coords) and (p in grid.axes["Y"].coords))
        for p in ["outer", "right"]
    }
    if pos_dict["outer"]:
        return True
    elif pos_dict["right"]:
        return False
    else:
        raise ValueError("Horizontal grid axes ('X', 'Y') must be either both symmetric or both non-symmetric (by MOM6 conventions).")