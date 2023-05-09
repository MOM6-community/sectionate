def get_geo_corners(grid):
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
    
def check_symmetric(grid):
    x_sym = "outer" in grid.axes['X'].coords
    y_sym = "outer" in grid.axes['Y'].coords
    if x_sym and y_sym:
        return True
    elif not(x_sym) and not(y_sym):
        return False
    else:
        raise ValueError("Horizontal grid axes ('X', 'Y') must be either both symmetric or both non-symmetric (by MOM6 conventions).")