"""Real-data topology checks on the ECCOv4r4 native 13-tile lat-lon-cap (LLC90)
grid. These validate `_OuterTopology` on a genuine, hard multi-tile grid --
rotated seams, the Arctic cap, the tiles-2/6/10 three-tile junction, and the
65E/115W polar grid cut -- rather than on a synthetic fixture.

The grid geometry file lives under ``data/`` (fetched from Zenodo by the ECCO
example loader). If it is absent (e.g. a bare CI checkout), every test here
skips, so the suite still passes without the data.
"""
import os
import numpy as np
import pytest

# The ECCO loader lives alongside the example notebooks.
_HERE = os.path.dirname(__file__)
_EXAMPLES = os.path.abspath(os.path.join(_HERE, "..", "..", "examples"))
_DATA = os.path.abspath(os.path.join(_HERE, "..", "..", "data"))
_GEOM = os.path.join(_DATA, "GRID_GEOMETRY_ECCO_V4r4_native_llc0090.nc")

pytestmark = pytest.mark.skipif(
    not os.path.exists(_GEOM),
    reason="ECCO LLC90 geometry file not present under data/",
)


def _load_grid():
    import sys
    if _EXAMPLES not in sys.path:
        sys.path.insert(0, _EXAMPLES)
    from load_example_ECCO_grid import load_ECCO_LLC90_grid
    return load_ECCO_LLC90_grid(_DATA)


def test_llc90_outer_topology_builds_and_is_well_formed():
    """`outer_topology` builds on the real LLC90 grid, and every corner node is
    well formed: no node has more than four neighbours (a >4 valence would mean
    the corner graph is inconsistent), and the only degree-0 nodes are the
    wall/cut corners stored on no face."""
    from sectionate.gridutils import outer_topology
    ot = outer_topology(_load_grid())

    deg = np.array([len(a) for a in ot.node_adj])
    native = ot.node_native[:, 0] >= 0

    assert deg.max() <= 4                                   # no folded-on-itself corner
    # Non-native nodes (points stored on no face) are exactly the degree-0 nodes:
    # open-wall corners, the unstored cut lip, and un-stored junctions.
    assert np.array_equal(np.where(~native)[0], np.where(deg == 0)[0])
    assert int(np.count_nonzero(~native)) == 78             # LLC90 wall/cut corners


def test_llc90_three_tile_junction_resolves_to_one_node():
    """The tiles-2/6/10 three-tile junction (a corner where three rotated tiles
    meet, stored on no face) must resolve to a SINGLE node whose representations
    span all three tiles -- the topological `by_junction` merge.

    Regression: when the three tiles' extrapolated coordinates for that corner
    disagreed, the coordinate merges could not see it was one physical point and
    the junction split into three separate degree-0 nodes. The velocity faces
    around it then read as zero from every neighbour's frame, leaking a spurious
    convergence into the junction's cells."""
    from sectionate.gridutils import outer_topology
    ot = outer_topology(_load_grid())

    junctions = [
        n for n in range(len(ot.node_reps))
        if {f for (f, _, _) in ot.node_reps[n]} == {2, 6, 10}
    ]
    assert len(junctions) == 1
    n = junctions[0]
    assert ot.node_native[n, 0] < 0                         # stored on no face
    assert len(ot.node_reps[n]) == 3                        # one slot per touching tile


def test_llc90_polar_cut_lips_share_nodes():
    """The 65E/115W polar cut has two coincident lips. The STORED lip (tiles 0/3
    at lon +65) is native and merged exactly by `by_loc`: at least one node must
    carry native representations on BOTH tile 0 and tile 3 (the same physical
    corner stored twice, once per tile)."""
    from sectionate.gridutils import outer_topology
    ot = outer_topology(_load_grid())

    shared = [
        n for n in range(len(ot.node_reps))
        if ot.node_native[n, 0] >= 0
        and {f for (f, _, _) in ot.node_reps[n]} >= {0, 3}
    ]
    assert shared, "no node shares native storage across tiles 0 and 3"


def test_llc90_face_corners_resolve_topologically():
    """Every native corner appears in the neighbour maps with the correct arity:
    building the maps requires resolving the four-tile-junction face corners via
    the topological diagonal recovery (no coordinates). A failure here would
    surface as a raised inconsistency during `outer_topology`/`maps` construction;
    this test simply asserts the maps exist and are index-valid."""
    from sectionate.gridutils import outer_topology
    ot = outer_topology(_load_grid())
    maps = ot.maps
    nf = ot.nf
    for d, (fm, jm, im) in maps.items():
        assert fm.shape == jm.shape == im.shape
        assert (fm >= 0).all() and (fm < nf).all()
        assert (jm >= 0).all() and (im >= 0).all()


def test_llc90_meridional_segment_traces_both_directions():
    """A meridional segment on a real multi-tile grid, under
    ``curve="latitude and great circle"``. Its endpoints do not share a latitude, so the
    combined option routes it to the geodesic -- which is what makes it traceable at all:
    the constant-latitude metrics measure progress purely in longitude, so along a
    meridian they are flat, the walk never converges, and it eventually gives up with
    "Should have reached the endpoint by now." Being direction-independent, it must also
    trace identically whichever end it starts from."""
    from sectionate.section import grid_section

    grid = _load_grid()
    fwd = grid_section(grid, [0., 0.], [60., 80.], curve="latitude and great circle")
    rev = grid_section(grid, [0., 0.], [80., 60.], curve="latitude and great circle")

    i, j, f, lons, lats = fwd
    assert len(i) == 49
    assert lats[0] < lats[-1]                               # it really does head north
    for a, b in zip(fwd, rev):
        assert np.array_equal(a, b[::-1])

    # curve="latitude circle" refuses the same segment outright rather than walking it.
    with pytest.raises(ValueError, match="constant latitude"):
        grid_section(grid, [0., 0.], [60., 80.], curve="latitude circle")


def test_llc90_zonal_segment_holds_its_parallel():
    """The complement of the test above: a segment whose endpoints do share a latitude is
    routed to the parallel by the combined option, giving the same path as an explicit
    ``curve="latitude circle"``. The grid stores its corner latitudes in float32, so this
    also exercises the classification tolerance on real single-precision coordinates."""
    from sectionate.section import grid_section

    grid = _load_grid()
    combined = grid_section(grid, [0., 60.], [20., 20.],
                            curve="latitude and great circle")
    parallel = grid_section(grid, [0., 60.], [20., 20.], curve="latitude circle")
    for a, b in zip(combined, parallel):
        assert np.array_equal(a, b)

    lats = combined[4]
    assert np.ptp(lats) < 2.                                # stays within a cell of 20N
