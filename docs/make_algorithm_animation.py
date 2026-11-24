"""
Generate the figures for ``docs/source/algorithm.md``: an animation and a static
contact sheet walking through every iteration of ``sectionate``'s section-tracing
algorithm on a small synthetic grid.

Run by hand from the repository root, in the docs environment (needs ``ffmpeg``)::

    python docs/make_algorithm_animation.py

It writes ``docs/source/_static/algorithm/walk.mp4`` and ``walk_steps.png``, both
of which are committed -- Read the Docs never executes anything, so the artifacts
must already exist at build time.

The walk is *not* reimplemented here. ``replay_walk`` mirrors the loop in
``sectionate.section.infer_grid_path`` step by step so that the intermediate
state (which neighbors exist, which are admitted, what each one's metrics are)
can be recorded for drawing, but every number it computes comes from the
package's own helpers, and ``main`` asserts that the replayed path is identical
to what ``sectionate.grid_section`` returns. If the algorithm ever changes, that
assertion fails and these figures cannot be regenerated until they are brought
back in sync.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr
import xgcm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FFMpegWriter

from sectionate import grid_section
from sectionate.gridutils import get_geo_corners, build_neighbor_maps
from sectionate.section import (
    COINCIDENT_TOLERANCE_M,
    WALK_DEVIATION_ATOL,
    distance_on_unit_sphere,
    find_closest_grid_point,
    spherical_angle,
)

R_EARTH = 6.371e6  # matches the default in `distance_on_unit_sphere`

# ---------------------------------------------------------------------------
# Style. Ink carries the section being built (path, waypoints, target arc);
# saturated hue carries only the algorithm's live decision state; grey carries
# options that are out of the running. Every state also differs in linestyle,
# marker fill or glyph, so color is never the only channel.
#
# The three saturated hues are a colorblind-safe set validated as a group on a
# white surface over all pairs: worst CVD dE 9.2 (deutan) and worst normal-vision
# dE 24.0, in OKLab x100. Aqua sits at 2.82:1 contrast, below the 3:1 bar, which
# is why the admission circle is always named in the legend rather than left to
# color alone. Re-validate as a set before changing any of them.
# ---------------------------------------------------------------------------
INK = "#0b0b0b"          # primary ink: committed path, waypoints, target arc
INK_2 = "#52514e"        # secondary ink: annotations
MUTED = "#898781"        # rejected: moves away from the endpoint
FAINT = "#c3c2b7"        # not a legal move (wall / backtrack), untouched corners
MESH = "#c3c2b7"         # background grid of C-grid faces
ADMITTED = "#2a78d6"     # admitted candidate (blue)
CHOSEN = "#eb6834"       # the winning candidate (orange)
CIRCLE = "#1baf7a"       # the admission circle (aqua)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": FAINT,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# Waypoints: deliberately off-corner, so the opening beat can show them snap.
WAYPOINT_LONS = [62.0, 104.0, 136.0]
WAYPOINT_LATS = [42.0, 58.0, 48.0]

# Display window, in longitude unwrapped about 0 degrees.
LON_VIEW = (30.0, 156.0)  # left margin leaves room for the westmost candidate labels
LAT_VIEW = (28.0, 72.0)
LAT_REF = 50.0  # aspect-ratio reference latitude


# ---------------------------------------------------------------------------
# The toy grid
# ---------------------------------------------------------------------------
def toy_grid(dlon=15.0, dlat=10.0, lat0=-80.0, lat1=80.0):
    """A coarse global lat-lon C-grid, built like the test fixture
    ``sectionate.tests.test_section._latlon_grid`` but much coarser: corners
    every ``dlon`` degrees of longitude (so 0 and 360 are not duplicated) and
    every ``dlat`` degrees of latitude. Periodic in X, walled in Y.
    """
    lon_c = np.arange(0.0, 360.0, dlon)
    lat_c = np.arange(lat0, lat1 + 0.5 * dlat, dlat)
    LON_C, LAT_C = np.meshgrid(lon_c, lat_c)
    # 'right' staggering (MOM6 non-symmetric): tracer cell (j, i) has its
    # upper-right corner at (LON_C[j, i], LAT_C[j, i]), so centers sit half a
    # cell to the south-west.
    LON_H, LAT_H = np.meshgrid(lon_c - 0.5 * dlon, lat_c - 0.5 * dlat)
    ny, nx = LON_C.shape
    ds = xr.Dataset(coords={
        "xq": np.arange(nx), "yq": np.arange(ny),
        "xh": np.arange(nx) + 0.5, "yh": np.arange(ny) + 0.5,
        "geolon_c": (("yq", "xq"), LON_C), "geolat_c": (("yq", "xq"), LAT_C),
        "geolon": (("yh", "xh"), LON_H), "geolat": (("yh", "xh"), LAT_H),
    })
    return xgcm.Grid(
        ds,
        coords={"X": {"center": "xh", "right": "xq"},
                "Y": {"center": "yh", "right": "yq"}},
        padding={"X": "periodic", "Y": "extend"},
        autoparse_metadata=False,
    )


# ---------------------------------------------------------------------------
# Spherical geometry used only for drawing
# ---------------------------------------------------------------------------
def _xyz(lon, lat):
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _lonlat(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.rad2deg(np.arctan2(v[..., 1], v[..., 0])), np.rad2deg(np.arcsin(np.clip(v[..., 2], -1, 1)))


def arc(lon1, lat1, lon2, lat2, n=48):
    """Densified great-circle arc (slerp) from point 1 to point 2."""
    a, b = _xyz(lon1, lat1), _xyz(lon2, lat2)
    omega = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    t = np.linspace(0.0, 1.0, n)[:, None]
    if omega < 1e-12:
        pts = np.repeat(a[None, :], n, axis=0)
    else:
        pts = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)
    return _lonlat(pts)


def bearing(lon1, lat1, lon2, lat2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2),
                      np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def destination(lon, lat, brg, ang):
    """Point at angular distance ``ang`` (radians) from (lon, lat) on bearing ``brg``."""
    p1, l1 = np.deg2rad(lat), np.deg2rad(lon)
    p2 = np.arcsin(np.sin(p1) * np.cos(ang) + np.cos(p1) * np.sin(ang) * np.cos(brg))
    l2 = l1 + np.arctan2(np.sin(brg) * np.sin(ang) * np.cos(p1),
                         np.cos(ang) - np.sin(p1) * np.sin(p2))
    return np.rad2deg(l2), np.rad2deg(p2)


def geodesic_circle(lon0, lat0, radius_m, n=241):
    """The locus of points a fixed geodesic distance from (lon0, lat0)."""
    ang = radius_m / R_EARTH
    brg = np.linspace(0.0, 2 * np.pi, n)
    return destination(lon0, lat0, brg, ang)


def wedge(vlon, vlat, alon, alat, blon, blat, ang_deg, n=48):
    """Polygon of the spherical angle at vertex V between the arcs V->A and V->B.

    Drawn at angular radius ``ang_deg`` from the vertex, so its opening angle is
    exactly ``spherical_angle(V, A, B)`` -- one of the two terms of `deviation`.
    """
    ba = bearing(vlon, vlat, alon, alat)
    bb = bearing(vlon, vlat, blon, blat)
    d = (bb - ba + np.pi) % (2 * np.pi) - np.pi  # signed, shortest way
    th = ba + np.linspace(0.0, d, n)
    lo, la = destination(vlon, vlat, th, np.deg2rad(ang_deg))
    return np.concatenate([[vlon], lo, [vlon]]), np.concatenate([[vlat], la, [vlat]])


def unwrap(lon, center=0.0):
    """Longitude mapped into [center-180, center+180) for plotting."""
    return ((np.asarray(lon, dtype=float) - center + 180.0) % 360.0) - 180.0 + center


# ---------------------------------------------------------------------------
# Instrumented replay of `sectionate.section.infer_grid_path`
# ---------------------------------------------------------------------------
DIRECTIONS = ("right", "left", "down", "up")  # the order infer_grid_path probes


def replay_walk(gridlon, gridlat, neighbor_maps, lons, lats):
    """Replay the walk, recording per-step decision state for drawing.

    Mirrors the loop in ``sectionate.section.infer_grid_path`` (single-tile,
    ``curve="great circle"``). Returns ``(segments, path)`` where ``path`` is the
    stitched list of ``(j, i)`` corners, matching ``grid_section``.
    """
    def neighbor(direction, j, i):
        _, jmap, imap = neighbor_maps[direction]
        return (int(jmap[j, i]), int(imap[j, i]))

    segments, path = [], []

    for seg in range(len(lons) - 1):
        i1, j1 = find_closest_grid_point(lons[seg], lats[seg], gridlon, gridlat)
        i2, j2 = find_closest_grid_point(lons[seg + 1], lats[seg + 1], gridlon, gridlat)
        lon1, lat1 = gridlon[j1, i1], gridlat[j1, i1]
        lon2, lat2 = gridlon[j2, i2], gridlat[j2, i2]

        def progress(lon, lat):
            return distance_on_unit_sphere(lon, lat, lon2, lat2)

        def deviation(lon, lat):
            return (spherical_angle(lon2, lat2, lon1, lat1, lon, lat)
                    + spherical_angle(lon1, lat1, lon2, lat2, lon, lat))

        j, i = j1, i1
        j_prev, i_prev = j, i
        corners = [(j, i)]
        steps = []

        while (j, i) != (j2, i2):
            here = (j, i)
            prev = (j_prev, i_prev)
            lon_h, lat_h = gridlon[j, i], gridlat[j, i]
            d_current = distance_on_unit_sphere(lon_h, lat_h, lon2, lat2)
            if d_current < COINCIDENT_TOLERANCE_M:
                break

            nbrs = {d: neighbor(d, j, i) for d in DIRECTIONS}
            skip = {here, prev}
            p_current = progress(lon_h, lat_h)

            cand = []       # (dev, (j, i), direction) for admitted neighbors
            rows = []       # one record per direction, for the panel
            chosen, why = None, ""

            # The endpoint short-circuit: the first non-skipped neighbor that
            # coincides with the endpoint is taken immediately.
            for d in DIRECTIONS:
                nb = nbrs[d]
                if nb in skip:
                    continue
                if distance_on_unit_sphere(gridlon[nb], gridlat[nb], lon2, lat2) < COINCIDENT_TOLERANCE_M:
                    chosen, why = d, "coincides with the endpoint"
                    break
            snap = chosen is not None

            for d in DIRECTIONS:
                nb = nbrs[d]
                nb_lon, nb_lat = gridlon[nb], gridlat[nb]
                rec = {"dir": d, "pt": nb, "dprog": None, "dev": None}
                if nb == here:
                    rec["cat"], rec["note"] = "wall", "wall"
                elif nb == prev:
                    rec["cat"], rec["note"] = "backtrack", "came from"
                elif snap:
                    # When the short-circuit fires, `infer_grid_path` never
                    # reaches the scoring block -- neither `progress` nor
                    # `deviation` is evaluated for any neighbor. Leave both
                    # blank rather than display numbers the algorithm did not
                    # compute (deviation *at* the endpoint is the degenerate
                    # case the short-circuit exists to avoid).
                    if d == chosen:
                        rec["cat"], rec["note"] = "snap", "coincides with the endpoint"
                    else:
                        rec["cat"], rec["note"] = "unevaluated", "not evaluated"
                else:
                    p_nb = progress(nb_lon, nb_lat)
                    rec["dprog"] = p_nb - p_current
                    seam_twin = distance_on_unit_sphere(
                        nb_lon, nb_lat, lon_h, lat_h) < COINCIDENT_TOLERANCE_M
                    if p_nb < p_current or seam_twin:
                        dev = deviation(nb_lon, nb_lat)
                        if np.isfinite(dev):
                            rec["cat"], rec["dev"], rec["note"] = "admitted", dev, ""
                            cand.append((dev, nb, d))
                        else:
                            rec["cat"], rec["note"] = "rejected", "deviation undefined"
                    else:
                        rec["cat"], rec["note"] = "rejected", "farther"
                rows.append(rec)

            if chosen is None:
                if cand:
                    best = min(c[0] for c in cand)
                    tied = [c for c in cand if c[0] <= best + WALK_DEVIATION_ATOL]
                    winner = min(tied, key=lambda c: (c[1][0], c[1][1]))
                    chosen = winner[2]
                    why = ("smallest deviation" if len(tied) == 1
                           else "deviation tied; broken by lowest (j, i)")
                else:
                    alt = [d for d in DIRECTIONS if nbrs[d] not in skip] or \
                          [d for d in DIRECTIONS if nbrs[d] != prev]
                    chosen = min(alt, key=lambda d: (
                        distance_on_unit_sphere(gridlon[nbrs[d]], gridlat[nbrs[d]], lon2, lat2),
                        nbrs[d][0], nbrs[d][1]))
                    why = "no admissible move; nearest neighbor"

            steps.append({
                "here": here, "prev": prev, "rows": rows,
                "chosen": chosen, "why": why, "snap": snap,
                "remaining": d_current,
                "wrapped": abs(nbrs[chosen][1] - i) > 1,
            })

            j_prev, i_prev = j, i
            j, i = nbrs[chosen]
            corners.append((j, i))

        segments.append({
            "start": (j1, i1), "end": (j2, i2),
            "wp_lon": lons[seg], "wp_lat": lats[seg],
            "wp_lon_end": lons[seg + 1], "wp_lat_end": lats[seg + 1],
            "corners": corners, "steps": steps,
            "d0": distance_on_unit_sphere(lon1, lat1, lon2, lat2),
        })
        # `create_section_composite` drops each segment's last point so the
        # shared waypoint corner is not duplicated.
        path.extend(corners[:-1])

    path.append(segments[-1]["corners"][-1])
    return segments, path


def build_frames(segments):
    """Flatten the replay into the frame sequence the animation renders."""
    frames = []
    committed = []          # (j, i) corners already in the section
    step_no = 0
    total = sum(len(s["steps"]) for s in segments)
    nwp = len(WAYPOINT_LONS)

    base_open = {"kind": "opening", "seg": 0, "phase": None, "committed": [],
                 "step": 0, "total": total}
    for k in range(nwp):
        frames.append({**base_open, "n_wp": k + 1, "n_snap": k, "hold": 22,
                       "caption": f"Waypoint {k + 1}: the coordinates you asked for."})
        frames.append({**base_open, "n_wp": k + 1, "n_snap": k + 1, "hold": 26,
                       "caption": f"Waypoint {k + 1} snaps to its nearest grid corner "
                                  f"(find_closest_grid_point)."})
    frames.append({**base_open, "n_wp": nwp, "n_snap": nwp, "show_gc": True,
                   "hold": 46,
                   "caption": "Each consecutive pair of snapped corners defines one "
                              "segment, and the walk approximates the great circle "
                              "between them."})

    for si, seg in enumerate(segments):
        if si > 0:
            frames.append({
                "kind": "handoff", "seg": si, "phase": None, "hold": 52,
                "committed": list(committed), "step": step_no, "total": total,
                "caption": "Segment %d done. Its last corner is dropped so the shared "
                           "waypoint corner is not duplicated; the target, the great "
                           "circle and the admission circle all re-aim." % si})
        if not committed:
            committed.append(seg["corners"][0])

        for k, st in enumerate(seg["steps"]):
            step_no += 1
            base = {"seg": si, "kind": "step", "step": step_no, "total": total,
                    "committed": list(committed), "st": st, "segment": seg}
            cap = None
            extra = 0
            if st["wrapped"]:
                # Not exercised by the default waypoints, which stay away from
                # the seam; kept so that moving them across it stays explained.
                cap = ("The step crosses the periodic seam, so i wraps. The "
                       "neighbour comes from build_neighbor_maps, not from i+1.")
                extra = 32
            elif st["why"] == "coincides with the endpoint":
                cap = ("A neighbour coincides with the endpoint, so it is taken "
                       "immediately and the deviation test is skipped entirely.")
                extra = 32
            frames.append({**base, "phase": "probe", "hold": 14, "caption": None})
            frames.append({**base, "phase": "admit", "hold": 20, "caption": None})
            frames.append({**base, "phase": "commit", "hold": 20 + extra, "caption": cap})
            committed.append(seg["corners"][k + 1])

    frames.append({"kind": "closing", "seg": len(segments) - 1, "phase": None,
                   "hold": 80, "committed": list(committed), "step": step_no,
                   "total": total,
                   "caption": "The finished section: a chain of vorticity-point "
                              "corners approximating the great circle through every "
                              "waypoint."})
    return frames


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
CAT_STYLE = {
    # category      color      linestyle   marker facecolor  z
    "wall":        (FAINT,    (0, (1, 2)), "none"),
    "backtrack":   (FAINT,    (0, (1, 2)), "none"),
    "rejected":    (MUTED,    (0, (4, 2)), "none"),
    "unevaluated": (MUTED,    (0, (4, 2)), "none"),
    "admitted":    (ADMITTED, "-",         ADMITTED),
    "snap":        (CHOSEN,   "-",         CHOSEN),
}
LABEL_OFFSET = {"right": (2.6, 0.0), "left": (-2.6, 0.0),
                "up": (0.0, 2.2), "down": (0.0, -2.2)}
LABEL_ALIGN = {"right": ("left", "center"), "left": ("right", "center"),
               "up": ("center", "bottom"), "down": ("center", "top")}


def _mesh_segments(gridlon, gridlat, neighbor_maps):
    """Every C-grid face of the corner lattice, as drawable polylines.

    Built from the neighbor maps rather than from ``i+1``/``j+1``, so the
    periodic wrap edge is a real edge and a wall is a missing one -- the drawn
    mesh is exactly the graph the walk is allowed to move on.
    """
    ny, nx = gridlon.shape
    segs = []
    for d in ("right", "up"):
        _, jmap, imap = neighbor_maps[d]
        for j in range(ny):
            for i in range(nx):
                jn, iN = int(jmap[j, i]), int(imap[j, i])
                if (jn, iN) == (j, i):
                    continue  # wall: no face here
                lo, la = arc(gridlon[j, i], gridlat[j, i], gridlon[jn, iN], gridlat[jn, iN], n=8)
                x = unwrap(lo)
                if np.abs(np.diff(x)).max() > 180.0:
                    continue  # wraps the display cut, far outside the window
                if x.max() < LON_VIEW[0] - 20 or x.min() > LON_VIEW[1] + 20:
                    continue
                segs.append(np.column_stack([x, la]))
    return segs


def _setup_map(ax):
    ax.set_xlim(*LON_VIEW)
    ax.set_ylim(*LAT_VIEW)
    ax.set_aspect(1.0 / np.cos(np.deg2rad(LAT_REF)), adjustable="box")
    xt = np.arange(np.ceil(LON_VIEW[0] / 15.0) * 15.0, LON_VIEW[1], 15.0)
    yt = np.arange(30, 71, 10)
    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_xticklabels([f"{v % 360:g}°E" for v in xt], fontsize=7.5)
    ax.set_yticklabels([f"{v:g}°N" for v in yt], fontsize=7.5)
    ax.tick_params(length=2.5, colors=MUTED)
    for s in ax.spines.values():
        s.set_linewidth(0.6)


def draw_map(ax, fr, ctx, compact=False):
    gridlon, gridlat = ctx["gridlon"], ctx["gridlat"]
    ax.clear()
    _setup_map(ax)

    def P(pt):
        """(j, i) -> plotted (x, y)."""
        return unwrap(gridlon[pt]), gridlat[pt]

    # --- background: the C-grid face structure, and the corner lattice -------
    ax.add_collection(LineCollection(ctx["mesh"], colors=MESH, linewidths=0.8,
                                     alpha=0.65, zorder=0))
    ax.plot(ctx["corner_x"], ctx["corner_y"], ".", color="#a9a89f", markersize=3.6,
            zorder=1, linestyle="none")

    # The periodic seam, annotated only when it is actually in frame -- the
    # default waypoints sit well away from it.
    if LON_VIEW[0] < 0.0 < LON_VIEW[1]:
        ax.axvline(0.0, color=MUTED, lw=1.0, ls=(0, (5, 3)), alpha=0.75, zorder=1)
        if not compact:
            ax.text(1.0, LAT_VIEW[1] - 1.2, "i = 0  (periodic seam)", fontsize=7,
                    color=MUTED, ha="left", va="top", rotation=90)

    seg = ctx["segments"][fr["seg"]]
    start, target = seg["start"], seg["end"]

    # --- the section being built: all ink ------------------------------------
    # Requested waypoints are hollow stars; the corner one snapped to is a
    # filled star once it becomes the active target. On the opening beat they
    # appear one at a time, each followed by the corner it snapped to.
    n_wp = fr.get("n_wp", len(WAYPOINT_LONS))
    ax.plot(unwrap(WAYPOINT_LONS[:n_wp]), WAYPOINT_LATS[:n_wp], "*", mfc="none",
            mec=INK, mew=1.2, markersize=14, alpha=0.85, zorder=6, linestyle="none")
    if fr["kind"] == "opening":
        for k in range(fr.get("n_snap", 0)):
            s = ctx["snapped"][k]
            lo, la = arc(WAYPOINT_LONS[k], WAYPOINT_LATS[k],
                         gridlon[s], gridlat[s], n=12)
            ax.plot(unwrap(lo), la, "-", color=INK, lw=1.0, alpha=0.6, zorder=5)
            ax.plot(*P(s), "o", mfc="none", mec=INK, mew=1.4, markersize=12,
                    alpha=0.9, zorder=6, linestyle="none")

    # The great circle being approximated: every segment's once the whole plan
    # is on screen (end of the opening beat, and the closing beat), the active
    # segment's while walking.
    if fr["kind"] == "opening":
        gc_segs = ctx["segments"] if fr.get("show_gc") else []
    elif fr["kind"] == "closing":
        gc_segs = ctx["segments"]
    else:
        gc_segs = [seg]
    for s in gc_segs:
        lo, la = arc(gridlon[s["start"]], gridlat[s["start"]],
                     gridlon[s["end"]], gridlat[s["end"]], n=96)
        ax.plot(unwrap(lo), la, ls=(0, (6, 3)), color=INK, lw=1.5, alpha=0.85, zorder=3)

    # the committed path
    comm = fr["committed"]
    if len(comm) > 1:
        xs, ys = [], []
        for a, b in zip(comm[:-1], comm[1:]):
            lo, la = arc(gridlon[a], gridlat[a], gridlon[b], gridlat[b], n=12)
            xs.append(unwrap(lo))
            ys.append(la)
        ax.plot(np.concatenate(xs), np.concatenate(ys), "-", color=INK, lw=2.0, zorder=5)
    if comm:
        ax.plot([P(c)[0] for c in comm], [P(c)[1] for c in comm], "o", color=INK,
                markersize=5.0, zorder=6, linestyle="none")

    if fr["kind"] == "handoff":
        ax.plot(*P(target), "*", color=INK, markersize=15, zorder=7, linestyle="none")
        if not compact:
            ax.annotate("new target", xy=P(target), xytext=(7, -12),
                        textcoords="offset points", fontsize=7.5, color=INK_2)
    if fr["kind"] != "step":
        return

    st, phase = fr["st"], fr["phase"]
    here = st["here"]
    hx, hy = P(here)
    tx, ty = P(target)
    sx, sy = P(start)

    # --- the admission circle ------------------------------------------------
    # Centred on the target with radius equal to the *current* point's remaining
    # distance, so it passes exactly through the current corner: a neighbour is
    # admitted iff it falls strictly inside.
    if phase in ("admit", "commit") and not st["snap"]:
        clo, cla = geodesic_circle(gridlon[target], gridlat[target], st["remaining"])
        cx = unwrap(clo)
        brk = np.where(np.abs(np.diff(cx)) > 180.0)[0]
        cxn, clan = np.insert(cx, brk + 1, np.nan), np.insert(cla, brk + 1, np.nan)
        if not brk.size:
            ax.fill(cx, cla, color=CIRCLE, alpha=0.05, ec="none", zorder=1)
        ax.plot(cxn, clan, "-", color=CIRCLE, lw=1.2, alpha=0.75, zorder=4)

    # --- candidate paths -----------------------------------------------------
    for rec in st["rows"]:
        cat, d, nb = rec["cat"], rec["dir"], rec["pt"]
        if nb == here:  # a wall points at itself; nothing to draw
            drawn_cat = "wall"
        else:
            drawn_cat = cat
        if phase == "probe" and cat in ("admitted", "rejected", "unevaluated", "snap"):
            color, ls, mfc = INK_2, "-", "none"
        else:
            color, ls, mfc = CAT_STYLE[drawn_cat]
        if cat == "admitted" and phase == "commit" and d == st["chosen"]:
            continue  # drawn below, thicker
        if nb != here:
            # The endpoint snap is a committed step like any other, so draw it as
            # heavily as a deviation-chosen one (it just carries no wedges).
            heavy = cat == "snap" and phase == "commit"
            lo, la = arc(gridlon[here], gridlat[here], gridlon[nb], gridlat[nb], n=16)
            ax.plot(unwrap(lo), la, ls=ls, color=color, lw=3.0 if heavy else 1.3,
                    alpha=0.9, zorder=4)
            ax.plot(*P(nb), "o", mfc=mfc, mec=color, mew=1.3,
                    markersize=9.5 if heavy else 7.5, zorder=6, linestyle="none")
        if compact:
            continue
        # floating annotation
        ox, oy = LABEL_OFFSET[d]
        ha, va = LABEL_ALIGN[d]
        if nb == here:
            txt = f"{d}\n✗ wall"
            lx, ly = hx + ox, hy + oy
        else:
            lx, ly = P(nb)
            lx, ly = lx + ox, ly + oy
            if phase == "probe":
                txt = f"{d}\n(i={nb[1]}, j={nb[0]})"
                if cat == "backtrack":
                    txt += "\n✗ came from"
            elif cat == "backtrack":
                txt = f"{d}\n✗ came from"
            elif cat == "unevaluated":
                txt = f"{d}\nnot evaluated"
            elif cat == "snap":
                txt = f"{d}\n✓ is the endpoint"
            elif cat == "rejected":
                txt = f"{d}\n✗ {rec['dprog']/1e3:+.0f} km"
            else:
                txt = f"{d}\n✓ {rec['dprog']/1e3:+.0f} km"
                if phase == "commit":
                    txt += f"\ndev {rec['dev']:.3f}"
        ax.text(lx, ly, txt, fontsize=7.2, color=color, ha=ha, va=va,
                zorder=8, linespacing=1.35,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.72))

    # --- admitted candidates: transparent arcs to BOTH segment endpoints -----
    # At `admit` this is *every* admitted candidate, on equal footing: the walk
    # has not compared deviations yet, so singling any one out here would give
    # away the answer a frame early. The winner is only revealed at `commit`,
    # where its arcs are redrawn opaque with the two wedges.
    if phase == "admit":
        for rec in st["rows"]:
            if rec["cat"] != "admitted":
                continue
            for end in (start, target):
                lo, la = arc(gridlon[rec["pt"]], gridlat[rec["pt"]],
                             gridlon[end], gridlat[end], n=64)
                ax.plot(unwrap(lo), la, "-", color=ADMITTED, lw=0.9, alpha=0.35, zorder=3)

    # --- the winner: opaque arcs plus a shaded wedge at each endpoint --------
    if phase == "commit" and not st["snap"]:
        win = next(r for r in st["rows"] if r["dir"] == st["chosen"])
        wpt = win["pt"]
        lo, la = arc(gridlon[here], gridlat[here], gridlon[wpt], gridlat[wpt], n=16)
        ax.plot(unwrap(lo), la, "-", color=CHOSEN, lw=3.0, zorder=5)
        ax.plot(*P(wpt), "o", color=CHOSEN, markersize=9.5, zorder=7, linestyle="none")
        for end in (start, target):
            lo, la = arc(gridlon[wpt], gridlat[wpt], gridlon[end], gridlat[end], n=64)
            ax.plot(unwrap(lo), la, "-", color=CHOSEN, lw=1.2, alpha=0.85, zorder=4)

        # deviation = angle at the target + angle at the start
        a_end = spherical_angle(gridlon[target], gridlat[target],
                                gridlon[start], gridlat[start], gridlon[wpt], gridlat[wpt])
        a_start = spherical_angle(gridlon[start], gridlat[start],
                                  gridlon[target], gridlat[target], gridlon[wpt], gridlat[wpt])
        for vert, other, ang, lab in ((target, start, a_end, "∠end"),
                                      (start, target, a_start, "∠start")):
            wx, wy = wedge(gridlon[vert], gridlat[vert],
                           gridlon[other], gridlat[other],
                           gridlon[wpt], gridlat[wpt], ang_deg=9.0)
            ax.fill(unwrap(wx), wy, color=CHOSEN, alpha=0.30, ec=CHOSEN, lw=0.9, zorder=3)
            if not compact:
                bm = bearing(gridlon[vert], gridlat[vert], gridlon[other], gridlat[other])
                bo = bearing(gridlon[vert], gridlat[vert], gridlon[wpt], gridlat[wpt])
                dd = (bo - bm + np.pi) % (2 * np.pi) - np.pi
                mlo, mla = destination(gridlon[vert], gridlat[vert],
                                       bm + dd / 2.0, np.deg2rad(11.5))
                ax.text(unwrap(mlo), mla, f"{lab} {ang:.3f}", fontsize=7.2,
                        color=CHOSEN, ha="center", va="center", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    # --- current point and segment target -----------------------------------
    ax.plot(tx, ty, "*", color=INK, markersize=15, zorder=7, linestyle="none")
    ax.plot(hx, hy, "o", mfc="white", mec=INK, mew=2.0, markersize=11, zorder=7,
            linestyle="none")
    if not compact:
        ax.annotate("target", xy=(tx, ty), xytext=(6, -12), textcoords="offset points",
                    fontsize=7.5, color=INK_2)


def draw_panel(ax, fr, ctx):
    gridlon, gridlat = ctx["gridlon"], ctx["gridlat"]
    ax.clear()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def T(x, y, s, **kw):
        kw.setdefault("fontsize", 8.2)
        kw.setdefault("color", INK_2)
        kw.setdefault("va", "top")
        ax.text(x, y, s, transform=ax.transAxes, **kw)

    nseg = len(ctx["segments"])
    T(0.0, 0.99, f"step {fr['step']} of {fr['total']}", fontsize=13,
      color=INK, weight="bold")
    T(0.0, 0.925, f"segment {fr['seg']+1} of {nseg}   ·   curve = 'great circle'",
      fontsize=8.5)

    if fr["kind"] != "step":
        T(0.0, 0.83, {"opening": "Snapping waypoints to grid corners",
                      "handoff": "Starting the next segment",
                      "closing": "Section complete"}[fr["kind"]],
          fontsize=10, color=INK, weight="bold")
        ax.plot([0, 1], [0.80, 0.80], transform=ax.transAxes, color=FAINT, lw=0.8)
        T(0.0, 0.765, f"{'':4s}{'requested':>18s}{'':4s}{'nearest corner':>24s}",
          fontsize=7.4, family="monospace", color=MUTED)
        n_wp = fr.get("n_wp", len(WAYPOINT_LONS))
        n_snap = fr.get("n_snap", len(WAYPOINT_LONS))
        for k in range(n_wp):
            wl, wa = WAYPOINT_LONS[k], WAYPOINT_LATS[k]
            row = f"{k+1:<4d}{wl:7.1f}°E {wa:5.1f}°N"
            if k < n_snap:
                sn = ctx["snapped"][k]
                row += (f"  →  (i={sn[1]:2d}, j={sn[0]:2d}) "
                        f"{gridlon[sn]:6.1f}°E {gridlat[sn]:5.1f}°N")
            T(0.0, 0.705 - 0.052 * k, row, fontsize=7.6, family="monospace",
              color=INK if k == n_wp - 1 else INK_2)
        if fr["kind"] == "closing":
            T(0.0, 0.50,
              f"{len(fr['committed'])} corners  ·  {fr['total']} steps  ·  "
              f"{len(ctx['segments'])} segments", fontsize=9, color=INK)
        return

    st, seg = fr["st"], ctx["segments"][fr["seg"]]
    here, target = st["here"], seg["end"]
    T(0.0, 0.855,
      f"at      (i={here[1]:2d}, j={here[0]:2d})   {gridlon[here]:5.1f}°E {gridlat[here]:5.1f}°N",
      fontsize=8.4, family="monospace", color=INK)
    T(0.0, 0.805,
      f"target  (i={target[1]:2d}, j={target[0]:2d})   {gridlon[target]:5.1f}°E {gridlat[target]:5.1f}°N",
      fontsize=8.4, family="monospace")

    # The `progress` metric: the geodesic distance from where we stand to the
    # target. Not the length of the path still to be walked -- the staircase the
    # walk actually takes is longer than this straight-line distance.
    frac = float(np.clip(st["remaining"] / seg["d0"], 0.0, 1.0))
    T(0.0, 0.735, "progress: current distance to target", fontsize=7.6, color=MUTED)
    ax.add_patch(plt.Rectangle((0.0, 0.655), 0.62, 0.032, transform=ax.transAxes,
                               fc="#eeeeea", ec="none"))
    ax.add_patch(plt.Rectangle((0.0, 0.655), 0.62 * frac, 0.032, transform=ax.transAxes,
                               fc=CIRCLE, ec="none"))
    T(0.65, 0.695, f"{st['remaining']/1e3:,.0f} km", fontsize=8.6, color=INK,
      family="monospace")

    # candidate table. Columns are placed at fixed axes fractions rather than
    # padded into one string, so the headers line up with the values.
    COL = {"dir": 0.0, "idx": 0.145, "dprog": 0.60, "dev": 0.79}
    y = 0.575
    T(0.0, y, "neighbours, in the order the code probes them", fontsize=7.6, color=MUTED)
    y -= 0.050
    for key, label, ha in (("dir", "", "left"), ("idx", "index", "left"),
                           ("dprog", "Δprogress", "right"), ("dev", "deviation", "right")):
        if label:
            T(COL[key], y, label, fontsize=7.6, color=MUTED, ha=ha)
    y -= 0.030
    ax.plot([0, 1], [y, y], transform=ax.transAxes, color=FAINT, lw=0.8)

    for rec in st["rows"]:
        cat = rec["cat"]
        if fr["phase"] == "probe":
            # Admission has not been tested yet; only the structural verdicts
            # (wall, backtrack) are known, so do not pre-color the rest.
            color = MUTED if cat in ("wall", "backtrack") else INK_2
        else:
            color = {"admitted": ADMITTED, "snap": CHOSEN}.get(cat, MUTED)
            if cat == "admitted" and rec["dir"] == st["chosen"] and fr["phase"] == "commit":
                color = CHOSEN
        dp = "—" if rec["dprog"] is None else f"{rec['dprog']/1e3:+,.0f} km"
        dv = "—" if rec["dev"] is None else f"{rec['dev']:.4f}"
        if fr["phase"] == "probe":
            dp = dv = "—"
        elif fr["phase"] == "admit":
            dv = "—"
        y -= 0.068
        weight = "bold" if (cat == "admitted" and rec["dir"] == st["chosen"]
                            and fr["phase"] == "commit") else "normal"
        ax.text(COL["dir"], y, rec["dir"], transform=ax.transAxes, fontsize=8.0,
                family="monospace", color=color, va="top", weight=weight)
        ax.text(COL["idx"], y, f"(i={rec['pt'][1]:2d}, j={rec['pt'][0]:2d})",
                transform=ax.transAxes, fontsize=8.0, family="monospace",
                color=color, va="top")
        ax.text(COL["dprog"], y, dp, transform=ax.transAxes, fontsize=8.0,
                family="monospace", color=color, va="top", ha="right")
        ax.text(COL["dev"], y, dv, transform=ax.transAxes, fontsize=8.0,
                family="monospace", color=color, va="top", ha="right")
        mark = ""
        if fr["phase"] != "probe" or cat in ("wall", "backtrack"):
            mark = {"wall": "✗ wall", "backtrack": "✗ came from",
                    "rejected": "✗ farther", "unevaluated": "not evaluated",
                    "snap": "✓ endpoint"}.get(cat, "")
            if cat == "admitted":
                mark = ("✓ chosen" if (fr["phase"] == "commit"
                                       and rec["dir"] == st["chosen"])
                        else "✓ admitted")
        ax.text(1.0, y, mark, transform=ax.transAxes, fontsize=7.6,
                color=color, va="top", ha="right", weight=weight)

    if fr["phase"] == "commit":
        T(0.0, y - 0.085, f"chose '{st['chosen']}' — {st['why']}",
          fontsize=8.8, color=CHOSEN, weight="bold")
        if not st["snap"]:
            # Spell out the sum the two shaded wedges depict, so the picture and
            # the number in the table are visibly the same quantity.
            win = next(r for r in st["rows"] if r["dir"] == st["chosen"])
            start = seg["start"]
            a_end = spherical_angle(gridlon[target], gridlat[target],
                                    gridlon[start], gridlat[start],
                                    gridlon[win["pt"]], gridlat[win["pt"]])
            a_start = spherical_angle(gridlon[start], gridlat[start],
                                      gridlon[target], gridlat[target],
                                      gridlon[win["pt"]], gridlat[win["pt"]])
            T(0.0, y - 0.145,
              f"deviation {win['dev']:.4f} = ∠start {a_start:.4f} + ∠end {a_end:.4f}",
              fontsize=7.8, family="monospace", color=INK_2)


#           color     linestyle     lw   marker  facecolor  ms   label
LEGEND = [
    (INK,      "-",         2.0, "o", INK,    5.5, "section so far"),
    (INK,      "-",         0.0, "*", "none", 9.0, "requested waypoint (★ = target)"),
    (INK,      (0, (6, 3)), 1.5, None, None,  0.0, "target great circle"),
    (CIRCLE,   "-",         1.3, None, None,  0.0, "admission circle (inside = ok)"),
    (ADMITTED, "-",         1.4, "o", "none", 5.5, "admitted: strictly closer"),
    (CHOSEN,   "-",         2.6, "o", CHOSEN, 5.5, "chosen: least deviation"),
    (MUTED,    (0, (4, 2)), 1.4, "o", "none", 5.5, "rejected: moves farther"),
    (FAINT,    (0, (1, 2)), 1.4, "o", "none", 5.5, "illegal: wall / backtrack"),
]


def draw_legend(fig, ncol=4):
    """Encoding key, as a two-row strip along the foot of the figure."""
    x0, dx, y0, dy = 0.045, 0.236, 0.055, 0.030
    for k, (color, ls, lw, mk, mfc, ms, label) in enumerate(LEGEND):
        x, y = x0 + dx * (k % ncol), y0 - dy * (k // ncol)
        if lw:
            fig.add_artist(plt.Line2D([x, x + 0.026], [y, y], transform=fig.transFigure,
                                      color=color, ls=ls, lw=lw, solid_capstyle="butt"))
        if mk:
            fig.add_artist(plt.Line2D([x + 0.013], [y], transform=fig.transFigure,
                                      marker=mk, mfc=mfc, mec=color, mew=1.2,
                                      markersize=ms, linestyle="none"))
        fig.text(x + 0.032, y, label, fontsize=7.4, color=INK_2, va="center")


def render(fr, ctx, fig, axm, axp):
    draw_map(axm, fr, ctx)
    draw_panel(axp, fr, ctx)
    for t in list(fig.texts):
        t.remove()
    for a in list(fig.artists):
        a.remove()
    fig.text(0.045, 0.975, "How sectionate traces a section", fontsize=11.5,
             color=INK, weight="bold", va="top")
    if fr.get("caption"):
        fig.text(0.045, 0.938, fr["caption"], fontsize=8.4, color=INK_2, va="top")
    draw_legend(fig)


def make_figure():
    fig = plt.figure(figsize=(12.4, 6.1), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0],
                          left=0.045, right=0.985, top=0.885, bottom=0.115,
                          wspace=0.10)
    return fig, fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])


def _resolve_ffmpeg():
    """Point matplotlib at ffmpeg even when the env was not activated.

    ``docs/environment.yml`` installs ffmpeg into the environment, but running
    ``<env>/bin/python docs/make_algorithm_animation.py`` without activating
    leaves ``<env>/bin`` off PATH, and matplotlib's default is the bare name.
    """
    if shutil.which("ffmpeg"):
        return
    beside = Path(sys.executable).parent / "ffmpeg"
    if beside.exists():
        matplotlib.rcParams["animation.ffmpeg_path"] = str(beside)
        return
    raise SystemExit(
        "ffmpeg not found. It is listed in docs/environment.yml; either activate "
        "the docs environment or install ffmpeg on PATH.")


def write_animation(frames, ctx, out, fps=10):
    _resolve_ffmpeg()
    fig, axm, axp = make_figure()
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                          extra_args=["-pix_fmt", "yuv420p", "-crf", "26",
                                      "-preset", "slow"])
    n = 0
    with writer.saving(fig, str(out), dpi=100):
        for fr in frames:
            render(fr, ctx, fig, axm, axp)
            for _ in range(fr["hold"]):
                writer.grab_frame()
                n += 1
    plt.close(fig)
    print(f"wrote {out}  ({n} frames, {n/fps:.1f} s)")


def write_contact_sheet(frames, ctx, out, ncol=2):
    """Every iteration's decision at once. Two columns, because the docs content
    column is 900px wide and four would shrink the panel titles past reading."""
    steps = [f for f in frames if f["kind"] == "step" and f["phase"] == "commit"]
    nrow = int(np.ceil(len(steps) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.3 * nrow), dpi=110)
    for ax, fr in zip(axes.ravel(), steps):
        draw_map(ax, fr, ctx, compact=True)
        st = fr["st"]
        win = next(r for r in st["rows"] if r["dir"] == st["chosen"])
        detail = ("coincides with the endpoint" if st["snap"]
                  else f"deviation {win['dev']:.3f}")
        ax.set_title(
            f"step {fr['step']} · segment {fr['seg']+1} · chose '{st['chosen']}' — {detail}",
            fontsize=9, color=INK, pad=5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for ax in axes.ravel()[len(steps):]:
        ax.axis("off")
    fig.tight_layout(pad=0.8)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def _dump(segments, path):
    print(f"path ({len(path)} corners, {len(path)-1} steps):")
    for j, i in path:
        print(f"    (j={j:3d}, i={i:3d})")
    for si, seg in enumerate(segments):
        print(f"\n=== segment {si+1}: {seg['start']} -> {seg['end']}, "
              f"d0={seg['d0']/1e3:.0f} km, {len(seg['steps'])} steps")
        for k, st in enumerate(seg["steps"]):
            print(f"  step {k+1}: at (j={st['here'][0]}, i={st['here'][1]}), "
                  f"remaining {st['remaining']/1e3:.0f} km, "
                  f"chose {st['chosen']!r} ({st['why']})")
            for r in st["rows"]:
                dp = "     -   " if r["dprog"] is None else f"{r['dprog']/1e3:+8.1f}"
                dv = "    -   " if r["dev"] is None else f"{r['dev']:8.5f}"
                print(f"      {r['dir']:6s} (j={r['pt'][0]:2d}, i={r['pt'][1]:2d})  "
                      f"dprog {dp} km  dev {dv}  {r['cat']:9s} {r['note']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", action="store_true",
                    help="print the replayed decision table and exit")
    ap.add_argument("--outdir", type=Path,
                    default=Path(__file__).resolve().parent / "source" / "_static" / "algorithm",
                    help="where to write walk.mp4 and walk_steps.png")
    ap.add_argument("--preview", type=int, default=None, metavar="N",
                    help="render only frame N to preview.png, for iterating on layout")
    args = ap.parse_args()

    grid = toy_grid()
    geo = get_geo_corners(grid)
    gridlon = np.asarray(geo["X"].values, dtype=float)
    gridlat = np.asarray(geo["Y"].values, dtype=float)
    neighbor_maps = build_neighbor_maps(grid, geo)

    segments, path = replay_walk(gridlon, gridlat, neighbor_maps,
                                 WAYPOINT_LONS, WAYPOINT_LATS)

    # Fidelity gate: the replay above must reproduce the real algorithm exactly.
    i_c, j_c, _, _ = grid_section(grid, WAYPOINT_LONS, WAYPOINT_LATS)
    expected = list(zip(j_c.astype(int).tolist(), i_c.astype(int).tolist()))
    if path != expected:
        raise AssertionError(
            "replay_walk has drifted from sectionate.grid_section.\n"
            f"  replayed: {path}\n  grid_section: {expected}")

    if args.dump:
        _dump(segments, path)
        return

    print(f"replay matches grid_section: {len(path)} corners, {len(path)-1} steps")

    ny, nx = gridlon.shape
    cx = unwrap(gridlon).ravel()
    cy = gridlat.ravel()
    keep = ((cx > LON_VIEW[0] - 20) & (cx < LON_VIEW[1] + 20)
            & (cy > LAT_VIEW[0] - 20) & (cy < LAT_VIEW[1] + 20))
    ctx = {
        "gridlon": gridlon, "gridlat": gridlat, "segments": segments,
        "mesh": _mesh_segments(gridlon, gridlat, neighbor_maps),
        "corner_x": cx[keep], "corner_y": cy[keep],
        "snapped": [tuple(reversed(find_closest_grid_point(lo, la, gridlon, gridlat)))
                    for lo, la in zip(WAYPOINT_LONS, WAYPOINT_LATS)],
    }

    frames = build_frames(segments)

    if args.preview is not None:
        fig, axm, axp = make_figure()
        render(frames[args.preview % len(frames)], ctx, fig, axm, axp)
        fig.savefig("preview.png", facecolor="white")
        plt.close(fig)
        print(f"wrote preview.png (frame {args.preview % len(frames)} of {len(frames)})")
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_animation(frames, ctx, args.outdir / "walk.mp4")
    write_contact_sheet(frames, ctx, args.outdir / "walk_steps.png")


if __name__ == "__main__":
    main()
