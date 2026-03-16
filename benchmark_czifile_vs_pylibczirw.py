"""Benchmark czifile vs pylibCZIrw for common tasks on two public CZI files.

Files (resolved relative to tests/data/ next to this script):
  LSM:   CZI-samples/CellObserver SD/
         LLC-PK1_2chTZ-triggered-20msec(SD).zstd.czi
         ZstdHdr-compressed 16-bit, T=20 C=2 Z=25 Y=512 X=512
         (0.52 GB uncompressed)
         Source: CZI sample images (CellObserver SD)

  Slide: CZI-samples/Axio Scan.Z1/Young_mouse.czi
         JPEG-XR-compressed uint8 RGB, 11 pyramid levels, 190 k x 69 k pixels
         Source: CZI sample images (Axio Scan.Z1)

Tasks
-----
1. Load full LSM scene as a numpy array.
2. Iterate all LSM frames in non-native dimension order (C-outer, Z-mid,
   T-inner; file stores data in T-outer, C-inner order) and load each frame
   as a numpy array. Three approaches: czifile bulk (single-pass asarray with
   dimension reorder), czifile chunks (per-frame chunks iteration), pylibCZIrw
   (per-plane reads).
3. Iterate NTILES consecutive 512x512 windows on the slide base layer and load
   each as a numpy array. Three approaches: czifile ROI (manual roi loop),
   czifile chunks (spatial tiling via chunks API), pylibCZIrw.
4. Load a center ROI from the slide base layer covering 1/12 of the full
   area (~3.30 GB), keeping the same aspect ratio.

Usage::

    python benchmark_czifile_vs_pylibczirw.py [--ntiles N] [--maxworkers N]
                                               [--maxcache N]

    --maxworkers N  worker threads for czifile
                    (default: 1; 0 = auto / half CPU cores).
                    pylibCZIrw threading is not controllable from Python.
    --maxcache N    subblock decode cache size for czifile
                    (default: None = auto mode).
                    0 = disabled, positive int = persistent LRU of that size.

"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

DATA = pathlib.Path(__file__).parent / 'tests' / 'data'
LSM_FILE = (
    DATA
    / 'CZI-samples'
    / 'CellObserver SD'
    / 'LLC-PK1_2chTZ-triggered-20msec(SD).zstd.czi'
)
SLIDE_FILE = DATA / 'CZI-samples' / 'Axio Scan.Z1' / 'Young_mouse.czi'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _set_maxcache(czif: Any, maxcache: int | None) -> None:
    """Set decode cache size on CziFile if supported (czifile >= 2026.x.x)."""
    if hasattr(czif, 'maxcache'):
        czif.maxcache = maxcache


def _fmt(seconds: float, nbytes: int) -> str:
    gb = nbytes / 1e9
    gbps = gb / seconds if seconds > 0 else float('inf')
    return f'{seconds:6.2f} s  ({gb:.2f} GB  {gbps:.2f} GB/s)'


def _section(title: str) -> None:
    width = 72
    print()
    print('=' * width)
    print(f'  {title}')
    print('=' * width)


def _result(label: str, seconds: float, nbytes: int, extra: str = '') -> float:
    line = f'  {label:<20s}  {_fmt(seconds, nbytes)}'
    if extra:
        line += f'  [{extra}]'
    print(line)
    return seconds


def _best_of(
    fn: Callable[..., tuple[float, int]], *args: Any, repeat: int
) -> tuple[float, int]:
    """Run fn(*args) repeat times, return (min elapsed, nbytes of last run)."""
    best = float('inf')
    nbytes = 0
    for _ in range(repeat):
        t, nb = fn(*args)
        if t < best:
            best = t
            nbytes = nb
    return best, nbytes


# ---------------------------------------------------------------------------
# Task 1 - load full LSM scene
# ---------------------------------------------------------------------------


def task1_czifile(
    path: pathlib.Path, maxworkers: int | None, maxcache: int | None
) -> tuple[float, int]:
    """Load full LSM scene with czifile."""
    import czifile

    t0 = time.perf_counter()
    with czifile.CziFile(str(path)) as czif:
        _set_maxcache(czif, maxcache)
        arr = czif.scenes[0].asarray(maxworkers=maxworkers)
    elapsed = time.perf_counter() - t0
    return elapsed, arr.nbytes


def task1_pylibczirw(path: pathlib.Path) -> tuple[float, int]:
    """Load full LSM scene with pylibCZIrw."""
    from pylibCZIrw import czi as pyczi

    t0 = time.perf_counter()
    with pyczi.open_czi(str(path)) as czi:
        bb = czi.total_bounding_box
        n_t = bb['T'][1] - bb['T'][0]
        n_c = bb['C'][1] - bb['C'][0]
        n_z = bb['Z'][1] - bb['Z'][0] if 'Z' in bb else 1
        rect = czi.total_bounding_rectangle
        roi = (rect.x, rect.y, rect.w, rect.h)
        nbytes = 0
        for t in range(n_t):
            for c in range(n_c):
                for z in range(n_z):
                    plane = {'T': t, 'C': c}
                    if n_z > 1:
                        plane['Z'] = z
                    frame = czi.read(roi=roi, plane=plane)
                    nbytes += frame.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, nbytes


# ---------------------------------------------------------------------------
# Task 2 - iterate frames in non-native (C-outer, T-inner) order
# ---------------------------------------------------------------------------


def task2_czifile(
    path: pathlib.Path, maxworkers: int | None, maxcache: int | None
) -> tuple[float, int]:
    """Iterate LSM frames in non-native order using czifile bulk asarray."""
    import czifile

    with czifile.CziFile(str(path)) as czif:
        _set_maxcache(czif, maxcache)
        img = czif.scenes[0]
        t0 = time.perf_counter()
        arr = img(C=None, Z=None, T=None).asarray(maxworkers=maxworkers)
        nbytes = arr.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, nbytes


def task2_czifile_chunks(
    path: pathlib.Path, maxworkers: int | None, maxcache: int | None
) -> tuple[float, int]:
    """Iterate LSM frames in non-native order using czifile chunks API."""
    import czifile

    nbytes = 0
    with czifile.CziFile(str(path)) as czif:
        _set_maxcache(czif, maxcache)
        img = czif.scenes[0]
        t0 = time.perf_counter()
        for chunk in img(C=None, Z=None, T=None).chunks():
            arr = chunk.asarray(maxworkers=maxworkers)
            nbytes += arr.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, nbytes


def task2_pylibczirw(path: pathlib.Path) -> tuple[float, int]:
    """Iterate LSM frames in non-native order with pylibCZIrw."""
    from pylibCZIrw import czi as pyczi

    nbytes = 0
    with pyczi.open_czi(str(path)) as czi:
        bb = czi.total_bounding_box
        n_c = bb['C'][1] - bb['C'][0]
        n_z = bb['Z'][1] - bb['Z'][0] if 'Z' in bb else 1
        n_t = bb['T'][1] - bb['T'][0]
        rect = czi.total_bounding_rectangle
        roi = (rect.x, rect.y, rect.w, rect.h)
        t0 = time.perf_counter()
        for c in range(n_c):
            for z in range(n_z):
                for t in range(n_t):
                    plane = {'C': c, 'T': t}
                    if n_z > 1:
                        plane['Z'] = z
                    frame = czi.read(roi=roi, plane=plane)
                    nbytes += frame.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, nbytes


# ---------------------------------------------------------------------------
# Task 4 - load center ROI of slide base layer (1/10 of full area)
# ---------------------------------------------------------------------------


def _compute_task4_roi(path: pathlib.Path) -> tuple[int, int, int, int]:
    """Return center ROI covering 1/12 of the base-layer area.

    Both linear dimensions are divided by sqrt(12) to give 1/12 the area
    while preserving the aspect ratio.

    Returns:
        Bounding box as (x, y, width, height) in absolute pixel coordinates.
    """
    import math

    import czifile

    with czifile.CziFile(str(path)) as czif:
        full_x, full_y, full_w, full_h = czif.scenes[0].bbox
    factor = math.sqrt(12)
    roi_w = int(full_w / factor)
    roi_h = int(full_h / factor)
    roi_x = full_x + (full_w - roi_w) // 2
    roi_y = full_y + (full_h - roi_h) // 2
    return roi_x, roi_y, roi_w, roi_h


def task4_czifile(
    path: pathlib.Path,
    roi: tuple[int, int, int, int],
    maxworkers: int | None,
    maxcache: int | None,
) -> tuple[float, int]:
    """Load center ROI of slide base layer with czifile."""
    import czifile

    with czifile.CziFile(str(path)) as czif:
        _set_maxcache(czif, maxcache)
        img = czif.scenes[0]
        t0 = time.perf_counter()
        arr = img(roi=roi).asarray(maxworkers=maxworkers)
    elapsed = time.perf_counter() - t0
    return elapsed, arr.nbytes


def task4_pylibczirw(
    path: pathlib.Path,
    roi: tuple[int, int, int, int],
) -> tuple[float, int]:
    """Load center ROI of slide base layer with pylibCZIrw."""
    from pylibCZIrw import czi as pyczi

    t0 = time.perf_counter()
    with pyczi.open_czi(str(path)) as czi:
        arr = czi.read(roi=roi, plane={})
    elapsed = time.perf_counter() - t0
    return elapsed, arr.nbytes


# ---------------------------------------------------------------------------
# Task 3 - iterate 512x512 tiles on slide base layer
# ---------------------------------------------------------------------------


def _compute_slide_tile_rois(
    path: pathlib.Path, ntiles: int, tile_size: int
) -> list[tuple[int, int, int, int]]:
    """Return first *ntiles* tile ROIs of *tile_size* x *tile_size* pixels.

    ROIs are derived from a regular grid over scene-0's bounding box,
    matching the grid that CziImageChunks generates for the same tile_size.
    This ensures all benchmark paths (roi, chunks, pylibCZIrw) read
    identical regions.
    """
    import czifile

    rois: list[tuple[int, int, int, int]] = []
    with czifile.CziFile(str(path)) as czif:
        img = czif.scenes[0]
        bx, by, bw, bh = img.bbox
        layout = img._subblock_layout
        xi = layout.x_idx
        for y in range(by, by + bh, tile_size):
            for x in range(bx, bx + bw, tile_size):
                w = min(tile_size, bx + bw - x)
                h = min(tile_size, by + bh - y)
                # skip tiles that don't overlap any subblock (WSI gaps/edges)
                if xi >= 0 and not img._spatial_index_filter(x, y, w, h):
                    continue
                rois.append((x, y, w, h))
                if len(rois) >= ntiles:
                    return rois
    return rois


def task3_czifile(
    path: pathlib.Path,
    tile_rois: list[tuple[int, int, int, int]],
    maxworkers: int | None,
    maxcache: int | None,
) -> tuple[float, int]:
    """Iterate slide tiles with czifile ROI reads."""
    import czifile

    nbytes = 0
    with czifile.CziFile(str(path)) as czif:
        _set_maxcache(czif, maxcache)
        img = czif.scenes[0]
        t0 = time.perf_counter()
        for x, y, w, h in tile_rois:
            tile = img(roi=(x, y, w, h)).asarray(maxworkers=maxworkers)
            nbytes += tile.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, nbytes


def task3_czifile_chunks(
    path: pathlib.Path,
    ntiles: int,
    tile_size: int,
    maxworkers: int | None,
    maxcache: int | None,
) -> tuple[float, int]:
    """Iterate slide tiles with czifile chunks API."""
    import itertools

    import czifile

    nbytes = 0
    with czifile.CziFile(str(path)) as czif:
        _set_maxcache(czif, maxcache)
        img = czif.scenes[0]
        t0 = time.perf_counter()
        for chunk in itertools.islice(
            img.chunks(Y=tile_size, X=tile_size), ntiles
        ):
            arr = chunk.asarray(maxworkers=maxworkers)
            nbytes += arr.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, nbytes


def task3_pylibczirw(
    path: pathlib.Path, tile_rois: list[tuple[int, int, int, int]]
) -> tuple[float, int]:
    """Iterate slide tiles with pylibCZIrw."""
    from pylibCZIrw import czi as pyczi

    nbytes = 0
    with pyczi.open_czi(str(path)) as czi:
        t0 = time.perf_counter()
        for x, y, w, h in tile_rois:
            tile = czi.read(roi=(x, y, w, h), plane={})
            nbytes += tile.nbytes
    elapsed = time.perf_counter() - t0
    return elapsed, nbytes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all benchmark tasks and print results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ntiles',
        type=int,
        default=200,
        help='Number of slide tiles to read in task 3 (default: 200)',
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=512,
        help='Tile width and height in pixels for task 3 (default: 512)',
    )
    parser.add_argument(
        '--repeat',
        type=int,
        default=10,
        metavar='N',
        help=(
            'Repeat tasks 1 and 2 N times and report the best'
            ' (minimum) time (default: 10).'
            ' Tasks 3 and 4 always run once.'
        ),
    )
    parser.add_argument(
        '--maxworkers',
        type=int,
        default=1,
        help=(
            'Worker threads for czifile'
            ' (default: 1; 0 = auto / half CPU cores).'
            ' pylibCZIrw threading is not controllable from Python.'
        ),
    )
    parser.add_argument(
        '--maxcache',
        type=int,
        default=None,
        metavar='N',
        help=(
            'Subblock decode cache size for czifile'
            ' (default: None = auto mode).'
            ' 0 = disabled, positive int = persistent LRU of that size.'
        ),
    )
    args = parser.parse_args()

    for label, path in [('LSM', LSM_FILE), ('Slide', SLIDE_FILE)]:
        if not path.exists():
            print(f'ERROR: {label} file not found: {path}')
            return

    try:
        from pylibCZIrw import czi as pyczi  # noqa: F401
    except ImportError:
        print('ERROR: pylibCZIrw is not installed. Install it with:')
        print('  pip install pylibCZIrw')
        return

    import czifile

    has_cache = hasattr(czifile.CziFile, 'maxcache')

    print()
    print('Benchmark: czifile vs pylibCZIrw')
    print(f'  LSM file  : {LSM_FILE.name}')
    print(f'  Slide file: {SLIDE_FILE.name}')
    print(f'  ntiles    : {args.ntiles}')
    print(f'  tile-size : {args.tile_size}x{args.tile_size}')
    maxworkers: int | None = args.maxworkers if args.maxworkers > 0 else None
    maxcache: int | None = args.maxcache
    print(f'  maxworkers: {maxworkers} (czifile)  /  n/a (pylibCZIrw)')
    cache_note = '' if has_cache else ' (not supported, ignored)'
    print(f'  maxcache  : {maxcache}{cache_note} (czifile)')
    repeat = args.repeat
    if repeat > 1:
        print(f'  repeat    : {repeat} (best-of-{repeat} for tasks 1 and 2)')

    # ---- Task 1 ----
    _section('Task 1 - Load full LSM scene as numpy array')
    print(f'  File: {LSM_FILE.name}')
    print('  Dims: T=20, C=2, Z=25, Y=512, X=512, uint16 (Zstd-compressed)')
    print()

    t, nb = _best_of(
        task1_czifile, LSM_FILE, maxworkers, maxcache, repeat=repeat
    )
    t1a = _result('czifile', t, nb)
    t, nb = _best_of(task1_pylibczirw, LSM_FILE, repeat=repeat)
    t1b = _result('pylibCZIrw', t, nb, 'per-plane reads (T, C, Z loops)')
    print(f'  speedup: {t1b / t1a:.2f}x  (czifile / pylibCZIrw)')

    # ---- Task 2 ----
    _section(
        'Task 2 - Iterate all LSM frames in non-native order'
        ' (C-outer, Z-mid, T-inner)'
    )
    print(f'  File: {LSM_FILE.name}')
    print(
        '  Native order: T-outer, C-inner, Z-innermost.'
        '  Benchmark uses C-outer, Z-mid, T-inner.'
    )
    print('  2 x 25 x 20 = 1000 frames of 512x512 uint16')
    print()

    t, nb = _best_of(
        task2_czifile, LSM_FILE, maxworkers, maxcache, repeat=repeat
    )
    t2a = _result('czifile bulk', t, nb, 'single-pass asarray')
    t, nb = _best_of(
        task2_czifile_chunks, LSM_FILE, maxworkers, maxcache, repeat=repeat
    )
    t2b = _result('czifile chunks', t, nb, 'per-frame iteration')
    t, nb = _best_of(task2_pylibczirw, LSM_FILE, repeat=repeat)
    t2c = _result('pylibCZIrw', t, nb, 'per-plane reads')
    print(f'  speedup: {t2c / t2a:.2f}x  bulk    (czifile / pylibCZIrw)')
    print(f'  speedup: {t2c / t2b:.2f}x  chunks  (czifile / pylibCZIrw)')

    # ---- Task 3 ----
    _section(
        f'Task 3 - Iterate {args.ntiles}'
        f' x {args.tile_size}x{args.tile_size}'
        ' tiles on slide base layer'
    )
    print(f'  File: {SLIDE_FILE.name}')
    print(
        '  Base layer: 190 309 x 69 378 px RGB uint8'
        ' (JPEG-XR, 11 pyramid levels)'
    )
    print(
        '  Computing tile layout from scene bounding box...',
        end='',
        flush=True,
    )
    slide_tiles = _compute_slide_tile_rois(
        SLIDE_FILE, args.ntiles, args.tile_size
    )
    tile_w = slide_tiles[0][2] if slide_tiles else 0
    tile_h = slide_tiles[0][3] if slide_tiles else 0
    print(f' {len(slide_tiles)} tiles ({tile_w}x{tile_h} each)')
    print()

    t, nb = _best_of(
        task3_czifile,
        SLIDE_FILE,
        slide_tiles,
        maxworkers,
        maxcache,
        repeat=1,
    )
    t3a = _result('czifile roi', t, nb, 'manual ROI loop')
    t, nb = _best_of(
        task3_czifile_chunks,
        SLIDE_FILE,
        args.ntiles,
        args.tile_size,
        maxworkers,
        maxcache,
        repeat=1,
    )
    t3b = _result('czifile chunks', t, nb, 'spatial tiling')
    t, nb = _best_of(task3_pylibczirw, SLIDE_FILE, slide_tiles, repeat=1)
    t3c = _result('pylibCZIrw', t, nb)
    print(f'  speedup: {t3c / t3a:.2f}x  roi     (czifile / pylibCZIrw)')
    print(f'  speedup: {t3c / t3b:.2f}x  chunks  (czifile / pylibCZIrw)')

    # ---- Task 4 ----
    _section('Task 4 - Load center ROI of slide base layer as numpy array')
    print(f'  File: {SLIDE_FILE.name}')
    print(
        '  Full base layer: 190 309 x 69 378 px RGB uint8'
        ' (approx. 39.6 GB uncompressed)'
    )
    print()

    print('  Computing center ROI (1/12 of full area)...', end='', flush=True)
    task4_roi = _compute_task4_roi(SLIDE_FILE)
    roi_x, roi_y, roi_w, roi_h = task4_roi
    roi_gb = roi_w * roi_h * 3 / 1e9
    print(
        f' roi=({roi_x}, {roi_y}, {roi_w}, {roi_h})  approx. {roi_gb:.2f} GB'
    )
    print()
    t, nb = task4_czifile(SLIDE_FILE, task4_roi, maxworkers, maxcache)
    t4a = _result('czifile', t, nb)
    t, nb = task4_pylibczirw(SLIDE_FILE, task4_roi)
    t4b = _result('pylibCZIrw', t, nb)
    print(f'  speedup: {t4b / t4a:.2f}x  (czifile / pylibCZIrw)')

    print()


if __name__ == '__main__':
    main()
