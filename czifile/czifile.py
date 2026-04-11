# czifile.py

# Copyright (c) 2013-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

r"""Read Carl Zeiss image files (CZI).

Czifile is a Python library for reading image data and metadata from
Carl Zeiss Image (CZI) files, the native file format of ZEN by
Carl Zeiss Microscopy GmbH.

Czifile is a pure-Python library under the BSD-3-Clause license. It provides
single-call array access to scenes and spatial ROIs, xarray DataArray output
with physical axis coordinates, multi-scene merging, per-dimension selection
by integer, slice, or sequence, chunk-based iteration, and pyramid-level
access. It handles Fast Airyscan upsampling and PALM downsampling with
optional stored-resolution output, and assembles FCS and line-scan files from
T-chunked subblocks. It also supports Zstd and JPEG XR compression, pixel type
promotion across channels, and direct access to all ZISRAW segments and
file-level attachments.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.4.11
:DOI: `10.5281/zenodo.14948581 <https://doi.org/10.5281/zenodo.14948581>`_

Quickstart
----------

Install the czifile package and all dependencies from the
`Python Package Index <https://pypi.org/project/czifile/>`_::

    python -m pip install -U czifile[all]

See `Examples`_ for using the programming interface.

Source code, examples, and support are available on
`GitHub <https://github.com/cgohlke/czifile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.12.10, 3.13.13, 3.14.4 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.4.4
- `Imagecodecs <https://pypi.org/project/imagecodecs>`_ 2026.3.6
- `Xarray <https://pypi.org/project/xarray>`_ 2026.2.0 (recommended)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.8 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2026.3.3 (optional)

Revisions
---------

2026.4.11

- Fall back to parsing numeric channel names as float coords['C'].

2026.3.17

- Add cache for decoded subblock arrays.
- Prefer imagecodecs' WIC over JPEGXR codec if available.
- Import imagecodecs functions on demand.

2026.3.15

- Replace CziImagePlanes with CziImageChunks (breaking).
- Add CziImage.chunks method for flexible chunk-based iteration.
- Add CziFile.metadata_segment property.
- Add offset properties to CziAttachmentEntryA1 and subblock entry classes.
- Add CziSegmentId.packed property returning the 16-byte on-disk field.
- Manage update_pending flag in CziFile context manager for writable handles.
- Improve documentation.

2026.3.14

- Add option to return pixel data at stored resolution.
- Allow sequence and slice of scene indices in imread and asarray/asxarray.
- Interpret dimension slice selection as absolute coordinates.
- Add command line options to select dimensions.

2026.3.12

- Rewrite with many breaking changes.
- Support Zstd compression schemes.
- Support reading subblock masks.
- Add CziFile.scenes interface.
- Add pyramid level access via CziImage.levels.
- Add option to read subset of image data.
- Add option to iterate over image planes in any dimension order.
- Add xarray-style attributes.
- Add asxarray method to return image as xarray DataArray with metadata.
- Add fillvalue and maxworkers parameters to asarray.
- Add option to specify pixel type.
- Promote pixel type when channels have mixed types.
- Remove Mosaic dimension from CziDirectoryEntryDV.dims; use mosaic_index.
- Reduce caching of CziDirectoryEntryDV properties.
- Remove resize and order parameters from asarray (breaking).
- Remove czi2tif function and command line script.
- Prefix public class names with Czi.
- Raise CziFileError for issues with CZI file structure.
- Use logging instead of warnings.
- Improve representation of instances.
- Add pytest-based unit tests.
- Add type hints.
- Convert docstrings to Google style with Sphinx directives.
- Remove imagecodecs-lite fallback; require imagecodecs.
- Remove scipy/ndimage dependency.
- Make tifffile an optional dependency.
- Drop support for Python < 3.12 and numpy < 2 (SPEC 0).

2019.7.2.3

- …

Refer to the CHANGES file for older revisions.

Notes
-----

The API is not stable yet and might change between revisions.

`Carl Zeiss AG <https://www.zeiss.com/>`_ is a manufacturer of microscopes
and scientific instruments.
CZI is a proprietary file format written by Zeiss acquisition software
such as ZEN to store microscopy images and metadata.

CZI files are based on the ZISRAW (Zeiss Image Segment Raw) container
specification, which is confidential and does not permit writing CZI files:

    | ZISRAW (CZI) File Format Design Specification Release Version 1.2.2.
    | "CZI 07-2016/CZI-DOC ZEN 2.3/DS_ZISRAW-FileFormat.pdf"

The ZISRAW format organizes data into typed, length-prefixed segments:
a file header, image subblocks, XML metadata, and attachments. Each image
subblock carries pixels for one tile or Z-plane across up to ten logical
dimensions (X, Y, Z, channel, time, scene, phase, illumination, rotation, and
mosaic index). Pixel data may be stored uncompressed or compressed with JPEG,
JPEG XR, or Zstd.

Only a subset of the 2016 specification is implemented. Specifically,
multi-file images and topography images are not supported.
Some features are untested due to lack of sample files.

Czifile relies on the `imagecodecs <https://pypi.org/project/imagecodecs/>`__
package for decoding LZW, Zstd, JPEG, and JPEG XR compressed images.

Other libraries for reading CZI files (all GPL or LGPL licensed):
`libczi <https://github.com/ZEISS/libczi>`__,
`pylibCZIrw <https://pypi.org/project/pylibCZIrw>`__,
`bioio-czi <https://github.com/bioio-devs/bioio-czi>`__,
`bio-formats <https://github.com/ome/bioformats>`_,
`libCZI <https://github.com/zeiss-microscopy/libCZI>`__ (deprecated), and
`pylibczi <https://github.com/elhuhdron/pylibczi>`__ (deprecated).

Examples
--------

Read image data of the first scene from a CZI file as numpy array:

>>> arr = imread('Example.czi')
>>> assert arr.shape == (2, 2, 3, 486, 1178)
>>> assert arr.dtype == 'uint16'

Access scenes, shape, and metadata:

>>> with CziFile('Example.czi') as czi:
...     assert len(czi.scenes) == 3
...     img = czi.scenes[0]  # 0 is the absolute coordinate of the first scene
...     assert img.shape == (2, 2, 3, 486, 1178)
...     assert img.dims == ('T', 'C', 'Z', 'Y', 'X')
...     assert img.dtype == 'uint16'
...     assert img.compression.name == 'ZSTDHDR'
...     assert list(img.channels) == ['DAPI', 'EGFP']
...     assert czi.metadata().startswith('<ImageDocument>')
...

Select dimensions and read as numpy array:

>>> with CziFile('Example.czi') as czi:
...     img = czi.scenes[0]
...     assert img.sizes == {'T': 2, 'C': 2, 'Z': 3, 'Y': 486, 'X': 1178}
...
...     # integer selection: fix T=0 and C=0; result has Z, but no T or C axis
...     volume = img(T=0, C=0).asarray()
...     assert volume.shape == (3, 486, 1178)
...
...     # None selection: keep all values but reorder dimensions
...     # dims order follows the kwargs order, then spatial dims
...     # T (unspecified) comes first, then C, Z (in kwargs order), then Y X
...     tczyx = img(C=None, Z=None).asarray()
...     assert tczyx.shape == (2, 2, 3, 486, 1178)
...
...     # read in C-outer, Z-inner, T-innermost order with parallelism
...     arr = img(C=None, Z=None, T=None).asarray(maxworkers=8)
...     assert arr.shape == (2, 3, 2, 486, 1178)  # 'C', 'Z', 'T', 'Y', 'X'
...
...     # img.bbox gives (x, y, width, height) in global CZI coordinates
...     x0, y0, *_ = img.bbox
...     plane_roi = img(T=0, C=0, roi=(x0, y0, 128, 128)).asarray()
...     assert plane_roi.shape == (3, 128, 128)  # 'Z', 'Y', 'X'
...
...     # fill pixels outside subblock coverage with a specific value
...     padded = img(C=0, roi=(0, 0, 2048, 2048)).asarray(fillvalue=0)
...     assert padded.shape == (2, 3, 2048, 2048)  # 'T', 'Z', 'Y', 'X'
...

Iterate image chunks:

>>> with CziFile('Example.czi') as czi:
...     img = czi.scenes[0]
...
...     # iterate individual Y/X planes as CziImage views
...     # by default, all non-spatial dims are iterated one-at-a-time
...     for chunk in img.chunks():
...         assert isinstance(chunk, CziImage)
...         assert chunk.asarray().shape == (486, 1178)
...
...     # keep C in each chunk: iterate T and Z only
...     for chunk in img.chunks(C=None):
...         assert chunk.asarray().shape == (2, 486, 1178)
...
...     # batch Z into groups of 3; last chunk may be smaller if Z indivisible
...     for chunk in img.chunks(Z=3):
...         assert chunk.sizes['Z'] <= 3
...
...     # spatial tiling: iterate T x C x Z x grid
...     for chunk in img.chunks(Y=256, X=256):
...         assert chunk.shape[-2] <= 256
...         assert chunk.shape[-1] <= 256
...
...     # keep C, tile spatially
...     for chunk in img.chunks(C=None, Y=256, X=256):
...         assert chunk.dims[0] == 'C'
...

Read image as xarray DataArray with physical coordinates and attributes:

>>> with CziFile('Example.czi') as czi:
...     xarr = czi.scenes[0].asxarray()
...     assert xarr.name == 'Scene 0'
...     assert xarr.sizes == {'T': 2, 'C': 2, 'Z': 3, 'Y': 486, 'X': 1178}
...     assert xarr.coords['X'].size == 1178  # physical axis coordinates
...

Access multiple scenes:

>>> with CziFile('Example.czi') as czi:
...     # iterate scenes individually and read as arrays
...     for img in czi.scenes.values():
...         arr = img.asarray()
...
...     # query which scenes (indices) are available
...     assert list(czi.scenes.keys()) == [0, 1, 2]
...
...     # select the second scene
...     assert czi.scenes[1].sizes == {
...         'T': 2,
...         'C': 2,
...         'Z': 3,
...         'Y': 256,
...         'X': 256,
...     }
...
...     # merge selected scenes into one
...     img = czi.scenes(scene=[0, 1])  # first 2 scenes
...     assert img.sizes == {'T': 2, 'C': 2, 'Z': 3, 'Y': 1109, 'X': 1760}
...
...     # merge all scenes into one
...     img = czi.scenes()
...     assert img.sizes == {'T': 2, 'C': 2, 'Z': 3, 'Y': 2055, 'X': 2581}
...

Access pyramid levels:

>>> with CziFile('Example.czi') as czi:
...     img = czi.scenes[0]
...     assert img.is_pyramid
...     assert len(img.levels) == 2  # full resolution + 1 downsampled level
...     assert img.levels[0] is img  # full resolution level is the same as img
...     overview = img.levels[1]  # lowest-res level
...     assert overview.sizes == {'T': 2, 'C': 2, 'Z': 3, 'Y': 243, 'X': 589}
...

Access attachments:

>>> with CziFile('Example.czi') as czi:
...     for attachment in czi.attachments():
...         name = attachment.attachment_entry.name
...         data = attachment.data()  # decoded (ndarray, tuple, bytes...)
...         raw = attachment.data(raw=True)  # bytes; may be written to file
...
...     # convenience shortcut for TimeStamps attachment data
...     assert czi.timestamps.shape == (2,)
...

Low-level access to CZI file segments:

>>> with CziFile('Example.czi') as czi:
...     # file header: version, GUIDs, and segment offsets
...     hdr = czi.header
...     assert hdr.version == (1, 0)
...     assert str(hdr.file_guid) == 'f8a61493-053e-c94e-bae0-bc7e96d18997'
...     assert not hdr.update_pending
...
...     # iterate all subblock segments sequentially via the directory
...     for segdata in czi.subblocks():
...         entry = segdata.directory_entry
...         assert entry.dims == ('H', 'T', 'C', 'Z', 'Y', 'X', 'S')
...         assert entry.start == (0, 0, 0, 0, 0, 582, 0)
...         assert entry.shape == (1, 1, 1, 1, 486, 1178, 1)
...         assert entry.stored_shape == (1, 1, 1, 1, 243, 589, 1)
...         assert entry.compression == CziCompressionType.ZSTDHDR
...         assert segdata.data_offset == 661865  # offset of image data
...         assert segdata.data_size == 183875  # size of compressed image data
...         tile = segdata.data()  # decompressed image data as numpy array
...         assert tile.shape == entry.stored_shape
...         assert tile.dtype == entry.pixel_type.dtype
...         assert isinstance(segdata.data(raw=True), bytes)  # compressed data
...         assert segdata.metadata().startswith('<METADATA>')
...         break  # just the first subblock segment for demonstration
...
...     # iterate only image tiles in a selected image view
...     img = czi.scenes[0](T=0, C=0, Z=0)
...     for entry in img.directory_entries:
...         segdata = entry.read_segment_data(czi)
...         assert isinstance(segdata, CziSubBlockSegmentData)
...         tile = segdata.data()
...         assert tile.shape == entry.stored_shape
...         break  # just the first filtered directory entry
...
...     # walk all file segments by type using their ZISRAW segment IDs
...     for segdata in czi.segments(CziSegmentId.ZISRAWSUBBLOCK):
...         assert isinstance(segdata, CziSubBlockSegmentData)
...
...     # direct low-level segment header at a known file offset
...     seg = CziSegment(czi, czi.header.directory_position)
...     assert seg.sid == CziSegmentId.ZISRAWDIRECTORY
...     assert seg.used_size == 68768
...     seg_data = seg.data()
...     assert isinstance(seg_data, CziSubBlockDirectorySegmentData)
...

View the images and metadata in a CZI file from the console::

    $ python -m czifile Example.czi

"""

from __future__ import annotations

__version__ = '2026.4.11'

__all__ = [
    'CONVERT_PIXELTYPE',
    'CziAttachmentDirectorySegmentData',
    'CziAttachmentEntryA1',
    'CziAttachmentSegmentData',
    'CziCompressionType',
    'CziContentFileType',
    'CziDeletedSegmentData',
    'CziDimensionEntryDV1',
    'CziDimensionType',
    'CziDirectoryEntryDV',
    'CziEventListEntry',
    'CziEventType',
    'CziFile',
    'CziFileError',
    'CziFileHeaderSegmentData',
    'CziImage',
    'CziImageChunks',
    'CziMetadataSegmentData',
    'CziPixelType',
    'CziPyramidType',
    'CziScenes',
    'CziSegment',
    'CziSegmentData',
    'CziSegmentId',
    'CziSegmentNotFoundError',
    'CziSubBlockDirectorySegmentData',
    'CziSubBlockSegmentData',
    'CziUnknownSegmentData',
    '__version__',
    'imread',
    'logger',
]

import abc
import bisect
import collections
import contextlib
import enum
import io
import itertools
import logging
import os
import re
import struct
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import cache, cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, cast, final, overload
from xml.etree import ElementTree

import imagecodecs
import numpy

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Generator,
        Iterable,
        Iterator,
        Mapping,
        Sequence,
    )
    from types import TracebackType
    from typing import IO, Any, Literal, Self

    from numpy.typing import ArrayLike, DTypeLike, NDArray
    from xarray import DataArray

type OutputType = str | IO[bytes] | NDArray[Any] | None
type SelectionValue = int | slice | Sequence[int] | None


FIRST_SCENE: int = -1010101  # sentinel: select first available scene


@overload
def imread(
    files: str | os.PathLike[Any] | IO[bytes],
    /,
    *,
    scene: SelectionValue = ...,
    roi: tuple[int, int, int, int] | None = ...,
    pixeltype: CziPixelType | None = ...,
    fillvalue: ArrayLike | None = ...,
    storedsize: bool = ...,
    squeeze: bool = ...,
    maxworkers: int | None = ...,
    asxarray: Literal[False] = ...,
    out: OutputType = ...,
    **selection: SelectionValue,
) -> NDArray[Any]: ...


@overload
def imread(
    files: str | os.PathLike[Any] | IO[bytes],
    /,
    *,
    scene: SelectionValue = ...,
    roi: tuple[int, int, int, int] | None = ...,
    pixeltype: CziPixelType | None = ...,
    fillvalue: ArrayLike | None = ...,
    storedsize: bool = ...,
    squeeze: bool = ...,
    maxworkers: int | None = ...,
    asxarray: Literal[True],
    out: OutputType = ...,
    **selection: SelectionValue,
) -> DataArray: ...


@overload
def imread(
    files: str | os.PathLike[Any] | IO[bytes],
    /,
    *,
    scene: SelectionValue = ...,
    roi: tuple[int, int, int, int] | None = ...,
    pixeltype: CziPixelType | None = ...,
    fillvalue: ArrayLike | None = ...,
    storedsize: bool = ...,
    squeeze: bool = ...,
    maxworkers: int | None = ...,
    asxarray: bool,
    out: OutputType = ...,
    **selection: SelectionValue,
) -> NDArray[Any] | DataArray: ...


def imread(
    files: str | os.PathLike[Any] | IO[bytes],
    /,
    *,
    scene: SelectionValue = FIRST_SCENE,
    roi: tuple[int, int, int, int] | None = None,
    pixeltype: CziPixelType | None = None,
    fillvalue: ArrayLike | None = 0,
    storedsize: bool = False,
    squeeze: bool = True,
    maxworkers: int | None = None,
    asxarray: bool = False,
    out: OutputType = None,
    **selection: SelectionValue,
) -> NDArray[Any] | DataArray:
    """Return image data from CZI file as numpy array or xarray DataArray.

    Parameters:
        files:
            File name or seekable binary stream.
        scene:
            Absolute S-coordinate(s) of scene(s) to read.
            Defaults to the first available scene.
            An ``int`` selects a single scene by its S-coordinate.
            A ``slice`` selects scenes whose S-coordinate falls within
            the range defined by the slice.
            A sequence of ``int`` merges the specified scenes into one
            array.
            If ``None``, read all scenes merged into one array.
        roi:
            Spatial region of interest in absolute pixel coordinates
            ``(x, y, width, height)``.
        pixeltype:
            Output pixel type.
            If ``None``, use the pixel type of the image.
            When specified, tiles are converted during compositing.
        fillvalue:
            Value for pixels not covered by any subblock.
        storedsize:
            Return pixel data at stored resolution.
            Skip resampling of Airyscan and PALM tiles.
        squeeze:
            If True, remove dimensions of length 1 from result.
        maxworkers:
            Maximum number of threads to decode subblock data.
            By default, up to half the CPU cores are used.
        asxarray:
            If True, return an xarray DataArray instead of a numpy array.
        out:
            Output destination for image data.
            Passed to :py:meth:`CziImage.asarray`
            or :py:meth:`CziImage.asxarray`.
        **selection:
            Dimension selections forwarded to :py:meth:`CziImage.__call__`.

    """
    with CziFile(files, squeeze=squeeze) as czi:
        if asxarray:
            return czi.asxarray(
                scene=scene,
                roi=roi,
                pixeltype=pixeltype,
                storedsize=storedsize,
                fillvalue=fillvalue,
                maxworkers=maxworkers,
                out=out,
                **selection,
            )
        return czi.asarray(
            scene=scene,
            roi=roi,
            pixeltype=pixeltype,
            storedsize=storedsize,
            fillvalue=fillvalue,
            maxworkers=maxworkers,
            out=out,
            **selection,
        )


class BinaryFile:
    """Binary file.

    Parameters:
        file:
            File name or seekable binary stream.
        mode:
            File open mode if `file` is a file name.
            If not specified, defaults to 'r'. Files are always opened
            in binary mode.

    Raises:
        TypeError:
            File is a text stream, or an unsupported type.
        ValueError:
            Invalid file name, extension, or stream.
            File stream is not seekable.

    """

    _fh: IO[bytes]
    _path: str  # absolute path of file
    _name: str  # name of file or handle
    _close: bool  # file needs to be closed
    _closed: bool  # file is closed
    _ext: ClassVar[set[str]] = set()  # valid extensions, empty for any

    def __init__(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        /,
        *,
        mode: Literal['r', 'r+'] | None = None,
    ) -> None:

        self._path = ''
        self._name = 'Unnamed'
        self._close = False
        self._closed = False
        self._lock: threading.RLock | NullLock = NullLock()

        if isinstance(file, (str, os.PathLike)):
            ext = os.path.splitext(file)[-1].lower()
            if self._ext and ext not in self._ext:
                msg = f'invalid file extension: {ext!r} not in {self._ext!r}'
                raise ValueError(msg)
            if mode is None:
                mode = 'r'
            else:
                if mode[-1:] == 'b':
                    # accept 'rb'/'r+b'
                    mode = mode[:-1]  # type: ignore[assignment]
                if mode not in ('r', 'r+'):
                    msg = f'invalid {mode=!r}'
                    raise ValueError(msg)
            self._path = os.path.abspath(file)
            self._close = True
            self._fh = open(self._path, mode + 'b')  # noqa: SIM115

        elif hasattr(file, 'seek'):
            # binary stream: open file, BytesIO, fsspec LocalFileOpener
            if isinstance(file, io.TextIOBase):  # type: ignore[unreachable]
                msg = (  # type: ignore[unreachable]
                    f'{file=!r} is not open in binary mode'
                )
                raise TypeError(msg)

            self._fh = file
            try:
                self._fh.tell()
            except Exception as exc:
                msg = f'{file=!r} is not seekable'
                raise ValueError(msg) from exc
            if hasattr(file, 'path'):
                self._path = os.path.abspath(file.path)
            elif hasattr(file, 'name'):
                self._path = os.path.abspath(file.name)

        elif hasattr(file, 'open'):
            # fsspec OpenFile
            self._fh = file.open()
            self._close = True
            try:
                self._fh.tell()
            except Exception as exc:
                with contextlib.suppress(Exception):
                    self._fh.close()
                msg = f'{file=!r} is not seekable'
                raise ValueError(msg) from exc
            if hasattr(file, 'path'):
                self._path = os.path.abspath(file.path)

        else:
            msg = f'cannot handle {type(file)=}'
            raise TypeError(msg)

        if hasattr(file, 'name') and file.name:
            self._name = os.path.basename(file.name)
        elif self._path:
            self._name = os.path.basename(self._path)
        else:
            self._name = type(file).__name__

    @property
    def filehandle(self) -> IO[bytes]:
        """File handle."""
        return self._fh

    @property
    def filepath(self) -> str:
        """Absolute path to file, or empty string if unavailable."""
        return self._path

    @property
    def filename(self) -> str:
        """Name of file, or empty if no path is available."""
        return os.path.basename(self._path)

    @property
    def dirname(self) -> str:
        """Directory containing file, or empty if no path is available."""
        return os.path.dirname(self._path)

    @property
    def name(self) -> str:
        """Display name of file."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def attrs(self) -> Mapping[str, Any]:
        """Selected metadata as dict."""
        return MappingProxyType({'name': self.name, 'filepath': self.filepath})

    @property
    def lock(self) -> threading.RLock | NullLock:
        """Lock for thread-safe file access."""
        return self._lock

    def set_lock(self, enabled: bool, /) -> None:  # noqa: FBT001
        """Enable or disable thread-safe file access.

        Parameters:
            enabled: If true, use a threading.RLock, else a no-op lock.

        """
        self._lock = threading.RLock() if enabled else NullLock()

    @property
    def closed(self) -> bool:
        """File is closed."""
        return self._closed

    def close(self) -> None:
        """Close file."""
        self._closed = True  # always report file as closed
        if self._close:
            with contextlib.suppress(Exception):
                self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self._name!r}>'


class CziFileError(ValueError):
    """Exception to indicate invalid CZI file structure."""


class CziSegmentNotFoundError(ValueError):
    """Exception to indicate that file does not contain segment at position."""


@final
class CziFile(BinaryFile):
    """Read image and metadata from Carl Zeiss Image (CZI) file.

    CziFile instances must be closed with :py:meth:`CziFile.close`, which
    is automatically called when using the 'with' context manager.

    CziFile instances are not thread-safe. All attributes are read-only.

    Parameters:
        file:
            CZI file to read.
            Open file objects must be positioned at CZI header.
        mode:
            File open mode if `file` is file name. Defaults to ``'r'``.
        squeeze:
            If True, remove dimensions of length 1 from results.

    Raises:
        CziFileError: File is not a valid ZISRAW file.

    Notes:
        Opening a file with ``mode='r+'`` enables in-place modification
        via :py:attr:`filehandle`. When used as a context manager, the
        ``update_pending`` flag in the file header is automatically set
        on entry and cleared on clean exit. It remains set if an
        exception occurs, triggering a warning on the next open.

    """

    header: CziFileHeaderSegmentData
    """Global file metadata such as file version and GUID."""

    _fh: IO[bytes]
    _cache: DecodeCache
    _auto_cache: bool

    def __init__(
        self,
        file: str | os.PathLike[Any] | IO[bytes],
        /,
        *,
        mode: Literal['r', 'r+', 'rb', 'r+b'] | None = None,
        squeeze: bool = True,
    ) -> None:
        # open CZI file and read header
        mode_: Literal['r', 'r+'] | None = None
        if mode is None:
            mode_ = None
        elif mode in ('r', 'rb'):
            mode_ = 'r'
        elif mode in ('r+', 'r+b'):
            mode_ = 'r+'
        super().__init__(file, mode=mode_)

        fh = self._fh

        magic = fh.read(10)
        if magic != b'ZISRAWFILE':
            fh.close()
            msg = f'not a CZI file {magic=!r}'
            raise CziFileError(msg)

        try:
            self.header = cast(
                CziFileHeaderSegmentData, CziSegment(self, 0).data()
            )
        except Exception as exc:
            fh.close()
            msg = 'Failed to read CZI file header'
            raise CziFileError(msg) from exc

        if self.header.update_pending:
            logger().warning('file is pending update')
        self._squeeze = squeeze
        self._cache = DecodeCache()
        self._auto_cache = True

    def __enter__(self) -> Self:
        super().__enter__()
        if self._fh.writable():
            fh = self._fh
            fh.seek(self.header.segment.data_offset + 68)
            fh.write(struct.pack('<i', 1))
            self.header.update_pending = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._fh.writable() and exc_type is None:
            fh = self._fh
            fh.seek(self.header.segment.data_offset + 68)
            fh.write(struct.pack('<i', 0))
            fh.flush()
            self.header.update_pending = False
        super().__exit__(exc_type, exc_value, traceback)

    @property
    def maxcache(self) -> int | None:
        """Maximum number of decoded subblock arrays to cache.

        ``None`` (default) enables automatic cache management:
        :py:meth:`CziImage.chunks` activates a scoped cache when
        spatial tiling is requested and leaves it disabled otherwise.
        ``0`` explicitly disables caching for all operations.
        Positive integer sets a persistent FIFO cache of that size.
        Cache is keyed by file offset and uses FIFO eviction.
        Caching is only applied to compressed subblocks.

        """
        return None if self._auto_cache else self._cache.maxsize

    @maxcache.setter
    def maxcache(self, value: int | None) -> None:
        self._auto_cache = value is None
        self._cache.resize(0 if value is None else value)

    def segments(
        self, kind: str | Sequence[str] | None = None, /
    ) -> Generator[CziSegmentData]:
        """Return iterator over segment data of specified kind.

        Parameters:
            kind:
                CziSegment identifier or sequence of identifiers to match.
                By default, all segments are returned.

        """
        kinds: frozenset[str] | None
        if kind is None:
            kinds = None
        elif isinstance(kind, str):
            kinds = frozenset((kind,))
        else:
            kinds = frozenset(kind)
        offset = 0
        while True:
            try:
                segment = CziSegment(self, offset)
            except CziSegmentNotFoundError:
                break
            if kinds is None or segment.sid in kinds:
                yield segment.data()
            offset = segment.data_offset + segment.allocated_size

    @overload
    def metadata(self, *, asdict: Literal[False] = ...) -> str: ...

    @overload
    def metadata(self, *, asdict: Literal[True]) -> dict[str, Any]: ...

    @overload
    def metadata(self, *, asdict: bool) -> str | dict[str, Any]: ...

    def metadata(self, *, asdict: bool = False) -> str | dict[str, Any]:
        """Return data of CziMetadataSegmentData from file.

        Returns an empty string, or ``{}`` when ``asdict=True``, if no
        CziMetadataSegmentData is found.

        Parameters:
            asdict: Return metadata as dict instead of XML.

        """
        if asdict:
            xml_element = self.xml_element
            return {} if xml_element is None else xml2dict(xml_element)

        if self.header.metadata_position:
            segment = CziSegment(self, self.header.metadata_position)
            if segment.sid == CziMetadataSegmentData.SID:
                mds = cast(CziMetadataSegmentData, segment.data())
                return mds.data()
        logger().warning('CziMetadataSegmentData not found')
        try:
            metadata = next(self.segments(CziMetadataSegmentData.SID))
            msd = cast(CziMetadataSegmentData, metadata)
            self.header.metadata_position = msd.segment.offset
            return msd.data()
        except StopIteration:
            pass
        # raise CziFileError('CziMetadataSegmentData not found')
        return ''

    @cached_property
    def xml_element(self) -> ElementTree.Element | None:
        """Parsed XML root element of CZI metadata, or ``None``."""
        xml_str = self.metadata()
        if not xml_str:
            return None
        return ElementTree.fromstring(xml_str)

    @cached_property
    def metadata_segment(self) -> CziMetadataSegmentData | None:
        """Metadata segment data, or ``None`` if not found."""
        if self.header.metadata_position:
            segment = CziSegment(self, self.header.metadata_position)
            if segment.sid == CziMetadataSegmentData.SID:
                return cast(CziMetadataSegmentData, segment.data())
        try:
            msd = cast(
                CziMetadataSegmentData,
                next(self.segments(CziMetadataSegmentData.SID)),
            )
            self.header.metadata_position = msd.segment.offset
        except StopIteration:
            return None
        return msd

    @cached_property
    def subblock_directory(self) -> tuple[CziDirectoryEntryDV, ...]:
        """All directory entries in file.

        Uses subblock directory segment if available, else searches for
        subblock segments in file.

        """
        if self.header.directory_position:
            segment = CziSegment(self, self.header.directory_position)
            if segment.sid == CziSubBlockDirectorySegmentData.SID:
                return cast(
                    CziSubBlockDirectorySegmentData, segment.data()
                ).entries
        logger().warning('SubBlockDirectory segment not found')
        return tuple(
            cast(CziSubBlockSegmentData, segment).directory_entry
            for segment in self.segments(CziSubBlockSegmentData.SID)
        )

    @cached_property
    def filtered_subblock_directory(self) -> tuple[CziDirectoryEntryDV, ...]:
        """Directory entries filtered to pyramid level 0.

        Entries with a mosaic index are sorted by mosaic index so that
        tiles with higher M-index are composited on top.

        """
        filtered = [
            entry
            for entry in self.subblock_directory
            if entry.pyramid_type == 0
        ]
        if not filtered:
            return self.subblock_directory

        # Some producers store pyramid levels with pyramid_type=0 instead of
        # using the proper pyramid_type field. Detect this by checking whether
        # any entry has stored_shape != shape (is_pyramid). When a full-
        # resolution subset exists AND the is_pyramid entries only duplicate
        # existing non-spatial positions (i.e. they are thumbnails, not unique
        # data like Airyscan detector channels), keep only the full-resolution
        # entries so that _scale_factors comes out to 1.0.
        if any(entry.is_pyramid for entry in filtered):
            full_res = [entry for entry in filtered if not entry.is_pyramid]
            if full_res:
                # Build non-spatial position keys (exclude Y, X, S dims).
                # If every is_pyramid key also exists in full_res, the
                # is_pyramid entries are redundant thumbnails (safe to drop).
                # If any is_pyramid entry has a unique position not in
                # full_res (e.g. Airyscan H/C channels), keep all entries.
                ref_dims = filtered[0].dims
                yx_set = frozenset(
                    i for i, d in enumerate(ref_dims) if d in ('Y', 'X', 'S')
                )

                def _ns_key(
                    e: CziDirectoryEntryDV,
                ) -> tuple[int, ...]:
                    if e.dims == ref_dims:
                        return tuple(
                            s for i, s in enumerate(e.start) if i not in yx_set
                        )
                    d = dict(zip(e.dims, e.start, strict=True))
                    return tuple(
                        d.get(dim, 0)
                        for i, dim in enumerate(ref_dims)
                        if i not in yx_set
                    )

                full_res_keys = {_ns_key(e) for e in full_res}
                pyramid_keys = {_ns_key(e) for e in filtered if e.is_pyramid}
                if pyramid_keys <= full_res_keys:
                    filtered = full_res
        if any(entry.mosaic_index >= 0 for entry in filtered):
            filtered.sort(key=lambda x: x.mosaic_index)
        return tuple(filtered)

    @cached_property
    def attachment_directory(self) -> tuple[CziAttachmentEntryA1, ...]:
        """All attachment entries in file.

        Uses attachment directory segment if available, else searches for
        attachment segments in file.

        """
        if self.header.attachment_directory_position:
            segment = CziSegment(
                self, self.header.attachment_directory_position
            )
            if segment.sid == CziAttachmentDirectorySegmentData.SID:
                return cast(
                    CziAttachmentDirectorySegmentData, segment.data()
                ).entries
        logger().warning('AttachmentDirectory segment not found')
        return tuple(
            cast(CziAttachmentSegmentData, segment).attachment_entry
            for segment in self.segments(CziAttachmentSegmentData.SID)
        )

    def subblocks(self) -> Generator[CziSubBlockSegmentData]:
        """Yield all subblock segment data in file."""
        for subblock in self.subblock_directory:
            segmentdata = subblock.read_segment_data(self)
            if isinstance(segmentdata, CziSubBlockSegmentData):
                # segment may have been deleted
                yield segmentdata

    def attachments(self) -> Generator[CziAttachmentSegmentData]:
        """Yield all attachment segment data in file."""
        for attachment in self.attachment_directory:
            segmentdata = attachment.read_segment_data(self)
            if isinstance(segmentdata, CziAttachmentSegmentData):
                # segment may have been deleted
                yield segmentdata

    def save_attachments(
        self,
        directory: str | os.PathLike[Any] | None = None,
    ) -> None:
        """Save all attachments to files."""
        if directory is None:
            directory = self.filepath + '.attachments'
        os.makedirs(directory, exist_ok=True)
        for attachment in self.attachments():
            attachment.save(directory=directory)

    @cached_property
    def timestamps(self) -> NDArray[Any] | None:
        """Timestamps from TimeStamps attachment, or ``None``.

        Values are float64. The unit depends on the CZI producer:
        some files store seconds relative to the start of acquisition,
        others store OLE Automation Dates (days since 1899-12-30).

        """
        for attachment in self.attachments():
            if attachment.attachment_entry.name == 'TimeStamps':
                return attachment.data()  # type: ignore[no-any-return]
        return None

    @cached_property
    def scenes(self) -> CziScenes:
        """Mapping of scene S-coordinates to CziImage instances."""
        return CziScenes(self, squeeze=self._squeeze)

    def asarray(
        self,
        *,
        scene: SelectionValue = FIRST_SCENE,
        roi: tuple[int, int, int, int] | None = None,
        pixeltype: CziPixelType | None = None,
        fillvalue: ArrayLike | None = 0,
        storedsize: bool = False,
        maxworkers: int | None = None,
        out: OutputType = None,
        **selection: SelectionValue,
    ) -> NDArray[Any]:
        """Return image data from file as numpy array.

        Parameters:
            scene:
                Absolute S-coordinate(s) of scene(s) to read.
                Defaults to the first available scene.
                An ``int`` selects a single scene by its S-coordinate.
                A ``slice`` selects scenes whose S-coordinate falls
                within the range defined by the slice.
                A sequence of ``int`` merges the specified scenes into
                one array.
                If ``None``, read all scenes merged into one array.
                Use ``czi.scenes.keys()`` to enumerate valid values.
            roi:
                Spatial region of interest in absolute pixel coordinates
                ``(x, y, width, height)``.
                If ``None``, read the entire image.
            pixeltype:
                Output pixel type.
                If ``None``, use the pixel type of the image.
                When specified, tiles are converted during compositing.
            fillvalue:
                Value for pixels not covered by any subblock.
            storedsize:
                Return pixel data at stored resolution.
                Skip resampling of Airyscan and PALM tiles.
            maxworkers:
                Maximum number of threads to decode subblock data.
                By default, up to half the CPU cores are used.
            out:
                Output destination for image data.
                If ``None``, create a new NumPy array in main memory.
                If ``'memmap'``, create a memory-mapped array in a
                temporary file.
                If a ``numpy.ndarray``, a writable, initialized array
                of compatible shape and dtype.
                If a ``file name`` or ``open file``, create a
                memory-mapped array in the specified file.
            **selection:
                Dimension selections forwarded to :py:meth:`CziImage.__call__`.

        """
        if scene is None:
            image = self.scenes()
        elif scene == FIRST_SCENE:
            image = next(iter(self.scenes.values()))
        elif isinstance(scene, int):
            image = self.scenes[scene]
        else:
            image = self.scenes(scene=scene)
        if roi is not None or selection or pixeltype is not None or storedsize:
            image = image(
                roi=roi,
                pixeltype=pixeltype,
                storedsize=storedsize,
                **selection,
            )
        return image.asarray(
            fillvalue=fillvalue, maxworkers=maxworkers, out=out
        )

    def asxarray(
        self,
        *,
        scene: SelectionValue = FIRST_SCENE,
        roi: tuple[int, int, int, int] | None = None,
        pixeltype: CziPixelType | None = None,
        fillvalue: ArrayLike | None = 0,
        storedsize: bool = False,
        maxworkers: int | None = None,
        out: OutputType = None,
        **selection: SelectionValue,
    ) -> DataArray:
        """Return image data from file as xarray DataArray.

        Parameters:
            scene:
                Absolute S-coordinate(s) of scene(s) to read.
                Defaults to the first available scene.
                An ``int`` selects a single scene by its S-coordinate.
                A ``slice`` selects scenes whose S-coordinate falls
                within the range defined by the slice.
                A sequence of ``int`` merges the specified scenes into
                one DataArray.
                If ``None``, read all scenes merged into one DataArray.
            roi:
                Spatial region of interest in absolute pixel coordinates
                ``(x, y, width, height)``.
                If ``None``, read the entire image.
            pixeltype:
                Output pixel type.
                If ``None``, use the pixel type of the image.
                When specified, tiles are converted during compositing.
            fillvalue:
                Value for pixels not covered by any subblock.
            storedsize:
                Return pixel data at stored resolution.
                Skip resampling of Airyscan and PALM tiles.
            maxworkers:
                Maximum number of threads to decode subblock data.
                By default, up to half the CPU cores are used.
            out:
                Output destination for backing NumPy array.
                If ``None``, create a new NumPy array in main memory.
                If ``'memmap'``, create a memory-mapped array in a
                temporary file.
                If a ``numpy.ndarray``, a writable, initialized array
                of compatible shape and dtype.
                If a ``file name`` or ``open file``, create a
                memory-mapped array in the specified file.
            **selection:
                Dimension selections forwarded to :py:meth:`CziImage.__call__`.

        """
        if scene is None:
            image = self.scenes()
        elif scene == FIRST_SCENE:
            image = next(iter(self.scenes.values()))
        elif isinstance(scene, int):
            image = self.scenes[scene]
        else:
            image = self.scenes(scene=scene)
        if roi is not None or selection or pixeltype is not None or storedsize:
            image = image(
                roi=roi,
                pixeltype=pixeltype,
                storedsize=storedsize,
                **selection,
            )
        return image.asxarray(
            fillvalue=fillvalue, maxworkers=maxworkers, out=out
        )

    def __repr__(self) -> str:
        ls = len(self.scenes)
        scenes = '1 scene' if ls == 1 else f'{ls} scenes'
        return f'<{self.__class__.__name__} {self._name!r} {scenes}>'

    def __str__(self) -> str:
        entries = self.filtered_subblock_directory
        return indent(
            repr(self),
            os.path.normpath(os.path.normcase(self.filepath)),
            '',
            self.header,
            '',
            entries[0] if entries else 'no subblock directory entries',
            '',
            *(repr(im) for im in self.scenes.values()),
            # pformat(self.metadata(), height=1024),
        )


class CziImage:
    """Filtered view of image data in a CZI file.

    A CziImage represents a subset of subblock directory entries,
    filtered by scene, dimension selection, or pyramid level.

    All attributes are read-only.

    Parameters:
        parent:
            CziFile instance providing file access.
        directory_entries:
            Subblock directory entries for this image view.
        name:
            Display name incorporating scene and selection info.
        squeeze:
            If true, remove dimensions of length 1 from results.
        roi:
            Spatial region of interest in absolute pixel coordinates
            ``(x, y, width, height)``.
            If ``None``, use the full extent.
        pixeltype:
            Output pixel type override.
            If ``None``, the pixel type is taken from the first directory
            entry.
            When set, :py:attr:`dtype`, :py:attr:`shape`, and
            :py:meth:`asarray` all reflect the requested type.
        storedsize:
            If true, return pixel data at the stored resolution
            instead of resampling to the logical size.
            Airyscan sub-sampled tiles are returned at their smaller
            stored size; PALM super-resolution tiles at their larger
            stored size.
            Raises ``ValueError`` when subblocks have non-uniform
            stored-to-logical ratios (apply a dimension selection
            first to make ratios uniform).
        selection:
            Dimension selections as mapping of dimension name to
            ``int``, ``slice``, ``Sequence[int]``, or ``None``.

    Notes:
        All coordinates are absolute (CZI file-level), not relative to a
        scene or subblock. This follows the ZISRAW specification and the
        behaviour of the libczi reference implementation.
        This applies consistently across the API:

        **Dimension selection**
        (``**kwargs`` to :py:meth:`CziImage.__call__`):
        An integer value such as ``T=0`` or ``C=1`` selects subblocks whose
        file-level start coordinate for that dimension equals the given value.
        If the file's T axis starts at 5, use ``T=5`` to select the first
        time point - ``T=0`` would match nothing.
        A ``Sequence[int]`` selects subblocks whose coordinate equals any of
        the given absolute values.
        A ``slice`` selects subblocks whose coordinate falls within the range
        defined by the slice start/stop/step, all interpreted as absolute
        coordinate values (not positional offsets).
        ``ValueError`` is raised if the selection matches no subblocks.
        The dimensions ``X``, ``Y``, and ``S`` (color samples) are excluded
        from ``**selection``: use ``roi`` for spatial cropping and
        :py:class:`CziScenes` (:py:attr:`CziFile.scenes`) for scene selection.

        **Dimension ordering**: the output axis order follows three groups:
        (1) unspecified non-spatial dimensions, in their original file order;
        (2) explicitly specified dimensions (including ``None``), in the order
        they appear as keyword arguments;
        (3) spatial dimensions (Y, X, S), always last.
        Dimensions selected with an integer are collapsed and absent from the
        output. Use :py:attr:`CziImage.dims` or :py:attr:`CziImage.sizes`
        to read back the resulting order.

        **ROI** (``roi`` parameter): ``(x, y, width, height)`` are global
        pixel coordinates, matching the values in :py:attr:`CziImage.bbox`
        and :py:attr:`CziImage.start`.

        **Squeeze and size-1 dimensions**:
        when ``squeeze=True`` (the default), dimensions of length 1 are
        removed from :py:attr:`CziImage.dims`, :py:attr:`CziImage.start`,
        and the :py:class:`CziImageChunks` iteration axes.
        Disable squeeze with ``CziFile(file, squeeze=False)`` when you
        need to access all coordinate axes unconditionally.

        **Scene access** (:py:class:`CziScenes`): scene selection uses the
        absolute S-coordinate, consistent with all other dimension access.
        Here, S-coordinate refers to scene indices, not the sample
        dimension ``S``.
        ``czi.scenes[4]`` selects the scene whose S-coordinate is 4 in the
        file. ``KeyError`` is raised for an unknown S-coordinate.
        ``czi.scenes(scene=slice(2, 5))`` returns a merged image of all
        scenes whose S-coordinate falls in ``range(2, 5)``. S-coordinates
        within the range that are absent from the file are silently skipped,
        and ``KeyError`` is raised only when the range is entirely empty.
        Use ``czi.scenes.keys()`` to enumerate valid S-coordinate keys.
        :py:attr:`scene_indices` on a :py:class:`CziImage` returns the
        S-coordinate(s) as a sorted tuple: ``(4,)`` for a single scene,
        ``(0, 1)`` for a merged two-scene image, or ``None`` for files
        without an explicit S dimension.
        Roundtrip: ``czi.scenes(scene=image.scene_indices)`` reconstructs
        the same image.

    """

    _parent: CziFile
    _entries: tuple[CziDirectoryEntryDV, ...]
    _name: str
    _squeeze: bool
    _roi: tuple[int, int, int, int] | None
    _selection: dict[str, int | slice | Sequence[int] | None]
    _pixeltype: CziPixelType | None
    _storedsize: bool
    _levels: list[CziImage]

    def __init__(
        self,
        parent: CziFile,
        directory_entries: tuple[CziDirectoryEntryDV, ...],
        /,
        *,
        name: str = '',
        squeeze: bool = True,
        roi: tuple[int, int, int, int] | None = None,
        pixeltype: CziPixelType | None = None,
        storedsize: bool = False,
        selection: dict[str, SelectionValue] | None = None,
    ) -> None:
        self._parent = parent
        self._entries = directory_entries
        self._name = name
        self._squeeze = squeeze
        self._roi = roi
        self._selection = dict(selection) if selection else {}
        self._pixeltype = pixeltype
        self._storedsize = storedsize
        self._levels = [self]

    def __call__(
        self,
        *,
        roi: tuple[int, int, int, int] | None = None,
        pixeltype: CziPixelType | None = None,
        storedsize: bool | None = None,
        **selection: SelectionValue,
    ) -> CziImage:
        """Return new CziImage with dimension selection and/or ROI.

        Parameters:
            roi:
                Spatial region of interest in absolute pixel coordinates
                ``(x, y, width, height)``.
            pixeltype:
                Output pixel type override.
                If ``None``, inherit any existing override from this image.
            storedsize:
                Return pixel data at stored resolution without resampling.
                If ``None``, inherit from parent image.
            **selection:
                Dimension selections.
                Keys are dimension names (single uppercase characters,
                excluding X/Y/S).
                Values can be ``int`` (single index), ``slice`` (range),
                ``Sequence[int]`` (specific indices), or ``None`` (all
                values; use for dimension ordering only).
                All values are absolute CZI file-level coordinates.

                **Output dimension order** is determined as follows:
                (1) unspecified non-spatial dimensions, in file order;
                (2) specified dimensions, in the order they appear as
                keyword arguments;
                (3) spatial dimensions (Y, X, and optionally S), always
                last.
                Dimensions selected with an ``int`` are collapsed and do
                not appear in the output shape.

        Raises:
            TypeError:
                If this CziImage already has a ROI applied, or if a
                dimension selection is requested on an image that already
                has a dimension selection or ROI applied.
                Applying a ROI to an image that already has a dimension
                selection is permitted (one level of chaining only):
                ``img(C=0)(roi=...)`` works, but further calls on the
                result always raise ``TypeError``.
            ValueError:
                If roi width or height is not positive, or if a
                dimension name is unknown or spatial.

        Example::

            # img.dims = ('T', 'C', 'Z', 'Y', 'X')

            # fix C=1: result has dims ('T', 'Z', 'Y', 'X')
            sub = img(C=1)

            # reorder to Z-outer, T-inner; result has dims ('Z', 'T', 'Y', 'X')
            sub = img(Z=None, T=None)

            # combine: fix T to first time-point, then reorder C outer Z inner
            # all coordinate values are absolute CZI file-level indices
            sub = img(T=0, C=None, Z=None)  # dims: ('C', 'Z', 'Y', 'X')

            # crop to a 256x256 ROI (absolute pixel coordinates)
            x0, y0, *_ = img.bbox
            sub = img(C=0, roi=(x0, y0, 256, 256))

        """
        inherited_pixeltype = (
            pixeltype if pixeltype is not None else self._pixeltype
        )
        inherited_storedsize = (
            storedsize if storedsize is not None else self._storedsize
        )
        if (
            not selection
            and roi is None
            and pixeltype is None
            and storedsize is None
        ):
            return self
        # Chaining rule:
        #   - Applying a ROI on top of a dimension selection is allowed.
        #   - Re-applying a selection, re-applying a ROI, or applying a
        #     selection on top of a ROI are all disallowed.
        if selection and (self._selection or self._roi is not None):
            msg = (
                'cannot apply dimension selection to a CziImage that'
                ' already has selection or roi applied'
            )
            raise TypeError(msg)
        if roi is not None and self._roi is not None:
            msg = (
                'cannot apply roi to a CziImage that already has a roi applied'
            )
            raise TypeError(msg)

        # validate selection dimensions
        raw = self._raw_sizes
        dims_list = list(raw.keys())
        for dim in selection:
            if dim in ('X', 'Y', 'S'):
                if dim == 'S':
                    msg = (
                        "'S' is the scene/sample dimension: "
                        'use czi.scenes for scene filtering'
                    )
                else:
                    msg = f'{dim!r} is a spatial dimension: use roi'
                raise ValueError(msg)
            if dim not in raw:
                msg = (
                    f'unknown dimension {dim!r}, '
                    f'available: {tuple(raw.keys())}'
                )
                raise ValueError(msg)

        # filter entries by dimension selection
        entries: (
            tuple[CziDirectoryEntryDV, ...] | list[CziDirectoryEntryDV]
        ) = self._entries
        for dim, val in selection.items():
            if val is None:
                continue
            dim_idx = dims_list.index(dim)
            if isinstance(val, int):
                entries = tuple(e for e in entries if e.start[dim_idx] == val)
            elif isinstance(val, slice):
                # Absolute coordinate range, consistent with int selection:
                # slice start/stop/step are coordinate values, not positions.
                dim_start = self._start[dim_idx]
                dim_size = raw[dim]
                s_start = val.start if val.start is not None else dim_start
                s_stop = (
                    val.stop
                    if val.stop is not None
                    else (dim_start + dim_size)
                )
                s_step = val.step if val.step is not None else 1
                selected_set = set(range(s_start, s_stop, s_step))
                entries = tuple(
                    e for e in entries if e.start[dim_idx] in selected_set
                )
            else:
                # Sequence[int] - values are coordinate values
                allowed = set(val)
                entries = tuple(
                    e for e in entries if e.start[dim_idx] in allowed
                )

        if not entries:
            msg = 'selection matches no subblocks'
            raise ValueError(msg)

        # filter entries by spatial ROI
        if roi is not None:
            rx, ry, rw, rh = roi
            if rw <= 0 or rh <= 0:
                msg = 'roi width and height must be positive'
                raise ValueError(msg)
            layout = self._subblock_layout
            xi = layout.x_idx
            yi = layout.y_idx
            # Use cached spatial index when filtering the full _entries
            # unchanged (no dimension selection applied first). This is the
            # common case for repeated tile reads on the same CziImage.
            if entries is self._entries and xi >= 0:
                entries = tuple(self._spatial_index_filter(rx, ry, rw, rh))
            else:
                roi_filtered: list[CziDirectoryEntryDV] = []
                for e in entries:
                    if xi >= 0:
                        ex = e.start[xi]
                        ew = e.shape[xi]
                        if ex + ew <= rx or ex >= rx + rw:
                            continue
                    if yi >= 0:
                        ey = e.start[yi]
                        eh = e.shape[yi]
                        if ey + eh <= ry or ey >= ry + rh:
                            continue
                    roi_filtered.append(e)
                entries = tuple(roi_filtered)
            if not entries:
                msg = 'roi matches no subblocks'
                raise ValueError(msg)

        # build name
        parts = []
        if self._name:
            parts.append(self._name)
        sel_parts = [f'{d}={v!r}' for d, v in selection.items()]
        if sel_parts:
            parts.append(', '.join(sel_parts))
        if roi is not None:
            parts.append(f'roi={roi!r}')
        name = ' '.join(parts) if parts else ''

        # When chaining ROI onto an image that already has a dimension
        # selection, carry the parent selection forward so that dimension
        # ordering is preserved in the result.
        merged_selection = {**self._selection, **selection}

        return CziImage(
            self._parent,
            tuple(entries) if isinstance(entries, list) else entries,
            name=name,
            squeeze=self._squeeze,
            roi=roi,
            pixeltype=inherited_pixeltype,
            storedsize=inherited_storedsize,
            selection=merged_selection or selection,
        )

    @property
    def name(self) -> str:
        """Display name of image incorporating scene and selection info."""
        return self._name

    @property
    def compression(self) -> CziCompressionType:
        """Compression type of image."""
        return self._entries[0].compression

    @cached_property
    def scene_indices(self) -> tuple[int, ...] | None:
        """Absolute S-coordinates of this image's scene(s), or None.

        Returns a sorted tuple of S-coordinate values as stored in the CZI
        file - the same values used as keys in :py:class:`CziScenes`.

        * Single scene at S=2: ``(2,)``
        * Merged scenes S=0 and S=1: ``(0, 1)``
        * All merged scenes: sorted S-coordinate values present in the
          image, for example ``(2, 4, 7)``
        * File with no explicit S dimension (implicit): ``None``

        Roundtrip: ``czi.scenes(scene=image.scene_indices)`` reproduces this
        image. Returns ``None`` for implicit single-scene files.

        """
        seen = sorted(
            {e.scene_index for e in self._entries if e.scene_index != -1}
        )
        return tuple(seen) if seen else None

    @cached_property
    def pixeltype(self) -> CziPixelType:
        """Pixel type of image data.

        Return the explicit override if one was provided, otherwise the
        best common type across all channels.

        """
        if self._pixeltype is not None:
            return self._pixeltype
        entries = self._entries
        first_pt = entries[0].pixel_type
        if len(entries) == 1:
            return first_pt
        ref_dims = entries[0].dims
        c_idx = ref_dims.index('C') if 'C' in ref_dims else -1
        if c_idx < 0:
            # no channel axis: all tiles share the same pixel type
            return first_pt
        # pixel type varies only along C - examine one entry per C value,
        # not every tile (O(C) instead of O(N tiles))
        seen_c: dict[int, CziPixelType] = {entries[0].start[c_idx]: first_pt}
        for e in entries[1:]:
            if e.dims == ref_dims:
                c = e.start[c_idx]
            elif 'C' in e.dims:
                c = e.start[e.dims.index('C')]
            else:
                continue
            if c not in seen_c:
                seen_c[c] = e.pixel_type
        pts = set(seen_c.values())
        if len(pts) == 1:
            return first_pt
        # promote: widest dtype via numpy.result_type, most samples via max
        promoted_dtype = numpy.result_type(*[pt.dtype for pt in pts])
        max_samples = max(pt.samples for pt in pts)
        return CziPixelType.get(promoted_dtype, max_samples)

    @property
    def dtype(self) -> numpy.dtype[Any]:
        """NumPy data type of image."""
        return self.pixeltype.dtype

    @cached_property
    def sizes(self) -> Mapping[str, int]:
        """Ordered mapping of dimension name to length."""
        raw = self._raw_sizes
        ordered = {d: raw[d] for d in self._dim_order}
        if self._squeeze:
            return MappingProxyType(
                {k: v for k, v in ordered.items() if v > 1}
            )
        return MappingProxyType(ordered)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image."""
        return tuple(self.sizes.values())

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names of image."""
        return tuple(self.sizes.keys())

    @property
    def axes(self) -> str:
        """Character codes for each dimension in image."""
        return ''.join(self.sizes.keys())

    @property
    def ndim(self) -> int:
        """Number of image dimensions."""
        return len(self.sizes)

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by image."""
        return self.size * self.dtype.itemsize

    @property
    def size(self) -> int:
        """Number of elements in image."""
        size = 1
        for v in self.sizes.values():
            size *= v
        return size

    @cached_property
    def start(self) -> tuple[int, ...]:
        """Minimum start indices per dimension."""
        raw = self._raw_sizes
        dim_list = list(raw.keys())
        reordered = tuple(
            self._start[dim_list.index(d)] for d in self._dim_order
        )
        if self._squeeze:
            return tuple(
                s
                for s, d in zip(reordered, self._dim_order, strict=True)
                if raw[d] > 1
            )
        return reordered

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Spatial bounding box in absolute pixel coordinates.

        Returns:
            Bounding box as ``(x, y, width, height)``.

        """
        raw = self._raw_sizes
        dims = list(raw.keys())
        start = self._start
        x = start[dims.index('X')] if 'X' in dims else 0
        y = start[dims.index('Y')] if 'Y' in dims else 0
        w = raw.get('X', 0)
        h = raw.get('Y', 0)
        return (x, y, w, h)

    @cached_property
    def coords(self) -> Mapping[str, NDArray[Any]]:
        """Mapping of dimension names to physical coordinate arrays.

        Spatial dimensions (X, Y, Z) use pixel spacing in meters from CZI
        scaling metadata.

        T uses acquisition datetimes when both acquisition time and interval
        metadata are available, seconds elapsed since acquisition start when
        only interval metadata is available, falls back to the TimeStamps
        attachment otherwise, or integer indices if no timing metadata exists.

        C uses the midpoint of `DetectionWavelength` in nanometres from
        :py:attr:`channels` when that field is present for every channel in
        the slice and the ranges are mutually non-overlapping (narrow
        spectral bins, for example LSM980 spectral imaging). When the ranges
        overlap, channel names are used instead. When `DetectionWavelength`
        is absent, channel names are used; if all names parse as floats
        (for example, wavelengths encoded as names like ``'480'``), a float
        array is returned. Falls back to integer indices if names are absent.

        """
        result: dict[str, NDArray[Any]] = {}
        root = self._parent.xml_element
        if root is None:
            return result

        # pixel spacing in meters per axis id
        scaling: dict[str, float] = {}
        for dist in root.findall('.//Scaling/Items/Distance'):
            axis = dist.get('Id')
            val = dist.findtext('Value')
            if axis and val:
                try:
                    scaling[axis] = float(val)
                except ValueError:
                    pass

        # T interval in seconds from experiment setup metadata
        t_incr: float | None = None
        t_incr_txt = root.findtext('.//Interval/TimeSpan/Value')
        t_unit_txt = root.findtext('.//Interval/TimeSpan/DefaultUnitFormat')
        if t_incr_txt:
            try:
                v = float(t_incr_txt)
                if v > 0.0:
                    if t_unit_txt and t_unit_txt.lower() == 'ms':
                        v /= 1000.0
                    t_incr = v
            except ValueError:
                pass

        acq_time_str = root.findtext('.//AcquisitionDateAndTime')
        sentinel = '0001-01-01T00:00:00'
        acq_time: numpy.datetime64 | None = None
        if acq_time_str and not acq_time_str.startswith(sentinel):
            try:
                # strip timezone indicator (Z or HH:MM) to avoid NumPy 2.x
                # UserWarning about timezone representation in datetime64
                acq_time_clean = re.sub(
                    r'Z$|[+-]\d{2}:\d{2}$', '', acq_time_str
                ).replace(' ', 'T')
                acq_time = numpy.datetime64(acq_time_clean, 'ns')
            except (ValueError, OverflowError):
                pass

        for dim, start_val, size_val in zip(
            self.dims, self.start, self.shape, strict=True
        ):
            if dim in scaling:
                result[dim] = (
                    numpy.arange(start_val, start_val + size_val)
                    * scaling[dim]
                )
            elif dim == 'T':
                if t_incr is not None and acq_time is not None:
                    dt_ns = round(t_incr * 1e9)
                    result[dim] = numpy.array(
                        [
                            acq_time
                            + numpy.timedelta64((start_val + i) * dt_ns, 'ns')
                            for i in range(size_val)
                        ],
                        dtype='datetime64[ns]',
                    )
                elif t_incr is not None:
                    result[dim] = (
                        numpy.arange(start_val, start_val + size_val) * t_incr
                    )
                else:
                    # try TimeStamps attachment as fallback
                    att_ts = self._parent.timestamps
                    if att_ts is not None and start_val + size_val <= len(
                        att_ts
                    ):
                        result[dim] = att_ts[start_val : start_val + size_val]
                    else:
                        result[dim] = numpy.arange(
                            start_val, start_val + size_val
                        )
            elif dim == 'C':
                channel_list = list(self.channels.items())
                ch_slice = channel_list[start_val : start_val + size_val]
                # collect (lo, hi) pairs; scalar DetectionWavelength -> (v, v)
                dw_ranges: list[tuple[float, float]] = []
                for _name, ch_data in ch_slice:
                    dw = ch_data.get('DetectionWavelength')
                    if isinstance(dw, tuple):
                        dw_ranges.append(dw)
                    elif isinstance(dw, (int, float)):
                        dw_ranges.append((float(dw), float(dw)))
                    else:
                        break
                if len(dw_ranges) == size_val:
                    # use midpoints only for non-overlapping ranges
                    # (LSM980-style narrow bins); overlapping ranges indicate
                    # LSM800 ChS cumulative integration windows where the
                    # channel name is the more meaningful coordinate
                    sorted_dw = sorted(dw_ranges)
                    non_overlapping = all(
                        sorted_dw[i][1] <= sorted_dw[i + 1][0]
                        for i in range(len(sorted_dw) - 1)
                    )
                    if non_overlapping:
                        result[dim] = numpy.array(
                            [(lo + hi) / 2.0 for lo, hi in dw_ranges]
                        )
                    else:
                        names_slice = [name for name, _ in ch_slice]
                        result[dim] = numpy.array(names_slice)
                else:
                    names_slice = [name for name, _ in ch_slice]
                    if len(names_slice) == size_val and all(names_slice):
                        # if all names parse as floats (e.g. wavelengths
                        # encoded as names), prefer numeric coordinates
                        try:
                            result[dim] = numpy.array(
                                [float(n) for n in names_slice]
                            )
                        except ValueError:
                            result[dim] = numpy.array(names_slice)
                    else:
                        result[dim] = numpy.arange(
                            start_val, start_val + size_val
                        )
            else:
                result[dim] = numpy.arange(
                    start_val,
                    start_val + size_val,
                    dtype=numpy.min_scalar_type(start_val + size_val - 1),
                )
        return MappingProxyType(result)

    @cached_property
    def channels(self) -> Mapping[str, dict[str, Any]]:
        """Per-channel metadata keyed by channel name.

        Returns a dict keyed by channel name (falling back to channel Id
        when no name is available).

        Each value is a dict whose keys are XML tag names from the
        ZISRAW ``Channel`` element (``ExcitationWavelength``,
        ``DetectionWavelength``, ``Fluor``, ``AcquisitionMode``, etc.)
        plus display-setting fields (``DyeName``, ``DyeMaxExcitation``,
        ``DyeMaxEmission``, ``ShortName``, ``Color``).

        ``DetectionWavelength`` is a ``(lo, hi)`` float tuple when the XML
        encodes a range, or a scalar float for single-value elements.

        """
        root = self._parent.xml_element
        if root is None:
            return {}

        def _float(el: ElementTree.Element, tag: str) -> float | None:
            txt = el.findtext(tag)
            if txt:
                try:
                    return float(txt)
                except ValueError:
                    pass
            return None

        image_channels: dict[str, dict[str, Any]] = {}
        for ch_el in root.findall(
            './/Information/Image/Dimensions/Channels/Channel'
        ):
            ch_id = ch_el.get('Id', '')
            channel: dict[str, Any] = {}
            ch_name = ch_el.get('Name') or ch_el.findtext('Name')
            if ch_name:
                channel['name'] = ch_name
            for xml_tag in (
                'ExcitationWavelength',
                'EmissionWavelength',
                'IlluminationWavelength',
                'ExposureTime',
                'PinholeSize',
            ):
                val = _float(ch_el, xml_tag)
                if val is not None:
                    channel[xml_tag] = val
            # DetectionWavelength is almost always a <Ranges>lo-hi</Ranges>
            # sub-element (single-band files have lo==hi; spectral bins differ)
            ranges_txt = ch_el.findtext('DetectionWavelength/Ranges')
            if ranges_txt:
                sep = ranges_txt.rfind('-')
                if sep > 0:
                    try:
                        lo = float(ranges_txt[:sep])
                        hi = float(ranges_txt[sep + 1 :])
                        channel['DetectionWavelength'] = (
                            lo if lo == hi else (lo, hi)
                        )
                    except ValueError:
                        pass
            else:
                val = _float(ch_el, 'DetectionWavelength')
                if val is not None:
                    channel['DetectionWavelength'] = val
            # Excitation wavelength fallback:
            # laser line from LightSourcesSettings
            if 'ExcitationWavelength' not in channel:
                ls_wl_txt = ch_el.findtext(
                    'LightSourcesSettings/LightSourceSettings/Wavelength'
                )
                if ls_wl_txt:
                    try:
                        channel['ExcitationWavelength'] = float(ls_wl_txt)
                    except ValueError:
                        pass
            for xml_tag in (
                'Fluor',
                'AcquisitionMode',
                'ContrastMethod',
                'Color',
            ):
                txt = ch_el.findtext(xml_tag)
                if txt:
                    channel[xml_tag] = txt
            image_channels[ch_id] = channel
        for ch_el in root.findall('.//DisplaySetting/Channels/Channel'):
            ch_id = ch_el.get('Id', '')
            ch_name = ch_el.get('Name') or ch_el.findtext('Name')
            entry = image_channels.setdefault(ch_id, {})
            if ch_name:
                entry['name'] = ch_name
            for xml_tag in (
                'DyeName',
                'DyeMaxExcitation',
                'DyeMaxEmission',
                'ShortName',
                'Color',
            ):
                txt = ch_el.findtext(xml_tag)
                if txt and xml_tag not in entry:
                    entry[xml_tag] = txt
        # re-key by channel name; fall back to Id when name is absent
        result: dict[str, dict[str, Any]] = {}
        for ch_id, ch_data in image_channels.items():
            key = ch_data.pop('name', '') or ch_id
            result[key] = ch_data
        return MappingProxyType(result)

    @cached_property
    def objective(self) -> Mapping[str, Any]:
        """Objective lens metadata keyed by XML tag name.

        Returns a dict containing a subset of the Objective XML element
        tags defined in the ZISRAW specification.

        """
        root = self._parent.xml_element
        if root is None:
            return MappingProxyType({})
        obj_el = root.find('.//Objectives/Objective')
        if obj_el is None:
            return MappingProxyType({})
        result: dict[str, Any] = {}
        name = obj_el.get('Name') or obj_el.findtext('Name')
        if name:
            result['Name'] = name
        for xml_tag in (
            'NominalMagnification',
            'CalibratedMagnification',
            'LensNA',
            'ImmersionRefractiveIndex',
            'WorkingDistance',
        ):
            txt = obj_el.findtext(xml_tag)
            if txt:
                try:
                    result[xml_tag] = float(txt)
                except ValueError:
                    pass
        for xml_tag in ('Immersion', 'PupilGeometry'):
            txt = obj_el.findtext(xml_tag)
            if txt:
                result[xml_tag] = txt
        mfr_model = obj_el.findtext('Manufacturer/Model')
        if mfr_model:
            result['Manufacturer'] = {'Model': mfr_model}
        return MappingProxyType(result)

    @cached_property
    def attrs(self) -> Mapping[str, Any]:
        """Image metadata as dict.

        Contains ``filepath`` always. When CZI metadata is available,
        also includes:

        ``datetime``
            ISO-8601 acquisition date/time string.
        ``pixel_size``
            Dict mapping axis ids (``'X'``, ``'Y'``, ``'Z'``) to physical
            pixel spacing in meters.
        ``objective``
            Objective lens metadata. See :py:attr:`objective`.
        ``channels``
            Per-channel metadata. See :py:attr:`channels`.

        """
        result: dict[str, Any] = {
            'filepath': self._parent._path,
        }
        root = self._parent.xml_element
        if root is None:
            return MappingProxyType(result)

        # acquisition datetime
        sentinel = '0001-01-01T00:00:00'
        dt_str = root.findtext('.//AcquisitionDateAndTime')
        if dt_str and not dt_str.startswith(sentinel):
            result['datetime'] = dt_str

        # pixel spacing in meters
        pixel_size: dict[str, float] = {}
        for dist in root.findall('.//Scaling/Items/Distance'):
            axis = dist.get('Id')
            val = dist.findtext('Value')
            if axis and val:
                try:
                    pixel_size[axis] = float(val)
                except ValueError:
                    pass
        if pixel_size:
            result['pixel_size'] = pixel_size

        objective = self.objective
        if objective:
            result['objective'] = objective

        if self.channels:
            result['channels'] = self.channels

        return MappingProxyType(result)

    @property
    def directory_entries(self) -> tuple[CziDirectoryEntryDV, ...]:
        """Filtered directory entries for this image view."""
        return self._entries

    @property
    def levels(self) -> list[CziImage]:
        """Pyramid levels. ``levels[0]`` is this image."""
        return self._levels

    @property
    def is_pyramid(self) -> bool:
        """Image has multiple pyramid levels."""
        return len(self._levels) > 1

    @property
    def is_pyramid_level(self) -> bool:
        """Image is downsampled pyramid overview."""
        return self._subblock_layout.is_pyramid_level

    @property
    def is_upsampled(self) -> bool:
        """Subblocks are stored sub-sampled and expanded on read (Airyscan)."""
        return self._subblock_layout.is_upsampled

    @property
    def is_downsampled(self) -> bool:
        """Subblocks are stored above logical resolution (PALM)."""
        return self._subblock_layout.is_downsampled

    @property
    def storedsize(self) -> bool:
        """Return pixel data at stored resolution without resampling."""
        return self._storedsize

    def asarray(
        self,
        *,
        fillvalue: ArrayLike | None = 0,
        maxworkers: int | None = None,
        out: OutputType = None,
    ) -> NDArray[Any]:
        """Return image data from file as numpy array.

        Parameters:
            fillvalue:
                Value for pixels not covered by any subblock.
            maxworkers:
                Maximum number of threads to decode subblock data.
                By default, up to half the CPU cores are used.
            out:
                Output destination for image data.
                If ``None``, create a new NumPy array in main memory.
                If ``'memmap'``, create a memory-mapped array in a
                temporary file.
                If a ``numpy.ndarray``, a writable, initialized array
                of compatible shape and dtype.
                If a ``file name`` or ``open file``, create a
                memory-mapped array in the specified file.

        """
        dest_pixeltype = self.pixeltype
        raw_sizes = self._raw_sizes

        # determine if dimension reordering is needed
        needs_reorder = False
        inv_perm: tuple[int, ...] = ()
        if self._selection:
            file_dims = list(raw_sizes.keys())
            desired_dims = list(self._dim_order)
            if file_dims != desired_dims:
                needs_reorder = True
                inv_perm = tuple(desired_dims.index(d) for d in file_dims)

        if needs_reorder:
            desired_shape = tuple(raw_sizes[d] for d in self._dim_order)
            out = create_output(
                out, desired_shape, self.dtype, fillvalue=fillvalue
            )
            # file-order view for tile compositing
            compositing_out = out.transpose(inv_perm)
        else:
            raw_shape = tuple(raw_sizes.values())
            out = create_output(
                out, raw_shape, self.dtype, fillvalue=fillvalue
            )
            compositing_out = out

        c = self.compression

        # warn once if SYSTEMRAW: data is read as raw bytes, not decoded pixels
        if int(c) > 999:
            logger().warning(f'{c!r} reading raw bytes as uncompressed pixels')

        roi = self._roi
        layout = self._subblock_layout
        x_idx = layout.x_idx
        y_idx = layout.y_idx
        is_upsampled = layout.is_upsampled
        is_pyramid_level = layout.is_pyramid_level
        is_downsampled = layout.is_downsampled
        dims = self._entries[0].dims if self._entries else ()
        sf = self._scale_factors
        ss_sf = self._storedsize_scale
        use_storedsize = self._storedsize and bool(ss_sf)

        # precompute pyramid-scaled ROI to avoid redundant per-tile work
        if roi is not None and is_pyramid_level:
            rx, ry, rw, rh = roi
            pyr_ry = round(ry / sf[y_idx]) if y_idx >= 0 else 0
            pyr_rh = round(rh / sf[y_idx]) if y_idx >= 0 else 0
            pyr_rx = round(rx / sf[x_idx]) if x_idx >= 0 else 0
            pyr_rw = round(rw / sf[x_idx]) if x_idx >= 0 else 0
        else:
            pyr_rx = pyr_ry = pyr_rw = pyr_rh = 0

        # precompute storedsize-scaled ROI
        if roi is not None and use_storedsize:
            rx, ry, rw, rh = roi
            ss_ry = round(ry * ss_sf[y_idx]) if y_idx >= 0 else 0
            ss_rh = round(rh * ss_sf[y_idx]) if y_idx >= 0 else 0
            ss_rx = round(rx * ss_sf[x_idx]) if x_idx >= 0 else 0
            ss_rw = round(rw * ss_sf[x_idx]) if x_idx >= 0 else 0
        else:
            ss_rx = ss_ry = ss_rw = ss_rh = 0

        # precompute inverse ss_sf for start coordinate mapping
        inv_ss_sf = tuple(1.0 / f for f in ss_sf) if use_storedsize else ()

        def func(
            directory_entry: CziDirectoryEntryDV,
            start: tuple[int, ...] = self._start,
            out: Any = compositing_out,
        ) -> None:
            subblock = directory_entry.read_segment_data(self._parent)
            if not isinstance(subblock, CziSubBlockSegmentData):
                return

            tile = subblock.data()
            if directory_entry.pixel_type != dest_pixeltype:
                converter = CONVERT_PIXELTYPE.get(
                    (directory_entry.pixel_type, dest_pixeltype)
                )
                if converter is not None:
                    tile = converter(tile)

            entry_sf: tuple[float, ...] = ()
            if not use_storedsize and (
                is_upsampled
                or is_downsampled
                or directory_entry.stored_shape != directory_entry.shape
            ):
                esf = tuple(
                    float(s) / float(ss) if ss > 0 else 1.0
                    for s, ss in zip(
                        directory_entry.shape,
                        directory_entry.stored_shape,
                        strict=True,
                    )
                )
                if is_upsampled:
                    entry_sf = esf
                    for d, f in enumerate(esf):
                        if f > 1.0:
                            tile = numpy.repeat(tile, round(f), axis=d)
                else:
                    for d, f in enumerate(esf):
                        if f < 1.0:
                            n_in = tile.shape[d]
                            n_out = round(n_in * f)
                            idx = numpy.floor(
                                numpy.arange(n_out) * n_in / n_out
                            ).astype(numpy.intp)
                            tile = numpy.take(tile, idx, axis=d)

            entry_start: tuple[int, ...]
            if directory_entry.dims != dims:
                d_start = dict(
                    zip(
                        directory_entry.dims,
                        directory_entry.start,
                        strict=True,
                    )
                )
                entry_start = tuple(d_start.get(d, 0) for d in dims)
                d_tile = dict(
                    zip(directory_entry.dims, tile.shape, strict=True)
                )
                tile = tile.reshape(tuple(d_tile.get(d, 1) for d in dims))
                if entry_sf:
                    d_sf = dict(
                        zip(directory_entry.dims, entry_sf, strict=True)
                    )
                    entry_sf = tuple(d_sf.get(d, 1.0) for d in dims)
            else:
                entry_start = directory_entry.start

            if roi is not None:
                if is_pyramid_level:
                    # use precomputed pyramid-scaled ROI
                    rx, ry, rw, rh = pyr_rx, pyr_ry, pyr_rw, pyr_rh
                    entry_start = tuple(
                        round(s / f)
                        for s, f in zip(entry_start, sf, strict=True)
                    )
                elif use_storedsize:
                    rx, ry, rw, rh = ss_rx, ss_ry, ss_rw, ss_rh
                    entry_start = tuple(
                        round(s / f)
                        for s, f in zip(entry_start, inv_ss_sf, strict=True)
                    )
                else:
                    rx, ry, rw, rh = roi
                # compute tile-to-output mapping with ROI clipping
                index_list = []
                tile_slices = []
                for d in range(len(start)):
                    t_start = entry_start[d]
                    t_size = tile.shape[d]
                    if d == y_idx:
                        # clip to ROI Y range
                        src_lo = max(t_start, ry)
                        src_hi = min(t_start + t_size, ry + rh)
                        if src_lo >= src_hi:
                            return  # tile does not overlap ROI
                        tile_slices.append(
                            slice(src_lo - t_start, src_hi - t_start)
                        )
                        index_list.append(slice(src_lo - ry, src_hi - ry))
                    elif d == x_idx:
                        # clip to ROI X range
                        src_lo = max(t_start, rx)
                        src_hi = min(t_start + t_size, rx + rw)
                        if src_lo >= src_hi:
                            return  # tile does not overlap ROI
                        tile_slices.append(
                            slice(src_lo - t_start, src_hi - t_start)
                        )
                        index_list.append(slice(src_lo - rx, src_hi - rx))
                    else:
                        tile_slices.append(slice(None))
                        index_list.append(
                            slice(
                                t_start - start[d],
                                t_start - start[d] + t_size,
                            )
                        )
                index = tuple(index_list)
                tile = tile[tuple(tile_slices)]
            elif is_pyramid_level or use_storedsize:
                csf = sf if is_pyramid_level else inv_ss_sf
                index = tuple(
                    slice(
                        round(i / f) - j,
                        round(i / f) - j + k,
                    )
                    for i, j, k, f in zip(
                        entry_start,
                        start,
                        tile.shape,
                        csf,
                        strict=True,
                    )
                )
            else:
                index = tuple(
                    slice(i - j, i - j + k)
                    for i, j, k in zip(
                        entry_start,
                        start,
                        tile.shape,
                        strict=True,
                    )
                )

            # Skip mask() method call when attachment_size is too
            # small to contain a mask (common case). This avoids the
            # overhead of attachments() parsing on every tile.
            mask = subblock.mask() if subblock.attachment_size >= 32 else None
            if mask is not None:
                if is_upsampled and entry_sf:
                    # mask was decoded at stored size; repeat to logical size
                    if y_idx >= 0 and entry_sf[y_idx] > 1.0:
                        mask = numpy.repeat(
                            mask, round(entry_sf[y_idx]), axis=0
                        )
                    if x_idx >= 0 and entry_sf[x_idx] > 1.0:
                        mask = numpy.repeat(
                            mask, round(entry_sf[x_idx]), axis=1
                        )
                if roi is not None:
                    mask = mask[
                        tile_slices[y_idx] if y_idx >= 0 else slice(None),
                        tile_slices[x_idx] if x_idx >= 0 else slice(None),
                    ]
                # If the decoded tile is smaller than the mask (CZI writer bug
                # where stored_shape > actual decoded size), clip the mask to
                # match the tile dimensions before broadcasting.
                if y_idx >= 0 and mask.shape[0] > tile.shape[y_idx]:
                    mask = mask[: tile.shape[y_idx], :]
                if x_idx >= 0 and mask.shape[1] > tile.shape[x_idx]:
                    mask = mask[:, : tile.shape[x_idx]]
                # Expand (Y, X) mask to the full tile shape by inserting
                # singleton axes for all non-Y, non-X dims (e.g. C, S).
                # This ensures numpy broadcasting works for any tile rank.
                expand_axes = tuple(
                    i
                    for i in range(tile.ndim)
                    if (y_idx < 0 or i != y_idx) and (x_idx < 0 or i != x_idx)
                )
                mask_nd = numpy.expand_dims(mask, axis=expand_axes)
                dest = out[index]
                numpy.copyto(dest, tile, where=mask_nd)
            else:
                out[index] = tile

        if maxworkers is None:
            maxworkers = self._default_maxworkers
        if maxworkers > 1:
            # The file-handle lock (set_lock) serialises I/O only.
            # Concurrent numpy writes to `out` are safe as long as tiles do
            # not share output pixels, which holds for non-overlapping tiles.
            # For mosaic images the directory entries are pre-sorted by
            # mosaic_index (ascending) so that tiles with a higher M-index
            # are composited on top. Because ThreadPoolExecutor does not
            # guarantee execution order, the final value of any pixel covered
            # by more than one tile is non-deterministic under multi-threading.
            # This is acceptable in practice because overlapping mosaic
            # tiles are rare and the overlap region is typically a thin seam.
            # maxworkers=1 forces sequential, fully deterministic
            # compositing when exact overlap behaviour is required.
            self._parent.set_lock(True)
            try:
                with ThreadPoolExecutor(maxworkers) as executor:
                    for _ in executor.map(func, self._entries):
                        pass
            finally:
                self._parent.set_lock(False)
        else:
            for directory_entry in self._entries:
                func(directory_entry)

        if hasattr(out, 'flush'):
            out.flush()

        if self._squeeze:
            out = out.squeeze()

        return out

    def asxarray(self, **kwargs: Any) -> DataArray:
        """Return image data as xarray DataArray.

        Parameters:
            **kwargs:
                Keyword arguments forwarded to :py:meth:`asarray`:
                ``fillvalue``, ``maxworkers``, and ``out``.

        """
        from xarray import DataArray

        return DataArray(
            self.asarray(**kwargs),
            coords=self.coords,
            dims=self.dims,
            name=self._name,
            attrs=self.attrs,
        )

    def chunks(
        self,
        *,
        squeeze: bool = True,
        maxcache: int | None = None,
        **sizes: int | None,
    ) -> CziImageChunks:
        """Return image chunks as CziImage views.

        Each chunk is a :py:class:`CziImage` with the full API including
        :py:meth:`asarray`, :py:meth:`asxarray`, :py:attr:`bbox`,
        :py:attr:`dims`, :py:attr:`sizes`, and :py:meth:`__call__`.

        Parameters:
            squeeze:
                Whether to drop size-1 dimensions from each chunk.
                When ``False``, size-1 axes such as the single T or Z index
                per chunk are retained in :py:attr:`~CziImage.dims` and
                :py:attr:`~CziImage.shape`.
            maxcache:
                Number of decoded subblock arrays to cache during iteration.
                Only active when :py:attr:`CziFile.maxcache` is ``None``
                (auto mode, the default).
                ``None`` (default) auto-detects: computes how many
                subblocks can overlap one grid tile
                (``(floor((tile_x-2)/sub_w)+2) * (floor((tile_y-2)/sub_h)+2)
                * layers``), where ``layers`` is the product of any
                non-spatial dimensions specified in ``sizes``, using
                their batch size or full axis size for ``None``;
                capped at the subblock count and 64;
                0 when no spatial tiling is requested.
                Pass 0 to disable caching, or an explicit positive integer
                to set the temporary FIFO cache size for this iteration.
            **sizes:
                Chunk size specification as keyword arguments.
                Keys are dimension names, values control behavior:

                - *Absent* non-spatial dimension: iterated one-at-a-time.
                - *Absent* spatial dimension (Y, X): kept in full.
                - ``None`` value: full axis kept in each chunk.
                - ``int`` value: axis batched into groups of that size.
                  For spatial dimensions, this produces a tiling grid.

        Notes:
            With no size arguments, non-spatial dimensions of length > 1 are
            iterated one-at-a-time (equivalent to plane-by-plane iteration).
            Dimensions of length 1 are skipped when the parent image uses
            squeeze (the default).
            ``squeeze`` controls only how each chunk presents its dimensions:
            size-1 axes are dropped when ``True`` (default) and retained
            when ``False``.

        Example::

            # iterate individual Y/X planes
            for chunk in img.chunks():
                process(chunk.asarray())

            # keep C in each chunk, iterate T and Z
            for chunk in img.chunks(C=None):
                assert 'C' in chunk.dims

            # batch Z into groups of 10
            for chunk in img.chunks(Z=10):
                arr = chunk.asarray()

            # spatial tiling: 512x512 tiles
            for chunk in img.chunks(Y=512, X=512):
                tile = chunk.asarray()

        For bulk reads, use :py:meth:`CziImage.asarray` directly, which is
        faster as it composites all tiles in a single parallel pass.

        """
        return CziImageChunks(self, sizes, squeeze=squeeze, maxcache=maxcache)

    @cached_property
    def _storedsize_scale(self) -> tuple[float, ...]:
        """Inverse scale factors for storedsize coordinate mapping.

        Each element is ``stored_shape / shape`` for the corresponding dim.
        Values > 1.0 for PALM, < 1.0 for Airyscan, 1.0 for normal.
        Empty tuple when storedsize is False or no entries.

        Raises:
            ValueError: Entries have non-uniform stored-to-logical ratios.

        """
        if not self._storedsize or not self._entries:
            return ()
        layout = self._subblock_layout
        if not layout.is_upsampled and not layout.is_downsampled:
            return ()
        # collect per-entry ratios; check uniformity
        ref_ratio: tuple[float, ...] | None = None
        for e in self._entries:
            ratio = tuple(
                float(ss) / float(s) if s > 0 else 1.0
                for s, ss in zip(e.shape, e.stored_shape, strict=True)
            )
            if ref_ratio is None:
                ref_ratio = ratio
            elif ratio != ref_ratio:
                msg = (
                    'storedsize=True requires uniform stored-to-logical'
                    ' ratios across all subblocks; apply a dimension'
                    ' selection first'
                )
                raise ValueError(msg)
        return ref_ratio if ref_ratio is not None else ()

    @cached_property
    def _scale_factors(self) -> tuple[float, ...]:
        """Per-dimension scale factors (> 1.0 for downsampled pyramid dims)."""
        if not self._entries:
            return ()
        e = self._entries[0]
        raw = tuple(
            float(s) / float(ss) if ss > 0 else 1.0
            for s, ss in zip(e.shape, e.stored_shape, strict=True)
        )
        return normalize_pyramid_scale(raw)

    @cached_property
    def _raw_sizes(self) -> dict[str, int]:
        """Ordered mapping of dimension name to length, before squeeze."""
        if not self._entries:
            msg = 'image has no subblocks'
            raise ValueError(msg)
        dims = self._entries[0].dims
        sf = self._scale_factors
        is_pyr_level = (
            bool(sf)
            and any(f != 1.0 for f in sf)
            and self._entries[0].pyramid_type != CziPyramidType.NONE
        )
        ss_sf = self._storedsize_scale
        if is_pyr_level or ss_sf:
            # both pyramid and storedsize use stored-space coordinates
            # for pyramid: sf = shape/stored_shape (>1.0)
            # for storedsize: ss_sf = stored_shape/shape, so 1/ss_sf
            #   maps start to stored space the same way
            csf = sf if is_pyr_level else tuple(1.0 / f for f in ss_sf)
            stored_starts = [
                tuple(round(s / f) for s, f in zip(e.start, csf, strict=True))
                for e in self._entries
            ]
            stored_ends = [
                tuple(
                    round(s / f) + ss
                    for s, f, ss in zip(
                        e.start, csf, e.stored_shape, strict=True
                    )
                )
                for e in self._entries
            ]
            global_start = tuple(
                int(x) for x in numpy.min(stored_starts, axis=0)
            )
            global_end = tuple(int(x) for x in numpy.max(stored_ends, axis=0))
            shape = tuple(
                e - s for e, s in zip(global_end, global_start, strict=True)
            )
        else:
            shape = filtered_shape(self._entries, self._start)
        sizes = dict(zip(dims, shape, strict=True))
        if self._roi is not None:
            _rx, _ry, rw, rh = self._roi
            if 'X' in sizes:
                x_i = dims.index('X')
                if is_pyr_level:
                    sizes['X'] = round(rw / sf[x_i])
                elif ss_sf:
                    sizes['X'] = round(rw * ss_sf[x_i])
                else:
                    sizes['X'] = rw
            if 'Y' in sizes:
                y_i = dims.index('Y')
                if is_pyr_level:
                    sizes['Y'] = round(rh / sf[y_i])
                elif ss_sf:
                    sizes['Y'] = round(rh * ss_sf[y_i])
                else:
                    sizes['Y'] = rh
        if 'S' in sizes:
            sizes['S'] = self.pixeltype.samples
        return sizes

    @cached_property
    def _dim_order(self) -> tuple[str, ...]:
        """Dimension names in output order.

        If no selection is applied, file order is used.
        Otherwise: unspecified non-spatial dims in file order,
        then specified dims in kwargs order, then spatial dims.

        """
        if not self._selection:
            return tuple(self._raw_sizes.keys())
        raw = self._raw_sizes
        plane_dims = {'Y', 'X', 'S'}
        specified = list(self._selection.keys())
        specified_set = set(specified)
        unspecified = [
            d for d in raw if d not in plane_dims and d not in specified_set
        ]
        spatial = [d for d in raw if d in plane_dims]
        return tuple(unspecified + specified + spatial)

    @cached_property
    def _start(self) -> tuple[int, ...]:
        """Minimum start indices per dimension (internal, before squeeze)."""
        sf = self._scale_factors
        is_pyr_level = (
            bool(sf)
            and any(f != 1.0 for f in sf)
            and self._entries[0].pyramid_type != CziPyramidType.NONE
        )
        ss_sf = self._storedsize_scale
        if is_pyr_level or ss_sf:
            csf = sf if is_pyr_level else tuple(1.0 / f for f in ss_sf)
            starts_stored = [
                tuple(round(s / f) for s, f in zip(e.start, csf, strict=True))
                for e in self._entries
            ]
            start = tuple(int(x) for x in numpy.min(starts_stored, axis=0))
        else:
            start = filtered_start(self._entries)
        if self._roi is not None:
            rx, ry, _rw, _rh = self._roi
            dims = self._entries[0].dims
            start_list = list(start)
            for i, d in enumerate(dims):
                if d == 'X':
                    if is_pyr_level:
                        start_list[i] = round(rx / sf[i])
                    elif ss_sf:
                        start_list[i] = round(rx * ss_sf[i])
                    else:
                        start_list[i] = rx
                elif d == 'Y':
                    if is_pyr_level:
                        start_list[i] = round(ry / sf[i])
                    elif ss_sf:
                        start_list[i] = round(ry * ss_sf[i])
                    else:
                        start_list[i] = ry
            start = tuple(start_list)
        return start

    @cached_property
    def _subblock_layout(self) -> CziSubBlockLayout:
        """Layout flags for subblock-to-image mapping."""
        if not self._entries:
            return CziSubBlockLayout(
                is_upsampled=False,
                is_pyramid_level=False,
                is_downsampled=False,
                x_idx=-1,
                y_idx=-1,
            )
        sf = self._scale_factors
        ptype = self._entries[0].pyramid_type
        dims = self._entries[0].dims
        # True pyramid overview: pyramid_type is uniform across all tiles,
        # so entry 0 is sufficient
        is_pyr = (
            bool(sf)
            and any(f != 1.0 for f in sf)
            and ptype != CziPyramidType.NONE
        )
        # Airyscan / PALM detection: use entry 0's scale factors as a free
        # fast path (already computed). Only scan entries[1:] when entry 0
        # is a normal tile (all sf == 1.0), which happens in mixed-tile files
        # like Palm_mitDrift.czi where entry 0 is full-resolution but later
        # entries are super-resolution. Short-circuit on the first hit.
        is_upsampled = False
        is_downsampled = False
        if not is_pyr:
            is_upsampled = any(f > 1.0 for f in sf)
            is_downsampled = any(f < 1.0 for f in sf)
            if not is_upsampled and not is_downsampled:
                for e in self._entries[1:]:
                    if e.pyramid_type != CziPyramidType.NONE:
                        continue
                    for s, ss in zip(e.shape, e.stored_shape, strict=True):
                        if ss > 0 and s != ss:
                            if s > ss:
                                is_upsampled = True
                            else:
                                is_downsampled = True
                    if is_upsampled or is_downsampled:
                        break
        return CziSubBlockLayout(
            is_upsampled=is_upsampled,
            is_pyramid_level=is_pyr,
            is_downsampled=is_downsampled,
            x_idx=dims.index('X') if 'X' in dims else -1,
            y_idx=dims.index('Y') if 'Y' in dims else -1,
        )

    @cached_property
    def _spatial_index(
        self,
    ) -> tuple[
        tuple[int, ...],  # sorted entry start-x values
        tuple[CziDirectoryEntryDV, ...],  # entries in sorted order
    ]:
        """Cached spatial index over _entries sorted by X start.

        Used by _spatial_index_filter for fast ROI lookup.
        Dimension indices x_idx/y_idx are read from _subblock_layout.

        """
        x_idx = self._subblock_layout.x_idx
        if x_idx < 0:
            return (), self._entries
        sorted_entries = sorted(self._entries, key=lambda e: e.start[x_idx])
        starts_x = tuple(e.start[x_idx] for e in sorted_entries)
        return starts_x, tuple(sorted_entries)

    def _spatial_index_filter(
        self,
        rx: int,
        ry: int,
        rw: int,
        rh: int,
    ) -> list[CziDirectoryEntryDV]:
        """Return entries overlapping the given ROI using the spatial index.

        Parameters:
            rx: ROI left edge (absolute X coordinate).
            ry: ROI top edge (absolute Y coordinate).
            rw: ROI width in pixels.
            rh: ROI height in pixels.

        """
        x_idx = self._subblock_layout.x_idx
        y_idx = self._subblock_layout.y_idx
        starts_x, sorted_entries = self._spatial_index
        rx_end = rx + rw
        # Find the range of sorted entries whose stop_x > rx
        # (i.e., not completely to the left of the ROI).
        # starts_x is sorted, so entries with start_x >= rx_end are completely
        # to the right and can be skipped. stop_x is computed on the fly
        # (avoid caching N extra int objects).
        right_bound = bisect.bisect_left(starts_x, rx_end)
        result: list[CziDirectoryEntryDV] = []
        for i in range(right_bound):
            e = sorted_entries[i]
            if e.start[x_idx] + e.shape[x_idx] <= rx:
                continue  # completely to the left
            if y_idx >= 0:
                ey = e.start[y_idx]
                eh = e.shape[y_idx]
                if ey + eh <= ry or ey >= ry + rh:
                    continue
            result.append(e)
        return result

    @cached_property
    def _default_maxworkers(self) -> int:
        """Default maxworkers for parallel tile compositing.

        Returns ``1`` for:

        - uncompressed images - no CPU-bound decompression, so threading
          overhead exceeds any benefit;
        - mosaic images (first entry carries a mosaic index) - tiles may
          overlap at seams and higher-M-index tiles must be composited on
          top, so sequential compositing order must be preserved;
        - images with fewer than 3 entries - thread-pool overhead exceeds
          any parallelism benefit for 1 or 2 tasks.

        Otherwise returns :py:func:`default_maxworkers()`
        (typically half the available CPU cores).

        """
        if (
            self.compression == CziCompressionType.UNCOMPRESSED
            or int(self.compression) > 999  # SYSTEMRAW
            or len(self._entries) < 3
            or self._entries[0].mosaic_index >= 0
        ):
            return 1
        return default_maxworkers()

    def __repr__(self) -> str:
        if len(self._levels) > 1:
            levels = f' {len(self._levels)} levels'
        else:
            levels = ''
        ss = ' storedsize' if self._storedsize else ''
        dims = ', '.join(f'{k}: {v}' for k, v in self.sizes.items())
        return (
            f'<{self.__class__.__name__} '
            f'{self._name!r} '
            f'({dims}) '
            # f'{self.dtype} '
            f'{self.pixeltype.name.lower()} '
            f'{self.compression.name.lower()}'
            f'{levels}'
            f'{ss}>'
        )


@final
class CziImageChunks:
    """Lazy container of image chunks from a CziImage.

    Each chunk is a :py:class:`CziImage` view covering a subset of the
    parent image's dimensions. Keyword arguments to
    :py:meth:`CziImage.chunks` control which dimensions are iterated
    and how they are batched.

    With no keyword arguments, non-spatial dimensions of length > 1 are
    iterated one-at-a-time, producing individual Y/X planes.
    Dimensions of length 1 are skipped when the parent image uses squeeze
    (the default).
    ``squeeze`` controls only how each chunk presents its dimensions:
    size-1 axes are dropped when ``True`` (default) and retained
    when ``False``.
    Spatial dimensions (Y, X) are always kept in full unless
    explicitly tiled with an ``int`` value.

    Parameters:
        image:
            Source CziImage.
        sizes:
            Chunk size specification as mapping of dimension name to
            ``int`` or ``None``.
            Values control behavior:

            - *Absent* non-spatial dimension: iterated one-at-a-time.
            - *Absent* spatial dimension (Y, X): kept in full.
            - ``None`` value: full axis kept in each chunk.
            - ``int`` value: axis batched into groups of that size.
              For spatial dimensions, this produces a tiling grid.
        squeeze:
            Whether to drop size-1 dimensions from each chunk.
        maxcache:
            Maximum number of decoded subblock arrays to cache during
            iteration.
            ``None`` (default) auto-detects: computes how many subblocks
            can overlap one grid tile
            (``(floor((tile_x-2)/sub_w)+2) * (floor((tile_y-2)/sub_h)+2)
            * layers``), where ``layers`` is the product of any
            non-spatial dimensions specified in ``sizes``, using their
            batch size or full axis size for ``None``;
            capped at the subblock count and 64; 0 when no spatial tiling.
            ``0`` disables caching.
            Positive integer sets an explicit limit.
            Only active when :py:attr:`CziFile.maxcache` is ``None``
            (auto mode, the default).
            The cache is set on the parent :py:class:`CziFile` for the
            duration of iteration and restored on exit.

    """

    _image: CziImage
    _sizes: dict[str, int | None]
    _squeeze: bool
    _maxcache: int  # 0 = disabled, >0 = cache size

    def __init__(
        self,
        image: CziImage,
        sizes: dict[str, int | None] | None = None,
        /,
        *,
        squeeze: bool = True,
        maxcache: int | None = None,
    ) -> None:
        self._image = image
        self._squeeze = squeeze
        if sizes is None:
            sizes = {}
        raw = image._raw_sizes
        for dim, val in sizes.items():
            if dim == 'S':
                msg = f'chunk size for sample dimension {dim!r} not supported'
                raise ValueError(msg)
            if dim not in raw:
                msg = f'unknown dimension {dim!r}'
                raise ValueError(msg)
            if val is not None and not isinstance(val, int):
                msg = (  # type: ignore[unreachable]
                    f'chunk size for {dim!r} must be int or None,'
                    f' got {type(val).__name__}'
                )
                raise TypeError(msg)
            if isinstance(val, int) and val < 1:
                msg = f'chunk size for {dim!r} must be positive, got {val}'
                raise ValueError(msg)
        self._sizes = dict(sizes)
        # resolve maxcache: auto computes how many subblocks can overlap one
        # grid tile. a tile of width w can straddle at most
        # floor((w-2)/sub_w)+2 subblock columns (and same for rows).
        # multiply by layers: when non-spatial dims (C, Z, ...) are batched
        # or kept whole in the chunk, each spatial position contributes
        # multiple entries and the cache must hold all of them to avoid
        # re-decoding boundary subblocks for consecutive tiles
        if maxcache is None:
            tile_x = sizes.get('X')
            tile_y = sizes.get('Y')
            if tile_x is not None or tile_y is not None:
                layout = image._subblock_layout
                xi, yi = layout.x_idx, layout.y_idx
                entries = image._entries
                if entries:
                    e0 = entries[0]
                    sub_w = e0.shape[xi] if xi >= 0 else 1
                    sub_h = e0.shape[yi] if yi >= 0 else 1
                    nx = (tile_x - 2) // sub_w + 2 if tile_x else 1
                    ny = (tile_y - 2) // sub_h + 2 if tile_y else 1
                    spatial_set = {'Y', 'X', 'S'}
                    layers = 1
                    for dim, sz in sizes.items():
                        if dim in spatial_set:
                            continue
                        dim_size = raw.get(dim, 1)
                        layers *= (
                            dim_size if sz is None else min(int(sz), dim_size)
                        )
                    self._maxcache = min(nx * ny * layers, len(entries), 64)
                else:
                    self._maxcache = 0
            else:
                self._maxcache = 0
        elif maxcache < 0:
            msg = f'maxcache must be >= 0, got {maxcache}'
            raise ValueError(msg)
        else:
            self._maxcache = maxcache

    def __len__(self) -> int:
        """Number of chunks."""
        _, combos, grid = self._chunk_axes
        n = len(combos)
        if grid is not None:
            n *= len(grid)
        return n

    def __iter__(self) -> Iterator[CziImage]:
        """Iterate over chunks as CziImage views."""
        iter_dims, combos, grid = self._chunk_axes
        image = self._image
        sizes = self._sizes
        parent = image._parent
        squeeze = self._squeeze
        pixeltype = image._pixeltype
        storedsize = image._storedsize
        parent_selection = image._selection
        entries_all = image._entries

        layout = image._subblock_layout
        xi = layout.x_idx
        yi = layout.y_idx

        # precompute selection key ordering once (image._dim_order is fixed)
        parent_rank = {d: i for i, d in enumerate(image._dim_order)}

        # Fast path for integer-only combos: build a dict that maps each
        # combination of iterated-dimension coordinates to the matching
        # entries, replacing the sequential O(N) linear scans per combo
        # with an O(1) dict lookup.
        entries_by_combo: (
            dict[tuple[int | slice, ...], tuple[CziDirectoryEntryDV, ...]]
            | None
        ) = None
        if iter_dims and not any(
            isinstance(v, slice) for combo in combos for v in combo
        ):
            _build: dict[tuple[int, ...], list[CziDirectoryEntryDV]] = {}
            try:
                for e in entries_all:
                    key = tuple(e.start[e.dims.index(d)] for d in iter_dims)
                    _build.setdefault(key, []).append(e)
                entries_by_combo = {k: tuple(v) for k, v in _build.items()}
            except (ValueError, IndexError):
                pass  # entry missing an iterated dim; fall back to scan

        no_entries: tuple[CziDirectoryEntryDV, ...] = ()

        with parent._cache.scoped_resize(
            self._maxcache if parent._auto_cache else 0
        ):
            for combo in combos:
                # filter entries by iterated dimension coordinates
                selection: dict[str, SelectionValue] = dict(parent_selection)
                entries: tuple[CziDirectoryEntryDV, ...] = entries_all

                if entries_by_combo is not None:
                    # O(1) lookup replacing sequential O(N) scans
                    entries = entries_by_combo.get(combo, no_entries)
                    selection.update(zip(iter_dims, combo, strict=True))
                else:
                    for dim, val in zip(iter_dims, combo, strict=True):
                        if isinstance(val, int):
                            entries = tuple(
                                e
                                for e in entries
                                if dim in e.dims
                                and e.start[e.dims.index(dim)] == val
                            )
                            selection[dim] = val
                        elif isinstance(val, slice):
                            r = range(val.start, val.stop)
                            entries = tuple(
                                e
                                for e in entries
                                if dim in e.dims
                                and e.start[e.dims.index(dim)] in r
                            )
                            selection[dim] = val

                # add None for dims kept in chunk
                for dim, sz in sizes.items():
                    if sz is None and dim not in ('Y', 'X', 'S'):
                        selection[dim] = None

                if not entries:
                    continue

                # reorder selection keys to match parent dim order so that
                # the chunk's _dim_order follows the parent's dimension order
                selection = dict(
                    sorted(
                        selection.items(),
                        key=lambda kv: parent_rank.get(
                            kv[0], len(parent_rank)
                        ),
                    )
                )

                # build name parts from iteration coordinates
                parts: list[str] = []
                if image._name:
                    parts.append(image._name)
                sel_parts = [
                    f'{d}={v!r}' for d, v in zip(iter_dims, combo, strict=True)
                ]
                if sel_parts:
                    parts.append(', '.join(sel_parts))

                if grid is not None:
                    # Use the image's cached spatial index when entries are
                    # unfiltered (i.e. no non-spatial dim selection was
                    # applied), avoiding an O(N) linear scan per grid tile.
                    use_spi = (entries is entries_all) and xi >= 0
                    for roi in grid:
                        rx, ry, rw, rh = roi
                        if use_spi:
                            roi_entries: tuple[CziDirectoryEntryDV, ...] = (
                                tuple(
                                    image._spatial_index_filter(rx, ry, rw, rh)
                                )
                            )
                        else:
                            roi_entries = tuple(
                                e
                                for e in entries
                                if (
                                    xi < 0
                                    or not (
                                        e.start[xi] + e.shape[xi] <= rx
                                        or e.start[xi] >= rx + rw
                                    )
                                )
                                and (
                                    yi < 0
                                    or not (
                                        e.start[yi] + e.shape[yi] <= ry
                                        or e.start[yi] >= ry + rh
                                    )
                                )
                            )
                        if not roi_entries:
                            continue
                        roi_parts = list(parts)
                        roi_parts.append(f'roi={roi!r}')
                        name = ' '.join(roi_parts)
                        yield CziImage(
                            parent,
                            roi_entries,
                            name=name,
                            squeeze=squeeze,
                            roi=roi,
                            pixeltype=pixeltype,
                            storedsize=storedsize,
                            selection=selection or None,
                        )
                else:
                    name = ' '.join(parts) if parts else ''
                    yield CziImage(
                        parent,
                        entries,
                        name=name,
                        squeeze=squeeze,
                        roi=image._roi,
                        pixeltype=pixeltype,
                        storedsize=storedsize,
                        selection=selection or None,
                    )

    @cached_property
    def _chunk_axes(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[int | slice, ...], ...],
        tuple[tuple[int, int, int, int], ...] | None,
    ]:
        """Return (iter_dims, coord_combos, spatial_grid).

        iter_dims:
            Non-spatial dimension names that are iterated.
        coord_combos:
            Cartesian product of coordinate values or slices for each
            iterated dimension.
        spatial_grid:
            Tuple of ``(x, y, w, h)`` ROI tuples for spatial tiling
            in absolute CZI pixel coordinates, or ``None`` when no
            spatial tiling is requested.

        """
        raw = self._image._raw_sizes
        start_tuple = self._image._start
        dim_list = list(raw.keys())
        dim_order = self._image._dim_order
        parent_squeeze = self._image._squeeze
        sizes = self._sizes
        spatial_dims = {'Y', 'X', 'S'}

        iter_dims: list[str] = []
        iter_values: list[list[Any]] = []

        for dim in dim_order:
            if dim in spatial_dims:
                continue
            dim_idx = dim_list.index(dim)
            dim_start = start_tuple[dim_idx]
            dim_size = raw[dim]

            if dim in sizes:
                if sizes[dim] is None:
                    continue
                # batch by sizes[dim]
                batch: int = sizes[dim]  # type: ignore[assignment]
                vals: list[int | slice] = []
                for b in range(dim_start, dim_start + dim_size, batch):
                    b_end = min(b + batch, dim_start + dim_size)
                    vals.append(slice(b, b_end))
                iter_dims.append(dim)
                iter_values.append(vals)
            else:
                # iterate one-at-a-time
                coord_vals = list(range(dim_start, dim_start + dim_size))
                if parent_squeeze and len(coord_vals) <= 1:
                    continue
                iter_dims.append(dim)
                iter_values.append(coord_vals)

        if iter_dims:
            coord_combos = tuple(itertools.product(*iter_values))
        else:
            coord_combos = ((),)

        # spatial tiling grid
        tile_x = sizes.get('X')
        tile_y = sizes.get('Y')
        spatial_grid: tuple[tuple[int, int, int, int], ...] | None = None

        if tile_x is not None or tile_y is not None:
            if self._image._roi is not None:
                ext_x, ext_y, ext_w, ext_h = self._image._roi
            else:
                abs_start = filtered_start(self._image._entries)
                abs_shape = filtered_shape(self._image._entries, abs_start)
                xi = dim_list.index('X') if 'X' in dim_list else -1
                yi = dim_list.index('Y') if 'Y' in dim_list else -1
                ext_x = abs_start[xi] if xi >= 0 else 0
                ext_y = abs_start[yi] if yi >= 0 else 0
                ext_w = abs_shape[xi] if xi >= 0 else 0
                ext_h = abs_shape[yi] if yi >= 0 else 0

            # step sizes in absolute CZI coordinates
            step_x = tile_x if tile_x is not None else ext_w
            step_y = tile_y if tile_y is not None else ext_h

            # scale tile sizes for pyramid levels and storedsize
            sf = self._image._scale_factors
            ss_sf = self._image._storedsize_scale
            layout = self._image._subblock_layout
            if layout.is_pyramid_level and sf:
                x_di = layout.x_idx
                y_di = layout.y_idx
                if x_di >= 0 and tile_x is not None:
                    step_x = round(step_x * sf[x_di])
                if y_di >= 0 and tile_y is not None:
                    step_y = round(step_y * sf[y_di])
            elif self._image._storedsize and bool(ss_sf):
                x_di = layout.x_idx
                y_di = layout.y_idx
                if x_di >= 0 and tile_x is not None:
                    step_x = round(step_x / ss_sf[x_di])
                if y_di >= 0 and tile_y is not None:
                    step_y = round(step_y / ss_sf[y_di])

            grid: list[tuple[int, int, int, int]] = []
            for y in range(ext_y, ext_y + ext_h, step_y):
                for x in range(ext_x, ext_x + ext_w, step_x):
                    w = min(step_x, ext_x + ext_w - x)
                    h = min(step_y, ext_y + ext_h - y)
                    grid.append((x, y, w, h))
            spatial_grid = tuple(grid)

        return tuple(iter_dims), coord_combos, spatial_grid

    def __repr__(self) -> str:
        dims = ', '.join(f'{k}: {v}' for k, v in self._sizes.items())
        return f'<{self.__class__.__name__} {len(self)} chunks ({dims})>'


class CziScenes(collections.abc.Mapping[int, CziImage]):
    """Mapping of scene S-coordinates to :py:class:`CziImage` instances.

    Keys are the absolute S-coordinate values stored in the CZI file.
    For files that have no explicit scene (S) dimension, the single
    implicit scene is exposed under the synthetic key ``0``.

    Full :py:class:`collections.abc.Mapping` interface is provided::

        scenes[s]        # CziImage for scene S=s
        s in scenes      # True when S=s is a valid key
        len(scenes)      # number of scenes
        scenes.keys()    # S-coordinate values (ascending)
        scenes.values()  # CziImage instances in key order
        scenes.items()   # (S-coordinate, CziImage) pairs

    Use the call syntax to obtain a merged :py:class:`CziImage` spanning
    all or a selection of scenes::

        scenes()              # all scenes merged into one CziImage
        scenes(scene=0)       # scene with S-coordinate 0
        scenes(scene=[0, 2])  # scenes S=0 and S=2 merged

    The first scene (lowest S-coordinate) is conveniently accessed as::

        next(iter(scenes.values()))

    """

    __slots__ = (
        '_parent',
        '_scene_indices',
        '_squeeze',
        '_cache',
        '_implicit',
    )

    _parent: CziFile
    _scene_indices: tuple[int, ...]
    _squeeze: bool
    _cache: dict[int, CziImage]
    _implicit: bool

    def __init__(
        self,
        parent: CziFile,
        /,
        *,
        squeeze: bool = True,
    ) -> None:
        self._parent = parent
        self._squeeze = squeeze
        self._cache = {}
        # collect all S-coordinate values used by non-pyramid subblocks
        indices: set[int] = set()
        for entry in parent.filtered_subblock_directory:
            if entry.pyramid_type == 0:
                indices.add(entry.scene_index)
        if indices == {-1}:
            # no explicit S dimension: single implicit scene, exposed as key 0
            self._scene_indices = (0,)
            self._implicit = True
        else:
            # explicit scenes: use actual S-coordinate values as keys
            self._scene_indices = tuple(sorted(indices - {-1}))
            self._implicit = False

    def __call__(
        self,
        *,
        scene: SelectionValue = None,
        roi: tuple[int, int, int, int] | None = None,
        **selection: SelectionValue,
    ) -> CziImage:
        """Return CziImage for one or multiple scenes merged.

        Parameters:
            scene:
                Absolute S-coordinate(s) of scene(s) to select.
                ``None`` merges all scenes into one :py:class:`CziImage`.
                ``int`` selects a single scene by its S-coordinate.
                ``slice`` selects scenes whose S-coordinate falls within
                the range defined by the slice (absolute S-coordinates),
                consistent with ``int`` selection.
                ``Sequence[int]`` selects and merges specific scenes.
                For files without an explicit S dimension, use ``0`` (or
                ``None``).
            roi:
                Spatial region of interest in absolute pixel coordinates
                ``(x, y, width, height)``.
                Applied to the resulting CziImage.
            **selection:
                Dimension selections forwarded to
                :py:meth:`CziImage.__call__`.

        Raises:
            KeyError:
                If a requested scene key is not present in the file.

        """
        keys = self._resolve_keys(scene)
        key_set = set(keys)
        if self._implicit:
            entries = tuple(
                e
                for e in self._parent.filtered_subblock_directory
                if e.pyramid_type == 0
            )
        else:
            entries = tuple(
                e
                for e in self._parent.filtered_subblock_directory
                if e.pyramid_type == 0
                and (e.scene_index == -1 or e.scene_index in key_set)
            )
        if not entries:
            raise KeyError(scene)
        # build name
        if scene is None:
            name = 'All scenes'
        elif isinstance(scene, int):
            name = f'Scene {scene}'
        else:
            name = f'Scenes {scene!r}'
        image = CziImage(
            self._parent,
            entries,
            name=name,
            squeeze=self._squeeze,
        )
        if roi is not None or selection:
            return image(roi=roi, **selection)  # type: ignore[arg-type]
        return image

    def __getitem__(self, key: int, /) -> CziImage:
        """Return CziImage for scene with S-coordinate *key*.

        Raises:
            KeyError: If *key* is not a valid scene S-coordinate.

        """
        if key not in self._scene_indices:
            valid = list(self._scene_indices)
            msg = f'scene {key} not found: valid S-coordinates are {valid}'
            raise KeyError(msg)
        if key not in self._cache:
            self._cache[key] = self._make_image(key)
        return self._cache[key]

    def __iter__(self) -> Iterator[int]:
        """Iterate over scene S-coordinate keys in ascending order."""
        return iter(self._scene_indices)

    def __len__(self) -> int:
        """Number of scenes."""
        return len(self._scene_indices)

    def _make_image(self, key: int, /) -> CziImage:
        """Create a CziImage for the given scene key."""
        if self._implicit:
            entries = tuple(
                e
                for e in self._parent.filtered_subblock_directory
                if e.pyramid_type == 0
            )
        else:
            entries = tuple(
                e
                for e in self._parent.filtered_subblock_directory
                if e.pyramid_type == 0 and e.scene_index in (-1, key)
            )
        name = f'Scene {key}'
        if not entries:
            raise KeyError(key)
        image = CziImage(
            self._parent,
            entries,
            name=name,
            squeeze=self._squeeze,
        )
        # populate pyramid levels
        if self._implicit:
            pyramid_entries = [
                e
                for e in self._parent.subblock_directory
                if e.pyramid_type != 0
            ]
        else:
            pyramid_entries = [
                e
                for e in self._parent.subblock_directory
                if e.pyramid_type != 0 and e.scene_index in (-1, key)
            ]
        if pyramid_entries:
            # group by scale factor
            levels_by_scale: dict[
                tuple[float, ...], list[CziDirectoryEntryDV]
            ] = {}
            for e in pyramid_entries:
                lkey = normalize_pyramid_scale(e.scale)
                levels_by_scale.setdefault(lkey, []).append(e)
            for scale in sorted(levels_by_scale):
                level_entries = tuple(levels_by_scale[scale])
                level = CziImage(
                    self._parent,
                    level_entries,
                    name=f'{name} level {len(image._levels)}',
                    squeeze=self._squeeze,
                )
                image._levels.append(level)
        return image

    def _resolve_keys(self, scene: SelectionValue, /) -> tuple[int, ...]:
        """Resolve scene argument to a tuple of valid scene keys."""
        if scene is None:
            return self._scene_indices
        if isinstance(scene, int):
            if scene not in self._scene_indices:
                msg = (
                    f'scene key {scene!r} not found: '
                    f'available {self._scene_indices}'
                )
                raise KeyError(msg)
            return (scene,)
        if isinstance(scene, slice):
            # interpret as absolute S-coordinate range, consistent with int
            selected = set(range(*scene.indices(self._scene_indices[-1] + 1)))
            result = tuple(s for s in self._scene_indices if s in selected)
            if not result:
                raise KeyError(scene)
            return result
        keys: list[int] = []
        for s in scene:
            if s not in self._scene_indices:
                msg = (
                    f'scene key {s!r} not found: '
                    f'available {self._scene_indices}'
                )
                raise KeyError(msg)
            keys.append(s)
        return tuple(keys)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {len(self._scene_indices)}>'

    def __str__(self) -> str:
        return indent(repr(self), *(repr(im) for im in self.values()))


class CziSegment:
    """ZISRAW segment.

    Parameters:
        czifile:
            CZI file to read from.
        offset:
            Position of segment in file.
            If ``None``, use current file position.

    """

    __slots__ = ('sid', 'offset', 'allocated_size', 'used_size', '_czifile')

    sid: CziSegmentId
    """CziSegment identifier."""

    offset: int
    """Position of segment in file."""

    allocated_size: int
    """Number of bytes allocated for segment."""

    used_size: int
    """Currently used number of bytes for segment."""

    _czifile: CziFile

    def __init__(self, czifile: CziFile, offset: int | None = None, /) -> None:
        self._czifile = czifile
        fh = czifile.filehandle
        if offset is not None:
            fh.seek(offset)
            self.offset = offset
        else:
            self.offset = fh.tell()
        data = fh.read(32)
        try:
            (
                sid,
                self.allocated_size,
                self.used_size,
            ) = struct.unpack('<16sqq', data)
        except struct.error as exc:
            msg = f'failed to read ZISRAW segment header {data!r}'
            raise CziSegmentNotFoundError(msg) from exc
        self.sid = CziSegmentId(sid.rstrip(b'\x00').decode('cp1252'))
        if self.used_size == 0:
            self.used_size = self.allocated_size

    @property
    def czifile(self) -> CziFile:
        """Associated CZI file instance."""
        return self._czifile

    @property
    def filehandle(self) -> IO[bytes]:
        """File handle."""
        return self._czifile.filehandle

    @property
    def data_offset(self) -> int:
        """Position of segment data in file."""
        return self.offset + 32

    def data(self) -> CziSegmentData:
        """Read and return segment payload."""
        self._czifile.filehandle.seek(self.offset + 32)
        return self.sid.read_segment_data(self)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.sid} @ {self.offset}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'used_size: {self.used_size}',
            f'allocated_size: {self.allocated_size}',
        )


class CziSegmentData(metaclass=abc.ABCMeta):
    """Abstract base class for segment data."""

    SID: str
    """CziSegment identifier."""

    @abc.abstractmethod
    def __init__(self, segment: CziSegment, /) -> None:
        pass

    @property
    @abc.abstractmethod
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} {self.SID!r} '
            f'@ {self.segment.data_offset}>'
        )


class CziFileHeaderSegmentData(CziSegmentData):
    """ZISRAWFILE file header segment data.

    Contains global file metadata such as file version and GUID.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = (
        '_segment',
        'version',
        'primary_file_guid',
        'file_guid',
        'file_part',
        'metadata_position',
        'directory_position',
        'attachment_directory_position',
        'update_pending',
    )

    SID = 'ZISRAWFILE'

    version: tuple[int, int]
    """Major and minor file version."""

    primary_file_guid: uuid.UUID
    """Globally unique identifier of main file."""

    file_guid: uuid.UUID
    """Globally unique identifier of file."""

    file_part: int
    """Part number in multi-file scenarios."""

    metadata_position: int
    """Position of metadata segment in file."""

    directory_position: int
    """Position of subblock directory segment in file."""

    attachment_directory_position: int
    """Position of attachment directory segment in file."""

    update_pending: bool
    """File has inconsistent state."""

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        if segment.sid != self.SID:
            msg = f'{segment.sid!r} != {self.SID!r}'
            raise ValueError(msg)
        self._segment = segment
        (
            major,
            minor,
            _,  # reserved1
            _,  # reserved2
            primary_file_guid,
            file_guid,
            self.file_part,
            self.directory_position,
            self.metadata_position,
            self.update_pending,
            self.attachment_directory_position,
        ) = struct.unpack('<iiii16s16siqqiq', segment.filehandle.read(80))
        self.version = (major, minor)
        self.update_pending = bool(self.update_pending)
        self.primary_file_guid = uuid.UUID(bytes=primary_file_guid)
        self.file_guid = uuid.UUID(bytes=file_guid)

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    def __str__(self) -> str:
        return indent(
            repr(self),
            *(
                f'{name}: {getattr(self, name)}'
                for name in CziFileHeaderSegmentData.__slots__
                if not name.startswith('_')
            ),
        )


class CziMetadataSegmentData(CziSegmentData):
    """ZISRAWMETADATA segment data.

    Contains global image metadata in UTF-8 encoded XML format.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = ('_segment', 'xml_size', 'xml_offset', 'attachment_size')

    SID = 'ZISRAWMETADATA'

    xml_size: int
    """Size of XML data."""

    xml_offset: int
    """Position of XML metadata in file."""

    attachment_size: int
    """Size of binary attachments (unused)."""

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        if segment.sid != self.SID:
            msg = f'{segment.sid!r} != {self.SID!r}'
            raise ValueError(msg)
        self._segment = segment
        fh = segment.filehandle
        data = fh.read(256)  # header (8) + spare (248)
        self.xml_size, self.attachment_size = struct.unpack('<ii', data[:8])
        self.xml_offset = segment.data_offset + 256

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    def data(self) -> str:
        """Read and return XML metadata payload."""
        fh = self._segment.filehandle
        fh.seek(self.xml_offset)
        xml = fh.read(self.xml_size)
        xml = xml.replace(b'\r\n', b'\n').replace(b'\r', b'\n')  # ???
        return xml.decode()

    def __str__(self) -> str:
        return indent(repr(self), self.data())


class CziSubBlockSegmentData(CziSegmentData):
    """ZISRAWSUBBLOCK segment data (ImageSubBlock).

    Contains XML metadata, optional attachments, and homogeneous,
    contiguous pixel data.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = (
        '_segment',
        'directory_entry',
        'data_offset',
        'data_size',
        'metadata_size',
        'attachment_size',
    )

    SID = 'ZISRAWSUBBLOCK'

    directory_entry: CziDirectoryEntryDV
    """Image subset indices and size information."""

    data_offset: int
    """Position of data section in file."""

    data_size: int
    """Size of data section."""

    metadata_size: int
    """Size of metadata section."""

    attachment_size: int
    """Size of optional attachment section."""

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        if segment.sid != self.SID:
            msg = f'{segment.sid!r} != {self.SID!r}'
            raise ValueError(msg)
        self._segment = segment
        fh = segment.filehandle
        # with fh.lock:
        (
            self.metadata_size,
            self.attachment_size,
            self.data_size,
        ) = struct.unpack('<iiq', fh.read(16))
        self.directory_entry = CziDirectoryEntryDV(fh)
        self.data_offset = fh.tell()
        self.data_offset += max(240 - self.directory_entry.storage_size, 0)
        self.data_offset += self.metadata_size

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    @property
    def directory_entry_offset(self) -> int:
        """Position of inline directory entry in file."""
        return self._segment.data_offset + 16

    @property
    def metadata_offset(self) -> int:
        """Position of metadata section in file."""
        return self.data_offset - self.metadata_size

    @overload
    def metadata(
        self, *, asdict: Literal[False] = ..., fixesc: bool = True
    ) -> str: ...

    @overload
    def metadata(
        self, *, asdict: Literal[True], fixesc: bool = True
    ) -> dict[str, Any]: ...

    @overload
    def metadata(
        self, *, asdict: bool = False, fixesc: bool = True
    ) -> str | dict[str, Any]: ...

    def metadata(
        self, *, asdict: bool = False, fixesc: bool = True
    ) -> str | dict[str, Any]:
        """Return metadata from file.

        Parameters:
            asdict:
                If true, return metadata as dict, else XML (default).
            fixesc:
                If true (default), replace `&lt;` and `&gt;` with `<` and `>`.
                Fixes frequent, unparsable XML elements.

        """
        if self.metadata_size <= 0:
            return {} if asdict else ''
        fh = self.segment.filehandle
        with self.segment.czifile.lock:
            fh.seek(self.data_offset - self.metadata_size)
            metadata = fh.read(self.metadata_size)
        if fixesc and b'&lt;' in metadata and b'&gt;' in metadata:
            # work around bug in CZI writer
            metadata = metadata.replace(b'&lt;', b'<').replace(b'&gt;', b'>')
        if asdict:
            return xml2dict(metadata.decode())['METADATA']  # type: ignore[no-any-return]
        return metadata.decode()

    @overload
    def data(self, *, raw: Literal[False] = ...) -> NDArray[Any]: ...

    @overload
    def data(self, *, raw: Literal[True]) -> bytes: ...

    @overload
    def data(self, *, raw: bool = False) -> bytes | NDArray[Any]: ...

    def data(self, *, raw: bool = False) -> bytes | NDArray[Any]:
        """Return image data from file.

        Parameters:
            raw:
                If false (default), return decoded image data from file,
                else return undecoded data.

        """
        de: CziDirectoryEntryDV = self.directory_entry
        # bypass property descriptors on hot path
        czi = self._segment._czifile
        fh = czi._fh
        lock = czi._lock
        if raw:
            with lock:
                fh.seek(self.data_offset)
                return fh.read(self.data_size)
        dtype = de.pixel_type.dtype
        shape = de.stored_shape
        total = product(shape)
        size = total * dtype.itemsize
        compression = de.compression
        decode = de.decode
        decode_cache = czi._cache
        if decode is not None:
            cached = decode_cache.get(self.data_offset)
            if cached is not None:
                return cached
        if decode is not None:
            with lock:
                fh.seek(self.data_offset)
                data = fh.read(self.data_size)
            if len(data) != self.data_size:
                msg = 'failed to read all segment data'
                raise CziFileError(msg)
            if self.data_size == size:
                # TODO: report bug in CZI writer
                image = numpy.frombuffer(data, dtype)
            elif compression == 6:
                # ZSTD1: byte 0 is the total header size (including itself).
                # The header body (bytes 1..header_size-1) is a sequence
                # of typed chunks. Currently only chunk type 1 is defined,
                # whose single payload byte's LSB is the hi/lo byte-shuffle
                # flag. Unknown chunk types are skipped for forward
                # compatibility. Data starts at byte offset header_size.
                if self.data_size < 2:
                    msg = f'Zstd1 data too short {self.data_size}'
                    raise CziFileError(msg)
                header_size = data[0]
                if header_size == 0 or header_size >= self.data_size:
                    msg = f'invalid Zstd1 header {data[:4]!r}'
                    raise CziFileError(msg)
                hilo = False
                pos = 1
                while pos < header_size:
                    chunk_type = data[pos]
                    pos += 1
                    if chunk_type == 1:
                        if pos >= header_size:
                            msg = 'truncated Zstd1 chunk type 1'
                            raise CziFileError(msg)
                        hilo = (data[pos] & 1) != 0
                        pos += 1
                    # unknown chunk types: no defined payload length -
                    # stop parsing and let the decoder handle the rest
                    else:
                        break
                offset = header_size
                # Use memoryview to avoid copying compressed bytes
                # when skipping the small fixed-size ZSTD1 header.
                image = decode(memoryview(data)[offset:], out=size)
                image = numpy.frombuffer(image, dtype)
                if hilo:
                    image = imagecodecs.byteshuffle_decode(image)
            else:
                image = decode(data, out=size)
                if compression in (2, 5):
                    # LZW, ZSTD
                    image = numpy.frombuffer(image, dtype)
        elif compression == 0 or compression > 999:
            # SYSTEMRAW (>999) and CAMERARAW (100-999) are read as raw,
            # uncompressed pixels. Callers compositing via CziImage.asarray
            # see a warning there. Direct callers of data() do not.
            with lock:
                fh.seek(self.data_offset)
                image = read_array(fh, dtype, self.data_size // dtype.itemsize)
        else:
            msg = f'{compression!r} invalid or not supported'
            raise ValueError(msg)

        if image.size != total:
            # CZI writer bug: stored_shape in the directory entry does not
            # match the number of pixels the decoder actually returned.
            # This is seen on pyramid edge-tiles where the writer stores
            # ceil(N/2) in stored_size but encodes floor(N/2) rows (or cols).
            # Find the single dimension that can be adjusted to reconcile the
            # mismatch and update shape accordingly.
            adjusted = list(shape)
            for i, s in enumerate(shape):
                rest = total // s
                if rest > 0 and image.size % rest == 0:
                    adjusted[i] = image.size // rest
                    if product(adjusted) == image.size:
                        logger().debug(
                            f'stored_shape {shape} does not match decoded '
                            f'size {image.size}: adjusting dim {i} '
                            f'{s} -> {adjusted[i]}'
                        )
                        shape = tuple(adjusted)
                        break
                    adjusted[i] = s  # reset and try next dim
        image = image.reshape(shape)
        if compression not in (1, 4):
            if shape[-1] == 3:
                # BGR -> RGB
                image = image[..., ::-1].copy()
            elif shape[-1] == 4:
                # BGRA -> RGBA; alpha is forced to 255 (fully opaque),
                # matching Zeiss software
                image = image[..., [2, 1, 0, 3]].copy()
                image[..., 3] = 255

        if decode is not None:
            decode_cache.put(self.data_offset, image)
        return image

    def attachments(self) -> tuple[tuple[bytes, bytes], ...]:
        """Return optional attachments from file.

        Returns:
            Sequence of tuples of guid and binary attachments.

        """
        if self.attachment_size < 1:
            return ()
        fh = self.segment.filehandle
        with self.segment.czifile.lock:
            fh.seek(self.data_offset + self.data_size)
            data = fh.read(self.attachment_size)
            xml = self.metadata(asdict=False)
        if 'CHUNKCONTAINER' not in xml:
            return ((b'', data),)
        attachments = []
        index = 0
        while index < self.attachment_size - 20:
            guid, size = struct.unpack('<16si', data[index : index + 20])
            index += 20
            if index + size <= self.attachment_size:
                attachments.append((guid, data[index : index + size]))
            index += size
        return tuple(attachments)

    def mask(self) -> NDArray[Any] | None:
        """Return subblock validity mask, or ``None`` if absent.

        The mask is a boolean ndarray of shape ``(height, width)``
        where ``True`` indicates a pixel that was written by the encoder.
        ``False`` pixels are gaps - they were not covered by the sensor
        and should be treated as background.
        The compositing code in :py:meth:`CziImage.asarray` uses this
        mask to avoid overwriting already-composited pixels with gap
        values.

        """
        if self.attachment_size < 32:
            return None
        attachments = self.attachments()
        if len(attachments) < 1 or len(attachments[0][1]) < 16:
            return None
        # valid-pixel-mask GUID not listed in specification
        guid = b'g\xea\xe3\xcb\xfc[+I\xa1j\xec\xe3x\x03\x14H'
        if attachments[0][0] == guid:
            data = attachments[0][1]
        elif attachments[0][1][:16] == guid:
            data = attachments[0][1][16:]
        else:
            return None
        width, height, typerepr, stride = struct.unpack('<IIII', data[:16])
        if (
            typerepr != 0
            or width > 2**16
            or height > 2**16
            or abs(stride * 8 - width) > 8
        ):
            # logger().warning(f'unknown mask type {typerepr!r}')
            return None
        mask = imagecodecs.packints_decode(data[16:], bool, 1, runlen=width)
        return mask.reshape((height, width))

    def __str__(self) -> str:
        return indent(
            repr(self),
            # pformat(self.metadata()),
            self.directory_entry,
        )


class CziDirectoryEntryDV:
    """Directory Entry - Schema DV [Directory Variable length].

    Image subset indices and size information.

    Parameters:
        fh: File handle to read from.

    """

    __slots__ = (
        'offset',
        'pixel_type',
        'compression',
        'pyramid_type',
        'file_position',
        'file_part',
        'dimensions_count',
        'dims',
        'shape',
        'start',
        'stop',
        'stored_shape',
        'mosaic_index',
        'scene_index',
        'is_pyramid',
        'decode',
        '_start_coordinate_raw',
    )

    offset: int
    """Position of directory entry in file."""

    pixel_type: CziPixelType
    """Type of pixel data."""

    compression: CziCompressionType
    """Type of pixel data compression."""

    pyramid_type: CziPyramidType
    """Type of image pyramid."""

    file_position: int
    """Position of associated subblock segment in file."""

    file_part: int
    """Part number in multi-file scenarios."""

    dimensions_count: int
    """Number of dimension entries."""

    dims: tuple[str, ...]
    """Dimension names, excluding mosaic and scene."""

    shape: tuple[int, ...]
    """Logical size of dimensions, excluding mosaic and scene."""

    start: tuple[int, ...]
    """Start indices of dimensions, excluding mosaic and scene."""

    stop: tuple[int, ...]
    """Upper indices of dimensions, excluding mosaic and scene."""

    stored_shape: tuple[int, ...]
    """Stored size of dimensions, excluding mosaic and scene."""

    mosaic_index: int
    """Mosaic tile index, -1 if undefined."""

    scene_index: int
    """CziScene index, -1 if undefined."""

    is_pyramid: bool
    """Pixel data is sub- or supersampled along some dimensions."""

    decode: Callable[..., Any] | None
    """Decoder function for pixel data, or None if uncompressed."""

    _start_coordinate_raw: bytes

    def __init__(self, fh: IO[bytes], /) -> None:
        self.offset = fh.tell()
        (
            schema_type,
            pixel_type,  # PixelTypes
            self.file_position,
            self.file_part,
            compression,  # CompressionTypes
            pyramid_type,  # PyramidTypes
            _,  # reserved1
            _,  # reserved2
            self.dimensions_count,
        ) = struct.unpack('<2siqiiBB4si', fh.read(32))

        if schema_type != b'DV':
            # TODO: support schema DE, a fixed 128-byte legacy format?
            # need sample files for testing
            msg = f'not a CziDirectoryEntryDV {schema_type=!r}'
            raise CziFileError(msg)

        self.pixel_type = CziPixelType(pixel_type)
        self.compression = CziCompressionType(compression)
        self.pyramid_type = CziPyramidType(pyramid_type)
        self.decode = CziCompressionType.decoder(compression)
        self.scene_index = -1
        self.mosaic_index = -1

        dims = []
        shape = []
        start = []
        stop = []
        start_coordinate = []
        stored_shape = []

        data = fh.read(self.dimensions_count * 20)
        for i in reversed(range(0, self.dimensions_count * 20, 20)):
            (
                dimension,
                dim_start,
                size,
                dim_start_coordinate,
                stored_size,
            ) = struct.unpack('<4siifi', data[i : i + 20])
            dimension = dimension.rstrip(b'\x00').decode('cp1252')
            if dimension == 'M':
                self.mosaic_index = dim_start
                continue
            if dimension == 'S':
                self.scene_index = dim_start
                continue
            dims.append(dimension)
            shape.append(size)
            start.append(dim_start)
            stop.append(dim_start + size)
            start_coordinate.append(dim_start_coordinate)
            stored_shape.append(size if stored_size == 0 else stored_size)

        dims.append('S')
        shape.append(self.pixel_type.samples)
        start.append(0)
        stop.append(self.pixel_type.samples)
        start_coordinate.append(0.0)
        stored_shape.append(self.pixel_type.samples)

        self.dims = tuple(dims)
        self.shape = tuple(shape)
        self.start = tuple(start)
        self.stop = tuple(stop)
        self._start_coordinate_raw = struct.pack(
            f'<{len(start_coordinate)}f', *start_coordinate
        )
        self.stored_shape = tuple(stored_shape)
        self.is_pyramid = self.shape != self.stored_shape

    @staticmethod
    def read_file_position(fh: IO[bytes], /) -> int:
        """Return position of associated :py:class:`CziSubBlockSegmentData`.

        Parameters:
            fh: File handle to read from.

        """
        (
            schema_type,
            file_position,
            dimensions_count,
        ) = struct.unpack('<2s4xq14xi', fh.read(32))
        if schema_type != b'DV':
            msg = f'not a CziDirectoryEntryDV {schema_type=!r}'
            raise CziFileError(msg)
        fh.seek(dimensions_count * 20, 1)
        return int(file_position)

    @property
    def start_coordinate(self) -> tuple[float, ...]:
        """Physical start coordinate of dimension, excluding mosaic and scene.

        Values preserve the original float32 precision from the file.

        """
        n = len(self._start_coordinate_raw) // 4
        return struct.unpack(f'<{n}f', self._start_coordinate_raw)

    @property
    def schema_type(self) -> str:
        """Directory entry schema type."""
        return 'DV'

    @property
    def storage_size(self) -> int:
        """Number of bytes used to store CziDirectoryEntryDV."""
        return 32 + self.dimensions_count * 20

    @property
    def subblock_entry_offset(self) -> int:
        """Position of inline directory entry within associated subblock.

        Allows accessing dimension start values without going through
        the SubBlockDirectory.

        """
        return self.file_position + 48

    @property
    def dtype(self) -> numpy.dtype:
        """Numpy data type of pixel type."""
        return self.pixel_type.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions, excluding mosaic and scene."""
        return len(self.dims)

    @property
    def size(self) -> int:
        """Number of items in array described by dimensions."""
        return product(self.shape)

    @property
    def scale(self) -> tuple[float, ...]:
        """Sub / supersampling factors."""
        factors = [
            j / i for i, j in zip(self.stored_shape, self.shape, strict=True)
        ]
        return tuple(
            round(f) if abs(f - round(f)) < 0.0001 else f for f in factors
        )

    def read_dimension_entries(
        self, fh: IO[bytes], /
    ) -> dict[CziDimensionType, CziDimensionEntryDV1]:
        """Return all dimension entries from file, including M and S.

        Unlike the constructor, iterates forward and includes every
        dimension without filtering, keyed by :py:class:`CziDimensionType`.

        Parameters:
            fh: File handle to read from.

        """
        fh.seek(self.offset + 32)
        data = fh.read(self.dimensions_count * 20)
        dimension_entries = {}
        for i in range(0, self.dimensions_count * 20, 20):
            dim = CziDimensionEntryDV1(data[i : i + 20])
            dimension_entries[dim.dimension] = dim
        return dimension_entries

    def read_segment_data(
        self,
        czifile: CziFile,
        /,
    ) -> CziSubBlockSegmentData | CziDeletedSegmentData:
        """Return associated subblock segment data from file.

        Parameters:
            czifile: CZI file to read from.

        """
        fh = czifile._fh
        lock = czifile._lock
        pos = self.file_position
        with lock:
            fh.seek(pos)
            # All needed header fields fit in the first 48 bytes:
            # [0:16] SID, [16:32] allocated/used sizes,
            # [32:48] metadata_size / attachment_size / data_size.
            header = fh.read(48)
            # Fast path: segment is a live ZISRAWSUBBLOCK.
            # Compute data_offset from the already-known storage_size
            # instead of constructing a new CziDirectoryEntryDV.
            if header[:14] == b'ZISRAWSUBBLOCK':
                allocated_size, used_size = struct.unpack_from(
                    '<qq', header, 16
                )
                if used_size == 0:
                    used_size = allocated_size
                metadata_size, attachment_size, data_size = struct.unpack_from(
                    '<iiq', header, 32
                )
                storage = self.storage_size  # 32 + dimensions_count * 20
                data_offset = pos + 48 + max(240, storage) + metadata_size
                segmentdata = CziSubBlockSegmentData.__new__(
                    CziSubBlockSegmentData
                )
                seg = CziSegment.__new__(CziSegment)
                seg.sid = CziSegmentId.ZISRAWSUBBLOCK
                seg.offset = pos
                seg.allocated_size = allocated_size
                seg.used_size = used_size
                seg._czifile = czifile
                segmentdata._segment = seg
                segmentdata.directory_entry = self
                segmentdata.metadata_size = metadata_size
                segmentdata.attachment_size = attachment_size
                segmentdata.data_size = data_size
                segmentdata.data_offset = data_offset
                return segmentdata

            # slow fallback path: DELETED segment or unexpected type
            fallback = CziSegment(czifile, pos).data()

        if fallback.SID == CziSubBlockSegmentData.SID:
            return cast(CziSubBlockSegmentData, fallback)
        return cast(CziDeletedSegmentData, fallback)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} @ {self.offset}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'compression: {self.compression.name} '
            f'({self.compression._value_})',
            f'pixel_type: {self.pixel_type.name}',
            f'pyramid_type: {self.pyramid_type.name}',
            f'dtype: {self.dtype}',
            f'dims: {self.dims}',
            f'shape: {self.shape}',
            f'stored_shape: {self.stored_shape}',
            # *self.dimension_entries.values(),
        )


class CziDimensionEntryDV1:
    """Dimension Entry - Schema DV.

    Parameters:
        data: Byte stream of dimension entry.

    """

    __slots__ = (
        'dimension',
        'start',
        'size',
        'start_coordinate',
        'stored_size',
    )

    dimension: CziDimensionType
    """Single character code identifying dimension."""

    start: int
    """Start index. May be < 0."""

    size: int
    """Logical size of dimension."""

    start_coordinate: float
    """Physical start coordinate."""

    stored_size: int
    """Stored size if sub or supersampling, else 0."""

    def __init__(self, data: bytes, /) -> None:
        (
            dimension,
            self.start,
            self.size,
            self.start_coordinate,
            self.stored_size,
        ) = struct.unpack('<4siifi', data)
        self.dimension = CziDimensionType(
            dimension.rstrip(b'\x00').decode('cp1252')
        )

    @property
    def schema_type(self) -> str:
        """Dimension entry schema type."""
        return 'DV1'

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} '
            f'{self.dimension.name!r} '
            f'{self.start} '
            f'{self.size} '
            f'{self.start_coordinate} '
            f'{self.stored_size}>'
        )

    def __str__(self) -> str:
        return repr(self)


class CziSubBlockDirectorySegmentData(CziSegmentData):
    """ZISRAWDIRECTORY segment data.

    Contains entries of any kind, currently only CziDirectoryEntryDV.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = ('_segment', 'entries')

    SID = 'ZISRAWDIRECTORY'

    entries: tuple[CziDirectoryEntryDV, ...]
    """Directory entries."""

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        if segment.sid != self.SID:
            msg = f'{segment.sid!r} != {self.SID!r}'
            raise ValueError(msg)
        self._segment = segment
        fh = segment.filehandle
        entry_count = int.from_bytes(
            fh.read(128)[:4], byteorder='little', signed=True  # +124 reserved
        )
        self.entries = tuple(
            CziDirectoryEntryDV(fh) for _ in range(entry_count)
        )

    @staticmethod
    def file_positions(fh: IO[bytes], /) -> tuple[int, ...]:
        """Return file positions of associated ``SubBlock`` segments.

        Lightweight alternative to full parsing: reads only the file
        offsets without constructing :py:class:`CziDirectoryEntryDV`
        objects. Useful for recovery or when only segment locations
        are needed.

        Parameters:
            fh: File handle to read from.

        """
        entry_count = int.from_bytes(
            fh.read(128)[:4], byteorder='little', signed=True  # +124 reserved
        )
        return tuple(
            CziDirectoryEntryDV.read_file_position(fh)
            for _ in range(entry_count)
        )

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, key: int, /) -> CziDirectoryEntryDV:
        return self.entries[key]

    def __iter__(self) -> Iterator[CziDirectoryEntryDV]:
        return iter(self.entries)

    def __str__(self) -> str:
        return indent(repr(self), *self.entries)


class CziAttachmentSegmentData(CziSegmentData):
    """ZISRAWATTACH segment data.

    Contains binary or text data as specified in :py:attr:`attachment_entry`.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = ('_segment', 'data_size', 'data_offset', 'attachment_entry')

    SID = 'ZISRAWATTACH'

    data_size: int
    """Size of data section."""

    data_offset: int
    """Position of attachment embedded in file."""

    attachment_entry: CziAttachmentEntryA1
    """Core information about attachment."""

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        if segment.sid != self.SID:
            msg = f'{segment.sid!r} != {self.SID!r}'
            raise ValueError(msg)
        self._segment = segment
        fh = segment.filehandle
        self.data_size = int.from_bytes(
            fh.read(16)[:4], byteorder='little', signed=True  # +12 reserved
        )
        self.attachment_entry = CziAttachmentEntryA1(fh)
        fh.seek(112, 1)  # reserved
        self.data_offset = fh.tell()

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    def save(
        self,
        filename: str | os.PathLike[Any] | None = None,
        directory: str | os.PathLike[Any] = '.',
    ) -> None:
        """Write attachment to file.

        Parameters:
            filename:
                Name of file to save attachment.
                Defaults to :py:meth:`CziAttachmentEntryA1.filename`.
            directory:
                Directory for attachment.
                Defaults to current directory.

        """
        if not filename:
            filename = self.attachment_entry.filename
        filename = os.path.join(directory, filename)
        with open(filename, 'wb') as fh:
            fh.write(self.data(raw=True))

    @overload
    def data(self, *, raw: Literal[True]) -> bytes: ...

    @overload
    def data(self, *, raw: Literal[False] = ...) -> Any: ...

    @overload
    def data(self, *, raw: bool) -> Any: ...

    def data(self, *, raw: bool = False) -> Any:
        """Return content of embedded attachment.

        Parameters:
            raw:
                If true, return contents as bytes, else return decoded
                contents according to
                :py:attr:`CziContentFileType.read_content`.

        """
        fh = self.segment.filehandle
        fh.seek(self.data_offset)
        if raw:
            return fh.read(self.data_size)
        cft = self.attachment_entry.content_file_type
        return cft.read_content(fh, self.data_size)

    def __str__(self) -> str:
        return indent(repr(self), self.attachment_entry)


class CziAttachmentEntryA1:
    """AttachmentEntry - Schema A1.

    Part of :py:class:`CziAttachmentSegmentData`
    and :py:class:`CziAttachmentDirectorySegmentData`.

    Parameters:
        fh: File handle to read from.

    """

    __slots__ = (
        'offset',
        'name',
        'file_position',
        'file_part',
        'content_guid',
        'content_file_type',
    )

    offset: int
    """Position of attachment entry in file."""

    name: str
    """Name of attachment."""

    file_position: int
    """Position of associated :py:class:`CziAttachmentSegmentData` in file."""

    file_part: int
    """Part number in multi-file scenarios."""

    content_guid: uuid.UUID
    """Unique identifier used in references."""

    content_file_type: CziContentFileType
    """Content file identifier."""

    def __init__(self, fh: IO[bytes], /) -> None:
        self.offset = fh.tell()
        (
            schema_type,
            _,  # reserved
            self.file_position,
            self.file_part,
            content_guid,
            content_file_type,
            name,
        ) = struct.unpack('<2s10sqi16s8s80s', fh.read(128))
        if schema_type != b'A1':
            msg = 'not a CziAttachmentEntryA1'
            raise CziFileError(msg)
        self.content_guid = uuid.UUID(bytes=content_guid)
        self.content_file_type = CziContentFileType(
            content_file_type.rstrip(b'\x00').decode('cp1252')
        )
        self.name = name.rstrip(b'\x00').decode()

    @staticmethod
    def read_file_position(fh: IO[bytes], /) -> int:
        """Return position of associated :py:class:`CziAttachmentSegmentData`.

        Parameters:
            fh: File handle to read from.

        """
        schema_type, file_position = struct.unpack('<2s10xq', fh.read(20))
        if schema_type != b'A1':
            msg = f'not a CziAttachmentEntryA1 {schema_type=!r}'
            raise CziFileError(msg)
        fh.seek(108, 1)
        return int(file_position)

    @property
    def schema_type(self) -> str:
        """Attachment entry schema type."""
        return 'A1'

    @property
    def filename(self) -> str:
        """Unique file name for saving attachment."""
        ext = self.content_file_type.name.lower()
        return f'{self.name}@{self.file_position}.{ext}'

    @property
    def name_offset(self) -> int:
        """Position of name field in file (80 bytes, zero-padded)."""
        return self.offset + 48

    def read_segment_data(
        self,
        czifile: CziFile,
        /,
    ) -> CziAttachmentSegmentData | CziDeletedSegmentData:
        """Return associated attachment segment data from file.

        Parameters:
            czifile: CZI file to read from.

        """
        with czifile.lock:
            segmentdata = CziSegment(czifile, self.file_position).data()
        if segmentdata.SID == CziAttachmentSegmentData.SID:
            return cast(CziAttachmentSegmentData, segmentdata)
        return cast(CziDeletedSegmentData, segmentdata)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.name!r} @ {self.offset}>'

    def __str__(self) -> str:
        return indent(
            repr(self),
            *(
                f'{name}: {getattr(self, name)}'
                for name in CziAttachmentEntryA1.__slots__
                if name not in ('name', 'offset')
            ),
        )


class CziAttachmentDirectorySegmentData(CziSegmentData):
    """ZISRAWATTDIR segment data. Sequence of CziAttachmentEntryA1.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = ('_segment', 'entries')

    SID = 'ZISRAWATTDIR'

    entries: tuple[CziAttachmentEntryA1, ...]
    """Sequence of attachment entries."""

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        if segment.sid != self.SID:
            msg = f'{segment.sid!r} != {self.SID!r}'
            raise ValueError(msg)
        self._segment = segment
        fh = segment.filehandle
        entry_count = int.from_bytes(
            fh.read(256)[:4], byteorder='little', signed=True  # +252 reserved
        )
        self.entries = tuple(
            CziAttachmentEntryA1(fh) for _ in range(entry_count)
        )

    @staticmethod
    def file_positions(fh: IO[bytes], /) -> tuple[int, ...]:
        """Return file positions of associated attachment segments.

        Lightweight alternative to full parsing: reads only the file
        offsets without constructing :py:class:`CziAttachmentEntryA1`
        objects. Useful for recovery or when only segment locations
        are needed.

        Parameters:
            fh: File handle to read from.

        """
        entry_count = int.from_bytes(
            fh.read(256)[:4], byteorder='little', signed=True  # +252 reserved
        )
        return tuple(
            CziAttachmentEntryA1.read_file_position(fh)
            for _ in range(entry_count)
        )

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, key: int, /) -> CziAttachmentEntryA1:
        return self.entries[key]

    def __iter__(self) -> Iterator[CziAttachmentEntryA1]:
        return iter(self.entries)

    def __str__(self) -> str:
        return indent(repr(self), *self.entries)


class CziDeletedSegmentData(CziSegmentData):
    """DELETED segment data. Ignore.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = ('_segment',)

    SID = 'DELETED'

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        if segment.sid != self.SID:
            msg = f'{segment.sid!r} != {self.SID!r}'
            raise ValueError(msg)
        self._segment = segment

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    def __str__(self) -> str:
        return repr(self)


class CziUnknownSegmentData(CziSegmentData):
    """Unknown segment data. Ignore.

    Parameters:
        segment: Associated segment.

    """

    __slots__ = ('_segment',)

    SID = 'UNKNOWN'

    _segment: CziSegment

    def __init__(self, segment: CziSegment, /) -> None:
        self._segment = segment
        logger().warning(f'Unknown segment {segment!r}')

    @property
    def segment(self) -> CziSegment:
        """Parent segment associated with data."""
        return self._segment

    def __str__(self) -> str:
        return repr(self)


class CziEventListEntry:
    """CziEventListEntry content schema.

    Parameters:
        fh: File handle to read from.

    """

    __slots__ = ('time', 'event_type', 'description')

    time: float
    """Time of event in seconds relative to controller start time."""

    event_type: CziEventType
    """Type of event."""

    description: str
    """Description of event."""

    def __init__(self, fh: IO[bytes], /) -> None:
        (
            _,  # size
            self.time,
            event_type,
            description_size,
        ) = struct.unpack('<idii', fh.read(20))
        self.description = fh.read(description_size).rstrip(b'\x00').decode()
        self.event_type = CziEventType(event_type)

    def __repr__(self) -> str:
        return (
            f'<{self.__class__.__name__} '
            f'{self.event_type.name!r} @ {self.time}s>'
        )

    def __str__(self) -> str:
        return indent(repr(self), self.description)


class CziSubBlockLayout:
    """Layout flags describing how subblocks map to image grid.

    Attributes:
        is_upsampled:
            Subblocks are stored sub-sampled and must be expanded on read
            (Airyscan fast-scan: ``stored_shape < shape`` with
            ``pyramid_type == NONE``).
        is_pyramid_level:
            Image is downsampled pyramid overview, not full resolution
            (``stored_shape < shape`` with ``pyramid_type != NONE``).
        is_downsampled:
            Subblocks are stored at higher than logical resolution and
            must be reduced on read (PALM super-resolution:
            ``stored_shape > shape`` with ``pyramid_type == NONE``).
        x_idx:
            Index of ``'X'`` dimension in reference dims tuple,
            or ``-1`` if absent.
        y_idx:
            Index of ``'Y'`` dimension in reference dims tuple,
            or ``-1`` if absent.

    """

    __slots__ = (
        'is_upsampled',
        'is_pyramid_level',
        'is_downsampled',
        'x_idx',
        'y_idx',
    )

    def __init__(
        self,
        *,
        is_upsampled: bool,
        is_pyramid_level: bool,
        is_downsampled: bool,
        x_idx: int,
        y_idx: int,
    ) -> None:
        self.is_upsampled = is_upsampled
        self.is_pyramid_level = is_pyramid_level
        self.is_downsampled = is_downsampled
        self.x_idx = x_idx
        self.y_idx = y_idx

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'is_upsampled={self.is_upsampled!r}, '
            f'is_pyramid_level={self.is_pyramid_level!r}, '
            f'is_downsampled={self.is_downsampled!r}, '
            f'x_idx={self.x_idx!r}, '
            f'y_idx={self.y_idx!r})'
        )


def read_event_list(
    fh: IO[bytes], size: int = 0, /
) -> tuple[CziEventListEntry, ...]:
    """Return sequence of CziEventListEntry from file.

    CZEVL EventList content schema.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    del size
    _, number = struct.unpack('<ii', fh.read(8))
    return tuple(CziEventListEntry(fh) for _ in range(number))


def read_time_stamps(fh: IO[bytes], size: int = 0, /) -> NDArray[Any]:
    """Return time stamps from file as float64 values.

    CZTIMS TimeStamps content schema.

    The unit depends on the CZI producer. Values may be seconds relative
    to the start of acquisition or OLE Automation Dates.

    Parameters:
        fh:
            File handle to read from.
        size:
            Size of attachment.

    """
    del size
    _, number = struct.unpack('<ii', fh.read(8))
    return read_array(fh, dtype='<f8', count=number)


def read_focus_positions(fh: IO[bytes], size: int = 0, /) -> NDArray[Any]:
    """Return focus positions from file in um relative to acquisition start.

    CZFOC FocusPositions content schema.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    del size
    _, number = struct.unpack('<ii', fh.read(8))
    return read_array(fh, dtype='<f8', count=number)


def read_lookup_table(
    fh: IO[bytes], size: int = 0, /
) -> tuple[tuple[str, NDArray[Any]], ...]:
    """Return sequence of lookup table identifiers and arrays from file.

    CZLUT LookupTables content schema.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    if size < 8:
        return ()
    start = fh.tell()
    end = start + size
    _, number_luts = struct.unpack('<ii', fh.read(8))
    luts = []
    for _ in range(number_luts):
        if fh.tell() + 88 > end:
            break
        # LookupTableEntry
        _, identifier, number_components = struct.unpack('<i80si', fh.read(88))
        components = [None] * number_components
        for _ in range(number_components):
            if fh.tell() + 12 > end:
                break
            # ComponentEntry
            _, component_type, number_ints = struct.unpack('<iii', fh.read(12))
            if fh.tell() + number_ints * 2 > end:
                break
            intensity = read_array(fh, dtype='<i2', count=number_ints)
            if component_type == -1:  # RGB
                # TODO: no test file available
                components = intensity.reshape((-1, 3)).T  # type: ignore[assignment]
                break
            # else:  # R, G, B
            components[component_type] = intensity
        luts.append(
            (
                identifier.rstrip(b'\x00').decode(),
                numpy.array(components, copy=True),
            )
        )
    return tuple(luts)


def read_xml(fh: IO[bytes], size: int, /) -> str:
    """Return XML from file.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    return fh.read(size).rstrip(b'\x00').decode()


def read_jpeg(fh: IO[bytes], size: int = 0, /) -> NDArray[Any]:
    """Return decoded JPEG image from file.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    return imagecodecs.jpeg8_decode(fh.read(size))


def read_czi(fh: IO[bytes], size: int = 0, /) -> NDArray[Any]:
    """Return image from embedded CZI file.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    return imread(io.BytesIO(fh.read(size)))


def read_gzip(fh: IO[bytes], size: int = 0, /) -> bytes:
    """Return decoded GZIP stream from file.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    return bytes(imagecodecs.gzip_decode(fh.read(size)))


def read_bytes(fh: IO[bytes], size: int, /) -> bytes:
    """Return bytes from file.

    Parameters:
        fh: File handle to read from.
        size: Size of attachment.

    """
    return bytes(fh.read(size))


class CziSegmentId(enum.StrEnum):
    """CziSegment identifier."""

    UNKNOWN = '', CziUnknownSegmentData
    """Unknown segment."""

    ZISRAWFILE = 'ZISRAWFILE', CziFileHeaderSegmentData
    """File header segment."""

    ZISRAWMETADATA = 'ZISRAWMETADATA', CziMetadataSegmentData
    """Metadata segment."""

    ZISRAWDIRECTORY = 'ZISRAWDIRECTORY', CziSubBlockDirectorySegmentData
    """SubBlock directory segment."""

    ZISRAWSUBBLOCK = 'ZISRAWSUBBLOCK', CziSubBlockSegmentData
    """SubBlock segment."""

    ZISRAWATTDIR = 'ZISRAWATTDIR', CziAttachmentDirectorySegmentData
    """Attachment directory segment."""

    ZISRAWATTACH = 'ZISRAWATTACH', CziAttachmentSegmentData
    """Attachment segment."""

    DELETED = 'DELETED', CziDeletedSegmentData
    """Deleted segment data."""

    read_segment_data: Callable[[CziSegment], CziSegmentData]
    """Callable that parses payload for this segment identifier."""

    def __new__(
        cls,
        value: str,
        read_segment_data: Callable[
            [CziSegment], CziSegmentData
        ] = CziUnknownSegmentData,
    ) -> Self:
        """Create enum member with associated segment-data reader."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.read_segment_data = read_segment_data
        return obj

    @property
    def packed(self) -> bytes:
        """Segment ID as 16-byte little-endian field.

        Return the identifier encoded as cp1252 and zero-padded to
        16 bytes, matching the on-disk segment header layout.

        """
        return self.encode('cp1252').ljust(16, b'\x00')

    @classmethod
    def _missing_(cls, value: object, /) -> Self:
        if not isinstance(value, str):
            msg = f'invalid CziSegment ID {value!r}'
            raise TypeError(msg)
        if not (value.isalnum() and value.isupper()):
            msg = f'invalid CziSegment ID {value!r}'
            raise ValueError(msg)
        logger().warning(f'unknown segment identifier {value!r}')
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._name_ = cls.UNKNOWN._name_
        obj.read_segment_data = cls.UNKNOWN.read_segment_data
        return obj


class CziContentFileType(enum.StrEnum):
    """Attachment file type identifier."""

    UNKNOWN = '', read_bytes
    """Unknown attachment type."""

    BINARY = 'BINARY', read_bytes
    """Binary attachment type."""

    CZI = 'CZI', read_czi
    """Embedded CZI file containing slide label, preview or pre-scan image."""

    ZISRAW = 'ZISRAW', read_czi
    """Embedded CZI file."""

    CZTIMS = 'CZTIMS', read_time_stamps
    """Time stamps. Unit depends on producing CZI writer."""

    CZEVL = 'CZEVL', read_event_list
    """Events reported during timeseries."""

    CZLUT = 'CZLUT', read_lookup_table
    """Color lookup tables."""

    CZFOC = 'CZFOC', read_focus_positions
    """Focus positions relative to acquisition start."""

    CZEXP = 'CZEXP', read_xml
    """Experiment definitions (XML)."""

    CZHWS = 'CZHWS', read_xml
    """HardwareSetting used to record image (XML)."""

    CZMVM = 'CZMVM', read_xml
    """MultiviewMicroscopy information (XML)."""

    CZFBMX = 'CZFBMX', read_xml
    """FiberMatrix (XML)."""

    CZPML = 'CZPML', read_bytes
    """PalMoleculeList information (undocumented)."""

    ZIP = 'ZIP', read_gzip
    """ZIP compressed attachment."""

    ZIP_COMP = 'Zip-Comp', read_gzip
    """ZIP compressed XML."""

    JPG = 'JPG', read_jpeg
    """JPEG compressed thumbnail or preview image."""

    read_content: Callable[..., Any]
    """Callable that decodes attachment content for this file type."""

    def __new__(
        cls,
        value: str,
        read_content: Callable[..., Any] = read_bytes,
    ) -> Self:
        """Create enum member with associated attachment-content reader."""
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj.read_content = read_content
        return obj

    @classmethod
    def _missing_(cls, value: object, /) -> CziContentFileType:
        # logger().warning(f'unknown content file type {value!r}')
        obj = str.__new__(cls, value if isinstance(value, str) else '')
        obj._value_ = value  # type: ignore[assignment]
        obj._name_ = cls.UNKNOWN._name_
        obj.read_content = cls.UNKNOWN.read_content
        return obj


class CziDimensionType(enum.StrEnum):
    """Single character code identifying dimension in pixel and metadata."""

    UNKNOWN = 'Q'  # formerly '?'
    """Unknown dimension."""

    WIDTH = 'X'
    """Pixel index in X direction used for tiled images."""

    HEIGHT = 'Y'
    """Pixel index in Y direction used for tiled images."""

    DEPTH = 'Z'
    """Slice index in Z direction."""

    CHANNEL = 'C'
    """Channel in multi-channel data set."""

    TIME = 'T'
    """Sequentially acquired series of data."""

    ROTATION = 'R'
    """Data is recorded from various angles."""

    SCENE = 'S'
    """Cluster index contiguous region in mosaic image."""

    ILLUMINATION = 'I'
    """Illumination direction index."""

    BLOCK = 'B'
    """Acquisition block index in segmented experiments."""

    MOSAIC = 'M'
    """Mosaic tile index identifying all tiles in specific plane."""

    PHASE = 'H'
    """Phase index for specific acquisition methods."""

    VIEW = 'V'
    """View index for multi-view images such as SPIM."""

    # SAMPLE = 'S'
    # """Color component."""

    @classmethod
    def _missing_(cls, value: object, /) -> CziDimensionType:
        logger().warning(f'unknown dimension type {value!r}')
        obj = str.__new__(cls, value if isinstance(value, str) else 'Q')
        obj._value_ = value  # type: ignore[assignment]
        obj._name_ = cls.UNKNOWN._name_
        return obj


PIXELTYPE: dict[tuple[numpy.dtype[Any], int], CziPixelType] = {}


class CziPixelType(enum.IntEnum):
    """Type of pixel data."""

    UNKNOWN = -1, '<u1', 1
    """Unknown pixel type."""

    GRAY8 = 0, '<u1', 1
    """8-bit unsigned integer."""

    GRAY16 = 1, '<u2', 1
    """16-bit unsigned integer."""

    GRAY32FLOAT = 2, '<f4', 1
    """32-bit floating point."""

    BGR24 = 3, '<u1', 3
    """8-bit unsigned integers for Blue, Green, Red color samples."""

    BGR48 = 4, '<u2', 3
    """16-bit unsigned integers for Blue, Green, Red color samples."""

    BGR96FLOAT = 8, '<f4', 3
    """32-bit floating points for Blue, Green, Red color samples."""

    BGRA32 = 9, '<u1', 4
    """8-bit unsigned integers for Blue, Green, Red, Alpha color samples."""

    GRAY64COMPLEXFLOAT = 10, '<c8', 1
    """64-bit complex number."""

    BGR192COMPLEXFLOAT = 11, '<c8', 3
    """64-bit complex numbers for Blue, Green, Red color samples."""

    GRAY32 = 12, '<i4', 1
    """32-bit signed integer."""

    GRAY64 = 13, '<f8', 1
    """64-bit floating point."""

    dtype: numpy.dtype[Any]
    """Numpy data type of pixel type."""

    samples: int
    """Number of samples in pixel type."""

    def __new__(
        cls, value: int, dtype: str = 'u1', samples: int = 1, /
    ) -> Self:
        """Create enum member with associated numpy dtype and sample count."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.dtype = numpy.dtype(dtype)
        obj.samples = samples
        PIXELTYPE[obj.dtype, samples] = obj
        return obj

    @classmethod
    def _missing_(cls, value: object, /) -> CziPixelType:
        logger().warning(f'unknown pixel type {value!r}')
        obj = int.__new__(cls, -1)
        obj._value_ = value  # type: ignore[assignment]
        obj._name_ = cls.UNKNOWN._name_
        obj.dtype = cls.UNKNOWN.dtype
        obj.samples = cls.UNKNOWN.samples
        return obj

    @staticmethod
    def get(dtype: DTypeLike, samples: int) -> CziPixelType:
        """Return pixel type matching dtype and samples."""
        return PIXELTYPE[numpy.dtype(dtype), samples]


class CziCompressionType(enum.IntEnum):
    """Pixel data compression type."""

    UNKNOWN = -1
    """Unknown compression type."""

    UNCOMPRESSED = 0
    """No compression."""

    JPEG = 1
    """JPEG lossy."""

    LZW = 2
    """Lempel-Ziv-Welch."""

    JPEGLL = 3
    """JPEG lossless (undocumented)."""

    JPEGXR = 4
    """JPEG XR."""

    ZSTD = 5
    """ZStandard."""

    ZSTDHDR = 6
    """ZStandard with header and shuffle filter."""

    CAMERARAW = 100  # 100-999
    """Camera specific RAW data."""

    SYSTEMRAW = 1000  # 1000-
    """System specific RAW data."""

    def __new__(cls, value: int, /) -> Self:
        """Create enum member."""
        obj = int.__new__(cls, value)
        obj._value_ = value
        return obj

    @classmethod
    def _missing_(cls, value: object, /) -> CziCompressionType:
        if not isinstance(value, int):
            msg = f'invalid compression type {value!r}'
            raise TypeError(msg)
        if value < 100:
            logger().warning(f'unknown compression type {value!r}')
            base = cls.UNKNOWN
        elif value < 1000:
            base = cls.CAMERARAW
        else:
            base = cls.SYSTEMRAW
        obj = int.__new__(cls, int(base))
        obj._value_ = value
        obj._name_ = base._name_
        return obj

    def __str__(self) -> str:
        return f'{self.name} ({self._value_})'

    @staticmethod
    @cache
    def decoder(value: int, /) -> Callable[..., Any] | None:
        """Return decoder function for compression type value.

        Resolved lazily on first call per value and cached.
        For JPEG XR (value 4), prefers WIC on Windows when available and
        falls back to :py:func:`imagecodecs.jpegxr_decode` otherwise.

        Parameters:
            value:
                Integer compression type value.

        Returns:
            Decoder callable, or ``None`` if no decoder is needed.

        """
        result: Callable[..., Any] | None
        match value:
            case 1:  # | 3
                result = imagecodecs.jpeg8_decode
            case 2:
                result = imagecodecs.lzw_decode
            case 4:
                if hasattr(imagecodecs, 'WIC') and imagecodecs.WIC.available:
                    result = imagecodecs.wic_decode
                else:
                    result = imagecodecs.jpegxr_decode
            case 5 | 6:
                result = imagecodecs.zstd_decode
            case _:
                result = None
        return result


class CziPyramidType(enum.IntEnum):
    """Image pyramid type using SubBlocks of different resolution.

    The use of these values is undocumented.

    """

    UNKNOWN = -1
    """Unknown pyramid type."""

    NONE = 0
    """Not pyramidal."""

    SINGLE_SUBBLOCK = 1
    """Single subblock pyramid."""

    MULTI_SUBBLOCK = 2
    """Multi subblock pyramid."""

    @classmethod
    def _missing_(cls, value: object, /) -> CziPyramidType:
        obj = int.__new__(cls, -1)
        obj._value_ = value  # type: ignore[assignment]
        obj._name_ = cls.UNKNOWN._name_
        return obj


class CziEventType(enum.IntEnum):
    """Type of CziEventListEntry (EV_TYPE)."""

    UNKNOWN = -1
    """Unknown event type."""

    MARKER = 0
    """Experimental annotation."""

    TIMER_CHANGE = 1
    """Time interval changed."""

    BLEACH_START = 2
    """Start of bleach operation."""

    BLEACH_STOP = 3
    """Stop of bleach operation."""

    TRIGGER = 4
    """Trigger signal detected on user port of electronic module."""

    @classmethod
    def _missing_(cls, value: object, /) -> CziEventType:
        obj = int.__new__(cls, -1)
        obj._value_ = value  # type: ignore[assignment]
        obj._name_ = cls.UNKNOWN._name_
        return obj


def convert_int_float(data: NDArray[Any], /) -> NDArray[Any]:
    """Return FLOAT from BYTE or WORD."""
    return data.astype(numpy.float32)


def convert_word_byte(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BYTE from WORD."""
    return (data >> 8).astype(numpy.uint8)


def convert_byte_word(data: NDArray[Any], /) -> NDArray[Any]:
    """Return WORD from BYTE."""
    return data.astype(numpy.uint16)


def convert_float_byte(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BYTE from FLOAT."""
    return numpy.clip(  # type: ignore[no-any-return]
        numpy.nan_to_num(data), 0, 255
    ).astype(numpy.uint8)


def convert_float_word(data: NDArray[Any], /) -> NDArray[Any]:
    """Return WORD from FLOAT."""
    return numpy.clip(  # type: ignore[no-any-return]
        numpy.nan_to_num(data), 0, 65535
    ).astype(numpy.uint16)


def convert_gray32float_bgr24(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR24 from GRAY32FLOAT."""
    return numpy.tile(
        numpy.clip(numpy.nan_to_num(data), 0, 255).astype(numpy.uint8),
        (1, 1, 3),
    )


def convert_gray32float_bgr48(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR48 from GRAY32FLOAT."""
    return numpy.tile(
        numpy.clip(numpy.nan_to_num(data), 0, 65535).astype(numpy.uint16),
        (1, 1, 3),
    )


def convert_gray32float_bgra32(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGRA32 from GRAY32FLOAT."""
    out = numpy.zeros((*data.shape[:-1], 4), numpy.uint8)
    out[..., :3] = numpy.clip(numpy.nan_to_num(data), 0, 255).astype(
        numpy.uint8
    )
    out[..., 3] = 255
    return out


def convert_bgr_gray8(data: NDArray[Any], /) -> NDArray[Any]:
    """Return GRAY8 from BGR(A)."""
    data = data[..., :3].sum(axis=-1, dtype=numpy.uint32, keepdims=True)
    return ((data + 1) // 3).astype(numpy.uint8)


def convert_bgr_gray16(data: NDArray[Any], /) -> NDArray[Any]:
    """Return GRAY16 from BGR(A)."""
    data = data[..., :3].sum(axis=-1, dtype=numpy.uint32, keepdims=True)
    return ((data + 1) // 3).astype(numpy.uint16)


def convert_bgr_gray32float(data: NDArray[Any], /) -> NDArray[Any]:
    """Return GRAY32FLOAT from BGR(A)."""
    data = data[..., :3].sum(axis=-1, dtype=numpy.float32, keepdims=True)
    data /= 3.0
    return data


def convert_gray_bgr(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR from GRAY keeping dtype."""
    return numpy.tile(data, (1, 1, 3))


def convert_gray_bgra(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGRA from GRAY keeping dtype."""
    out = numpy.zeros((*data.shape[:-1], 4), data.dtype)
    out[..., :3] = data
    out[..., 3] = numpy.iinfo(data.dtype).max
    return out


def convert_gray16_bgr24(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR24 from GRAY16."""
    return numpy.tile((data >> 8).astype(numpy.uint8), (1, 1, 3))


def convert_gray16_bgra32(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGRA32 from GRAY16."""
    out = numpy.zeros((*data.shape[:-1], 4), numpy.uint8)
    out[..., :3] = (data >> 8).astype(numpy.uint8)
    out[..., 3] = 255
    return out


def convert_bgr_bgra(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGRA from BGR keeping dtype."""
    out = numpy.zeros((*data.shape[:-1], 4), data.dtype)
    out[..., :3] = data
    out[..., 3] = numpy.iinfo(data.dtype).max
    return out


def convert_bgra_bgr(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR from BGRA keeping dtype."""
    return data[..., :3]


def convert_bgr48_bgra32(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGRA32 from BGR48."""
    out = numpy.zeros((*data.shape[:-1], 4), numpy.uint8)
    out[..., :3] = (data >> 8).astype(numpy.uint8)
    out[..., 3] = 255
    return out


def convert_bgra32_bgr48(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR48 from BGRA32."""
    data = data[..., :3].astype(numpy.uint16)
    data <<= 8
    return data


def convert_gray8_bgr48(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR48 from GRAY8."""
    return numpy.tile(data.astype(numpy.uint16), (1, 1, 3))


def convert_gray_bgr96float(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR96FLOAT from GRAY."""
    return numpy.tile(data.astype(numpy.float32), (1, 1, 3))


def convert_bgr96float_bgra32(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGRA32 from BGR96FLOAT."""
    out = numpy.zeros((*data.shape[:-1], 4), numpy.uint8)
    out[..., :3] = numpy.clip(data, 0, 255).astype(numpy.uint8)
    out[..., 3] = 255
    return out


def convert_bgra32_bgr96float(data: NDArray[Any], /) -> NDArray[Any]:
    """Return BGR96FLOAT from BGRA32."""
    return data[..., :3].astype(numpy.float32)


CONVERT_PIXELTYPE: Mapping[
    tuple[CziPixelType, CziPixelType], Callable[[NDArray[Any]], NDArray[Any]]
] = MappingProxyType(
    {
        # (CziPixelType.GRAY8, CziPixelType.GRAY8)
        (CziPixelType.GRAY8, CziPixelType.GRAY16): convert_byte_word,
        (CziPixelType.GRAY8, CziPixelType.GRAY32FLOAT): convert_int_float,
        (CziPixelType.GRAY8, CziPixelType.BGR24): convert_gray_bgr,
        (CziPixelType.GRAY8, CziPixelType.BGR48): convert_gray8_bgr48,
        (CziPixelType.GRAY8, CziPixelType.BGR96FLOAT): convert_gray_bgr96float,
        (CziPixelType.GRAY8, CziPixelType.BGRA32): convert_gray_bgra,
        (CziPixelType.GRAY16, CziPixelType.GRAY8): convert_word_byte,
        # (CziPixelType.GRAY16, CziPixelType.GRAY16)
        (CziPixelType.GRAY16, CziPixelType.GRAY32FLOAT): convert_int_float,
        (CziPixelType.GRAY16, CziPixelType.BGR24): convert_gray16_bgr24,
        (CziPixelType.GRAY16, CziPixelType.BGR48): convert_gray_bgr,
        (
            CziPixelType.GRAY16,
            CziPixelType.BGR96FLOAT,
        ): convert_gray_bgr96float,
        (CziPixelType.GRAY16, CziPixelType.BGRA32): convert_gray16_bgra32,
        (CziPixelType.GRAY32FLOAT, CziPixelType.GRAY8): convert_float_byte,
        (CziPixelType.GRAY32FLOAT, CziPixelType.GRAY16): convert_float_word,
        # (CziPixelType.GRAY32FLOAT, CziPixelType.GRAY32FLOAT)
        (
            CziPixelType.GRAY32FLOAT,
            CziPixelType.BGR24,
        ): convert_gray32float_bgr24,
        (
            CziPixelType.GRAY32FLOAT,
            CziPixelType.BGR48,
        ): convert_gray32float_bgr48,
        (CziPixelType.GRAY32FLOAT, CziPixelType.BGR96FLOAT): convert_gray_bgr,
        (
            CziPixelType.GRAY32FLOAT,
            CziPixelType.BGRA32,
        ): convert_gray32float_bgra32,
        (CziPixelType.BGR24, CziPixelType.GRAY8): convert_bgr_gray8,
        (CziPixelType.BGR24, CziPixelType.GRAY16): convert_bgr_gray16,
        (
            CziPixelType.BGR24,
            CziPixelType.GRAY32FLOAT,
        ): convert_bgr_gray32float,
        # (CziPixelType.BGR24, CziPixelType.BGR24)
        (CziPixelType.BGR24, CziPixelType.BGR48): convert_byte_word,
        (CziPixelType.BGR24, CziPixelType.BGR96FLOAT): convert_int_float,
        (CziPixelType.BGR24, CziPixelType.BGRA32): convert_bgr_bgra,
        (CziPixelType.BGR48, CziPixelType.GRAY8): convert_bgr_gray8,
        (CziPixelType.BGR48, CziPixelType.GRAY16): convert_bgr_gray16,
        (
            CziPixelType.BGR48,
            CziPixelType.GRAY32FLOAT,
        ): convert_bgr_gray32float,
        (CziPixelType.BGR48, CziPixelType.BGR24): convert_word_byte,
        # (CziPixelType.BGR48, CziPixelType.BGR48)
        (CziPixelType.BGR48, CziPixelType.BGR96FLOAT): convert_int_float,
        (CziPixelType.BGR48, CziPixelType.BGRA32): convert_bgr48_bgra32,
        (CziPixelType.BGR96FLOAT, CziPixelType.GRAY8): convert_bgr_gray8,
        (CziPixelType.BGR96FLOAT, CziPixelType.GRAY16): convert_bgr_gray16,
        (
            CziPixelType.BGR96FLOAT,
            CziPixelType.GRAY32FLOAT,
        ): convert_bgr_gray32float,
        (CziPixelType.BGR96FLOAT, CziPixelType.BGR24): convert_float_byte,
        (CziPixelType.BGR96FLOAT, CziPixelType.BGR48): convert_float_word,
        # (CziPixelType.BGR96FLOAT, CziPixelType.BGR96FLOAT)
        (
            CziPixelType.BGR96FLOAT,
            CziPixelType.BGRA32,
        ): convert_bgr96float_bgra32,
        (CziPixelType.BGRA32, CziPixelType.GRAY8): convert_bgr_gray8,
        (CziPixelType.BGRA32, CziPixelType.GRAY16): convert_bgr_gray16,
        (
            CziPixelType.BGRA32,
            CziPixelType.GRAY32FLOAT,
        ): convert_bgr_gray32float,
        (CziPixelType.BGRA32, CziPixelType.BGR24): convert_bgra_bgr,
        (CziPixelType.BGRA32, CziPixelType.BGR48): convert_bgra32_bgr48,
        (
            CziPixelType.BGRA32,
            CziPixelType.BGR96FLOAT,
        ): convert_bgra32_bgr96float,
        # (CziPixelType.BGRA32, CziPixelType.BGRA32)
    }
)
"""Map of (source, target) CziPixelType pairs to converter functions."""


def normalize_pyramid_scale(
    factors: tuple[float, ...], /
) -> tuple[float, ...]:
    """Snap near-integer scale factors to predominant exact-integer scale.

    CZI pyramid tiles are isotropically downsampled so all spatial dimensions
    share the same integer scale factor. Edge tiles at image boundaries have
    a ``stored_size`` written as ``ceil(size/factor)`` by ZEN, making their
    per-entry ``shape/stored_shape`` ratio slightly wrong (for example,
    1.994 instead of 2.0, or 30.08 instead of 32.0). Find the one exact-integer
    scale present and snap any other factor that is within 10 % of it to that
    value.

    """
    int_scales = sorted(
        {f for f in factors if f > 1.0 and abs(f - round(f)) < 0.0001}
    )
    if len(int_scales) != 1:
        return factors
    canonical = int_scales[0]
    return tuple(
        canonical if f != 1.0 and abs(f - canonical) / canonical < 0.10 else f
        for f in factors
    )


def filtered_shape(
    directory_entries: tuple[CziDirectoryEntryDV, ...],
    start: tuple[int, ...],
    /,
) -> tuple[int, ...]:
    """Return logical shape spanned by directory entries."""
    ref_dims = directory_entries[0].dims
    shapes = []
    for e in directory_entries:
        if e.dims == ref_dims:
            shapes.append(
                [s + sz for s, sz in zip(e.start, e.shape, strict=True)]
            )
        else:
            d_start = dict(zip(e.dims, e.start, strict=True))
            d_shape = dict(zip(e.dims, e.shape, strict=True))
            shapes.append(
                [d_start.get(d, 0) + d_shape.get(d, 1) for d in ref_dims]
            )
    return tuple(
        int(i - j)
        for i, j in zip(numpy.max(shapes, axis=0), start, strict=True)
    )
    # shape = shape + (subblocks[0].pixel_type.samples,)


def filtered_start(
    directory_entries: tuple[CziDirectoryEntryDV, ...], /
) -> tuple[int, ...]:
    """Return minimum start indices per dimension across entries."""
    ref_dims = directory_entries[0].dims
    starts = []
    for e in directory_entries:
        if e.dims == ref_dims:
            starts.append(e.start)
        else:
            d = dict(zip(e.dims, e.start, strict=True))
            starts.append(tuple(d.get(dim, 0) for dim in ref_dims))
    return tuple(int(x) for x in numpy.min(starts, axis=0))


def match_filename(filename: str, /) -> tuple[str, int]:
    """Return base CZI filename and parsed multipart index."""
    match_obj = re.search(
        r'(.*?)(?:\((\d+)\))?\.czi$', filename, re.IGNORECASE
    )
    if match_obj is None:
        msg = f'not a CZI file name: {filename!r}'
        raise ValueError(msg)
    match = match_obj.groups()
    name = match[0] + '.czi'
    part = int(match[1]) if match[1] is not None else 0
    return name, part


class DecodeCache:
    """FIFO subblock decode cache keyed by file offset.

    Caching is applied only to compressed subblocks.
    Disabled by default (maxsize=0).

    """

    __slots__ = ('_data', '_lock', '_maxsize')

    _data: dict[int, NDArray[Any]] | None
    _lock: threading.Lock | NullLock
    _maxsize: int

    def __init__(
        self,
        maxsize: int = 0,
        lock: threading.Lock | NullLock | None = None,
    ) -> None:
        if lock is None:
            # use a real lock only when the GIL is disabled
            # (free-threaded Python 3.13+)
            if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled():
                lock = threading.Lock()
            else:
                lock = NullLock()
        self._lock = lock
        self._data = {} if maxsize > 0 else None
        self._maxsize = max(0, maxsize)

    def get(self, key: int, /) -> NDArray[Any] | None:
        """Return cached array for key, or None if absent or disabled."""
        data = self._data
        if data is None:
            return None
        return data.get(key)

    def put(self, key: int, value: NDArray[Any], /) -> None:
        """Store value under key, evicting oldest entry if at capacity."""
        data = self._data
        if data is None:
            return
        with self._lock:
            if len(data) >= self._maxsize:
                data.pop(next(iter(data)))
            data[key] = value

    @property
    def maxsize(self) -> int:
        """Maximum number of cache entries, or 0 if disabled."""
        return self._maxsize

    def resize(self, maxsize: int, /) -> None:
        """Enable, resize, or disable the cache.

        Parameters:
            maxsize:
                Maximum number of entries. 0 disables and clears cache.

        Raises:
            ValueError: maxsize is negative.

        """
        if maxsize < 0:
            msg = f'maxsize must be >= 0, got {maxsize}'
            raise ValueError(msg)
        if maxsize == 0:
            self._data = None
            self._maxsize = 0
        else:
            if self._data is None:
                self._data = {}
            elif len(self._data) > maxsize:
                with self._lock:
                    while len(self._data) > maxsize:
                        self._data.pop(next(iter(self._data)))
            self._maxsize = maxsize

    @contextlib.contextmanager
    def scoped_resize(self, maxsize: int, /) -> Iterator[None]:
        """Context manager that temporarily changes the cache size.

        Restores previous size and cached entries on exit, including
        when a generator using this context is closed.
        When maxsize is 0, the cache is left unchanged.

        Parameters:
            maxsize:
                Temporary cache size. 0 to leave unchanged.

        """
        if maxsize == 0:
            yield
            return
        prev_data = self._data
        prev_maxsize = self._maxsize
        self.resize(maxsize)
        try:
            yield
        finally:
            self._data = prev_data
            self._maxsize = prev_maxsize


class NullLock:
    """No-op context manager used as default lock."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        pass


def logger() -> logging.Logger:
    """Return module logger."""
    return logging.getLogger('czifile')


def read_array(
    fh: IO[bytes],
    dtype: DTypeLike | None,
    count: int,
    offset: int = 0,
    *,
    out: NDArray[Any] | None = None,
) -> NDArray[Any]:
    """Return NumPy array from file in native byte order.

    Parameters:
        fh:
            File handle to read from.
        dtype:
            Data type of array to read.
        count:
            Number of elements to read.
        offset:
            Start position of array-data in file.
        out:
            NumPy array to read into. By default, a new array is created.

    """
    dtype = numpy.dtype(dtype)
    nbytes = count * dtype.itemsize

    result = numpy.empty(count, dtype) if out is None else out

    if result.nbytes != nbytes:
        msg = 'size mismatch'
        raise ValueError(msg)

    if offset:
        fh.seek(offset)

    try:
        n = fh.readinto(result)  # type: ignore[attr-defined]
    except AttributeError:
        result[:] = numpy.frombuffer(fh.read(nbytes), dtype).reshape(
            result.shape
        )
        n = nbytes

    if n != nbytes:
        msg = f'failed to read {nbytes} bytes, got {n}'
        raise ValueError(msg)

    if not result.dtype.isnative:
        if not dtype.isnative:
            result.byteswap(inplace=True)
        result = result.view(result.dtype.newbyteorder())
    elif result.dtype.isnative != dtype.isnative:
        result.byteswap(inplace=True)

    if out is not None and hasattr(out, 'flush'):
        out.flush()

    return result


def create_output(
    out: OutputType,
    /,
    shape: Sequence[int],
    dtype: DTypeLike | None,
    *,
    mode: Literal['r+', 'w+', 'r', 'c'] = 'w+',
    suffix: str | None = None,
    fillvalue: ArrayLike | None = None,
) -> NDArray[Any] | numpy.memmap[Any, Any]:
    """Return NumPy array where data of shape and dtype can be copied.

    Parameters:
        out:
            Kind of array of `shape` and `dtype` to return:

                `None`:
                    Return new array.
                `numpy.ndarray`:
                    Return view of existing array.
                `'memmap'` or `'memmap:tempdir'`:
                    Return memory-map to array stored in temporary binary file.
                `str` or open file:
                    Return memory-map to array stored in specified binary file.
        shape:
            Shape of array to return.
        dtype:
            Data type of array to return.
            If `out` is an existing array, `dtype` must be castable to its
            data type.
        mode:
            File mode to create memory-mapped array.
            Default: ``'w+'``. Creates new or overwrites existing file for
            reading and writing.
        suffix:
            Suffix of `NamedTemporaryFile` if `out` is ``'memmap'``.
            Default: ``'.memmap'``.
        fillvalue:
            Value to initialize output array.
            By default, return uninitialized array.

    Returns:
        NumPy array or memory-mapped array of `shape` and `dtype`.

    Raises:
        ValueError:
            Existing array cannot be reshaped to `shape` or cast to `dtype`.

    """
    shape = tuple(shape)
    dtype = numpy.dtype(dtype)
    if out is None:
        if fillvalue is None:
            return numpy.empty(shape, dtype)
        if isinstance(fillvalue, (int, float)) and fillvalue == 0:
            return numpy.zeros(shape, dtype)
        return numpy.full(shape, fillvalue, dtype)
    if isinstance(out, numpy.ndarray):
        if product(shape) != product(out.shape):
            msg = f'cannot reshape {shape} to {out.shape}'
            raise ValueError(msg)
        if not numpy.can_cast(dtype, out.dtype):
            msg = f'cannot cast {dtype} to {out.dtype}'
            raise ValueError(msg)
        if out.shape != shape:
            out = out.reshape(shape)
        if fillvalue is not None:
            out[...] = fillvalue
        return out
    if isinstance(out, str) and out[:6] == 'memmap':
        import tempfile

        tempdir = out[7:] if len(out) > 7 else None
        if suffix is None:
            suffix = '.memmap'
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix=suffix) as fh:
            out = numpy.memmap(fh, shape=shape, dtype=dtype, mode=mode)
            if fillvalue is not None:
                out[...] = fillvalue
            return out
    out = numpy.memmap(out, shape=shape, dtype=dtype, mode=mode)
    if fillvalue is not None:
        out[...] = fillvalue
    return out


def product(iterable: Iterable[int], /) -> int:
    """Return product of integers.

    Like math.prod, but does not overflow with numpy arrays.

    """
    prod = 1
    for i in iterable:
        prod *= int(i)
    return prod


def parse_kwargs(
    kwargs: dict[str, Any], /, *keys: str, **keyvalues: Any
) -> dict[str, Any]:
    """Extract keys from kwargs into a new dict, removing them from kwargs.

    Keys listed in `keys` are extracted by name only.
    Keys listed in `keyvalues` are extracted by name if present in `kwargs`,
    otherwise their default value is used.
    Extracted keys are deleted from `kwargs`.

    >>> kwargs = {'one': 1, 'two': 2, 'four': 4}
    >>> parse_kwargs(kwargs, 'two', 'three', four=None, five=5)
    {'two': 2, 'four': 4, 'five': 5}
    >>> kwargs
    {'one': 1}
    >>> parse_kwargs({}, 'a', b=2)
    {'b': 2}

    """
    result = {}
    for key in keys:
        if key in kwargs:
            result[key] = kwargs.pop(key)
    for key, value in keyvalues.items():
        if key in kwargs:
            result[key] = kwargs.pop(key)
        else:
            result[key] = value
    return result


def xml2dict(
    xml: ElementTree.Element | str,
    /,
    *,
    sanitize: bool = True,
    prefix: tuple[str, str] | None = None,
    sep: str = ',',
) -> dict[str, Any]:
    """Return XML as dictionary.

    Parameters:
        xml: XML element to convert.
        sanitize: Remove namespace prefix from etree element tags.
        prefix: Prefixes for dictionary keys.
        sep: Sequence separator. Pass empty string to disable sequence parsing.

    Examples:
        >>> xml2dict(
        ...     '<?xml version="1.0" ?><root attr="name"><key>1</key></root>'
        ... )
        {'root': {'key': 1, 'attr': 'name'}}
        >>> xml2dict('<level1><level2>3.5322,-3.14</level2></level1>')
        {'level1': {'level2': (3.5322, -3.14)}}

    """
    at, tx = prefix or ('', '')

    def astype(value: Any, /) -> Any:
        # return string value as int, float, bool, tuple, or unchanged
        if not isinstance(value, str):
            return value
        if sep and sep in value:
            # sequence of numbers?
            values = []
            for val in value.split(sep):
                v = astype(val)
                if isinstance(v, str):
                    return value
                values.append(v)
            return tuple(values)  # may contain mixed int/float/bool types
        for t in (int, float, asbool):
            try:
                return t(value)
            except (TypeError, ValueError):
                pass
        return value

    def etree2dict(t: ElementTree.Element, /) -> dict[str, Any]:
        # adapted from https://stackoverflow.com/a/10077069/453463
        key = t.tag
        if sanitize:
            key = key.rsplit('}', 1)[-1]
        d: dict[str, Any] = {key: None}
        children = list(t)
        if children:
            dd = collections.defaultdict(list)
            for dc in map(etree2dict, children):
                for k, v in dc.items():
                    dd[k].append(v)
            d = {key: {k: v[0] if len(v) == 1 else v for k, v in dd.items()}}
        if t.attrib:
            if not isinstance(d[key], dict):
                d[key] = {}
            d[key].update((at + k, astype(v)) for k, v in t.attrib.items())
        if t.text:
            text = t.text.strip()
            if children or t.attrib:
                if text:
                    d[key][tx + 'value'] = astype(text)
            else:
                d[key] = astype(text)
        return d

    if isinstance(xml, str):
        xml = ElementTree.fromstring(xml)
    return etree2dict(xml)


@overload
def asbool(
    value: str,
    /,
    true: Sequence[str] | None = None,
    false: Sequence[str] | None = None,
) -> bool: ...


@overload
def asbool(
    value: bytes,
    /,
    true: Sequence[bytes] | None = None,
    false: Sequence[bytes] | None = None,
) -> bool: ...


def asbool(
    value: str | bytes,
    /,
    true: Sequence[str | bytes] | None = None,
    false: Sequence[str | bytes] | None = None,
) -> bool:
    """Return string as bool if possible, else raise TypeError.

    Custom `true` and `false` sequences must contain lowercase strings.

    >>> asbool(b' False ')
    False
    >>> asbool('ON', ['on'], ['off'])
    True

    """
    value = value.strip().lower()
    true_vals = (
        true
        if true is not None
        else (b'true' if isinstance(value, bytes) else ('true',))
    )
    false_vals = (
        false
        if false is not None
        else (b'false' if isinstance(value, bytes) else ('false',))
    )
    if value in true_vals:
        return True
    if value in false_vals:
        return False
    msg = f'{value!r} is not a recognized boolean value'
    raise TypeError(msg)


def indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    # [2:] removes leading indent from first line
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


@cache
def default_maxworkers() -> int:
    """Return default maximum number of worker threads.

    Uses the ``CZIFILE_NUM_THREADS`` environment variable if set,
    else half the CPU cores (up to 32), with a minimum of 1.

    """
    if 'CZIFILE_NUM_THREADS' in os.environ:
        try:
            return max(1, int(os.environ['CZIFILE_NUM_THREADS']))
        except ValueError:
            pass
    cpu_count: int | None = None
    try:
        cpu_count = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        cpu_count = os.cpu_count()
    if cpu_count is None:
        return 1
    return min(32, max(1, cpu_count // 2))


def askopenfilename(**kwargs: Any) -> str:
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


FILE_EXTENSIONS = {
    '.czi': 'CZI files',
}
"""Supported file extensions of CZI files."""


def main(argv: list[str] | None = None) -> int:
    """Command line usage main function.

    Preview image and metadata in specified files or all files in directory.

    ``python -m czifile file_or_directory``

    """
    import argparse
    from glob import glob

    imshow: Any
    try:
        from matplotlib import pyplot
        from tifffile import imshow
    except ImportError:
        imshow = None

    xarray: Any
    try:
        import xarray
    except ImportError:
        xarray = None

    def _parse_dim_value(s: str) -> int | slice | list[int] | None:
        """Parse dimension selection from a CLI string.

        Accepted forms:
            none       -> None (all coordinates)
            3          -> int
            1:10       -> slice(1, 10)
            0:10:2     -> slice(0, 10, 2)
            0,2,4      -> [0, 2, 4]

        """
        if s.lower() in ('none', ''):
            return None
        if ':' in s:
            parts = [int(p) if p else None for p in s.split(':')]
            return slice(*parts)
        if ',' in s:
            return [int(p) for p in s.split(',')]
        return int(s)

    class _DimAction(argparse.Action):
        """Record dimension arguments in the order they appear on the CLI."""

        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any,
            option_string: str | None = None,
        ) -> None:
            if not hasattr(namespace, '_dim_order'):
                namespace._dim_order = []
            namespace._dim_order.append((self.dest, values))
            setattr(namespace, self.dest, values)

    parser = argparse.ArgumentParser(
        prog='czifile',
        description='Preview image and metadata from CZI files.',
    )
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}',
    )
    parser.add_argument(
        'files',
        nargs='*',
        metavar='file_or_directory',
        help='CZI file(s) or directory to preview. '
        'Supports glob patterns. If omitted, a file dialog is shown.',
    )
    parser.add_argument(
        '--maxplots',
        type=int,
        default=10,
        metavar='N',
        help='Maximum number of scenes to plot per file (default: 10).',
    )
    parser.add_argument(
        '--scene',
        type=int,
        default=None,
        metavar='INDEX',
        help='Scene index to display. If omitted, all scenes are shown.',
    )
    parser.add_argument(
        '--merge',
        action='store_true',
        default=False,
        help='Merge all scenes into a single image instead of plotting each '
        'scene separately. Ignored when --scene is set.',
    )
    parser.add_argument(
        '--level',
        type=int,
        default=None,
        metavar='INDEX',
        help='Pyramid level index to use. '
        'If omitted, the first level that fits in memory is selected.',
    )
    parser.add_argument(
        '--pixeltype',
        default=None,
        metavar='TYPE',
        choices=[pt.name for pt in CziPixelType if pt != CziPixelType.UNKNOWN],
        help='Convert pixel data to this type before display '
        '(for example, GRAY8, GRAY16, BGR24). '
        'If omitted, the native pixel type is used.',
    )
    parser.add_argument(
        '--storedsize',
        action='store_true',
        default=False,
        help='Return pixel data at stored resolution. '
        'Skip resampling of Airyscan and PALM tiles.',
    )
    _dim_help = (
        'Select {name} dimension. '
        'Accepted values: integer index, start:stop[:step] slice, '
        'comma-separated list of indices, or "none" for all coordinates. '
        'When two or more dimension flags are given, their order on the '
        'command line determines the axis order in the output array.'
    )
    _dim_names: dict[str, str] = {
        'T': 'time',
        'Z': 'Z-stack',
        'C': 'channel',
        'H': 'phase',
        'R': 'rotation',
        'I': 'illumination',
        'B': 'block',
        'M': 'mosaic tile',
        'A': 'acquisition',
        'V': 'view',
    }
    for _dim, _name in _dim_names.items():
        parser.add_argument(
            f'--{_dim}',
            dest=_dim,
            type=_parse_dim_value,
            default=argparse.SUPPRESS,
            metavar='SEL',
            action=_DimAction,
            help=_dim_help.format(name=_name),
        )

    args = parser.parse_intermixed_args(argv[1:] if argv is not None else None)

    pixeltype: CziPixelType | None = (
        CziPixelType[args.pixeltype] if args.pixeltype is not None else None
    )
    selection: dict[str, int | slice | list[int] | None] = dict(
        getattr(args, '_dim_order', [])
    )

    if not args.files:
        path = askopenfilename(
            title='Select a CZI file',
            filetypes=[
                (f'{desc}', f'*{ext}') for ext, desc in FILE_EXTENSIONS.items()
            ]
            + [('All files', '*')],
        )
        files = [path] if path else []
    elif len(args.files) == 1 and '*' in args.files[0]:
        files = glob(args.files[0])
    elif len(args.files) == 1 and os.path.isdir(args.files[0]):
        files = [
            f
            for ext in FILE_EXTENSIONS
            for f in glob(f'{args.files[0]}/**/*{ext}', recursive=True)
        ]
    else:
        files = args.files

    def _show_czi(czi: CziImage, /, *, plot: bool = True) -> bool:
        """Print and optionally plot one CziImage. Return True if plotted."""
        print(czi)
        print()
        try:
            if xarray is not None:
                xa = czi.asxarray()
                data = xa.data
                print(xa)
            else:
                data = czi.asarray()
                print(data)
            print()
        except Exception as exc:
            print(czi._parent.name, exc)
            return False
        if not plot or imshow is None or data.ndim < 2:
            return False
        try:
            pm = 'RGB' if czi.dims[-1:] == ('S',) else 'MINISBLACK'
            imshow(
                data,
                title=repr(czi),
                show=False,
                photometric=pm,
                interpolation='None',
            )
        except Exception as exc:
            print(czi._parent.name, exc)
            return False
        return True

    for fname in files:
        try:
            czi_plotted = False
            with CziFile(fname) as czi:
                print(czi)
                print()
                plot_count = 0
                scenes = czi.scenes
                if args.merge and args.scene is None:
                    scene_iter: Any = [scenes()]
                else:
                    scene_iter = (
                        [scenes[args.scene]]
                        if args.scene is not None
                        else list(scenes.values())
                    )
                for image in scene_iter:
                    if args.level is not None:
                        image = image.levels[args.level]  # noqa: PLW2901
                    elif image.is_pyramid and image.nbytes > 2**31:
                        for lvl in image.levels[1:]:
                            if lvl.nbytes < 2**31:
                                image = lvl  # noqa: PLW2901
                                break
                    plotted = _show_czi(
                        image(
                            pixeltype=pixeltype,
                            storedsize=args.storedsize,
                            **selection,
                        ),
                        plot=plot_count < args.maxplots,
                    )
                    if plotted:
                        plot_count += 1
                        czi_plotted = True
            if czi_plotted:
                pyplot.show()
        except Exception:
            import traceback

            print('Failed to read', fname)
            traceback.print_exc()
            print()
            continue

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
