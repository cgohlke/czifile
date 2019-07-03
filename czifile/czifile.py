# -*- coding: utf-8 -*-
# czifile.py

# Copyright (c) 2013-2019, Christoph Gohlke
# Copyright (c) 2013-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
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

"""Read Carl Zeiss(r) Image (CZI) files.

Czifile is a Python library to read Carl Zeiss Image (CZI) files, the native
file format of the ZEN(r) software by Carl Zeiss Microscopy GmbH. CZI files
contain multidimensional images and metadata from microscopy experiments.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: 3-clause BSD

:Version: 2019.7.2

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* `Tifffile 2019.7.2 <https://pypi.org/project/tifffile/>`_
* `Imagecodecs 2019.5.22 <https://pypi.org/project/imagecodecs/>`_
  (optional; used for decoding LZW, JPEG, and JPEG XR)

Revisions
---------
2019.7.2
    Require tifffile 2019.7.2.
2019.6.18
    Add package main function to view CZI files.
    Fix BGR to RGB conversion.
    Fix czi2tif conversion on Python 2.
2019.5.22
    Fix czi2tif conversion when CZI metadata contain non-ASCII characters.
    Use imagecodecs_lite as a fallback for imagecodecs.
    Make CziFile.metadata a function (breaking).
    Make scipy an optional dependency; fallback on ndimage or fail on zoom().
2019.1.26
    Fix czi2tif console script.
    Update copyright year.
2018.10.18
    Rename zisraw package to czifile.
2018.8.29
    Move czifile.py and related modules into zisraw package.
    Move usage examples to main docstring.
    Require imagecodecs package for decoding JpegXrFile, JpgFile, and LZW.
2018.6.18
    Save CZI metadata to TIFF description in czi2tif.
    Fix AttributeError using max_workers=1.
    Make Segment.SID and DimensionEntryDV1.dimension str types.
    Return metadata as XML unicode string or dict, not etree.
    Return timestamps, focus positions, events, and luts as tuple or ndarray
2017.7.21
    Use multi-threading in CziFile.asarray to decode and copy segment data.
    Always convert BGR to RGB. Remove bgr2rgb options.
    Decode JpegXR directly from byte arrays.
2017.7.13
    Add function to convert CZI file to memory-mappable TIFF file.
2017.7.11
    Add 'out' parameter to CziFile.asarray.
    Remove memmap option from CziFile.asarray (breaking).
    Change spline interpolation order to 0 (breaking).
    Make axes return a string.
    Require tifffile 2017.7.11.
2014.10.10
    Read data into a memory mapped array (optional).
2013.12.4
    Decode JpegXrFile and JpgFile via _czifle extension module.
    Attempt to reconstruct tiled mosaic images.
2013.11.20
    Initial release.

Notes
-----
The API is not stable yet and might change between revisions.

Python 2.7 and 3.4 are deprecated.

"ZEISS" and "Carl Zeiss" are registered trademarks of Carl Zeiss AG.

The ZISRAW file format design specification [1] is confidential and the
license agreement does not permit to write data into CZI files.

Only a subset of the 2016 specification is implemented. Specifically,
multifile images, image pyramids, and topography images are not yet supported.

Tested on Windows with a few example files only.

Czifile relies on the `imagecodecs <https://pypi.org/project/imagecodecs/>`_
package for decoding LZW, JPEG, and JPEG XR compressed images. Alternatively,
the `imagecodecs_lite <https://pypi.org/project/imagecodecs_lite/>`_ package
can be used for decoding LZW compressed images.

Other libraries for reading CZI files (all GPL licensed):

* `libCZI <https://github.com/zeiss-microscopy/libCZI>`_
* `Python-bioformats <https://github.com/CellProfiler/python-bioformats>`_
* `Pylibczi <https://github.com/elhuhdron/pylibczi>`_

References
----------
1) ZISRAW (CZI) File Format Design Specification Release Version 1.2.2.
   CZI 07-2016/CZI-DOC ZEN 2.3/DS_ZISRAW-FileFormat.pdf (confidential).
   Documentation can be requested at
   `<https://www.zeiss.com/microscopy/us/products/microscope-software/zen/
   czi.html>`_

Examples
--------

Read image data from a CZI file as numpy array:

>>> image = imread('test.czi')
>>> image.shape
(3, 3, 3, 250, 200, 3)
>>> image[0, 0, 0, 0, 0]
array([10, 10, 10], dtype=uint8)

"""

from __future__ import division, print_function

__version__ = '2019.7.2'
__docformat__ = 'restructuredtext en'
__all__ = (
    'imread',
    'CziFile',
    'czi2tif',
    'Segment',
    'SegmentNotFoundError',
    'FileHeaderSegment',
    'MetadataSegment',
    'SubBlockSegment',
    'SubBlockDirectorySegment',
    'DirectoryEntryDV',
    'DimensionEntryDV1',
    'AttachmentSegment',
    'AttachmentEntryA1',
    'AttachmentDirectorySegment',
    'DeletedSegment',
    'UnknownSegment',
    'EventListEntry',
)

import os
import sys
import re
import uuid
import struct
import warnings
import multiprocessing

from concurrent.futures import ThreadPoolExecutor

import numpy

try:
    # TODO: use zoom fom imagecodecs implementation when available
    from scipy.ndimage.interpolation import zoom
except ImportError:
    try:
        from ndimage.interpolation import zoom
    except ImportError:
        zoom = None

try:
    import imagecodecs
except ImportError:
    try:
        import imagecodecs_lite as imagecodecs
    except ImportError:
        imagecodecs = None

from tifffile import (
    FileHandle, memmap, lazyattr, repeat_nd, product, stripnull, format_size,
    squeeze_axes, create_output, xml2dict, pformat, imshow, askopenfilename,
    nullfunc, Timer)


def imread(filename, *args, **kwargs):
    """Return image data from CZI file as numpy array.

    'args' and 'kwargs' are arguments to the CziFile.asarray function.

    """
    with CziFile(filename) as czi:
        result = czi.asarray(*args, **kwargs)
    return result


class CziFile(object):
    """Read Carl Zeiss Image (CZI) file.

    Attributes
    ----------
    header : FileHeaderSegment
        Global file metadata such as file version and GUID.
    metadata : str
        Global image metadata in UTF-8 encoded XML format.

    All attributes are read-only.

    """

    def __init__(self, arg, multifile=True, filesize=None, detectmosaic=True):
        """Open CZI file and read header.

        Raise ValueError if file is not a ZISRAW file.

        Parameters
        ----------
        multifile : bool
            If True (default), the master file of a multi-file container
            will be opened if applicable.
        filesize : int
            Size of file if arg is a file handle pointing to an
            embedded CZI file.
        detectmosaic : bool
            If True (default), mosaic images will be reconstructed from
            SubBlocks with a tile index.

        Notes
        -----
        CziFile instances created from file name must be closed using the
        'close' method, which is automatically called when using the
        'with' statement.

        """
        self._fh = FileHandle(arg, size=filesize)
        try:
            if self._fh.read(10) != b'ZISRAWFILE':
                raise ValueError('not a CZI file')
            self.header = Segment(self._fh, 0).data()
        except Exception:
            self._fh.close()
            raise

        if multifile and self.header.file_part and isinstance(arg, basestring):
            # open master file instead
            self._fh.close()
            name, _ = match_filename(arg)
            self._fh = FileHandle(name)
            self.header = Segment(self._fh, 0).data()
            assert self.header.primary_file_guid == self.header.file_guid
            assert self.header.file_part == 0

        if self.header.update_pending:
            warnings.warn('file is pending update')
        self._filter_mosaic = detectmosaic

    def segments(self, kind=None):
        """Return iterator over Segment data of specified kind.

        Parameters
        ----------
        kind : bytestring or sequence thereof
            Segment id(s) as listed in SEGMENT_ID.
            If None (default), all segments are returned.

        """
        fpos = 0
        while True:
            self._fh.seek(fpos)
            try:
                segment = Segment(self._fh)
            except SegmentNotFoundError:
                break
            if (kind is None) or (segment.sid in kind):
                yield segment.data()
            fpos = segment.data_offset + segment.allocated_size

    def metadata(self, raw=True):
        """Return data from MetadataSegment as XML (default) or dict.

        Return None if no Metadata segment is found.

        """
        if self.header.metadata_position:
            segment = Segment(self._fh, self.header.metadata_position)
            if segment.sid == MetadataSegment.SID:
                return segment.data().data(raw=raw)
        warnings.warn('Metadata segment not found')
        try:
            metadata = next(self.segments(MetadataSegment.SID))
            return metadata.data(raw=raw)
        except StopIteration:
            pass

    @lazyattr
    def subblock_directory(self):
        """Return list of all DirectoryEntryDV in file.

        Use SubBlockDirectorySegment if exists, else find SubBlockSegments.

        """
        if self.header.directory_position:
            segment = Segment(self._fh, self.header.directory_position)
            if segment.sid == SubBlockDirectorySegment.SID:
                return segment.data().entries
        warnings.warn('SubBlockDirectory segment not found')
        return list(segment.directory_entry for segment in
                    self.segments(SubBlockSegment.SID))

    @lazyattr
    def attachment_directory(self):
        """Return list of all AttachmentEntryA1 in file.

        Use AttachmentDirectorySegment if exists, else find AttachmentSegments.

        """
        if self.header.attachment_directory_position:
            segment = Segment(self._fh,
                              self.header.attachment_directory_position)
            if segment.sid == AttachmentDirectorySegment.SID:
                return segment.data().entries
        warnings.warn('AttachmentDirectory segment not found')
        return list(segment.attachment_entry for segment in
                    self.segments(AttachmentSegment.SID))

    def subblocks(self):
        """Return iterator over all SubBlock segments in file."""
        for entry in self.subblock_directory:
            yield entry.data_segment()

    def attachments(self):
        """Return iterator over all Attachment segments in file."""
        for entry in self.attachment_directory:
            yield entry.data_segment()

    def save_attachments(self, directory=None):
        """Save all attachments to files."""
        if directory is None:
            directory = self._fh.path + '.attachments'
        if not os.path.exists(directory):
            os.makedirs(directory)
        for attachment in self.attachments():
            attachment.save(directory=directory)

    @lazyattr
    def filtered_subblock_directory(self):
        """Return sorted list of DirectoryEntryDV if mosaic, else all."""
        if not self._filter_mosaic:
            return self.subblock_directory
        filtered = [directory_entry
                    for directory_entry in self.subblock_directory
                    if directory_entry.mosaic_index is not None]
        if not filtered:
            return self.subblock_directory
        return list(sorted(filtered, key=lambda x: x.mosaic_index))

    @lazyattr
    def shape(self):
        """Return shape of image data in file."""
        shape = [[dim.start + dim.size
                  for dim in directory_entry.dimension_entries
                  if dim.dimension != 'M']
                 for directory_entry in self.filtered_subblock_directory]
        shape = numpy.max(shape, axis=0)
        shape = tuple(int(i - j) for i, j in zip(shape, self.start[:-1]))
        dtype = self.filtered_subblock_directory[0].dtype
        sampleshape = numpy.dtype(dtype).shape
        shape = shape + (sampleshape if sampleshape else (1,))
        return shape

    @lazyattr
    def start(self):
        """Return minimum start indices per dimension of sub images in file."""
        start = [[dim.start
                  for dim in directory_entry.dimension_entries
                  if dim.dimension != 'M']
                 for directory_entry in self.filtered_subblock_directory]
        start = tuple(numpy.min(start, axis=0)) + (0,)
        return start

    @lazyattr
    def axes(self):
        """Return axes of image data in file."""
        return self.filtered_subblock_directory[0].axes

    @lazyattr
    def dtype(self):
        """Return numpy dtype of image data in file."""
        # subblock data can be of different pixel type
        dtype = numpy.dtype(self.filtered_subblock_directory[0].dtype[-2:])
        for directory_entry in self.filtered_subblock_directory:
            dtype = numpy.promote_types(dtype, directory_entry.dtype[-2:])
        return dtype

    def asarray(self, resize=True, order=0, out=None, max_workers=None):
        """Return image data from file(s) as numpy array.

        Parameters
        ----------
        resize : bool
            If True (default), resize sub/supersampled subblock data.
        order : int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers : int
            Maximum number of threads to read and decode subblock data.
            By default up to half the CPU cores are used.

        """
        out = create_output(out, self.shape, self.dtype)

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() // 2

        def func(directory_entry, resize=resize, order=order,
                 start=self.start, out=out):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            tile = subblock.data(resize=resize, order=order)
            index = tuple(slice(i - j, i - j + k) for i, j, k in
                          zip(directory_entry.start, start, tile.shape))
            try:
                out[index] = tile
            except ValueError as e:
                warnings.warn(str(e))

        if max_workers > 1:
            self._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:
            for directory_entry in self.filtered_subblock_directory:
                func(directory_entry)

        if hasattr(out, 'flush'):
            out.flush()
        return out

    def close(self):
        """Close file handle."""
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __str__(self):
        return '\n '.join((
            self._fh.name.capitalize(),
            '(Carl Zeiss Image File)',
            str(self.header),
            'MetadataSegment',
            str(self.axes),
            str(self.shape),
            str(self.dtype),
            pformat(self.metadata()),
        ))


class Segment(object):
    """ZISRAW Segment."""

    __slots__ = 'sid', 'allocated_size', 'used_size', 'data_offset', '_fh'

    def __init__(self, fh, fpos=None):
        """Read segment header from file."""
        if fpos is not None:
            fh.seek(fpos)
        try:
            (self.sid,
             self.allocated_size,
             self.used_size
             ) = struct.unpack('<16sqq', fh.read(32))
        except struct.error:
            raise SegmentNotFoundError('can not read ZISRAW segment')
        self.sid = bytes2str(stripnull(self.sid))
        if self.sid not in SEGMENT_ID:
            if not self.sid.startswith('ZISRAW'):
                raise SegmentNotFoundError('not a ZISRAW segment')
            warnings.warn('unknown segment type %s' % self.sid)
        self.data_offset = fh.tell()
        self._fh = fh
        if self.used_size == 0:
            self.used_size = self.allocated_size

    def data(self):
        """Read segment data from file and return as *Segment instance."""
        self._fh.seek(self.data_offset)
        return SEGMENT_ID.get(self.sid, UnknownSegment)(self._fh)

    def __str__(self):
        return 'Segment %s %i of %i' % (
            self.sid, self.used_size, self.allocated_size)


class SegmentNotFoundError(Exception):
    """Exception to indicate that file position does not contain Segment."""


class FileHeaderSegment(object):
    """ZISRAWFILE file header segment data.

    Contains global file metadata such as file version and GUID.

    """

    __slots__ = ('version', 'primary_file_guid', 'file_guid',
                 'file_part', 'directory_position', 'metadata_position',
                 'update_pending', 'attachment_directory_position')

    SID = 'ZISRAWFILE'

    def __init__(self, fh):
        (major,
         minor,
         reserved1,
         reserved2,
         primary_file_guid,
         file_guid,
         self.file_part,
         self.directory_position,
         self.metadata_position,
         self.update_pending,
         self.attachment_directory_position,
         ) = struct.unpack('<iiii16s16siqqiq', fh.read(80))
        self.version = (major, minor)
        self.update_pending = bool(self.update_pending)
        self.primary_file_guid = uuid.UUID(bytes=primary_file_guid)
        self.file_guid = uuid.UUID(bytes=file_guid)

    def __str__(self):
        return 'FileHeaderSegment\n ' + '\n '.join(
            '%s %s' % (name, str(getattr(self, name)))
            for name in FileHeaderSegment.__slots__)


class MetadataSegment(object):
    """ZISRAWMETADATA segment data.

    Contains global image metadata in UTF-8 encoded XML format.

    """

    __slots__ = 'xml_size', 'attachment_size', 'xml_offset', '_fh'

    SID = 'ZISRAWMETADATA'

    def __init__(self, fh):
        self.xml_size, self.attachment_size = struct.unpack('<ii', fh.read(8))
        fh.seek(248, 1)  # spare
        self.xml_offset = fh.tell()
        self._fh = fh

    def data(self, raw=True):
        """Read metadata from file and return as XML (default) or dict."""
        self._fh.seek(self.xml_offset)
        xml = self._fh.read(self.xml_size)
        xml = xml.replace(b'\r\n', b'\n').replace(b'\r', b'\n')  # ???
        return unicode(xml, 'utf-8') if raw else xml2dict(xml)

    def __str__(self):
        return 'MetadataSegment\n %s' % self.data()


class SubBlockSegment(object):
    """ZISRAWSUBBLOCK segment data.

    Contains XML metadata, optional attachments, and homogenous,
    contiguous pixel data.

    """

    __slots__ = ('metadata_size', 'attachment_size', 'data_size',
                 'directory_entry', 'data_offset', '_fh')

    SID = 'ZISRAWSUBBLOCK'

    def __init__(self, fh):
        """Read ZISRAWSUBBLOCK segment data from file."""
        with fh.lock:
            (self.metadata_size,
             self.attachment_size,
             self.data_size,
             ) = struct.unpack('<iiq', fh.read(16))
            self.directory_entry = DirectoryEntryDV(fh)
            # fh.seek(max(240 - self.directory_entry.storage_size, 0), 1)
            # self.metadata = unicode(fh.read(self.metadata_size), 'utf-8')
            self.data_offset = fh.tell()
        self.data_offset += max(240 - self.directory_entry.storage_size, 0)
        self.data_offset += self.metadata_size
        self._fh = fh

    def metadata(self, raw=False):
        """Read metadata from file and return as dict (default) or XML."""
        if self.metadata_size <= 0:
            return u'' if raw else None
        fh = self._fh
        with fh.lock:
            fh.seek(self.data_offset - self.metadata_size)
            metadata = fh.read(self.metadata_size)
        if raw:
            return unicode(metadata, 'utf-8')
        try:
            return xml2dict(metadata)['METADATA']
        except Exception:
            return unicode(metadata, 'utf-8')

    def data(self, raw=False, resize=True, order=0):
        """Read image data from file and return as numpy array."""
        de = self.directory_entry
        fh = self._fh
        if raw:
            with fh.lock:
                fh.seek(self.data_offset)
                data = fh.read(self.data_size)
            return data
        if de.compression:
            # if de.compression not in DECOMPRESS:
            #     raise ValueError('compression unknown or not supported')
            with fh.lock:
                fh.seek(self.data_offset)
                data = fh.read(self.data_size)
            data = DECOMPRESS[de.compression](data)
            if de.compression == 2:
                # LZW
                data = numpy.fromstring(data, de.dtype)
        else:
            dtype = numpy.dtype(de.dtype)
            with fh.lock:
                fh.seek(self.data_offset)
                data = fh.read_array(dtype, self.data_size // dtype.itemsize)

        data = data.reshape(de.stored_shape)
        if de.compression != 4 and de.stored_shape[-1] in (3, 4):
            if de.stored_shape[-1] == 3:
                # BGR -> RGB
                data = data[..., ::-1]
            else:
                # BGRA -> RGBA
                tmp = data[..., 0].copy()
                data[..., 0] = data[..., 2]
                data[..., 2] = tmp
        if de.stored_shape == de.shape or not resize:
            return data

        # sub / supersampling
        factors = [j / i for i, j in zip(de.stored_shape, de.shape)]
        factors = [(int(round(f)) if abs(f - round(f)) < 0.0001 else f)
                   for f in factors]

        # use repeat if possible
        if order == 0 and all(isinstance(f, int) for f in factors):
            data = repeat_nd(data, factors).copy()
            data.shape = de.shape
            return data

        # remove leading dimensions with size 1 for speed
        shape = list(de.stored_shape)
        i = 0
        for s in shape:
            if s != 1:
                break
            i += 1
        shape = shape[i:]
        factors = factors[i:]
        data.shape = shape

        # resize RGB components separately for speed
        if zoom is None:
            raise ImportError("cannot import 'zoom' from scipy or ndimage")
        if shape[-1] in (3, 4) and factors[-1] == 1.0:
            factors = factors[:-1]
            old = data
            data = numpy.empty(de.shape, de.dtype[-2:])
            for i in range(shape[-1]):
                data[..., i] = zoom(old[..., i], zoom=factors, order=order)
        else:
            data = zoom(data, zoom=factors, order=order)

        data.shape = de.shape
        return data

    def attachments(self):
        """Read optional attachments from file and return as byte strings."""
        # TODO: process chunk-containers and masks
        if self.attachment_size < 1:
            return b''
        fh = self._fh
        with fh.lock:
            fh.seek(self.data_offset + self.data_size)
            attachments = fh.read(self.attachment_size)
        return attachments

    def __getattr__(self, name):
        """Directly access DirectoryEntryDV attributes."""
        return getattr(self.directory_entry, name)

    def __str__(self):
        return 'SubBlockSegment\n %s\n %s' % (
            pformat(self.metadata()), str(self.directory_entry))


class DirectoryEntryDV(object):
    """Directory Entry - Schema DV."""

    # __slots__ = ('file_position', 'file_part', 'compression', 'pyramid_type',
    #             'dimension_entries', 'dtype', 'shape', 'stored_shape',
    #             'axes', 'mosaic_index', 'storage_size', 'start', '_fh')

    @staticmethod
    def read_file_position(fh):
        """Return file position of associated SubBlock segment."""
        (schema_type,
         file_position,
         dimensions_count,
         ) = struct.unpack('<2s4xq14xi', fh.read(32))
        fh.seek(dimensions_count * 20, 1)
        assert schema_type == b'DV'
        return file_position

    def __init__(self, fh):
        """Read DirectoryEntryDV from file."""
        self._fh = fh

        (schema_type,
         pixel_type,
         self.file_position,
         self.file_part,
         self.compression,
         self.pyramid_type,  # None=0, SingleSubblock=1, MultiSubblock=2
         reserved1,
         reserved2,
         dimensions_count,
         ) = struct.unpack('<2siqiiBB4si', fh.read(32))

        if schema_type != b'DV':
            raise ValueError('not a DirectoryEntryDV')
        self.dtype = PIXEL_TYPE[pixel_type]

        # reverse dimension_entries to match C contiguous data
        self.dimension_entries = list(reversed(
            [DimensionEntryDV1(fh) for _ in range(dimensions_count)]))

    @lazyattr
    def storage_size(self):
        return 32 + len(self.dimension_entries) * 20

    @lazyattr
    def pixel_type(self):
        return PIXEL_TYPE[self.dtype]

    @lazyattr
    def axes(self):
        axes = ''.join(dim.dimension for dim in self.dimension_entries
                       if dim.dimension != 'M')
        return axes + '0'

    @lazyattr
    def shape(self):
        shape = tuple(dim.size for dim in self.dimension_entries
                      if dim.dimension != 'M')
        sampleshape = numpy.dtype(self.dtype).shape
        return shape + (sampleshape if sampleshape else (1,))

    @lazyattr
    def start(self):
        start = tuple(dim.start for dim in self.dimension_entries
                      if dim.dimension != 'M')
        return start + (0,)

    @lazyattr
    def stored_shape(self):
        shape = tuple(dim.stored_size for dim in self.dimension_entries
                      if dim.dimension != 'M')
        sampleshape = numpy.dtype(self.dtype).shape
        return shape + (sampleshape if sampleshape else (1,))

    @lazyattr
    def mosaic_index(self):
        for dim in self.dimension_entries:
            if dim.dimension == 'M':
                return dim.start
        return None

    def data_segment(self):
        """Read and return SubBlockSegment at file_position."""
        # return Segment(self._fh, self.file_position).data()
        fh = self._fh
        with fh.lock:
            fh.seek(self.file_position)
            try:
                sid, _, _ = struct.unpack('<16sqq', fh.read(32))
            except struct.error:
                raise SegmentNotFoundError('can not read ZISRAW segment')
            sid = bytes2str(stripnull(sid))
            if sid not in SEGMENT_ID:
                raise SegmentNotFoundError('not a ZISRAW segment')
            data_segment = SEGMENT_ID[sid](fh)
        return data_segment

    def __str__(self):
        return 'DirectoryEntryDV\n  %s %s %s %s\n  %s' % (
            COMPRESSION.get(self.compression, self.compression),
            self.pixel_type, self.axes, str(self.shape),
            '\n  '.join(str(d) for d in self.dimension_entries))


class DimensionEntryDV1(object):
    """Dimension Entry - Schema DV."""

    __slots__ = 'dimension', 'start', 'size', 'start_coordinate', 'stored_size'

    def __init__(self, fh):
        """Read DimensionEntryDV1 from file."""
        (self.dimension,
         self.start,
         self.size,
         self.start_coordinate,
         stored_size
         ) = struct.unpack('<4siifi', fh.read(20))
        self.dimension = bytes2str(stripnull(self.dimension))
        self.stored_size = stored_size if stored_size else self.size

    def __str__(self):
        return 'DimensionEntryDV1 %s %i %i %f %i' % (
            self.dimension, self.start, self.size,
            self.start_coordinate, self.stored_size)


class SubBlockDirectorySegment(object):
    """ZISRAWDIRECTORY segment data.

    Contains entries of any kind, currently only DirectoryEntryDV.

    """

    __slots__ = ('entries',)

    SID = 'ZISRAWDIRECTORY'

    @staticmethod
    def file_positions(fh):
        """Return list of file positions of associated SubBlock segments."""
        entry_count = struct.unpack('<i', fh.read(4))[0]
        fh.seek(124, 1)  # reserved
        return tuple(DirectoryEntryDV.read_file_position(fh)
                     for _ in range(entry_count))

    def __init__(self, fh):
        """Read SubBlockDirectorySegment from file."""
        entry_count = struct.unpack('<i', fh.read(4))[0]
        fh.seek(124, 1)  # reserved
        self.entries = tuple(DirectoryEntryDV(fh) for _ in range(entry_count))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        return self.entries[key]

    def __iter__(self):
        return iter(self.entries)

    def __str__(self):
        return 'SubBlockDirectorySegment\n %s' % (
            '\n '.join(str(e) for e in self.entries))


class AttachmentSegment(object):
    """ZISRAWATTACH segment data.

    Contains binary or text data as specified in attachment_entry.

    """
    __slots__ = 'data_size', 'attachment_entry', 'data_offset', '_fh'

    SID = 'ZISRAWATTACH'

    def __init__(self, fh):
        """Read AttachmentSegment from file."""
        self.data_size = struct.unpack('<i', fh.read(4))[0]
        fh.seek(12, 1)  # reserved
        self.attachment_entry = AttachmentEntryA1(fh)
        fh.seek(112, 1)  # reserved
        self.data_offset = fh.tell()
        self._fh = fh

    def save(self, filename=None, directory='.'):
        """Save attachment to file in directory."""
        self._fh.seek(self.data_offset)
        if not filename:
            filename = self.attachment_entry.filename
        filename = os.path.join(directory, filename)
        with open(filename, 'wb') as fh:
            fh.write(self._fh.read(self.data_size))

    def data(self, raw=False):
        """Read embedded file and return content.

        If 'raw' is False (default), try return content according to
        CONTENT_FILE_TYPE, else return raw bytes.

        """
        self._fh.seek(self.data_offset)
        cotype = self.attachment_entry.content_file_type
        if not raw and cotype in CONTENT_FILE_TYPE:
            return CONTENT_FILE_TYPE[cotype](self._fh, filesize=self.data_size)
        return self._fh.read(self.data_size)

    def __str__(self):
        return 'AttachmentSegment\n %s' % self.attachment_entry


class AttachmentEntryA1(object):
    """AttachmentEntry - Schema A1."""

    __slots__ = ('content_guid', 'content_file_type', 'name', 'file_position',
                 '_fh')

    @staticmethod
    def read_file_position(fh):
        """Return file position of associated Attachment segment."""
        schema_type, file_position = struct.unpack('<2s10xq', fh.read(20))
        fh.seek(108, 1)
        assert schema_type == b'A1'
        return file_position

    def __init__(self, fh):
        """Read AttachmentEntryA1 from file."""
        (shema_type,
         reserved,
         self.file_position,
         file_part,  # reserved
         content_guid,
         content_file_type,
         name
         ) = struct.unpack('<2s10sqi16s8s80s', fh.read(128))

        if shema_type != b'A1':
            raise ValueError('not a AttachmentEntryA1')
        self.content_guid = uuid.UUID(bytes=content_guid)
        self.content_file_type = bytes2str(stripnull(content_file_type))
        self.name = unicode(stripnull(name), 'utf-8')
        self._fh = fh

    @property
    def filename(self):
        """Return unique file name for attachment."""
        return u'%s@%i.%s' % (self.name, self.file_position,
                              unicode(self.content_file_type, 'utf-8').lower())

    def data_segment(self):
        """Read and return AttachmentSegment at file_position."""
        return Segment(self._fh, self.file_position).data()

    def __str__(self):
        return ' '.join(str(i) for i in (
            'AttachmentEntryA1', self.name, self.content_file_type,
            self.content_guid))


class AttachmentDirectorySegment(object):
    """ZISRAWATTDIR segment data. Sequence of AttachmentEntryA1."""

    __slots__ = ('entries',)

    SID = 'ZISRAWATTDIR'

    @staticmethod
    def file_positions(fh):
        """Return list of file positions of associated Attachment segments."""
        entry_count = struct.unpack('<i', fh.read(4))[0]
        fh.seek(252, 1)
        return tuple(AttachmentEntryA1.read_file_position(fh)
                     for _ in range(entry_count))

    def __init__(self, fh):
        """Read AttachmentDirectorySegment from file."""
        entry_count = struct.unpack('<i', fh.read(4))[0]
        fh.seek(252, 1)
        self.entries = tuple(AttachmentEntryA1(fh) for _ in range(entry_count))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, key):
        return self.entries[key]

    def __iter__(self):
        return iter(self.entries)

    def __str__(self):
        return 'AttachmentDirectorySegment\n %s' % (
            '\n '.join(str(i) for i in self.entries))


class DeletedSegment(object):
    """DELETED segment data. Ignore."""

    __slots__ = ()

    SID = 'DELETED'

    def __init__(self, fh):
        pass

    def __str__(self):
        return 'DeletedSegment'


class UnknownSegment(object):
    """Unknown segment data. Ignore."""

    __slots__ = ()

    def __init__(self, fh):
        pass

    def __str__(self):
        return 'UnknownSegment'


def read_time_stamps(fh, filesize=None):
    """Read time stamps in seconds relative to start of acquisition.

    CZTIMS TimeStamps content schema.

    """
    size, number = struct.unpack('<ii', fh.read(8))
    return fh.read_array(dtype='float64', count=number)


def read_focus_positions(fh, filesize=None):
    """Read focus positions in micrometers relative to Z start of acquisition.

    CZFOC FocusPositions content schema.

    """
    size, number = struct.unpack('<ii', fh.read(8))
    return fh.read_array(dtype='float64', count=number)


def read_event_list(fh, filesize=None):
    """Read sequence of EventListEntry from file.

    CZEVL EventList content schema.

    """
    size, number = struct.unpack('<ii', fh.read(8))
    return tuple(EventListEntry(fh) for _ in range(number))


class EventListEntry(object):
    """EventListEntry content schema."""

    __slots__ = 'time', 'event_type', 'description'

    EV_TYPE = {0: 'MARKER', 1: 'TIME_CHANGE', 2: 'BLEACH_START',
               3: 'BLEACH_STOP', 4: 'TRIGGER'}

    def __init__(self, fh):
        (size,
         self.time,
         self.event_type,
         description_size,
         ) = struct.unpack('<idii', fh.read(20))
        description = stripnull(fh.read(description_size))
        self.description = unicode(description, 'utf-8')

    def __str__(self):
        return '%s @ %s (%s)' % (EventListEntry.EV_TYPE[self.event_type],
                                 self.time, self.description)


def read_lookup_table(fh, filesize=None):
    """Read list of LookupTableEntry from file.

    Return as tuple((identifier, ndarray),)

    CZLUT LookupTables content schema.

    """
    # TODO: test this; CZI specification is unclear.
    size, number = struct.unpack('<ii', fh.read(8))
    luts = []
    for _ in range(number):
        size, identifier, number = struct.unpack('<i80si', fh.read(88))
        identifier = unicode(stripnull(identifier), 'utf-8')
        components = [None] * number
        for _ in range(number):
            size, component_type, number = struct.unpack('<iii', fh.read(12))
            intensity = fh.read_array(dtype='<i2', count=number)
            if component_type == -1:  # RGB
                components = intensity.reshape(-1, 3).T
            else:  # R, G, B
                components[component_type] = intensity
        luts.append((identifier, numpy.array(components, copy=True)))
    return tuple(luts)


def read_xml(fh, filesize, raw=True):
    """Read XML from file and return as unicode string (default) or dict."""
    xml = stripnull(fh.read(filesize))
    return unicode(xml, 'utf-8') if raw else xml2dict(xml)


def match_filename(filename):
    """Return master file name and file part number from CZI file name."""
    match = re.search(r'(.*?)(?:\((\d+)\))?\.czi$',
                      filename, re.IGNORECASE).groups()
    name = match[0] + '.czi'
    part = int(match[1]) if len(match) > 1 else 0
    return name, part


# map Segment.sid to data reader
SEGMENT_ID = {
    FileHeaderSegment.SID: FileHeaderSegment,
    SubBlockDirectorySegment.SID: SubBlockDirectorySegment,
    SubBlockSegment.SID: SubBlockSegment,
    MetadataSegment.SID: MetadataSegment,
    AttachmentSegment.SID: AttachmentSegment,
    AttachmentDirectorySegment.SID: AttachmentDirectorySegment,
    DeletedSegment.SID: DeletedSegment,
}

# map AttachmentEntryA1.content_file_type to attachment reader.
CONTENT_FILE_TYPE = {
    'CZI': CziFile,
    'ZISRAW': CziFile,
    'CZTIMS': read_time_stamps,
    'CZEVL': read_event_list,
    'CZLUT': read_lookup_table,
    'CZFOC': read_focus_positions,
    'CZEXP': read_xml,  # Experiment
    'CZHWS': read_xml,  # HardwareSetting
    'CZMVM': read_xml,  # MultiviewMicroscopy
    'CZFBMX': read_xml,  # FiberMatrix
    # 'CZPML': PalMoleculeList,  # undocumented
    # 'ZIP'
    # 'JPG'
}

# map DirectoryEntryDV.pixeltype to numpy dtypes
PIXEL_TYPE = {
    0: '<u1', 'Gray8': '<u1', '<u1': 'Gray8',
    1: '<u2', 'Gray16': '<u2', '<u2': 'Gray16',
    2: '<f4', 'Gray32Float': '<f4', '<f4': 'Gray32Float',
    3: '<3u1', 'Bgr24': '<3u1', '<3u1': 'Bgr24',
    4: '<3u2', 'Bgr48': '<3u2', '<3u2': 'Bgr48',
    8: '<3f4', 'Bgr96Float': '<3f4', '<3f4': 'Bgr96Float',
    9: '<4u1', 'Bgra32': '<4u1', '<4u1': 'Bgra32',
    10: '<F8', 'Gray64ComplexFloat': '<F8', '<F8': 'Gray64ComplexFloat',
    11: '<3F8', 'Bgr192ComplexFloat': '<3F8', '<3F8': 'Bgr192ComplexFloat',
    12: '<i4', 'Gray32': '<i4', '<i4': 'Gray32',
    13: '<i8', 'Gray64': '<i8', '<i8': 'Gray64',
}


# map dimension character to description
DIMENSIONS = {
    '0': 'Sample',  # e.g. RGBA
    'X': 'Width',
    'Y': 'Height',
    'C': 'Channel',
    'Z': 'Slice',  # depth
    'T': 'Time',
    'R': 'Rotation',
    'S': 'Scene',  # contiguous regions of interest in a mosaic image
    'I': 'Illumination',  # direction
    'B': 'Block',  # acquisition
    'M': 'Mosaic',  # index of tile for compositing a scene
    'H': 'Phase',  # e.g. Airy detector fibers
    'V': 'View',  # e.g. for SPIM
}

# map DirectoryEntryDV.compression to description
COMPRESSION = {
    0: 'Uncompressed',
    1: 'JpgFile',
    2: 'LZW',
    4: 'JpegXrFile',
    # 100 and up: camera/system specific specific RAW data
}

# map DirectoryEntryDV.compression to decompression function
DECOMPRESS = {
    0: lambda x: x,  # uncompressed
}

if imagecodecs is not None:
    DECOMPRESS[2] = imagecodecs.lzw_decode
    if hasattr(imagecodecs, 'jpeg_decode'):
        DECOMPRESS[1] = imagecodecs.jpeg_decode
        DECOMPRESS[4] = imagecodecs.jxr_decode


def czi2tif(czifile, tiffile=None, squeeze=True, verbose=True, **kwargs):
    """Convert CZI file to memory-mappable TIFF file.

    To read the image data from the created TIFF file: Read the 'StripOffsets'
    and 'ImageDescription' tags from the first TIFF page. Get the 'dtype' and
    'shape' attributes from the ImageDescription string using a JSON decoder.
    Memory-map 'product(shape) * sizeof(dtype)' bytes in the file starting at
    StripOffsets[0]. Cast the mapped bytes to an array of 'dtype' and 'shape'.

    """
    verbose = print_ if verbose else nullfunc

    if tiffile is None:
        tiffile = czifile + '.tif'
    elif tiffile.lower() == 'none':
        tiffile = None

    verbose('\nOpening CZI file... ', end='', flush=True)
    timer = Timer()

    with CziFile(czifile) as czi:
        if squeeze:
            shape, axes = squeeze_axes(czi.shape, czi.axes, '')
        else:
            shape = czi.shape
            axes = czi.axes
        dtype = str(czi.dtype)
        size = product(shape) * czi.dtype.itemsize

        verbose(timer)
        verbose('Image\n  axes:  %s\n  shape: %s\n  dtype: %s\n  size:  %s'
                % (axes, shape, dtype, format_size(size)), flush=True)

        if not tiffile:
            verbose('Copying image from CZI file to RAM... ',
                    end='', flush=True)
            timer.start()
            czi.asarray(order=0)
        else:
            verbose('Creating empty TIF file... ', end='', flush=True)
            timer.start()
            if 'software' not in kwargs:
                kwargs['software'] = 'czi2tif'
            try:
                description = bytestr(czi.metadata(), 'utf-8')
            except Exception:
                description = None
            metadata = kwargs.pop('metadata', {})
            metadata.update(axes=axes, dtype=dtype)
            data = memmap(tiffile, shape=shape, dtype=dtype, metadata=metadata,
                          description=description, **kwargs)
            data = data.reshape(czi.shape)
            verbose(timer)
            verbose('Copying image from CZI to TIF file... ',
                    end='', flush=True)
            timer.start()
            czi.asarray(order=0, out=data)
        verbose(timer, flush=True)


def main(argv=None):
    """Command line usage main function."""
    if argv is None:
        argv = sys.argv
    if len(argv) < 2:
        filename = askopenfilename(title='Select a CZI file', multiple=False,
                                   filetypes=[('CZI files', '*.czi')])
    else:
        filename = argv[1]
    if not filename:
        return

    timer = Timer()
    with CziFile(filename) as czi:
        timer.stop()
        print(czi)
        print()
        timer.print('Opening file: ')
        timer.start('Reading image:')
        data = czi.asarray()
        timer.print()

    from matplotlib import pyplot  # NOQA: delay import

    imshow(data, title=os.path.split(filename)[-1])
    pyplot.show()


if sys.version_info[0] == 2:
    def print_(*args, **kwargs):
        """Print function with flush support."""
        flush = kwargs.pop('flush', False)
        print(*args, **kwargs)
        if flush:
            sys.stdout.flush()

    def bytes2str(b, encoding=None):
        """Return string from bytes."""
        return b

    def str2bytes(s, encoding=None):
        """Return bytes from string."""
        return s

    def bytestr(s, encoding='cp1252'):
        """Return byte string from unicode string, else pass through."""
        return s.encode(encoding) if isinstance(s, unicode) else s

else:
    basestring = str, bytes
    print_ = print

    def unicode(s, encoding='utf-8'):
        """Return unicode string from bytes or unicode."""
        try:
            return str(s, encoding)
        except TypeError:
            return s

    def bytes2str(b, encoding='cp1252'):
        """Return unicode string from bytes."""
        return str(b, encoding)

    def str2bytes(s, encoding='cp1252'):
        """Return bytes from unicode string."""
        return s.encode(encoding)

    def bytestr(s, encoding='cp1252'):
        """Return byte string from unicode string, else pass through."""
        return s.encode(encoding) if isinstance(s, str) else s


if __name__ == '__main__':
    if '--doctest' in sys.argv:
        import doctest
        numpy.set_printoptions(suppress=True, precision=5)
        doctest.testmod()
    else:
        sys.exit(main(sys.argv))
