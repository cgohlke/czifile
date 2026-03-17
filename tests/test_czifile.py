# test_czifile.py

# Copyright (c) 2013-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the czifile package.

:Version: 2026.3.17

"""

import contextlib
import glob
import io
import itertools
import math
import os
import pathlib
import struct
import sys
import sysconfig
import warnings
from collections.abc import Mapping

import numpy
import pytest
import tifffile

# NOTE: ``assert numpy.array_equal`` much faster than ``assert_array_equal``
# from numpy.testing import assert_array_equal

try:
    import fsspec
except ImportError:
    fsspec = None

import czifile
from czifile import (
    CziAttachmentDirectorySegmentData,
    CziAttachmentEntryA1,
    CziAttachmentSegmentData,
    CziCompressionType,
    CziContentFileType,
    CziDimensionEntryDV1,
    CziDimensionType,
    CziDirectoryEntryDV,
    CziFile,
    CziFileError,
    CziImage,
    CziImageChunks,
    CziMetadataSegmentData,
    CziPixelType,
    CziPyramidType,
    CziScenes,
    CziSubBlockDirectorySegmentData,
    CziSubBlockSegmentData,
    __version__,
    imread,
)
from czifile.czifile import (
    BinaryFile,
    DecodeCache,
    NullLock,
    default_maxworkers,
    match_filename,
)

HERE = pathlib.Path(os.path.dirname(__file__))
DATA = HERE / 'data'

# Maximum size of image arrays to read in glob tests
MAX_SIZE = int(os.environ.get('CZIFILE_MAX_SIZE', '512')) * 1024 * 1024


def config() -> str:
    """Return test configuration summary."""
    items = [f'MAX_SIZE={MAX_SIZE // (1024 * 1024)}MB']
    return ', '.join(items)


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert czifile versions match docstrings."""
    ver = ':Version: ' + __version__
    assert ver in __doc__
    assert ver in czifile.__doc__


class TestBinaryFile:
    """Test BinaryFile with different file-like inputs."""

    def setup_method(self):
        self.filename = os.path.normpath(DATA / 'binary.bin')
        if not os.path.exists(self.filename):
            pytest.skip(f'{self.filename!r} not found')

    def validate(
        self,
        fh: BinaryFile,
        filepath: str | None = None,
        filename: str | None = None,
        dirname: str | None = None,
        name: str | None = None,
        *,
        closed: bool = True,
    ) -> None:
        """Assert BinaryFile attributes."""
        if filepath is None:
            filepath = self.filename
        if filename is None:
            filename = os.path.basename(self.filename)
        if dirname is None:
            dirname = os.path.dirname(self.filename)
        if name is None:
            name = fh.filename

        attrs = fh.attrs
        assert attrs['name'] == name
        assert attrs['filepath'] == filepath

        assert fh.filepath == filepath
        assert fh.filename == filename
        assert fh.dirname == dirname
        assert fh.name == name
        assert fh.closed is False
        assert len(fh.filehandle.read()) == 256
        fh.filehandle.seek(10)
        assert fh.filehandle.tell() == 10
        assert fh.filehandle.read(1) == b'\n'
        fh.close()
        # underlying filehandle may still be be open if
        # BinaryFile was given an open filehandle
        assert fh._fh.closed is closed
        # BinaryFile always reports itself as closed after close() is called
        assert fh.closed

    def test_str(self):
        """Test BinaryFile with str path."""
        file = self.filename
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_pathlib(self):
        """Test BinaryFile with pathlib.Path."""
        file = pathlib.Path(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_open_file(self):
        """Test BinaryFile with open binary file."""
        with open(self.filename, 'rb') as fh, BinaryFile(fh) as bf:
            self.validate(bf, closed=False)

    def test_bytesio(self):
        """Test BinaryFile with BytesIO."""
        with open(self.filename, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with BinaryFile(file) as fh:
            self.validate(
                fh,
                filepath='',
                filename='',
                dirname='',
                name='BytesIO',
                closed=False,
            )

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test BinaryFile with fsspec OpenFile."""
        file = fsspec.open(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test BinaryFile with fsspec LocalFileOpener."""
        with fsspec.open(self.filename) as file, BinaryFile(file) as fh:
            self.validate(fh, closed=False)

    def test_text_file_fails(self):
        """Test BinaryFile with open text file fails."""
        with open(self.filename) as fh:  # noqa: SIM117
            with pytest.raises(TypeError):
                BinaryFile(fh)

    def test_file_extension_fails(self):
        """Test BinaryFile with wrong file extension fails."""
        ext = BinaryFile._ext
        BinaryFile._ext = {'.lif'}
        try:
            with pytest.raises(ValueError):
                BinaryFile(self.filename)
        finally:
            BinaryFile._ext = ext

    def test_file_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock file object without tell methods
            def seek(self):
                pass

        with pytest.raises(ValueError):
            BinaryFile(File)

    def test_openfile_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock fsspec OpenFile without seek/tell methods
            @staticmethod
            def open(*args, **kwargs):
                del args, kwargs
                return File()

        with pytest.raises(ValueError):
            BinaryFile(File)

    def test_invalid_object(self):
        """Test BinaryFile with invalid file object fails."""

        class File:
            # mock non-file object
            pass

        with pytest.raises(TypeError):
            BinaryFile(File)

    def test_invalid_mode(self):
        """Test BinaryFile with invalid mode fails."""
        with pytest.raises(ValueError):
            BinaryFile(self.filename, mode='ab')


class TestCziFile:
    """Test CziFile with different file-like inputs."""

    def setup_method(self):
        self.filename = os.path.normpath(DATA / 'Test.czi')
        if not os.path.exists(self.filename):
            pytest.skip(f'{self.filename!r} not found')

    def validate(self, czi: CziFile, name: str = 'Test.czi') -> None:
        """Assert CziFile attributes."""
        assert not czi.filehandle.closed
        assert czi.name == name
        assert repr(czi).startswith('<CziFile ')
        img = czi.scenes[0]
        assert img.dims == ('C', 'Z', 'Y', 'X')
        assert img.shape == (3, 22, 181, 312)
        assert img.dtype == numpy.uint16
        data = img.asarray()
        assert data.shape == (3, 22, 181, 312)
        assert data.dtype == numpy.uint16
        assert int(data.min()) == 54
        assert int(data.max()) == 6932

    def test_str(self):
        """Test CziFile with str path."""
        file = self.filename
        with CziFile(file) as czi:
            self.validate(czi)

    def test_pathlib(self):
        """Test CziFile with pathlib.Path."""
        file = pathlib.Path(self.filename)
        with CziFile(file) as czi:
            self.validate(czi)

    def test_open_file(self):
        """Test CziFile with open binary file."""
        with open(self.filename, 'rb') as fh, CziFile(fh) as czi:
            self.validate(czi)

    def test_bytesio(self):
        """Test CziFile with BytesIO."""
        with open(self.filename, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with CziFile(file) as czi:
            self.validate(czi, name='BytesIO')

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test CziFile with fsspec OpenFile."""
        file = fsspec.open(self.filename)
        with CziFile(file) as czi:
            self.validate(czi)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test CziFile with fsspec LocalFileOpener."""
        with fsspec.open(self.filename) as file, CziFile(file) as czi:
            self.validate(czi)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_http(self):
        """Test CziFile via fsspec HTTP with Range requests."""
        file = fsspec.open('http://localhost:8386/test/Test.czi', 'rb')
        try:
            with CziFile(file) as czi:
                self.validate(czi)
        except OSError as exc:
            pytest.skip(f'HTTP server not available: {exc}')


def test_not_czi():
    """Test open non-CZI file raises exceptions."""
    with pytest.raises(CziFileError):
        imread(DATA / 'empty.bin')
    with pytest.raises(TypeError):
        imread(ValueError)


def test_imread():
    """Smoke test for imread covering all parameters."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    # str path, default parameters
    im = imread(str(filename))
    assert isinstance(im, numpy.ndarray)
    assert im.shape == (3, 22, 181, 312)
    assert im.dtype == numpy.uint16

    # pathlib.Path
    im = imread(filename)
    assert im.shape == (3, 22, 181, 312)

    # IO[bytes] (open file object)
    with open(filename, 'rb') as fh:
        im = imread(fh)
    assert im.shape == (3, 22, 181, 312)

    # squeeze=False preserves length-1 dimensions
    im = imread(filename, squeeze=False)
    assert im.shape == (1, 3, 22, 181, 312, 1)

    # scene=None merges all scenes (Test.czi has one scene)
    im = imread(filename, scene=None)
    assert im.shape == (3, 22, 181, 312)

    # roi=(x, y, width, height) restricts spatial extent
    im = imread(filename, roi=(10, 20, 80, 40))
    assert im.shape == (3, 22, 40, 80)

    # pixeltype converts output pixel type
    im = imread(filename, pixeltype=CziPixelType.GRAY8)
    assert im.dtype == numpy.uint8
    assert im.shape == (3, 22, 181, 312)

    # fillvalue sets pixels not covered by any subblock
    im = imread(filename, fillvalue=255)
    assert im.shape == (3, 22, 181, 312)

    # maxworkers=1 forces single-threaded decode
    im = imread(filename, maxworkers=1)
    assert im.shape == (3, 22, 181, 312)

    # asxarray=True returns an xarray DataArray
    try:
        import xarray
    except ImportError:
        pass
    else:
        xa = imread(filename, asxarray=True)
        assert isinstance(xa, xarray.DataArray)
        assert xa.shape == (3, 22, 181, 312)

    # out='memmap' writes into a temporary memory-mapped file
    im = imread(filename, out='memmap')
    assert isinstance(im, numpy.ndarray)
    assert im.shape == (3, 22, 181, 312)
    del im  # close memmap

    # out=ndarray: data is written into the caller's buffer without copying
    out = numpy.zeros((3, 22, 181, 312), numpy.uint16)
    im = imread(filename, out=out)
    assert numpy.shares_memory(im, out)
    assert im.flags.c_contiguous
    assert out.sum() > 0

    # out=ndarray with reordering selection: contiguous, no copy
    with CziFile(filename) as czi:
        sub = czi.scenes[0](Z=None, C=None)
        out2 = numpy.zeros(sub.shape, sub.dtype)
        im2 = sub.asarray(out=out2)
        assert numpy.shares_memory(im2, out2)
        assert im2.flags.c_contiguous
        assert im2.shape == sub.shape

    # **selection: select a single channel and z-plane
    im = imread(filename, C=0, Z=0)
    assert im.shape == (181, 312)


def test_imread_scene_default():
    """Test imread reads first scene by default."""
    # PupalWing.czi has a single scene at S=4 (not S=0).

    filename = DATA / 'Private-PhasorPy' / 'PupalWing.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        assert list(czi.scenes.keys()) == [4]

    im = imread(filename)
    assert im.shape == (80, 520, 692)
    assert im.dtype == numpy.uint16

    with pytest.raises(KeyError, match='scene 0 not found'):
        imread(filename, scene=0)

    with pytest.raises(KeyError, match='scene -1 not found'):
        imread(filename, scene=-1)

    # slice(4, 5) selects S-coords in {4} -> same result as scene=4
    im_slice = imread(filename, scene=slice(4, 5))
    assert numpy.array_equal(im_slice, im)

    # slice(4, 10) covers S in {4..9} -> S=4 is included; same result
    im_wide = imread(filename, scene=slice(4, 10))
    assert numpy.array_equal(im_wide, im)

    # slice(0, 4) covers S in {0,1,2,3} -> file has no such scene
    with pytest.raises(KeyError):
        imread(filename, scene=slice(0, 4))


def test_imread_scene_sequence():
    """Test imread and CziFile.asarray accept a sequence of scene indices."""
    filename = DATA / 'Zenodo-7015307/S=3_1Pos_2Mosaic_T=2=Z=3_CH=2.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # single-item sequence is equivalent to scalar scene selection
        im_list1 = czi.asarray(scene=[1])
        im_scalar1 = czi.asarray(scene=1)
        assert im_list1.shape == im_scalar1.shape
        assert numpy.array_equal(im_list1, im_scalar1)

        # two-scene sequence merges scenes 0 and 1 into one bounding box
        im_01 = czi.asarray(scene=[0, 1])
        assert im_01.shape == (2, 2, 3, 1109, 1760)

        # result equals what czi.scenes(scene=[0, 1]) returns
        ref_01 = czi.scenes(scene=[0, 1]).asarray()
        assert numpy.array_equal(im_01, ref_01)

    # also reachable via imread()
    im_imread = imread(filename, scene=[0, 1])
    assert numpy.array_equal(im_imread, im_01)


def test_directory_entry_dimension_entries():
    """Test CziDirectoryEntryDV.read_dimension_entries returns all dims."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        de = czi.subblock_directory[0]
        dim_entries = de.read_dimension_entries(czi.filehandle)

        assert isinstance(dim_entries, dict)
        assert all(isinstance(k, CziDimensionType) for k in dim_entries)
        assert all(
            isinstance(v, CziDimensionEntryDV1) for v in dim_entries.values()
        )
        assert len(dim_entries) == de.dimensions_count

        # start and size match values parsed by constructor
        for dim_char in ('T', 'C', 'Z', 'Y', 'X'):
            dt = CziDimensionType(dim_char)
            assert dt in dim_entries
            idx = de.dims.index(dim_char)
            assert dim_entries[dt].start == de.start[idx]
            assert dim_entries[dt].size == de.shape[idx]

        # Test.czi has no mosaic or scene dims in directory entries
        assert CziDimensionType.MOSAIC not in dim_entries
        assert CziDimensionType.SCENE not in dim_entries


def test_directory_entry_dimension_entries_mosaic():
    """Test read_dimension_entries includes M and S dims for mosaic tiles."""
    filename = DATA / 'Zenodo-7015307/S=3_1Pos_2Mosaic_T=2=Z=3_CH=2.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        de_mosaic = next(
            (d for d in czi.subblock_directory if d.mosaic_index >= 0), None
        )
        if de_mosaic is None:
            pytest.skip('no mosaic entry found')

        dim_entries = de_mosaic.read_dimension_entries(czi.filehandle)

        assert CziDimensionType.MOSAIC in dim_entries
        m = dim_entries[CziDimensionType.MOSAIC]
        assert m.start == de_mosaic.mosaic_index
        assert CziDimensionType.SCENE in dim_entries
        s = dim_entries[CziDimensionType.SCENE]
        assert s.start == de_mosaic.scene_index


def test_subblock_directory_file_positions():
    """Test CziSubBlockDirectorySegmentData.file_positions matches entries."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        fh = czi.filehandle
        fh.seek(czi.header.directory_position + 32)
        positions = CziSubBlockDirectorySegmentData.file_positions(fh)

        assert isinstance(positions, tuple)
        assert all(isinstance(p, int) for p in positions)
        assert len(positions) == len(czi.subblock_directory)
        assert positions == tuple(
            e.file_position for e in czi.subblock_directory
        )


def test_attachment_directory_file_positions():
    """Test CziAttachmentDirectorySegmentData.file_positions."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')
    with CziFile(filename) as czi:
        if not czi.header.attachment_directory_position:
            pytest.skip('no attachment directory')

        fh = czi.filehandle
        fh.seek(czi.header.attachment_directory_position + 32)
        positions = CziAttachmentDirectorySegmentData.file_positions(fh)

        assert isinstance(positions, tuple)
        assert all(isinstance(p, int) for p in positions)
        assert len(positions) == len(czi.attachment_directory)
        assert positions == tuple(
            e.file_position for e in czi.attachment_directory
        )

        # name_offset points to the 80 byte name field in the A1 record
        for entry in czi.attachment_directory:
            assert entry.name_offset == entry.offset + 48
            fh.seek(entry.name_offset)
            raw_name = fh.read(80).rstrip(b'\x00').decode()
            assert raw_name == entry.name


@pytest.mark.parametrize(
    ('filename', 'expected_name', 'expected_part'),
    [
        ('foo.czi', 'foo.czi', 0),
        ('foo.CZI', 'foo.czi', 0),  # extension is normalised to lowercase
        ('foo(1).czi', 'foo.czi', 1),
        ('foo(42).czi', 'foo.czi', 42),
        ('path/to/foo.czi', 'path/to/foo.czi', 0),
        ('path/to/foo(3).czi', 'path/to/foo.czi', 3),
    ],
)
def test_match_filename(filename, expected_name, expected_part):
    """Test match_filename parses CZI file name and part number."""
    name, part = match_filename(filename)
    assert name == expected_name
    assert part == expected_part


def test_match_filename_invalid():
    """Test match_filename raises ValueError for non-CZI file names."""
    with pytest.raises(ValueError, match='not a CZI'):
        match_filename('foo.tif')


def test_update_pending():
    """Test update_pending flag management in CziFile context manager."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    buf = io.BytesIO(filename.read_bytes())

    # flag is set True on __enter__ for a writable filehandle
    czi = CziFile(buf)
    offset = czi.header.segment.data_offset + 68
    assert not czi.header.update_pending
    czi.__enter__()
    try:
        assert czi.header.update_pending
        buf.seek(offset)
        assert struct.unpack('<i', buf.read(4))[0] == 1
    finally:
        czi.__exit__(None, None, None)
        czi.close()

    # flag is cleared False on clean __exit__
    buf = io.BytesIO(filename.read_bytes())
    with CziFile(buf) as czi:
        offset = czi.header.segment.data_offset + 68
        assert czi.header.update_pending
    assert not czi.header.update_pending
    buf.seek(offset)
    assert struct.unpack('<i', buf.read(4))[0] == 0

    # flag stays True when context exits via exception
    buf = io.BytesIO(filename.read_bytes())
    offset = None
    with contextlib.suppress(RuntimeError), CziFile(buf) as czi:
        offset = czi.header.segment.data_offset + 68
        raise RuntimeError
    assert offset is not None
    buf.seek(offset)
    assert struct.unpack('<i', buf.read(4))[0] == 1

    # flag is not written for a read-only file stream
    with open(filename, 'rb') as f, CziFile(f) as czi:
        assert not czi.header.update_pending
    assert not czi.header.update_pending


def test_czi():
    """Test all implemented public interfaces of CziFile using Test.czi."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # BinaryFile base attributes
        assert not czi.closed
        assert not czi.filehandle.closed
        assert czi.name == 'Test.czi'
        assert czi.filename == 'Test.czi'
        assert czi.filepath == os.path.normpath(filename)
        assert czi.dirname == os.path.normpath(filename.parent)
        assert czi.attrs == {
            'name': 'Test.czi',
            'filepath': os.path.normpath(filename),
        }

        # repr / str
        assert repr(czi).startswith("<CziFile 'Test.czi'")
        assert 'CziFile' in str(czi)

        # set_lock
        assert isinstance(czi.lock, NullLock)
        czi.set_lock(True)
        assert hasattr(czi.lock, 'acquire')
        assert hasattr(czi.lock, 'release')
        assert not isinstance(czi.lock, NullLock)
        czi.set_lock(False)
        assert isinstance(czi.lock, NullLock)

        # header
        hdr = czi.header
        assert hdr.version == (1, 0)
        assert hdr.file_part == 0
        assert hdr.update_pending is False
        assert hdr.metadata_position > 0
        assert hdr.directory_position > 0
        assert repr(hdr).startswith("<CziFileHeaderSegmentData 'ZISRAWFILE'")

        # subblock_directory
        sbd = czi.subblock_directory
        assert isinstance(sbd, tuple)
        assert len(sbd) == 66
        assert all(isinstance(e, CziDirectoryEntryDV) for e in sbd)

        de = sbd[0]
        assert de.dims == ('T', 'C', 'Z', 'Y', 'X', 'S')
        assert de.shape == (1, 1, 1, 181, 312, 1)
        assert de.stored_shape == (1, 1, 1, 181, 312, 1)
        assert de.start == (0, 0, 0, 0, 0, 0)
        assert de.stop == (1, 1, 1, 181, 312, 1)
        assert de.pixel_type == CziPixelType.GRAY16
        assert de.compression == CziCompressionType.UNCOMPRESSED
        assert de.pyramid_type == CziPyramidType.NONE
        assert de.mosaic_index == -1
        assert de.is_pyramid is False
        assert de.dtype == numpy.uint16
        assert de.file_position > 0
        assert len(de.start_coordinate) == 6

        # filtered_subblock_directory
        fsd = czi.filtered_subblock_directory
        assert isinstance(fsd, tuple)
        assert len(fsd) == 66

        # metadata
        xml = czi.metadata()
        assert isinstance(xml, str)
        assert len(xml) > 0
        assert '<ImageDocument' in xml

        d = czi.metadata(asdict=True)
        assert isinstance(d, dict)
        assert 'ImageDocument' in d

        # subblocks
        subblocks = list(czi.subblocks())
        assert len(subblocks) == 66
        assert all(isinstance(sb, CziSubBlockSegmentData) for sb in subblocks)

        sb = subblocks[0]
        assert repr(sb).startswith("<CziSubBlockSegmentData 'ZISRAWSUBBLOCK'")
        assert sb.data_size == 112944
        assert sb.metadata_size > 0
        assert isinstance(sb.directory_entry, CziDirectoryEntryDV)
        assert sb.directory_entry.pixel_type == CziPixelType.GRAY16

        sbm = sb.metadata(asdict=True)
        assert isinstance(sbm, dict)
        sbm_str = sb.metadata(asdict=False)
        assert isinstance(sbm_str, str)

        # attachment_directory
        att_dir = czi.attachment_directory
        assert isinstance(att_dir, tuple)
        assert len(att_dir) == 1
        assert isinstance(att_dir[0], CziAttachmentEntryA1)

        a = att_dir[0]
        assert a.name == 'Thumbnail'
        assert repr(a).startswith("<CziAttachmentEntryA1 'Thumbnail'")
        assert a.file_position > 0

        # attachments
        attachments = list(czi.attachments())
        assert len(attachments) == 1
        assert isinstance(attachments[0], CziAttachmentSegmentData)
        att_data = attachments[0].data()
        assert isinstance(att_data, numpy.ndarray)

        # segments
        segs = list(czi.segments('ZISRAWSUBBLOCK'))
        assert len(segs) == 66

        # scenes property
        scenes = czi.scenes
        assert isinstance(scenes, CziScenes)
        assert repr(scenes) == '<CziScenes 1>'
        assert len(scenes) == 1

        # CziScenes: iteration yields S-coordinate keys (Mapping protocol)
        assert list(scenes) == [0]
        # CziScenes.values() yields CziImage instances
        imgs = list(scenes.values())
        assert len(imgs) == 1
        assert all(isinstance(i, CziImage) for i in imgs)

        # CziScenes.__getitem__ uses S-coordinate as key
        img0 = scenes[0]
        assert isinstance(img0, CziImage)
        assert img0.scene_indices is None  # implicit single-scene file
        with pytest.raises(KeyError):
            scenes[5]

        # CziScenes.__call__
        img_all = scenes()
        assert isinstance(img_all, CziImage)
        assert img_all.sizes == {'C': 3, 'Z': 22, 'Y': 181, 'X': 312}

        img_s0 = scenes(scene=0)
        assert img_s0.sizes == {'C': 3, 'Z': 22, 'Y': 181, 'X': 312}

        img_filtered = scenes(scene=0, C=1)
        assert img_filtered.sizes == {'Z': 22, 'Y': 181, 'X': 312}

        with pytest.raises(KeyError):
            scenes(scene=5)

        # CziImage.__call__ with selection
        img = scenes[0]
        sub_c0 = img(C=0)
        assert sub_c0.sizes == {'Z': 22, 'Y': 181, 'X': 312}
        assert sub_c0.dims == ('Z', 'Y', 'X')

        # dimension reordering via kwargs order
        sub_reorder = img(Z=None, C=None)
        assert sub_reorder.dims == ('Z', 'C', 'Y', 'X')
        assert sub_reorder.shape == (22, 3, 181, 312)

        # selection + roi
        sub_roi = img(C=0, roi=(10, 20, 80, 40))
        assert sub_roi.sizes == {'Z': 22, 'Y': 40, 'X': 80}

        # cannot re-apply selection/roi
        with pytest.raises(TypeError):
            sub_c0(Z=0)

        # spatial dims rejected
        with pytest.raises(ValueError):
            img(X=0)

        # unknown dim rejected
        with pytest.raises(ValueError):
            img(Q=0)

        # pixeltype override via __call__
        img_gray8 = img(pixeltype=CziPixelType.GRAY8)
        assert img_gray8 is not img
        assert img_gray8.pixeltype == CziPixelType.GRAY8
        assert img_gray8.dtype == numpy.dtype('uint8')
        assert img_gray8.shape == img.shape  # no S dim with default squeeze

        # pixeltype + selection combined
        img_c0_gray8 = img(C=0, pixeltype=CziPixelType.GRAY8)
        assert img_c0_gray8.pixeltype == CziPixelType.GRAY8
        assert img_c0_gray8.dtype == numpy.dtype('uint8')
        assert img_c0_gray8.shape == (22, 181, 312)

        # pixeltype-only call allowed on already-selected image (no TypeError)
        sub_c0_gray8 = sub_c0(pixeltype=CziPixelType.GRAY8)
        assert sub_c0_gray8.pixeltype == CziPixelType.GRAY8
        assert sub_c0_gray8.shape == (22, 181, 312)

        # pixeltype=None is a no-op (returns self)
        assert img(pixeltype=None) is img

        # CziImage attributes
        img = scenes[0]
        assert isinstance(img, CziImage)
        assert repr(img).startswith("<CziImage 'Scene 0' ")
        assert str(img) == repr(img)
        assert img.axes == 'CZYX'
        assert img.dims == ('C', 'Z', 'Y', 'X')
        assert img.shape == (3, 22, 181, 312)
        assert img.sizes == {'C': 3, 'Z': 22, 'Y': 181, 'X': 312}
        assert img.dtype == numpy.uint16
        assert img.ndim == 4
        assert img.size == 3 * 22 * 181 * 312
        assert img.nbytes == img.size * 2
        assert img.pixeltype == CziPixelType.GRAY16
        assert img._default_maxworkers == 1  # UNCOMPRESSED
        # pixeltype override is reflected by dtype, repr, and nbytes
        img_u8 = img(pixeltype=CziPixelType.GRAY8)
        assert img_u8.dtype == numpy.dtype('uint8')
        assert img_u8.nbytes == img_u8.size * 1
        assert img.is_pyramid is False
        assert img.is_upsampled is False
        assert img.is_pyramid_level is False
        assert img.is_downsampled is False
        assert len(img.levels) == 1
        assert img.levels[0] is img
        assert len(img.directory_entries) == 66
        assert all(
            isinstance(e, CziDirectoryEntryDV) for e in img.directory_entries
        )

        # start indices (squeezed)
        assert len(img.start) == 4

        # bbox: (x, y, width, height) in global CZI coordinates
        assert img.bbox == (0, 0, 312, 181)

        # coords and attrs
        assert isinstance(img.coords, Mapping)
        assert isinstance(img.attrs, Mapping)
        assert img.name == 'Scene 0'
        assert img.attrs['filepath'] == os.path.normpath(filename)

        # CziImageChunks interface
        chunks = img.chunks()
        assert isinstance(chunks, CziImageChunks)
        assert len(chunks) == 3 * 22

        # chunks from selection
        img_c0 = img(C=0)
        assert img_c0.sizes == {'Z': 22, 'Y': 181, 'X': 312}
        assert len(img_c0.chunks()) == 22

        # slice selects by absolute Z coordinate range
        img_z = img(Z=slice(0, 5))
        assert img_z.sizes['Z'] == 5
        assert len(img_z.chunks()) == 3 * 5

    # closed after context exit
    assert czi.closed


def test_attachments_slide_scanner():
    """Test attachments in a slide-scanner CZI (Zeiss-5-Uncompressed)."""
    filename = DATA / 'OpenSlide/Zeiss-5-Uncompressed.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        attachments = {a.attachment_entry.name: a for a in czi.attachments()}
        assert set(attachments) == {
            'EventList',
            'TimeStamps',
            'Label',
            'SlidePreview',
            'Profile',
            'Thumbnail',
        }

        # Thumbnail - JPEG-decoded ndarray
        thumb_entry = attachments['Thumbnail'].attachment_entry
        assert thumb_entry.content_file_type == CziContentFileType.JPG
        thumb = attachments['Thumbnail'].data()
        assert isinstance(thumb, numpy.ndarray)
        assert thumb.ndim == 3
        assert thumb.shape[2] == 3  # BGR

        # TimeStamps - float64 ndarray
        ts_entry = attachments['TimeStamps'].attachment_entry
        assert ts_entry.content_file_type == CziContentFileType.CZTIMS
        ts = attachments['TimeStamps'].data()
        assert isinstance(ts, numpy.ndarray)
        assert ts.dtype == numpy.float64

        # EventList - tuple
        ev_entry = attachments['EventList'].attachment_entry
        assert ev_entry.content_file_type == CziContentFileType.CZEVL
        ev = attachments['EventList'].data()
        assert isinstance(ev, tuple)

        # Label - embedded CZI image (ndarray)
        lbl_entry = attachments['Label'].attachment_entry
        assert lbl_entry.content_file_type == CziContentFileType.CZI
        lbl = attachments['Label'].data()
        assert isinstance(lbl, numpy.ndarray)
        assert lbl.ndim == 3
        assert lbl.shape[2] == 3

        # SlidePreview - embedded CZI image (ndarray)
        sp_entry = attachments['SlidePreview'].attachment_entry
        assert sp_entry.content_file_type == CziContentFileType.CZI
        sp = attachments['SlidePreview'].data()
        assert isinstance(sp, numpy.ndarray)
        assert sp.ndim == 3
        assert sp.shape[2] == 3

        # Profile - ZIP-compressed bytes
        prof_entry = attachments['Profile'].attachment_entry
        assert prof_entry.content_file_type == CziContentFileType.ZIP_COMP
        prof = attachments['Profile'].data()
        assert isinstance(prof, bytes)
        assert len(prof) > 0

        # raw=True always returns bytes regardless of content type
        raw = attachments['Thumbnail'].data(raw=True)
        assert isinstance(raw, bytes)
        assert raw[:2] == b'\xff\xd8'  # JPEG SOI marker

        # raw=True on an embedded CZI gives bytes that can be opened as CziFile
        label_raw = attachments['Label'].data(raw=True)
        assert isinstance(label_raw, bytes)
        assert label_raw[:10] == b'ZISRAWFILE'
        with CziFile(io.BytesIO(label_raw)) as embedded:
            assert len(embedded.scenes) == 1
            assert embedded.scenes[0].shape == lbl.shape

        # metadata_segment exposes xml_offset and xml_size
        # needed for in-place patching via filehandle
        mds = czi.metadata_segment
        assert isinstance(mds, CziMetadataSegmentData)
        assert isinstance(mds.xml_offset, int)
        assert mds.xml_offset > 0
        assert isinstance(mds.xml_size, int)
        assert mds.xml_size > 0
        fh = czi.filehandle
        fh.seek(mds.xml_offset)
        xml_bytes = fh.read(14)
        assert xml_bytes == b'<ImageDocument'

        # name_offset points to the 80-byte name field in ATTDIR entry
        for entry in czi.attachment_directory:
            assert entry.name_offset == entry.offset + 48
            fh.seek(entry.name_offset)
            raw_name = fh.read(80).rstrip(b'\x00').decode()
            assert raw_name == entry.name


def test_attachments_palm():
    """Test attachments in a PALM CZI (PALM_OnlineVerrechnet)."""
    filename = DATA / 'Zenodo-10577621/PALM_OnlineVerrechnet.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        attachments = {a.attachment_entry.name: a for a in czi.attachments()}
        assert set(attachments) == {
            'Thumbnail',
            'LookupTables',
            'TimeStamps',
            'EventList',
            'PalMoleculeList',
            'Image',
            'PalSumTirfRawData',
        }

        # Thumbnail - JPEG-decoded ndarray
        thumb = attachments['Thumbnail'].data()
        assert isinstance(thumb, numpy.ndarray)
        assert thumb.shape == (256, 256, 3)

        # LookupTables - tuple of lookup table entries
        lut_entry = attachments['LookupTables'].attachment_entry
        assert lut_entry.content_file_type == CziContentFileType.CZLUT
        lut = attachments['LookupTables'].data()
        assert isinstance(lut, tuple)
        assert len(lut) >= 1

        # TimeStamps - float64 ndarray
        ts = attachments['TimeStamps'].data()
        assert isinstance(ts, numpy.ndarray)
        assert ts.dtype == numpy.float64

        # PalMoleculeList - opaque bytes (CZPML)
        pml_entry = attachments['PalMoleculeList'].attachment_entry
        assert pml_entry.content_file_type == CziContentFileType.CZPML
        pml = attachments['PalMoleculeList'].data()
        assert isinstance(pml, bytes)
        assert len(pml) > 0

        # Image - embedded ZISRAW CZI (ndarray)
        img_entry = attachments['Image'].attachment_entry
        assert img_entry.content_file_type == CziContentFileType.ZISRAW
        img = attachments['Image'].data()
        assert isinstance(img, numpy.ndarray)
        assert img.shape == (100, 512, 512)

        # PalSumTirfRawData - embedded ZISRAW CZI (ndarray)
        pst_entry = attachments['PalSumTirfRawData'].attachment_entry
        assert pst_entry.content_file_type == CziContentFileType.ZISRAW
        pst = attachments['PalSumTirfRawData'].data()
        assert isinstance(pst, numpy.ndarray)
        assert pst.shape == (512, 512)


def test_palm_superresolution():
    """Test PALM file where stored_shape > entry.shape (super-resolution)."""
    # The tile is stored at 1206x1206 but the logical output space is 512x512.
    # czifile must downsample the tile using floor-based nearest-neighbour
    # (exact inverse of numpy.repeat) to match the ZEN Blue OME-TIFF export.

    filename = DATA / 'Zenodo-10577621/Image 5_PALM_verrechnet.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        assert len(czi.scenes) == 1
        image = czi.scenes[0]
        # logical shape is 512x512 even though stored_shape is 1206x1206
        assert image.shape == (512, 512)
        assert image.dims == ('Y', 'X')
        assert image.pixeltype == CziPixelType.GRAY16
        assert image.is_upsampled is False
        assert image.is_pyramid_level is False
        assert image.is_downsampled is True
        e = image.directory_entries[0]
        assert e.stored_shape[-3:-1] == (1206, 1206)
        assert e.shape[-3:-1] == (512, 512)
        arr = image.asarray()
        assert arr.shape == (512, 512)
        assert arr.dtype == numpy.uint16
        # validate pixel-exact match against ZEN Blue OME-TIFF ground truth
        ome_path = filename.with_suffix('').with_suffix('.ome.tiff')
        if ome_path.exists():
            expected = tifffile.imread(ome_path).astype(numpy.uint16)
            assert numpy.array_equal(arr, expected)
        # chunks interface must match asarray: chunk asarray must downsample
        # the stored 1206x1206 tile to the logical 512x512 output
        plane = next(iter(image.chunks())).asarray()
        assert plane.shape == (512, 512)
        assert plane.dtype == numpy.uint16
        assert numpy.array_equal(plane, arr)


def test_palm_drift():
    """Test PALM file with mixed tiles: one normal, one super-resolution."""
    # Palm_mitDrift.czi has two entries:
    # - entry 0: stored_shape == shape (256x256, normal)
    # - entry 1: stored_shape=(2560x2560) > shape=(256x256) (super-resolution)
    # Both channels must be decoded to the logical 256x256 output size,
    # pixel-exact against the ZEN Blue OME-TIFF export.

    filename = DATA / 'Zenodo-10577621/Palm_mitDrift.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        assert len(czi.scenes) == 1
        image = czi.scenes[0]
        assert image.shape == (2, 256, 256)
        assert image.dims == ('C', 'Y', 'X')
        assert image.pixeltype == CziPixelType.GRAY16
        assert image.is_upsampled is False
        assert image.is_pyramid_level is False
        assert image.is_downsampled is True
        entries = image.directory_entries
        assert entries[0].stored_shape[-3:-1] == (256, 256)
        assert entries[0].shape[-3:-1] == (256, 256)
        assert entries[1].stored_shape[-3:-1] == (2560, 2560)
        assert entries[1].shape[-3:-1] == (256, 256)
        arr = image.asarray()
        assert arr.shape == (2, 256, 256)
        assert arr.dtype == numpy.uint16
        # validate pixel-exact match against ZEN Blue OME-TIFF ground truth
        ome_path = filename.with_suffix('').with_suffix('.ome.tiff')
        if ome_path.exists():
            expected = tifffile.imread(str(ome_path)).astype(numpy.uint16)
            assert numpy.array_equal(arr, expected)
        # chunks interface must match asarray for both tiles:
        # C=0 is stored at full res (256x256), C=1 must be downsampled
        # from stored 2560x2560 to logical 256x256
        plane_c0 = next(iter(image(C=0).chunks())).asarray()
        assert plane_c0.shape == (256, 256)
        assert plane_c0.dtype == numpy.uint16
        assert numpy.array_equal(plane_c0, arr[0])
        plane_c1 = next(iter(image(C=1).chunks())).asarray()
        assert plane_c1.shape == (256, 256)
        assert plane_c1.dtype == numpy.uint16
        assert numpy.array_equal(plane_c1, arr[1])


def test_timestamps():
    """Test reading timestamps from TimeStamps attachment."""
    # https://github.com/cgohlke/czifile/issues/11

    filename = DATA / 'CZI_Timelapse_ZEN_Blue.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        for attachment in czi.attachments():
            if attachment.attachment_entry.name == 'TimeStamps':
                timestamps = attachment.data()
                break
        else:
            pytest.fail('TimeStamps attachment not found')

    assert isinstance(timestamps, numpy.ndarray)
    assert timestamps.dtype == numpy.dtype('float64')
    assert len(timestamps) == 5
    assert timestamps[0] == pytest.approx(1.389, abs=1e-3)
    assert timestamps[-1] == pytest.approx(5.275, abs=1e-3)
    assert round(float(timestamps[-1] - timestamps[0]), 2) == 3.89
    # timestamps are strictly increasing
    assert all(
        timestamps[i] < timestamps[i + 1] for i in range(len(timestamps) - 1)
    )


def test_roi_no_intersection():
    """Test that roi with no intersecting tiles raises ValueError."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        # ROI completely outside the image extent (image is 312x181)
        with pytest.raises(ValueError, match='roi matches no subblocks'):
            img(roi=(10000, 10000, 64, 64))

        # zero-area ROIs are rejected before any subblock filtering
        with pytest.raises(
            ValueError, match='roi width and height must be positive'
        ):
            img(roi=(0, 0, 0, 64))
        with pytest.raises(
            ValueError, match='roi width and height must be positive'
        ):
            img(roi=(0, 0, 64, 0))
        with pytest.raises(
            ValueError, match='roi width and height must be positive'
        ):
            img(roi=(0, 0, -1, 64))


def test_roi_larger_than_image():
    """Test that roi larger than image returns zero-padded result."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        # ROI extending far beyond the image extent (312x181)
        big = img(C=0, Z=0, roi=(0, 0, 1024, 1024))
        assert big.sizes == {'Y': 1024, 'X': 1024}
        arr = big.asarray()
        assert arr.shape == (1024, 1024)
        # pixels outside the actual 312x181 extent must be zero
        assert int(arr[:, 312:].sum()) == 0
        assert int(arr[181:, :].sum()) == 0
        # pixels inside the extent are non-zero (image has data there)
        assert int(arr[:181, :312].sum()) > 0


def test_fillvalue():
    """Test fillvalue parameter fills pixels not covered by any subblock."""
    filename = DATA / 'pylibCZIrw/c1_bgr24.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    bgr_fill = (10, 20, 30)  # B=10, G=20, R=30
    with CziFile(filename) as czi:
        # image is 256x256x3; use a 512x512 ROI so the right/bottom halves
        # are uncovered and should receive the fillvalue
        img = czi.scenes[0](roi=(0, 0, 512, 512))
        assert img.sizes == {'Y': 512, 'X': 512, 'S': 3}

        # scalar fillvalue=0: uncovered region must be zero
        arr0 = img.asarray(fillvalue=0)
        assert arr0.shape == (512, 512, 3)
        assert int(arr0[:, 256:, :].sum()) == 0
        assert int(arr0[256:, :, :].sum()) == 0

        # BGR triplet fillvalue: uncovered region must equal the triplet
        arr_bgr = img.asarray(fillvalue=bgr_fill)
        assert numpy.array_equal(
            arr_bgr[:, 256:, :],
            numpy.full((512, 256, 3), bgr_fill, dtype=arr_bgr.dtype),
        )
        assert numpy.array_equal(
            arr_bgr[256:, :, :],
            numpy.full((256, 512, 3), bgr_fill, dtype=arr_bgr.dtype),
        )

        # covered region is unaffected by fillvalue
        assert numpy.array_equal(arr0[:256, :256, :], arr_bgr[:256, :256, :])

        # fillvalue=None with pre-allocated array: uncovered pixels untouched
        pre = numpy.full((512, 512, 3), 99, dtype=arr0.dtype)
        img.asarray(out=pre, fillvalue=None)
        assert int(pre[:, 256:, :].min()) == 99
        assert int(pre[256:, :, :].min()) == 99
        assert numpy.array_equal(pre[:256, :256, :], arr0[:256, :256, :])


def test_squeeze():
    """Test CziFile with squeeze=False preserves length-1 dimensions."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename, squeeze=False) as czi:
        img = czi.scenes[0]
        assert img.sizes == {
            'T': 1,
            'C': 3,
            'Z': 22,
            'Y': 181,
            'X': 312,
            'S': 1,
        }
        assert img.shape == (1, 3, 22, 181, 312, 1)
        assert img.ndim == 6

        # pixeltype override with same samples count preserves S dim
        img_u8 = img(pixeltype=CziPixelType.GRAY8)
        assert img_u8.dtype == numpy.dtype('uint8')
        assert img_u8.sizes['S'] == 1
        assert img_u8.shape == (1, 3, 22, 181, 312, 1)


@pytest.mark.parametrize(
    ('filename', 'compression', 'shape', 'dtype', 'checksum'),
    [
        (
            'pylibCZIrw/c1_gray8.czi',
            CziCompressionType.UNCOMPRESSED,
            (256, 256),
            'uint8',
            2290374,
        ),
        (
            'OpenSlide/Zeiss-5-SlidePreview-JXR.czi',
            CziCompressionType.JPEGXR,
            (615, 1260, 3),
            'uint16',
            None,  # JPEGXR is lossy; no exact checksum
        ),
        (
            'OpenSlide/Zeiss-5-SlidePreview-Zstd0.czi',
            CziCompressionType.ZSTD,
            (615, 1260, 3),
            'uint16',
            115483010048,
        ),
        (
            'OpenSlide/Zeiss-5-SlidePreview-Zstd1-HiLo.czi',
            CziCompressionType.ZSTDHDR,
            (615, 1260, 3),
            'uint16',
            115483010048,
        ),
        (
            'SPIM_Compression_101.czi',
            CziCompressionType.SYSTEMRAW,
            (2, 501, 180, 1488),
            'uint16',
            26282333614,
        ),
        (
            'Zenodo-10577621/40_Dual.czi',
            CziCompressionType.CAMERARAW,
            (2, 40, 1040, 1388),
            'uint16',
            ValueError,  # CAMERARAW is not supported
        ),
        # CziCompressionType.JPEG (1) and CziCompressionType.LZW (2)
        # are not covered; no test files are available.
    ],
)
def test_compression(filename, compression, shape, dtype, checksum):
    """Test decoding of each supported compression type."""
    fpath = DATA / filename
    if not fpath.exists():
        pytest.skip(f'{fpath!r} not found')
    with CziFile(fpath) as czi:
        compressions = {e.compression for e in czi.subblock_directory}
        assert compression in compressions
        img = czi.scenes[0]
        if (
            compression == CziCompressionType.UNCOMPRESSED
            or int(img.compression) > 999  # SYSTEMRAW
            or img._entries[0].mosaic_index >= 0
            or len(img._entries) < 3
        ):
            assert img._default_maxworkers == 1
        else:
            assert img._default_maxworkers == default_maxworkers()
        if isinstance(checksum, type) and issubclass(checksum, Exception):
            with pytest.raises(checksum):
                czi.asarray()
            return
        arr = czi.asarray()
    assert arr.shape == shape
    assert arr.dtype == numpy.dtype(dtype)
    if checksum is None:
        assert arr.any()
    else:
        assert int(arr.sum()) == checksum


def test_compression_jpegxr_raw_fallback():
    """Test JPEGXR subblocks stored as uncompressed raw pixels."""
    # Some CZI writers set the compression field to JPEGXR but store
    # uncompressed pixel data (data_size == expected uncompressed size).
    # Verify the workaround reads these subblocks correctly.

    filename = DATA / '2019_10_03__10_10__0279-1.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')
    with CziFile(filename) as czi:
        # file uses JPEGXR compression
        compressions = {e.compression for e in czi.subblock_directory}
        assert CziCompressionType.JPEGXR in compressions

        # some subblocks have data_size == uncompressed size (writer bug)
        buggy = []
        for e in czi.subblock_directory:
            seg = e.read_segment_data(czi)
            if seg.SID != 'ZISRAWSUBBLOCK':
                continue
            dtype = e.pixel_type.dtype
            expected = math.prod(e.stored_shape) * dtype.itemsize
            if seg.data_size == expected and e.compression != 0:
                buggy.append(seg)
        assert len(buggy) == 12

        # verify one affected subblock decodes correctly
        data = buggy[0].data()
        assert data.shape == (1, 2040, 2040, 1)
        assert data.dtype == numpy.uint16
        assert int(data.sum()) == 1087553117

        # verify full image composites correctly
        arr = czi.asarray()
        assert arr.shape == (3, 5714, 7545)
        assert arr.dtype == numpy.uint16
        assert int(arr.sum(dtype=numpy.float64)) == 212644498386


def test_pixeltype_gray8_gray16():
    """Test mixed GRAY8/GRAY16 pixel types in c2_gray8_gray16.czi."""
    # C=0 is GRAY8, C=1 is GRAY16.  The auto-promoted pixeltype is GRAY16
    # (numpy.result_type promotes uint8 -> uint16), so the default composite
    # output is lossless.  Downcasting to GRAY8 via pixeltype= clips the
    # GRAY16 channel.

    filename = DATA / 'pylibCZIrw/c2_gray8_gray16.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        pts = {e.pixel_type for e in czi.subblock_directory}
        assert pts == {CziPixelType.GRAY8, CziPixelType.GRAY16}

        img = czi.scenes[0]
        assert img(C=0).pixeltype == CziPixelType.GRAY8
        assert img(C=1).pixeltype == CziPixelType.GRAY16

        # auto-promoted pixeltype:
        # GRAY16 (numpy.result_type(uint8, uint16) = uint16)
        assert img.pixeltype == CziPixelType.GRAY16
        assert img.dtype == numpy.dtype('uint16')
        assert img.dims == ('C', 'Y', 'X')
        assert img.shape == (2, 256, 256)

        # default composite: lossless - GRAY8 channel widened to uint16
        arr = img.asarray()
        assert arr.dtype == numpy.dtype('uint16')
        assert arr.shape == (2, 256, 256)
        assert int(arr[1].max()) == 29184  # full dynamic range, no clipping

        # GRAY8 downcast: GRAY16 channel clipped to uint8 range
        arr8 = img(pixeltype=CziPixelType.GRAY8).asarray()
        assert arr8.dtype == numpy.dtype('uint8')
        assert int(arr8[1].max()) == 114


def test_pixeltype_gray8_gray32float():
    """Test mixed GRAY8/GRAY32FLOAT pixel types in Muenze 2.czi."""
    # C=0 is GRAY8, C=1 is GRAY32FLOAT.  The auto-promoted pixeltype is
    # GRAY32FLOAT (numpy.result_type promotes uint8 → float32), so the
    # default composite output preserves the full float range, including
    # negative values that cannot occur in uint8.
    # The GRAY32FLOAT channel is a ZEN Online Adaptive Deconvolution (OAD)
    # result.  ZEN stores NaN as a sentinel for pixels with no valid
    # deconvolution result (masked or out-of-kernel regions).  When converting
    # to an integer type, NaN must be mapped to 0 rather than propagated,
    # which would trigger a RuntimeWarning on the integer cast.

    filename = DATA / 'ZEN-OAD/Input/CZI Images/Muenze 2.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        pts = {e.pixel_type for e in czi.subblock_directory}
        assert pts == {CziPixelType.GRAY8, CziPixelType.GRAY32FLOAT}

        img = czi.scenes[0]
        assert img(C=0).pixeltype == CziPixelType.GRAY8
        assert img(C=1).pixeltype == CziPixelType.GRAY32FLOAT

        # auto-promoted pixeltype:
        # GRAY32FLOAT (numpy.result_type(uint8, float32) = float32)
        assert img.pixeltype == CziPixelType.GRAY32FLOAT
        assert img.dtype == numpy.dtype('float32')
        assert img.dims == ('C', 'Y', 'X')
        assert img.shape == (2, 2048, 2048)

        # explicit GRAY32FLOAT: same as default; confirms override still works
        img_f32 = img(pixeltype=CziPixelType.GRAY32FLOAT)
        assert img_f32.pixeltype == CziPixelType.GRAY32FLOAT
        assert img_f32.dtype == numpy.dtype('float32')
        assert img_f32.shape == (2, 2048, 2048)

        # small ROI on C=1 to limit I/O; float channel has negative values
        arr = czi.scenes[0](
            C=1, pixeltype=CziPixelType.GRAY32FLOAT, roi=(0, 0, 32, 32)
        ).asarray()
        assert arr.dtype == numpy.dtype('float32')
        assert arr.shape == (32, 32)
        assert arr.min() < 0

        # the full float channel contains NaN sentinels (pixels with no valid
        # deconvolution result); converting to uint8 must not raise
        # RuntimeWarning: invalid value encountered in cast
        arr_u8 = czi.scenes[0](C=1, pixeltype=CziPixelType.GRAY8).asarray()
        assert arr_u8.dtype == numpy.dtype('uint8')
        assert not numpy.any(numpy.isnan(arr_u8.astype(numpy.float32)))
        # NaN pixels must be mapped to 0, not to some arbitrary integer
        arr_f32_full = czi.scenes[0](
            C=1, pixeltype=CziPixelType.GRAY32FLOAT
        ).asarray()
        nan_mask = numpy.isnan(arr_f32_full)
        assert nan_mask.any(), 'expected NaN sentinels in OAD float channel'
        assert numpy.all(arr_u8[nan_mask] == 0)


def test_pixeltype_bgra_opaque():
    """Test that BGRA32 images return alpha=255 even when stored alpha is 0."""
    # CZI writers routinely leave the alpha channel unpopulated (all zeros).
    # The reader must force alpha to 255 so the image displays correctly.
    # Tested for both raw subblock data and via asarray().

    filename = DATA / 'CZI-testimages/Z-Stack-Anno_RGB.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # raw subblock: alpha must be 255 despite stored zeros
        sb = czi.filtered_subblock_directory[0].read_segment_data(czi)
        raw = sb.data()
        assert raw.shape[-1] == 4, 'expected BGRA (4 channels)'
        assert numpy.all(raw[..., 3] == 255), 'alpha channel must be 255'

        # asarray: alpha must also be 255 in the composed array
        arr = czi.scenes[0].asarray()
        assert arr.shape[-1] == 4, 'expected RGBA (4 channels)'
        assert numpy.all(arr[..., 3] == 255), 'alpha channel must be 255'


def test_pixeltype_conversions():
    """Test cross-family pixeltype conversions via pixeltype= override."""
    # Uses two files that are present in the standard test data set:
    #   c1_bgr24.czi - BGR24 source (B == G == R, grayscale encoded as BGR)
    #   c2_gray8_gray16.czi - mixed GRAY8 (C=0) / GRAY16 (C=1) source
    # Conversions tested:
    # BGR24 -> GRAY8        simple mean of B+G+R, cast to uint8
    # BGR24 -> BGRA32       BGR channels copied, alpha channel zero-filled
    # BGR24 -> GRAY32FLOAT  mean of B+G+R as float32
    # GRAY8 -> BGR24        each gray value tiled into B=G=R, dtype preserved
    # GRAY16 -> BGR24       each channel >> 8 cast to uint8, tiled into B=G=R

    bgr_fname = DATA / 'pylibCZIrw/c1_bgr24.czi'
    mix_fname = DATA / 'pylibCZIrw/c2_gray8_gray16.czi'
    if not bgr_fname.exists():
        pytest.skip(f'{bgr_fname!r} not found')
    if not mix_fname.exists():
        pytest.skip(f'{mix_fname!r} not found')

    with CziFile(bgr_fname) as czi:
        img_bgr = czi.scenes[0]
        assert img_bgr.pixeltype == CziPixelType.BGR24
        arr_bgr = img_bgr.asarray()
        assert arr_bgr.shape == (256, 256, 3)

        # BGR24 -> GRAY8: mean of 3 channels cast to uint8
        arr_g8 = img_bgr(pixeltype=CziPixelType.GRAY8).asarray()
        assert arr_g8.dtype == numpy.dtype('uint8')
        assert arr_g8.shape == (256, 256)
        expected_g8 = (
            arr_bgr[..., :3].sum(axis=-1, dtype=numpy.float32) / 3.0
        ).astype(numpy.uint8)
        assert numpy.array_equal(arr_g8, expected_g8)
        assert int(arr_g8.sum()) == 2290374

        # BGR24 -> BGRA32: BGR copied, alpha channel opaque (255)
        arr_bgra = img_bgr(pixeltype=CziPixelType.BGRA32).asarray()
        assert arr_bgra.dtype == numpy.dtype('uint8')
        assert arr_bgra.shape == (256, 256, 4)
        assert numpy.array_equal(arr_bgra[..., :3], arr_bgr)
        assert numpy.all(arr_bgra[..., 3] == 255)

        # BGR24 -> GRAY32FLOAT: mean of 3 channels as float32
        arr_gf = img_bgr(pixeltype=CziPixelType.GRAY32FLOAT).asarray()
        assert arr_gf.dtype == numpy.dtype('float32')
        assert arr_gf.shape == (256, 256)
        expected_gf = arr_bgr[..., :3].sum(axis=-1, dtype=numpy.float32) / 3.0
        assert numpy.allclose(arr_gf, expected_gf)

    with CziFile(mix_fname) as czi:
        img = czi.scenes[0]

        # GRAY8 -> BGR24: each gray value tiled into all three channels
        img_c0 = img(C=0)
        assert img_c0.pixeltype == CziPixelType.GRAY8
        arr_g = img_c0.asarray()  # shape (256, 256)
        arr_g_bgr = img_c0(pixeltype=CziPixelType.BGR24).asarray()
        assert arr_g_bgr.dtype == numpy.dtype('uint8')
        assert arr_g_bgr.shape == (256, 256, 3)
        assert numpy.array_equal(arr_g_bgr[:, :, 0], arr_g)
        assert numpy.array_equal(arr_g_bgr[:, :, 1], arr_g)
        assert numpy.array_equal(arr_g_bgr[:, :, 2], arr_g)

        # GRAY16 -> BGR24: >> 8 to fit uint8, tiled into all three channels
        img_c1 = img(C=1)
        assert img_c1.pixeltype == CziPixelType.GRAY16
        arr16 = img_c1.asarray()  # shape (256, 256)
        arr16_bgr = img_c1(pixeltype=CziPixelType.BGR24).asarray()
        assert arr16_bgr.dtype == numpy.dtype('uint8')
        assert arr16_bgr.shape == (256, 256, 3)
        expected_ch = (arr16 >> 8).astype(numpy.uint8)
        assert numpy.array_equal(arr16_bgr[:, :, 0], expected_ch)
        assert numpy.array_equal(arr16_bgr[:, :, 1], expected_ch)
        assert numpy.array_equal(arr16_bgr[:, :, 2], expected_ch)


def test_storedsize_regular():
    """Test storedsize on regular file with no resampling."""
    filename = DATA / 'Test.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.is_upsampled is False
        assert img.is_downsampled is False
        assert img.storedsize is False

        ss = img(storedsize=True)
        assert ss.storedsize is True
        assert ss.is_upsampled is False
        assert ss.is_downsampled is False
        assert ss.is_pyramid_level is False
        # shape unchanged: stored_shape == shape for all entries
        assert ss.shape == img.shape
        assert ss.dims == img.dims

        arr = ss.asarray()
        assert arr.shape == img.shape
        # data must be identical to non-storedsize read
        expected = img.asarray()
        assert numpy.array_equal(arr, expected)

        # chunks interface: same shape, same data
        plane_ss = next(iter(ss.chunks())).asarray()
        plane_normal = next(iter(img.chunks())).asarray()
        assert plane_ss.shape == plane_normal.shape
        assert numpy.array_equal(plane_ss, plane_normal)

        # repr includes storedsize indicator
        assert 'storedsize' in repr(ss)
        assert 'storedsize' not in repr(img)


def test_storedsize_airyscan():
    """Test storedsize on Airyscan file skips Y-upsampling."""
    filename = DATA / 'Airyscan/AiryFastScan.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.is_upsampled is True

        # C=0 is sub-sampled (stored_Y=32, logical_Y=128).
        # storedsize=True returns stored 32-row data without repeat.
        c0_ss = img(H=0, T=0, C=0, storedsize=True)
        assert c0_ss.storedsize is True
        assert c0_ss.is_upsampled is True
        assert c0_ss.is_downsampled is False
        assert c0_ss.shape == (32, 128)
        arr_c0 = c0_ss.asarray()
        assert arr_c0.shape == (32, 128)

        # chunks interface returns stored size too
        plane_c0 = next(iter(c0_ss.chunks())).asarray()
        assert plane_c0.shape == (32, 128)
        assert numpy.array_equal(plane_c0, arr_c0)

        # C=1 is full-resolution (128x128 stored).
        # storedsize does not change output shape.
        c1_ss = img(H=0, T=0, C=1, storedsize=True)
        assert c1_ss.is_upsampled is False
        assert c1_ss.is_downsampled is False
        assert c1_ss.shape == (128, 128)
        c1_normal = img(H=0, T=0, C=1)
        assert numpy.array_equal(c1_ss.asarray(), c1_normal.asarray())


def test_storedsize_airyscan_nonuniform_raises():
    """Test storedsize raises ValueError for mixed Airyscan channels."""
    filename = DATA / 'Airyscan/AiryFastScan.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        # both channels together have mixed stored-to-logical ratios
        mixed = img(H=0, T=0, storedsize=True)
        with pytest.raises(ValueError, match='uniform stored-to-logical'):
            mixed.shape  # noqa: B018


def test_storedsize_palm():
    """Test storedsize on PALM file returns stored super-resolution data."""
    filename = DATA / 'Zenodo-10577621/Image 5_PALM_verrechnet.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.is_downsampled is True
        assert img.shape == (512, 512)

        ss = img(storedsize=True)
        assert ss.storedsize is True
        assert ss.is_upsampled is False
        assert ss.is_downsampled is True
        assert ss.is_pyramid_level is False
        # stored size is 1206x1206 (larger than logical 512x512)
        assert ss.shape == (1206, 1206)

        arr = ss.asarray()
        assert arr.shape == (1206, 1206)
        assert arr.dtype == numpy.uint16

        # chunks interface returns stored size too
        plane = next(iter(ss.chunks())).asarray()
        assert plane.shape == (1206, 1206)
        assert numpy.array_equal(plane, arr)


def test_storedsize_palm_nonuniform_raises():
    """Test storedsize raises ValueError for mixed PALM channels."""
    filename = DATA / 'Zenodo-10577621/Palm_mitDrift.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.is_downsampled is True
        # mixed ratios: C=0 is super-res, C=1 is normal
        mixed = img(storedsize=True)
        with pytest.raises(ValueError, match='uniform stored-to-logical'):
            mixed.shape  # noqa: B018


def test_storedsize_palm_per_channel():
    """Test storedsize on PALM file with per-channel selection."""
    filename = DATA / 'Zenodo-10577621/Palm_mitDrift.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        # C=0 is super-resolution (stored 2560x2560 -> logical 256x256)
        c0 = img(C=0, storedsize=True)
        assert c0.is_downsampled is True
        assert c0.is_upsampled is False
        assert c0.shape == (2560, 2560)
        arr_c0 = c0.asarray()
        assert arr_c0.shape == (2560, 2560)

        # C=1 is normal (stored 256x256 == logical 256x256)
        c1 = img(C=1, storedsize=True)
        assert c1.is_downsampled is False
        assert c1.is_upsampled is False
        assert c1.shape == (256, 256)
        arr_c1 = c1.asarray()
        assert arr_c1.shape == (256, 256)
        # no-op: data same as without storedsize
        assert numpy.array_equal(arr_c1, img(C=1).asarray())


def test_storedsize_pyramid():
    """Test storedsize on pyramid file is no-op."""
    filename = DATA / 'CZI-samples/Axio Scan.Z1/Kidney_40x_z_stack.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.is_pyramid

        # base level: storedsize flag set but shape unchanged
        ss = img(storedsize=True)
        assert ss.storedsize is True
        assert ss.is_upsampled is False
        assert ss.is_downsampled is False
        assert ss.is_pyramid_level is False
        assert ss.shape == img.shape

        # pyramid level: storedsize flag set but shape unchanged
        lv7 = img.levels[7]
        lv7_ss = lv7(storedsize=True)
        assert lv7_ss.storedsize is True
        assert lv7_ss.is_pyramid_level is True
        assert lv7_ss.is_upsampled is False
        assert lv7_ss.is_downsampled is False
        assert lv7_ss.shape == lv7.shape
        arr_ss = lv7_ss.asarray()
        arr_normal = lv7.asarray()
        assert numpy.array_equal(arr_ss, arr_normal)


def test_storedsize_inheritance():
    """Test storedsize=None inherits from parent image."""
    filename = DATA / 'Zenodo-10577621/Image 5_PALM_verrechnet.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.storedsize is False

        # explicit True
        ss = img(storedsize=True)
        assert ss.storedsize is True

        # explicit False overrides
        no_ss = ss(storedsize=False)
        assert no_ss.storedsize is False
        assert no_ss.shape == (512, 512)

        # default None inherits True from parent
        child = ss(storedsize=None)
        assert child.storedsize is True
        assert child.shape == (1206, 1206)


def test_storedsize_imread():
    """Test imread passes storedsize parameter through."""
    filename = DATA / 'Zenodo-10577621/Image 5_PALM_verrechnet.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    arr = imread(filename, storedsize=True)
    assert arr.shape == (1206, 1206)

    arr_normal = imread(filename)
    assert arr_normal.shape == (512, 512)


def test_chunks():
    """Test CziImageChunks iteration options and error handling."""
    # 2chZT.czi: T=19, C=2, Z=21, Y=300, X=400, uncompressed, start at 0
    filename = DATA / '2chZT.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.dims == ('T', 'C', 'Z', 'Y', 'X')
        assert img.shape == (19, 2, 21, 300, 400)
        assert img.bbox == (0, 0, 400, 300)

        # default: iterate all non-spatial dims one-at-a-time
        chunks = img.chunks()
        assert isinstance(chunks, CziImageChunks)
        assert len(chunks) == 19 * 2 * 21  # 798
        assert repr(chunks) == '<CziImageChunks 798 chunks ()>'
        chunk_list = list(chunks)
        assert all(isinstance(c, CziImage) for c in chunk_list)
        assert all(c.dims == ('Y', 'X') for c in chunk_list)
        assert all(c.shape == (300, 400) for c in chunk_list)
        assert chunk_list[0].asarray().shape == (300, 400)

        # C=None: keep C in each chunk, iterate T and Z
        chunks_cn = img.chunks(C=None)
        assert len(chunks_cn) == 19 * 21  # 399
        c0 = next(iter(chunks_cn))
        assert 'C' in c0.dims
        assert c0.shape == (2, 300, 400)
        assert c0.asarray().shape == (2, 300, 400)

        # Z=7: batch Z into groups of 7 (21 / 7 = 3 full batches)
        chunks_z7 = img.chunks(Z=7)
        assert len(chunks_z7) == 19 * 2 * 3  # 114
        assert all(c.sizes.get('Z') == 7 for c in chunks_z7 if 'Z' in c.dims)
        assert next(iter(chunks_z7)).asarray().shape == (7, 300, 400)

        # T=10: batch T into groups of 10 (batches of size 10 and 9)
        chunks_t10 = img.chunks(T=10)
        assert len(chunks_t10) == 2 * 2 * 21  # 84
        first_t = next(iter(chunks_t10))
        assert 'T' in first_t.dims
        assert first_t.sizes['T'] == 10
        assert first_t.asarray().shape == (10, 300, 400)

        # Z=None: keep full Z in each chunk, iterate T and C
        chunks_zn = img.chunks(Z=None)
        assert len(chunks_zn) == 19 * 2  # 38
        c_zn = next(iter(chunks_zn))
        assert c_zn.dims == ('Z', 'Y', 'X')
        assert c_zn.shape == (21, 300, 400)
        assert c_zn.asarray().shape == (21, 300, 400)

        # C=None, Z=7: keep C, batch Z; iterate T only
        chunks_cn_z7 = img.chunks(C=None, Z=7)
        assert len(chunks_cn_z7) == 19 * 3  # 57
        assert repr(chunks_cn_z7) == (
            '<CziImageChunks 57 chunks (C: None, Z: 7)>'
        )
        c_cn_z7 = next(iter(chunks_cn_z7))
        assert 'C' in c_cn_z7.dims
        assert c_cn_z7.sizes.get('Z') == 7
        assert c_cn_z7.asarray().shape == (2, 7, 300, 400)

        # Y=150, X=200: spatial tiling generates 2x2=4 tile grid
        # bbox (0, 0, 400, 300): step_x=200, step_y=150
        chunks_sp = img.chunks(Y=150, X=200)
        assert len(chunks_sp) == 19 * 2 * 21 * 4  # 3192
        c_sp = next(iter(chunks_sp))
        assert c_sp.shape[-2] <= 150
        assert c_sp.shape[-1] <= 200
        assert c_sp.asarray().shape == c_sp.shape

        # C=None, Y=150, X=200: keep C and tile spatially, iterate T and Z
        chunks_cn_sp = img.chunks(C=None, Y=150, X=200)
        assert len(chunks_cn_sp) == 19 * 21 * 4  # 1596
        c_cn_sp = next(iter(chunks_cn_sp))
        assert 'C' in c_cn_sp.dims
        assert c_cn_sp.shape[c_cn_sp.dims.index('C')] == 2
        assert c_cn_sp.shape[-2] <= 150
        assert c_cn_sp.shape[-1] <= 200
        assert c_cn_sp.asarray().shape == c_cn_sp.shape

        # T=None, Z=None: keep both T and Z, iterate C only
        chunks_tn_zn = img.chunks(T=None, Z=None)
        assert len(chunks_tn_zn) == 2  # one per C value
        c_tn_zn = next(iter(chunks_tn_zn))
        assert 'T' in c_tn_zn.dims
        assert 'Z' in c_tn_zn.dims
        assert c_tn_zn.shape == (19, 21, 300, 400)

        # data consistency: chunk-by-chunk assembly equals asarray()
        # use T=0, C=0 to read a single Z-stack (21 planes)
        sub = img(T=0, C=0)
        full = sub.asarray()
        planes = list(sub.chunks())
        assert len(planes) == 21
        assembled = numpy.stack([c.asarray() for c in planes])
        assert numpy.array_equal(assembled, full)

        # error: unknown dimension
        with pytest.raises(ValueError, match='unknown dimension'):
            img.chunks(Q=1)

        # error: sample dimension S not supported
        with pytest.raises(ValueError, match='sample dimension'):
            img.chunks(S=1)

        # error: size must be positive (zero)
        with pytest.raises(ValueError, match='must be positive'):
            img.chunks(Z=0)

        # error: size must be positive (negative)
        with pytest.raises(ValueError, match='must be positive'):
            img.chunks(Z=-1)

        # error: size must be int or None (float)
        with pytest.raises(TypeError, match='must be int or None'):
            img.chunks(Z=1.5)

        # squeeze=False: size-1 iterated dims preserved in each chunk
        chunks_nosq = img.chunks(squeeze=False)
        assert len(chunks_nosq) == 19 * 2 * 21  # same count
        c_nosq = next(iter(chunks_nosq))
        assert c_nosq.dims == ('T', 'C', 'Z', 'Y', 'X', 'S')
        assert c_nosq.shape == (1, 1, 1, 300, 400, 1)
        assert c_nosq.asarray().shape == (1, 1, 1, 300, 400, 1)

        # squeeze=False with C=None: C kept, T and Z iterated as size 1
        c_nosq_cn = next(iter(img.chunks(squeeze=False, C=None)))
        assert c_nosq_cn.dims == ('T', 'C', 'Z', 'Y', 'X', 'S')
        assert c_nosq_cn.shape == (1, 2, 1, 300, 400, 1)


def test_chunks_multitile():
    """Test chunks spatial tiling on a file with multiple spatial subblocks."""
    # Raw_Tiles_small_1channels.czi: uncompressed uint16, no C/T/Z dims,
    # 12 entries at X=[0, 1843, 3686] x Y=[0, 1843, 3687, 5530],
    # each 2048x2048.
    # Entries at X=0 and X=1843 end before X=4096, so a chunk ROI starting
    # at X=4096 must not include them.  Before the ROI-filtering fix, all
    # entries were forwarded to every spatial chunk, causing a negative-slice
    # broadcast error in asarray.
    filename = DATA / 'Raw_Tiles_small_1channels.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.dims == ('Y', 'X')
        assert img.shape == (7578, 5734)
        assert img.bbox == (0, 0, 5734, 7578)

        # spatial grid: 3 cols (0, 2048, 4096) x 4 rows (0, 2048, 4096, 6144)
        chunks = img.chunks(Y=2048, X=2048)
        assert len(chunks) == 12  # 3 cols x 4 rows

        ch_list = list(chunks)

        # chunk index 2 -- col=2, row=0: roi=(4096, 0, 1638, 2048).
        # Entries at X=0 (ends 2048) and X=1843 (ends 3891) end before
        # x=4096 and must be excluded; only the X=3686 entries overlap.
        c_edge = ch_list[2]
        assert c_edge._roi == (4096, 0, 1638, 2048)
        assert (
            len(c_edge._entries) == 2
        )  # only X=3686 entries (Y=0 and Y=1843)
        assert c_edge.shape == (2048, 1638)
        arr_edge = c_edge.asarray()
        assert arr_edge.shape == (2048, 1638)
        assert arr_edge.dtype == numpy.uint16

        # last chunk -- col=2, row=3: roi=(4096, 6144, 1638, 1434).
        # Only the X=3686, Y=5530 entry overlaps (1 entry).
        c_last = ch_list[-1]
        assert c_last._roi == (4096, 6144, 1638, 1434)
        assert len(c_last._entries) == 1
        assert c_last.shape == (1434, 1638)

        # first chunk -- col=0, row=0: all three X-column entries at Y=0
        # and Y=1843 overlap X=[0,2048) so 2 entries.
        c_first = ch_list[0]
        assert c_first._roi == (0, 0, 2048, 2048)
        assert c_first.shape == (2048, 2048)
        assert c_first.asarray().shape == (2048, 2048)


def test_pyramid():
    """Test multi-level pyramidal image."""
    filename = DATA / 'CZI-samples/Axio Scan.Z1/Kidney_40x_z_stack.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # file contains both non-pyramid and pyramid subblocks
        pyramid_types = {e.pyramid_type for e in czi.subblock_directory}
        assert CziPyramidType.NONE in pyramid_types
        assert CziPyramidType.MULTI_SUBBLOCK in pyramid_types

        # single scene
        assert len(czi.scenes) == 1
        img = czi.scenes[0]

        # base image (level 0) is the full-resolution non-pyramid layer
        assert img.is_pyramid
        assert all(
            e.pyramid_type == CziPyramidType.NONE
            for e in img.directory_entries
        )

        # 8 pyramid levels total, coarsest level has fewest entries
        assert len(img.levels) == 8
        assert img.levels[0] is img

        # entry counts decrease monotonically from level 0 to level 7
        entry_counts = [len(level.directory_entries) for level in img.levels]
        assert entry_counts == sorted(entry_counts, reverse=True)

        # level 0 full-resolution shape
        assert img.dims == ('Z', 'Y', 'X', 'S')
        assert img.shape == (7, 31301, 83143, 3)
        assert img.dtype == numpy.dtype('uint8')
        assert img.pixeltype == CziPixelType.BGR24

        # full-resolution level 0: no special tile scaling
        assert img.is_upsampled is False
        assert img.is_pyramid_level is False
        assert img.is_downsampled is False

        # all pyramid levels beyond 0 only contain pyramid subblocks
        for level in img.levels[1:]:
            assert not level.is_pyramid
            assert level.is_pyramid_level is True
            assert level.is_upsampled is False
            assert level.is_downsampled is False
            assert all(
                e.pyramid_type == CziPyramidType.MULTI_SUBBLOCK
                for e in level.directory_entries
            )

        # shape halves spatially at each level; asarray must agree
        expected_shapes = [
            (7, 31301, 83143, 3),  # level 0 - full resolution
            (7, 15720, 41840, 3),  # level 1
            (7, 7860, 20920, 3),  # level 2
            (7, 3930, 10460, 3),  # level 3
            (7, 1965, 5230, 3),  # level 4
            (7, 983, 2615, 3),  # level 5
            (7, 492, 1308, 3),  # level 6
            (7, 247, 654, 3),  # level 7 - coarsest
        ]
        for level, expected in zip(img.levels, expected_shapes, strict=True):
            assert level.shape == expected

        lv7 = img.levels[7]
        arr7 = lv7.asarray()
        assert arr7.shape == lv7.shape
        assert arr7.dtype == numpy.dtype('uint8')
        assert int(arr7.sum()) == 635479642

        # all subblocks use JPEGXR compression
        compressions = {e.compression for e in czi.subblock_directory}
        assert compressions == {CziCompressionType.JPEGXR}


def test_pyramid_level_isolation():
    """Test that asarray() on pyramid levels is deterministic and isolated."""
    # Without level isolation asarray() composites subblocks from all pyramid
    # tiers depending on iteration order, producing non-deterministic results.
    # CziImage.levels isolates each tier so the result is always the same.
    # https://github.com/cgohlke/czifile/issues/7

    filename = DATA / 'CZI-samples/Axio Scan.Z1/Kidney_40x_z_stack.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # the file mixes non-pyramid and pyramid subblocks - root cause of #7
        pyramid_types = {e.pyramid_type for e in czi.subblock_directory}
        assert CziPyramidType.NONE in pyramid_types
        assert CziPyramidType.MULTI_SUBBLOCK in pyramid_types

        lv6 = czi.scenes[0].levels[6]
        lv7 = czi.scenes[0].levels[7]

        # each level only contains its own subblock tier (not mixed)
        assert all(
            e.pyramid_type == CziPyramidType.MULTI_SUBBLOCK
            for e in lv6.directory_entries
        )
        assert all(
            e.pyramid_type == CziPyramidType.MULTI_SUBBLOCK
            for e in lv7.directory_entries
        )

        # shapes reflect stored pixel dimensions, not full-res bounding box
        assert lv6.shape == (7, 492, 1308, 3)
        assert lv7.shape == (7, 247, 654, 3)

        arr7 = lv7.asarray()

        # result shape agrees with level shape (not raw full-res bounding box)
        assert arr7.shape == lv7.shape

        # asarray is deterministic: repeated call returns identical data
        assert numpy.array_equal(lv7.asarray(), arr7)


def test_pyramid_tiles_mislabelled():
    """Test file with mislabelled pyramid_type=0 tiles."""
    # Mouse_stomach_20x_ROI_3chZTiles(WF).czi stores three stored-size variants
    # per tile (57x57, 171x171, 512x512 pixels) all with pyramid_type=NOT (0).
    # Only the full-resolution (stored_shape == shape) tiles should be used for
    # compositing, otherwise _scale_factors arrives at ~9x and tiles land at
    # the wrong positions, producing a visible seam around pixel 53 in both
    # axes.

    filename = (
        DATA / 'CZI-samples/Tiles/Mouse_stomach_20x_ROI_3chZTiles(WF).czi'
    )
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # all entries claim pyramid_type=NONE
        # but 2/3 are downsampled thumbnails
        assert all(
            e.pyramid_type == CziPyramidType.NONE
            for e in czi.subblock_directory
        )
        assert len(czi.subblock_directory) == 648
        full_res = sum(1 for e in czi.subblock_directory if not e.is_pyramid)
        assert full_res == 216

        # filtered_subblock_directory must keep only the full-resolution tiles
        fsd = czi.filtered_subblock_directory
        assert len(fsd) == full_res
        assert all(not e.is_pyramid for e in fsd)

        img = czi.scenes[0]

        # scale factors must all be 1.0 - no pyramid mis-mapping
        assert all(f == 1.0 for f in img._scale_factors)

        # shape reflects the true mosaic extent, not the ~9x compressed one
        assert img.dims == ('C', 'Z', 'Y', 'X')
        assert img.shape == (3, 18, 993, 995)

        # no pyramid levels - only one resolution after filtering
        assert img.is_pyramid is False
        assert img.is_upsampled is False
        assert img.is_pyramid_level is False
        assert img.is_downsampled is False
        assert len(img.levels) == 1
        assert img.levels[0] is img


def test_pyramid_stored_shape_mismatch():
    """Test pyramid edge-tiles where decoded size differs from stored_shape."""
    # sc-M180-075-substracted_zen3.8.czi contains JPEGXR-compressed pyramid
    # tiles whose directory entry declares stored_size Y=181 but the decoder
    # returns only 180 rows (ceil(361/2) vs floor(361/2)).  Without the fix
    # subblock.data() raises "cannot reshape array of size 184320 into shape
    # (1,181,1024,1)".

    filename = DATA / 'sc-M180-075-substracted_zen3.8.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        scene = czi.scenes[0]

        # 4-channel GRAY16 slide scanner image with 8 pyramid levels
        # (edge tiles are correctly merged into the same zoom-level group,
        # reducing the spurious 13-level count to 8 correct levels;
        # C=0 and C=1 lacked the S dimension so were previously missing)
        assert scene.dims == ('C', 'Y', 'X')
        assert scene.shape == (4, 33129, 68992)
        assert scene.dtype == numpy.dtype('uint16')
        assert len(scene.levels) == 8

        # all pyramid levels beyond 0 have at least one tile where
        # stored_shape != shape (the root cause of the reshape error)
        for level in scene.levels[1:]:
            assert any(
                e.stored_shape != e.shape for e in level.directory_entries
            )

        # level shapes halve at each step (edge tiles merged into their level)
        assert scene.levels[1].shape == (4, 16565, 34496)
        assert scene.levels[7].shape == (4, 259, 539)

        # asarray on a small pyramid level must succeed without a reshape
        # error (the mismatch is on pyramid tiles, not full-resolution ones)
        lv7 = scene.levels[7]
        arr = lv7.asarray(maxworkers=1)
        assert arr.shape == (4, 259, 539)
        assert arr.dtype == numpy.dtype('uint16')


def test_pyramid_stored_shape_mismatch_with_mask():
    """Test pyramid tiles with stored_shape mismatch and per-tile masks."""
    # sc-M180-075-substracted-zen2.3.czi contains JPEGXR pyramid tiles where:
    # - stored_shape in the directory entry is larger than the decoded tile
    #   (same CZI writer ceil/floor bug as in the zen3.8 variant), and
    # - each tile has a binary mask attachment (shape Y x X derived from
    #   stored_shape).
    # Without the fix, compositing raises "boolean index did not match indexed
    # array along axis 0; size of axis is 1 but size of corresponding boolean
    # axis is 729" because the (Y, X) mask was not clipped to match the
    # adjusted tile size, and was not expanded to match the tile's rank (C, Y,
    # X, S) before boolean indexing.

    filename = DATA / 'sc-M180-075-substracted-zen2.3.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        scene = czi.scenes[0]

        # 4-channel GRAY16 slide scanner image with 6 pyramid levels
        # (C=0 and C=1 lacked the S dimension so were previously missing)
        assert scene.dims == ('C', 'Y', 'X')
        assert scene.shape == (4, 33129, 68992)
        assert scene.dtype == numpy.dtype('uint16')
        assert len(scene.levels) == 6

        # all pyramid levels beyond 0 have stored_shape mismatches and masks
        for level in scene.levels[1:]:
            assert any(
                e.stored_shape != e.shape for e in level.directory_entries
            )

        # level 5 asarray must succeed without a boolean-index or reshape
        # error (smallest level that has both mismatch and masks)
        lv5 = scene.levels[5]
        assert lv5.shape == (4, 137, 284)
        arr5 = lv5.asarray(maxworkers=1)
        assert arr5.shape == (4, 137, 284)
        assert arr5.dtype == numpy.dtype('uint16')


def test_pyramid_roi():
    """Test ROI on a pyramid level."""
    # Validate that reading a pyramid level with an ROI produces the same
    # pixels as extracting the equivalent region from the full pyramid level
    # array.

    filename = DATA / 'sc-M180-075-substracted_zen3.8.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        scene = czi.scenes[0]
        lv = scene.levels[5]

        # full pyramid level array as reference
        full = lv.asarray(maxworkers=1)

        # ROI in full-resolution coordinates: center quarter
        bbox = scene.levels[0].bbox
        rx = bbox[0] + bbox[2] // 4
        ry = bbox[1] + bbox[3] // 4
        rw = bbox[2] // 2
        rh = bbox[3] // 2

        roi_img = lv(roi=(rx, ry, rw, rh))
        roi_arr = roi_img.asarray(maxworkers=1)

        # extract equivalent region from full array for comparison
        sf = lv._scale_factors
        dims = list(lv._raw_sizes.keys())
        y_i = dims.index('Y')
        x_i = dims.index('X')
        sy = round(ry / sf[y_i]) - lv._start[y_i]
        sx = round(rx / sf[x_i]) - lv._start[x_i]
        sh = round(rh / sf[y_i])
        sw = round(rw / sf[x_i])
        expected = full[:, sy : sy + sh, sx : sx + sw]

        assert roi_arr.shape == expected.shape
        assert numpy.array_equal(roi_arr, expected)


def test_mixed_dimension_sets():
    """Test file where entries have inconsistent dimension sets."""
    # Gland_1x_3substack.czi contains entries with two different dimension
    # tuples: some tiles have ('V', 'B', 'H', 'I', 'R', 'T', 'C', 'Z', 'Y',
    # 'X', 'S') and others are missing the 'B' dimension.  Without the fix,
    # compositing raises "zip() argument 2 is longer than argument 1" because
    # directory_entry.start (10 elements) was zipped with self._start (11
    # elements, derived from the reference entry that has 'B').

    filename = DATA / 'Gland_1x_3substack.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        scene = czi.scenes[0]

        # mixed dimension entries: some have 'B', some don't
        all_dims = {e.dims for e in czi.subblock_directory}
        assert len(all_dims) > 1, 'expected mixed dimension sets'

        assert scene.dims == ('V', 'C', 'Z', 'Y', 'X')
        assert scene.shape == (3, 2, 51, 1920, 1920)
        assert scene.dtype == numpy.dtype('uint16')

        # asarray must succeed without a zip-length error
        arr = czi.asarray(maxworkers=1)
        assert arr.shape == (3, 2, 51, 1920, 1920)
        assert arr.dtype == numpy.dtype('uint16')
        assert int(arr.sum()) == 443993255740


def test_multifile_series_missing_parts():
    """Test file is one part of multi-file series with missing siblings."""
    # file_part > 0 and primary_file_guid != file_guid indicate this is not
    # the primary file; the other parts are absent from the test data.
    # The file can still be opened and its own subblocks read normally.
    # https://github.com/cgohlke/czifile/issues/6

    filename = DATA / 'Private-PhasorPy/PupalWing.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        hdr = czi.header

        # this is part 5 of a multi-file series, not the primary file
        assert hdr.file_part == 5
        assert hdr.primary_file_guid != hdr.file_guid

        # the subblocks present in this part are readable
        sbd = czi.subblock_directory
        assert len(sbd) == 80

        # no attachments in a non-primary part (directory lives in part 0)
        assert len(czi.attachment_directory) == 0


def test_empty_subblock_directory():
    """Test file where ZEN wrote metadata/attachments but no image tiles."""
    # The SubBlockDirectory segment exists but contains 0 entries - no
    # ZISRAWSUBBLOCK segments are present in the file body at all.  The file
    # is a valid CZI (primary part, GUIDs match) and its attachments
    # (EventList, TimeStamps, Thumbnail) are still readable.
    # https://forum.image.sc/t/error-reading-czi-images/114333

    filename = DATA / '3K_tile_g3bp1_arsenite_MCCC1_01.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # primary file - GUIDs match and file_part == 0
        assert czi.header.file_part == 0
        assert czi.header.file_guid == czi.header.primary_file_guid

        # no image subblocks
        assert len(czi.subblock_directory) == 0
        assert len(czi.scenes) == 0

        # three attachments are present and readable
        entries = {ae.name for ae in czi.attachment_directory}
        assert entries == {'EventList', 'TimeStamps', 'Thumbnail'}

        for attachment in czi.attachments():
            data = attachment.data()
            assert data is not None

        # metadata XML is present
        xml = czi.metadata()
        assert isinstance(xml, str)
        assert len(xml) > 0


def test_point_time_series():
    """Test point time-series: single-pixel detector sampled 80_000 times."""
    # spot.czi stores a single subblock with axes TX (time x detector bins).
    # It has no spatial Y/Z - just a 1-D spectral/spatial axis X=500 repeated
    # at every time point T=80_000, all uncompressed uint8.  Five attachments
    # are present, including 80_000 uniformly-spaced timestamps and a preview
    # image.
    # https://forum.image.sc/t/104650

    filename = DATA / 'spot.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # primary, single-part file
        assert czi.header.file_part == 0
        assert czi.header.file_guid == czi.header.primary_file_guid

        # one uncompressed, non-pyramid subblock
        assert len(czi.subblock_directory) == 1
        entry = czi.subblock_directory[0]
        assert entry.compression == CziCompressionType.UNCOMPRESSED
        assert entry.pyramid_type == CziPyramidType.NONE

        # single scene with TX layout
        assert len(czi.scenes) == 1
        sc = czi.scenes[0]
        assert sc.dims == ('T', 'X')
        assert sc.shape == (80000, 500)
        assert sc.dtype == numpy.dtype('uint8')

        # one level, no sub-resolution pyramid
        assert len(sc.levels) == 1
        assert sc.levels[0] is sc

        # pixel values are low-intensity (point measurement, not a full image)
        arr = sc.asarray()
        assert arr.shape == sc.shape
        assert arr.dtype == numpy.dtype('uint8')
        assert int(arr.sum()) == 16002180

        # five attachments:
        # # Thumbnail, LookupTables, TimeStamps, EventList, Image
        att_names = {ae.name for ae in czi.attachment_directory}
        assert att_names == {
            'Thumbnail',
            'LookupTables',
            'TimeStamps',
            'EventList',
            'Image',
        }

        # 80 000 timestamps, one per time point, uniformly spaced ~614 us apart
        for attachment in czi.attachments():
            if attachment.attachment_entry.name == 'TimeStamps':
                ts = attachment.data()
                break
        else:
            pytest.fail('TimeStamps attachment not found')

        assert isinstance(ts, numpy.ndarray)
        assert ts.dtype == numpy.dtype('float64')
        assert len(ts) == 80000

        intervals = numpy.diff(ts)
        assert intervals.min() == pytest.approx(0.000614, rel=1e-3)
        assert intervals.max() == pytest.approx(0.000614, rel=1e-3)
        assert float(ts[-1] - ts[0]) == pytest.approx(49.119, rel=1e-3)


def test_scanning_fcs():
    """Test scanning FCS: line scan assembled from 2487 T-chunked subblocks."""
    # sFCS_980_3415_0p98.czi is a scanning FCS acquisition on an LSM 980.
    # The line (X=128 pixels) is scanned repeatedly; each pass is stored as
    # a separate subblock with a variable number of T frames (10-100 per
    # subblock), all at the same spatial position. The 2487 subblocks are
    # assembled along the T axis by the generic tile compositing path to
    # produce a single TX array (T=204920 x X=128), uint16 uncompressed.
    # Three attachments are present: EventList, TimeStamps, and Thumbnail.

    filename = DATA / 'sFCS_980_3415_0p98.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # primary, single-part file
        assert czi.header.file_part == 0
        assert czi.header.file_guid == czi.header.primary_file_guid

        # 2487 uncompressed, non-pyramid subblocks all at the same XY position
        assert len(czi.filtered_subblock_directory) == 2487
        entry = czi.filtered_subblock_directory[0]
        assert entry.compression == CziCompressionType.UNCOMPRESSED
        assert entry.pyramid_type == CziPyramidType.NONE

        # single scene assembled into TX layout
        assert len(czi.scenes) == 1
        sc = czi.scenes[0]
        assert sc.dims == ('T', 'X')
        assert sc.shape == (204920, 128)
        assert sc.dtype == numpy.dtype('uint16')

        # no resampling
        assert not sc.is_pyramid
        assert not sc.is_upsampled
        assert not sc.is_downsampled

        # one level, no sub-resolution pyramid
        assert len(sc.levels) == 1
        assert sc.levels[0] is sc

        arr = sc.asarray()
        assert arr.shape == sc.shape
        assert arr.dtype == numpy.dtype('uint16')
        assert int(arr.sum()) == 9018413

        # three attachments: EventList, TimeStamps, Thumbnail
        att_names = {ae.name for ae in czi.attachment_directory}
        assert att_names == {'EventList', 'TimeStamps', 'Thumbnail'}

        # 204920 timestamps, one per scan line, uniformly spaced ~293 us apart
        ts = czi.timestamps
        assert ts is not None
        assert isinstance(ts, numpy.ndarray)
        assert ts.dtype == numpy.dtype('float64')
        assert len(ts) == 204920

        intervals = numpy.diff(ts)
        assert intervals.min() == pytest.approx(0.0002928, rel=1e-3)
        assert intervals.max() == pytest.approx(0.0002928, rel=1e-3)
        assert float(ts[-1] - ts[0]) == pytest.approx(60.0, rel=1e-3)


def test_lattice_lightsheet_raw_metadata():
    """Test deskew geometry metadata extracted from unprocessed raw file."""
    # HeLa-Test-01.czi is the raw acquisition before online deskewing.
    # Compared with the _processed variant, it differs in three key ways:
    # - ZAxisShear == 'Skew60' - oblique stage scan, shear not yet removed
    # - Z voxel size == 200 nm - raw stage step, not isotropic like X/Y
    # - zscale is absent - populated only after deskewing
    # The deskew shear factor can instead be derived as Z_step / XY_pixel,
    # which must equal the LightSheetDeskewFactor recorded in the processed
    # file (1.3794).
    # https://forum.image.sc/t/101901

    import xml.etree.ElementTree as ET

    filename = DATA / 'HeLa-Test-01.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        raw_xml = czi.metadata()

        # 296 raw oblique-scan slices, ZSTDHDR compressed
        assert len(czi.subblock_directory) == 296
        assert len(czi.scenes) == 1
        sc = czi.scenes[0]
        assert sc.dims == ('Z', 'Y', 'X')
        assert sc.shape == (296, 256, 512)
        assert sc.dtype == numpy.uint16

    tree = ET.fromstring(raw_xml)

    def txt(path):
        el = tree.find(path)
        return el.text.strip() if el is not None and el.text else None

    # 1. Non-isotropic voxel sizes - Z is the raw oblique stage step
    xy_voxel_m = 1.4499219272808386e-07
    z_step_m = 2e-07  # 200 nm raw stage increment

    distances = tree.findall('.//Scaling/Items/Distance')
    by_id = {d.get('Id'): float(d.findtext('Value')) for d in distances}
    assert by_id['X'] == pytest.approx(xy_voxel_m, rel=1e-6)
    assert by_id['Y'] == pytest.approx(xy_voxel_m, rel=1e-6)
    assert by_id['Z'] == pytest.approx(z_step_m, rel=1e-6)

    # 2. Z-axis carries oblique shear - deskewing still required
    assert txt('.//Dimensions/Z/ZAxisShear') == 'Skew60'
    assert txt('.//Dimensions/Z/XYZHandedness') == 'RightHanded'
    assert txt('.//Dimensions/Z/ZAxisDirection') == 'FromObjectiveToSpecimen'

    # zscale is not present in the raw file (set only after deskewing)
    assert txt('.//Dimensions/Z/CustomAttributes/zscale') is None

    # Z stage increment matches the raw Z voxel size (um)
    z_increment = float(txt('.//Dimensions/Z/Positions/Interval/Increment'))
    assert z_increment == pytest.approx(z_step_m * 1e6, rel=1e-6)

    # 3. Deskew is enabled but not yet applied
    assert txt('.//ProcessingStep/EnableDeskew') == 'true'
    assert txt('.//Deskew/XyIsRotatedBy180') == 'true'
    assert txt('.//Deskew/XyIsRotatedBy90') == 'true'
    assert txt('.//Deskew/Interpolation') == 'Linear'

    # 4. Shear factor derivable from raw voxel sizes
    #    z_step / xy_pixel == LightSheetDeskewFactor in processed file
    derived_factor = z_step_m / xy_voxel_m
    assert derived_factor == pytest.approx(1.3794, rel=1e-3)


def test_lattice_lightsheet_deskew_metadata():
    """Test extraction of deskew geometry from a Lattice Lightsheet CZI."""
    # HeLa-Test-01_processed.czi was acquired on a Zeiss Lattice Lightsheet
    # microscope and subsequently deskewed online by ZEN. All parameters
    # needed to reproduce (or verify) the deskew are present in the metadata
    # XML and are checked here.
    # Key locations:
    # - Scaling: ImageDocument/Metadata/Scaling/Items/Distance
    # - Z-axis geometry: ImageDocument/Metadata/Information/Image/Dimensions/Z
    # - Deskew settings: .../Experiment/ProcessingSteps/ProcessingStep/Deskew
    # - Deskew factor: AuditTrail (HTML-escaped XML) inside CustomAttributes
    # https://forum.image.sc/t/101901

    import html
    import xml.etree.ElementTree as ET

    filename = DATA / 'HeLa-Test-01_processed.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        raw_xml = czi.metadata()

    tree = ET.fromstring(raw_xml)

    def txt(path):
        el = tree.find(path)
        return el.text.strip() if el is not None and el.text else None

    # 1. Isotropic voxel size after deskewing (metres)
    voxel_m = 1.4499219272808386e-07
    for axis in ('X', 'Y', 'Z'):
        distances = tree.findall('.//Scaling/Items/Distance')
        value = next(
            float(d.findtext('Value'))
            for d in distances
            if d.get('Id') == axis
        )
        assert value == pytest.approx(voxel_m, rel=1e-6), axis

    # 2. Z-axis geometry - file is already deskewed
    assert txt('.//Dimensions/Z/ZAxisShear') == 'None'
    assert txt('.//Dimensions/Z/XYZHandedness') == 'RightHanded'
    assert txt('.//Dimensions/Z/ZAxisDirection') == 'FromObjectiveToSpecimen'

    # zscale encodes the raw stage-scan shear factor (dz_stage / dxy)
    zscale = float(txt('.//Dimensions/Z/CustomAttributes/zscale'))
    assert zscale == pytest.approx(1.3794, rel=1e-3)

    # Z stage increment matches the deskewed voxel size (um)
    z_increment = float(txt('.//Dimensions/Z/Positions/Interval/Increment'))
    assert z_increment == pytest.approx(voxel_m * 1e6, rel=1e-6)

    # 3. Online deskew was enabled and applied
    assert txt('.//ProcessingStep/EnableDeskew') == 'true'
    assert txt('.//Deskew/XyIsRotatedBy180') == 'true'
    assert txt('.//Deskew/XyIsRotatedBy90') == 'true'
    assert txt('.//Deskew/Interpolation') == 'Linear'
    assert txt('.//Deskew/ProcessingMethod') == 'None'

    # 4. Deskew factor and method from the HTML-escaped AuditTrail
    audit_el = tree.find('.//CustomAttributes/AuditTrail')
    assert audit_el is not None, 'AuditTrail not found'
    audit_tree = ET.fromstring(html.unescape(audit_el.text))

    method_el = audit_tree.find('.//LightSheetDeskewMethod')
    assert method_el is not None
    assert method_el.text == 'Skew60'

    factor_el = audit_tree.find('.//LightSheetDeskewFactor')
    assert factor_el is not None
    deskew_factor = float(factor_el.text)
    # must agree with zscale to 4 significant figures
    assert deskew_factor == pytest.approx(zscale, rel=1e-4)


def test_spectral_lsm980():
    """Test LSM980 spectral imaging: narrow non-overlapping bins."""
    # 02_Chroma Slide 405.czi is a 32-channel spectral scan acquired on a
    # Zeiss LSM980 with a 405 nm laser.  Each channel corresponds to a narrow
    # (~9 nm) detection bin.  DetectionWavelength is a (lo, hi) tuple; bins are
    # non-overlapping, so coords['C'] returns float midpoints in nm.

    filename = DATA / 'Private-PhasorPy/980 Zeiss/02_Chroma Slide 405.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]

        assert img.dims == ('C', 'Y', 'X', 'S')
        assert img.shape == (32, 512, 512, 3)
        assert img.dtype == numpy.dtype('uint8')

        ch = img.channels
        assert len(ch) == 32

        # all channels are SpectralImaging with 405 nm excitation
        for ch_data in ch.values():
            assert ch_data.get('AcquisitionMode') == 'SpectralImaging'
            assert ch_data.get('ExcitationWavelength') == pytest.approx(405.0)

        # first channel: narrow detection bin, tuple range
        first = ch['414 nm']
        lo, hi = first['DetectionWavelength']
        assert lo == pytest.approx(409.668, rel=1e-4)
        assert hi == pytest.approx(418.558, rel=1e-4)

        # last channel: also a narrow bin
        last = ch['690 nm']
        lo2, hi2 = last['DetectionWavelength']
        assert lo2 == pytest.approx(685.262, rel=1e-4)
        assert hi2 == pytest.approx(694.152, rel=1e-4)

        # coords['C']: float midpoints because bins are non-overlapping
        c_coord = img.coords['C']
        assert c_coord.dtype == numpy.dtype('float64')
        assert c_coord.shape == (32,)
        assert c_coord[0] == pytest.approx((lo + hi) / 2.0, rel=1e-6)
        assert c_coord[-1] == pytest.approx((lo2 + hi2) / 2.0, rel=1e-6)
        # midpoints must be strictly increasing
        assert numpy.all(numpy.diff(c_coord) > 0)


def test_spectral_lsm800():
    """Test LSM800 ChS spectral imaging: overlapping cumulative windows."""
    # focalcheck A3_405nm.czi is a 22-channel acquisition on a Zeiss LSM800
    # using the ChS (GaAsP) variable-beam-splitter mode with a 405 nm laser.
    # Channels are recorded as 11 split-point pairs (Ch1 = short-pass,
    # Ch2 = long-pass).  Because the DetectionWavelength ranges overlap across
    # channels, coords['C'] falls back to channel names rather than midpoints.

    filename = (
        DATA
        / 'Private-PhasorPy/prueba espectrales LSM800/focalcheck A3_405nm.czi'
    )
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]

        assert img.dims == ('C', 'Y', 'X')
        assert img.shape == (22, 512, 512)
        assert img.dtype == numpy.dtype('uint8')

        ch = img.channels
        assert len(ch) == 22

        # channels come in Ch1/Ch2 pairs at each split point
        ch_names = list(ch)
        assert ch_names[:4] == ['450-Ch1', '450-Ch2', '470-Ch1', '470-Ch2']
        assert ch_names[-2:] == ['650-Ch1', '650-Ch2']

        # all channels share 405 nm excitation and SpectralImaging mode
        for ch_data in ch.values():
            assert ch_data.get('AcquisitionMode') == 'SpectralImaging'
            assert ch_data.get('ExcitationWavelength') == pytest.approx(405.0)

        # Ch1 = short-pass (400 nm to split), Ch2 = long-pass (split to 650 nm)
        assert ch['450-Ch1']['DetectionWavelength'] == (400.0, 450.0)
        assert ch['450-Ch2']['DetectionWavelength'] == (450.0, 650.0)

        # last Ch2 is a degenerate single-value (lo==hi -> scalar float)
        assert ch['650-Ch2']['DetectionWavelength'] == pytest.approx(650.0)

        # coords['C']: channel names because ranges overlap
        c_coord = img.coords['C']
        assert c_coord.dtype.kind == 'U'  # unicode strings
        assert c_coord.shape == (22,)
        assert list(c_coord[:4]) == [
            '450-Ch1',
            '450-Ch2',
            '470-Ch1',
            '470-Ch2',
        ]
        assert list(c_coord[-2:]) == ['650-Ch1', '650-Ch2']


def test_airyscan_timestamps():
    """Test Airyscan fast-scan file with zero t_incr uses TimeStamps."""
    # AiryFastScan.czi has Interval/TimeSpan/Value == '0' in the XML metadata,
    # so the computed time increment is zero.  coords['T'] must fall back to
    # the TimeStamps attachment (8200 entries) rather than returning all zeros.

    filename = DATA / 'Airyscan/AiryFastScan.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        # timestamps cached property
        ts = czi.timestamps
        assert isinstance(ts, numpy.ndarray)
        assert ts.dtype == numpy.dtype('float64')
        assert len(ts) == 8200
        assert ts[0] == pytest.approx(35751.4348, rel=1e-6)
        assert ts[-1] == pytest.approx(35919.4815, rel=1e-6)

        # scene layout
        img = czi.scenes[0]
        assert img.dims == ('H', 'C', 'T', 'Y', 'X')
        assert img.shape == (16, 2, 8200, 128, 128)

        # coords['T'] must come from TimeStamps, not be all zeros
        ct = img.coords['T']
        assert ct.dtype == numpy.dtype('float64')
        assert len(ct) == 8200
        assert ct[0] == pytest.approx(ts[0], rel=1e-9)
        assert ct[-1] == pytest.approx(ts[-1], rel=1e-9)
        assert numpy.all(numpy.diff(ct) > 0)

        # coords['H'] - Airyscan detector elements get integer indices
        ch = img.coords['H']
        assert ch.dtype == numpy.uint8
        assert list(ch) == list(range(16))


def test_airyscan_fastscan():
    """Test Airyscan fast-scan Y-upsampling via numpy.repeat."""
    # AiryFastScan.czi stores C=0 tiles with Y compressed 4x (stored_Y=32,
    # logical_Y=128).  asarray() must upsample them with numpy.repeat so that
    # the output array is fully populated at the logical 128-row size.
    # C=1 tiles are full-resolution and require no special handling.

    filename = DATA / 'Airyscan/AiryFastScan.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        img = czi.scenes[0]
        assert img.dims == ('H', 'C', 'T', 'Y', 'X')
        assert img.shape == (16, 2, 8200, 128, 128)
        assert img.is_upsampled is True
        assert img.is_pyramid_level is False
        assert img.is_downsampled is False

        # read a single H=0, T=0 plane (both channels) to keep I/O small
        frame = img(H=0, T=0).asarray()

        assert frame.shape == (2, 128, 128)
        assert frame.dtype == numpy.dtype('uint8')
        # C=0: stored 32 rows, upsampled to 128 - data must reach row 32+
        assert int((frame[0, 32:] != 0).sum()) > 0
        # each group of 4 consecutive rows must be identical (repeat pattern)
        for base in range(0, 128, 4):
            assert numpy.array_equal(frame[0, base], frame[0, base + 1])
            assert numpy.array_equal(frame[0, base], frame[0, base + 3])
        # C=1 is full-resolution (128 rows stored); no special handling needed
        assert frame[1].shape == (128, 128)

        # chunks interface must produce same result as asarray for upsampled
        # tiles: chunk asarray must upsample C=0 stored_Y=32 -> logical_Y=128
        # all three non-spatial dims are pinned in one __call__ to avoid
        # chaining, which is not permitted
        plane_c0 = next(iter(img(H=0, T=0, C=0).chunks())).asarray()
        assert plane_c0.shape == (128, 128)
        assert int((plane_c0[32:] != 0).sum()) > 0
        for base in range(0, 128, 4):
            assert numpy.array_equal(plane_c0[base], plane_c0[base + 1])
            assert numpy.array_equal(plane_c0[base], plane_c0[base + 3])
        plane_c1 = next(iter(img(H=0, T=0, C=1).chunks())).asarray()
        assert plane_c1.shape == (128, 128)
        # chunks result matches asarray result
        assert numpy.array_equal(plane_c0, frame[0])
        assert numpy.array_equal(plane_c1, frame[1])


def test_multiscene_mosaic():
    """Test CziScenes, CziImage selection, and CziImageChunks interfaces."""
    # S=3_1Pos_2Mosaic_T=2=Z=3_CH=2.czi has 3 scenes with different spatial
    # extents:
    #   scene[0]: tiled mosaic, (T=2, C=2, Z=3, Y=486, X=1178)
    #   scene[1]: single position, (T=2, C=2, Z=3, Y=256, X=256)
    #   scene[2]: tiled mosaic, (T=2, C=2, Z=3, Y=947, X=1408)

    filename = DATA / 'Zenodo-7015307/S=3_1Pos_2Mosaic_T=2=Z=3_CH=2.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    with CziFile(filename) as czi:
        scenes = czi.scenes

        # CziScenes
        assert len(scenes) == 3

        sc0, sc1, sc2 = scenes[0], scenes[1], scenes[2]

        # sc0 and sc2 are mosaic (mosaic_index >= 0 -> 1);
        # sc1 is uncompressed single-position (UNCOMPRESSED -> 1)
        assert sc0._default_maxworkers == 1
        assert sc1._default_maxworkers == 1
        assert sc2._default_maxworkers == 1

        assert sc0.dims == ('T', 'C', 'Z', 'Y', 'X')
        assert sc0.shape == (2, 2, 3, 486, 1178)
        assert sc0.dtype == numpy.dtype('uint16')

        assert sc1.dims == ('T', 'C', 'Z', 'Y', 'X')
        assert sc1.shape == (2, 2, 3, 256, 256)

        assert sc2.dims == ('T', 'C', 'Z', 'Y', 'X')
        assert sc2.shape == (2, 2, 3, 947, 1408)

        # scene_indices roundtrip: scenes(scene=img.scene_indices) == img
        assert sc0.scene_indices == (0,)
        assert sc1.scene_indices == (1,)
        assert sc2.scene_indices == (2,)
        assert (
            scenes(scene=sc1.scene_indices).directory_entries
            == sc1.directory_entries
        )

        # combined scenes: scene_indices reflects all merged S-coordinates
        sc12 = scenes(scene=[1, 2])
        assert sc12.scene_indices == (1, 2)
        sc_all = scenes()
        assert sc_all.scene_indices == (0, 1, 2)

        # __getitem__ is a Mapping (int key only); slices raise KeyError
        # multi-scene access uses __call__: scenes(scene=[1, 2])
        with pytest.raises(KeyError):
            scenes[1:3]

        # scenes are spatially disjoint: each has a distinct Y start
        y_idx = sc0.dims.index('Y')
        assert sc0.start[y_idx] == 0
        assert sc1.start[y_idx] == 853
        assert sc2.start[y_idx] == 1108

        # asarray checksums
        arr0 = sc0.asarray(maxworkers=1)
        assert int(arr0.sum()) == 7772223062

        arr1 = sc1.asarray(maxworkers=1)
        assert int(arr1.sum()) == 699256173

        # sc1 coords: T in seconds, C as channel names, Z/Y/X in metres
        coords = sc1.coords
        assert list(coords['T']) == pytest.approx([0.5439875, 33.9559896])
        assert list(coords['C']) == ['DAPI', 'EGFP']
        assert list(coords['Z']) == pytest.approx([0.0, 1e-6, 2e-6])
        assert coords['Y'][0] == pytest.approx(0.0003412)
        assert coords['Y'][1] - coords['Y'][0] == pytest.approx(4e-7)
        assert coords['X'][0] == pytest.approx(0.0)
        assert coords['X'][1] - coords['X'][0] == pytest.approx(4e-7)

        arr2 = sc2.asarray(maxworkers=1)
        assert int(arr2.sum()) == 12769120769

        # CziImage() dimension selection
        sub = sc0(T=0, C=0)
        assert sub.dims == ('Z', 'Y', 'X')
        assert sub.shape == (3, 486, 1178)
        arr_sub = sub.asarray(maxworkers=1)
        assert int(arr_sub.sum()) == 1939165033

        # squeezing both T and C leaves only Z/Y/X in coords
        c_sub = sub.coords
        assert list(c_sub.keys()) == ['Z', 'Y', 'X']
        assert list(c_sub['Z']) == pytest.approx([0.0, 1e-6, 2e-6])
        assert c_sub['Y'][0] == pytest.approx(0.0)
        assert c_sub['X'][0] == pytest.approx(0.0002328)

        # CziImageChunks: T=2, C=2, Z=3 -> 12 chunks (planes)
        chunks = sc1.chunks()
        assert len(chunks) == 12
        chunk_list = list(chunks)
        assert chunk_list[0].asarray().shape == (256, 256)
        assert int(chunk_list[0].asarray().sum()) == 62403914
        assert int(chunk_list[-1].asarray().sum()) == 55145280

        # asarray() reproduces the scene array
        all_planes = sc1.asarray()
        assert all_planes.shape == sc1.shape
        assert int(all_planes.sum()) == 699256173

        # chunk-by-chunk assembly matches asarray()
        # (equivalent but much slower than asarray())
        img_sel = sc1(T=None, C=None, Z=None)
        full = img_sel.asarray()
        chunks_list = list(img_sel.chunks())
        assert len(chunks_list) == 12  # T=2 * C=2 * Z=3
        assembled = numpy.stack([c.asarray() for c in chunks_list])
        assembled = assembled.reshape(img_sel.shape)
        assert numpy.array_equal(assembled, full)

        # single-value Z selection squeezes that dimension out
        sub_z1 = sc1(Z=1)
        assert sub_z1.dims == ('T', 'C', 'Y', 'X')
        assert sub_z1.shape == (2, 2, 256, 256)
        assert int(sub_z1.asarray(maxworkers=1).sum()) == 232559034

        # Z=1 squeezes Z out of coords; T and C are unchanged
        assert 'Z' not in sub_z1.coords
        assert list(sub_z1.coords['T']) == pytest.approx(
            [0.5439875, 33.9559896]
        )
        assert list(sub_z1.coords['C']) == ['DAPI', 'EGFP']

        # slice selects by absolute Z coordinate range (Z in [1, 3))
        sub_zsl = sc1(Z=slice(1, 3))
        assert sub_zsl.dims == ('T', 'C', 'Z', 'Y', 'X')
        assert sub_zsl.shape == (2, 2, 2, 256, 256)
        assert int(sub_zsl.asarray(maxworkers=1).sum()) == 456326113

        # Z=slice(1,3) keeps only the sliced Z values in the coord
        assert list(sub_zsl.coords['Z']) == pytest.approx([1e-6, 2e-6])

        # slice with a single element squeezes out the dimension; data matches
        # the corresponding slice from the full array
        sub_tsl = sc1(T=slice(1, 2))
        assert sub_tsl.dims == ('C', 'Z', 'Y', 'X')
        assert sub_tsl.shape == (2, 3, 256, 256)
        assert int(sub_tsl.asarray(maxworkers=1).sum()) == 339723183

        # single-element T slice squeezes T out of coords as well
        assert 'T' not in sub_tsl.coords

        # dim reorder: specifying C=None, T=None puts Z (unspecified) first,
        # then C, then T, before the spatial dims
        sub_ct = sc1(C=None, T=None)
        assert sub_ct.dims == ('Z', 'C', 'T', 'Y', 'X')
        assert sub_ct.shape == (3, 2, 2, 256, 256)
        assert int(sub_ct.asarray(maxworkers=1).sum()) == 699256173

        # coord keys follow the reordered dim order
        assert list(sub_ct.coords.keys())[:3] == ['Z', 'C', 'T']

        # scalar Z and single-element Z slice (Z=0) both squeeze Z out
        # and produce identical pixel data; slice is absolute coordinate
        sub_z0 = sc1(Z=0)
        sub_z0sl = sc1(Z=slice(0, 1))
        assert sub_z0.dims == ('T', 'C', 'Y', 'X')
        assert sub_z0sl.dims == ('T', 'C', 'Y', 'X')
        assert 'Z' not in sub_z0.coords
        assert 'Z' not in sub_z0sl.coords
        arr_z0 = sub_z0.asarray(maxworkers=1)
        assert int(arr_z0.sum()) == 242930060
        assert numpy.array_equal(arr_z0, sub_z0sl.asarray(maxworkers=1))

        # ROI in absolute file coordinates: sc1 occupies Y=853..1108, X=0..255
        roi = (20, 873, 100, 80)  # (x, y, width, height)
        sub_roi = sc1(roi=roi)
        assert sub_roi.dims == ('T', 'C', 'Z', 'Y', 'X')
        assert sub_roi.shape == (2, 2, 3, 80, 100)
        arr_roi = sub_roi.asarray(maxworkers=1)
        assert int(arr_roi.sum()) == 79054867

        # ROI coords represent absolute pixel positions of the cropped region:
        # roi y=873 is 20 rows into sc1 (which starts at file Y=853), so
        # Y[0] = sc1.Y[0] + 20*step; similarly X[0] = sc1.X[0] + 20*step
        cr = sub_roi.coords
        assert cr['Y'][0] == pytest.approx(0.0003492)
        assert cr['Y'][1] - cr['Y'][0] == pytest.approx(4e-7)
        assert cr['X'][0] == pytest.approx(8e-6)
        assert cr['X'][1] - cr['X'][0] == pytest.approx(4e-7)

        # ROI result equals the matching crop of the full scene array
        arr1 = sc1.asarray(maxworkers=1)
        rx, ry, rw, rh = roi
        assert numpy.array_equal(
            arr_roi, arr1[:, :, :, ry - 853 : ry - 853 + rh, rx : rx + rw]
        )

        # dim selection combined with ROI
        sub_roi_t = sc1(T=1, roi=roi)
        assert sub_roi_t.dims == ('C', 'Z', 'Y', 'X')
        assert sub_roi_t.shape == (2, 3, 80, 100)
        assert int(sub_roi_t.asarray(maxworkers=1).sum()) == 39114141

        # chaining: selection then ROI produces same result as combined call
        base_t1 = sc1(T=1)
        assert base_t1.dims == ('C', 'Z', 'Y', 'X')
        chained = base_t1(roi=roi)
        assert chained.dims == sub_roi_t.dims
        assert chained.shape == sub_roi_t.shape
        assert numpy.array_equal(
            chained.asarray(maxworkers=1),
            sub_roi_t.asarray(maxworkers=1),
        )

        # chaining preserves selection-specified dimension order
        reorder_base = sc1(C=None, T=None)  # dims ('Z', 'C', 'T', 'Y', 'X')
        chained_reorder = reorder_base(roi=roi)
        assert chained_reorder.dims == ('Z', 'C', 'T', 'Y', 'X')
        assert chained_reorder.shape == (3, 2, 2, 80, 100)

        # chaining is depth-1 only: selection -> roi is the only allowed
        # chain; any further call on the result raises TypeError.
        with pytest.raises(TypeError):
            sc1(T=1)(C=0)  # selection on selection
        with pytest.raises(TypeError):
            sc1(roi=roi)(C=0)  # selection on roi
        with pytest.raises(TypeError):
            sc1(roi=roi)(roi=roi)  # roi on roi
        with pytest.raises(TypeError):
            sc1(T=1)(roi=roi)(roi=roi)  # second roi on chained image
        with pytest.raises(TypeError):
            sc1(T=1)(roi=roi)(C=0)  # selection on chained image

        # sliding window: non-overlapping 128x128 tiles via chaining
        # reconstruct the full selection array pixel-for-pixel.
        # sc1 occupies Y=[853, 1109), X=[0, 256) in file coordinates.
        tile_size = 128
        x_start, y_start = 0, 853
        base_tc = sc1(T=0, C=0)  # shape (Z=3, Y=256, X=256)
        arr_tc = base_tc.asarray(maxworkers=1)
        reconstructed = numpy.zeros_like(arr_tc)
        for ri, ty in enumerate(range(y_start, y_start + 256, tile_size)):
            for ci, tx in enumerate(range(x_start, x_start + 256, tile_size)):
                t = base_tc(roi=(tx, ty, tile_size, tile_size)).asarray(
                    maxworkers=1
                )
                lo_r = ri * tile_size
                lo_c = ci * tile_size
                reconstructed[
                    :, lo_r : lo_r + tile_size, lo_c : lo_c + tile_size
                ] = t
        assert numpy.array_equal(reconstructed, arr_tc)

        # scenes() with a subset of scene indices: composite of sc0 and sc2
        sc02 = czi.scenes(scene=[0, 2])
        assert sc02.dims == ('T', 'C', 'Z', 'Y', 'X')
        assert sc02.shape == (2, 2, 3, 2055, 1999)
        assert int(sc02.asarray(maxworkers=1).sum()) == 20541343831

        # scenes(scene=[1]) with a single-item list isolates that scene and
        # returns pixel-for-pixel identical data to scenes[1].asarray()
        sc1_iso = czi.scenes(scene=[1])
        assert sc1_iso.shape == sc1.shape
        assert numpy.array_equal(sc1_iso.asarray(maxworkers=1), arr1)

        # composite vs OME-TIFF ground truth:
        # czi.scenes() merges all scenes into one full-bounding-box array.
        # The exported OME-TIFF (axes CTZYX) must match pixel-for-pixel after
        # transposing to TCZYX.
        ome_path = filename.with_suffix('').with_suffix('.ome.tiff')
        if ome_path.exists():
            with tifffile.TiffFile(ome_path) as tif:
                ome = tif.series[0].asarray()  # (C, T, Z, Y, X)
            ome_tczyx = numpy.transpose(ome, (1, 0, 2, 3, 4))
            composite = czi.scenes().asarray(maxworkers=1)
            assert composite.shape == ome_tczyx.shape
            assert numpy.array_equal(composite, ome_tczyx)


def test_decode_cache():
    """Test DecodeCache and CziFile.maxcache behavior."""
    # DecodeCache unit tests
    # disabled by default: get and put are no-ops
    cache = DecodeCache()
    assert cache.maxsize == 0
    assert cache.get(1) is None
    arr = numpy.zeros((2, 2), dtype='uint8')
    cache.put(1, arr)
    assert cache.get(1) is None

    # enabled: get/put round-trip
    cache = DecodeCache(maxsize=3)
    assert cache.maxsize == 3
    a0 = numpy.array([0], dtype='uint8')
    a1 = numpy.array([1], dtype='uint8')
    a2 = numpy.array([2], dtype='uint8')
    a3 = numpy.array([3], dtype='uint8')
    cache.put(0, a0)
    cache.put(1, a1)
    cache.put(2, a2)
    assert cache.get(0) is a0
    assert cache.get(1) is a1
    assert cache.get(2) is a2

    # FIFO eviction: inserting a 4th entry evicts the oldest (key 0)
    cache.put(3, a3)
    assert cache.get(0) is None
    assert cache.get(3) is a3

    # resize down: excess entries are evicted
    cache.resize(1)
    assert cache.maxsize == 1
    assert sum(cache.get(k) is not None for k in (1, 2, 3)) == 1

    # resize to 0: disables and clears the cache
    cache.resize(0)
    assert cache.maxsize == 0
    assert cache.get(3) is None

    # resize up from disabled
    cache.resize(2)
    assert cache.maxsize == 2
    cache.put(10, a0)
    assert cache.get(10) is a0

    # negative maxsize raises
    with pytest.raises(ValueError, match='maxsize'):
        cache.resize(-1)

    # scoped_resize: restores previous maxsize on exit
    cache = DecodeCache(maxsize=2)
    cache.put(0, a0)
    with cache.scoped_resize(4):
        assert cache.maxsize == 4
        cache.put(1, a1)
        assert cache.get(1) is a1
    assert cache.maxsize == 2  # maxsize restored
    assert cache.get(0) is a0  # original data preserved

    # scoped_resize from disabled: creates a temporary dict that is discarded
    cache = DecodeCache(maxsize=0)
    with cache.scoped_resize(4):
        assert cache.maxsize == 4
        cache.put(1, a1)
        assert cache.get(1) is a1
    assert cache.maxsize == 0  # restored to disabled
    assert cache.get(1) is None  # temporary dict discarded on exit

    # scoped_resize(0): no-op, cache left unchanged
    cache = DecodeCache(maxsize=3)
    cache.put(5, a0)
    with cache.scoped_resize(0):
        assert cache.maxsize == 3
        assert cache.get(5) is a0
    assert cache.maxsize == 3
    assert cache.get(5) is a0

    # CziFile.maxcache integration
    # ExampleZstd.czi: T=2, C=2, Z=3, Y=486, X=1178, Zstd-compressed
    filename = DATA / 'ExampleZstd.czi'
    if not filename.exists():
        pytest.skip(f'{filename!r} not found')

    # default: maxcache is None (auto mode), cache disabled at rest
    with CziFile(filename) as czi:
        assert czi.maxcache is None
        assert czi._cache.maxsize == 0

        # auto mode: cache enabled while iterating spatial chunks.
        # subblocks are 256x256; tile 256x256 -> nx=ny=2 -> maxcache=4
        img = czi.scenes[0]
        assert img.chunks(Y=256, X=256)._maxcache == 4
        # batching one non-spatial dim multiplies: C has 2 values -> 4*2=8
        assert img.chunks(Y=256, X=256, C=2)._maxcache == 8
        # C=None (keep whole dim in chunk): same factor as C=2 here -> 8
        assert img.chunks(Y=256, X=256, C=None)._maxcache == 8
        # Z=2 factor: 4*2=8
        assert img.chunks(Y=256, X=256, Z=2)._maxcache == 8
        # C=None + Z=None: 4 * C(2) * Z(3) = 24
        assert img.chunks(Y=256, X=256, C=None, Z=None)._maxcache == 24
        # no spatial tiling: cache stays 0
        assert img.chunks()._maxcache == 0

        chunks_iter = iter(img.chunks(Y=256, X=256))
        next(chunks_iter)
        assert czi._cache.maxsize == 4  # cache active inside chunks iterator
        for _ in chunks_iter:
            pass
        assert czi._cache.maxsize == 0  # restored after iteration

        # auto mode with non-spatial chunks: cache stays disabled
        for _ in img.chunks():
            assert czi._cache.maxsize == 0

    # maxcache=0: explicit disable, auto mode off
    with CziFile(filename) as czi:
        czi.maxcache = 0
        assert czi.maxcache == 0
        assert czi._cache.maxsize == 0
        # even spatial chunks leave cache disabled
        img = czi.scenes[0]
        for _ in img.chunks(Y=256, X=256):
            assert czi._cache.maxsize == 0

    # maxcache=N: explicit persistent FIFO, auto mode off
    with CziFile(filename) as czi:
        czi.maxcache = 5
        assert czi.maxcache == 5
        assert czi._cache.maxsize == 5
        img = czi.scenes[0]
        # cache stays at 5 throughout and after iteration
        for _ in img.chunks():
            assert czi._cache.maxsize == 5
        assert czi._cache.maxsize == 5

    # setting maxcache back to None restores auto mode
    with CziFile(filename) as czi:
        czi.maxcache = 10
        assert czi.maxcache == 10
        czi.maxcache = None
        assert czi.maxcache is None
        assert czi._cache.maxsize == 0

    # data correctness: cached result equals non-cached result
    with CziFile(filename) as czi:
        img = czi.scenes[0]
        arr_no_cache = img.asarray()
    with CziFile(filename) as czi:
        czi.maxcache = 8
        img = czi.scenes[0]
        arr_with_cache = img.asarray()
    assert numpy.array_equal(arr_no_cache, arr_with_cache)


@pytest.mark.skipif(
    not hasattr(sys, '_is_gil_enabled'), reason='Python < 3.13'
)
def test_gil_enabled():
    """Test that GIL state is consistent with build configuration."""
    assert sys._is_gil_enabled() != sysconfig.get_config_var('Py_GIL_DISABLED')


@pytest.mark.parametrize(
    'filename',
    [
        f
        for f in glob.glob('**/*.czi', root_dir=DATA, recursive=True)
        if not (DATA / f).is_dir()
    ],
)
def test_glob(filename):
    """Test read all CZI files and warn about unusual files."""
    if 'defective' in filename:
        pytest.xfail(reason='file is marked defective')
    filename = DATA / filename
    with CziFile(filename) as czi:
        str(czi)
        _ = czi.header
        if czi.header.update_pending:
            pytest.xfail(reason='file header update_pending')
        _ = czi.metadata()
        _ = czi.subblock_directory
        _ = czi.filtered_subblock_directory
        _ = czi.attachment_directory
        _ = czi.timestamps
        _ = czi.xml_element
        next(czi.segments(), None)
        next(czi.subblocks(), None)
        next(czi.attachments(), None)

        scene_keys = list(czi.scenes.keys())
        if len(scene_keys) > 1:
            expected = list(range(scene_keys[0], scene_keys[-1] + 1))
            if scene_keys != expected:
                warnings.warn(
                    f'non-consecutive scene indices {scene_keys!r}',
                    stacklevel=2,
                )

        for image in czi.scenes.values():
            for level in image.levels:
                if not level.directory_entries:
                    continue
                str(level)
                _ = level.coords
                _ = level.attrs
                _ = level.directory_entries
                _ = level.channels

                if level.compression in (
                    CziCompressionType.JPEG,
                    CziCompressionType.JPEGLL,
                    CziCompressionType.LZW,
                    CziCompressionType.UNKNOWN,
                ):
                    warnings.warn(
                        f'unsupported {level.compression!r}', stacklevel=2
                    )
                    continue
                if 100 <= int(level.compression) <= 999:
                    warnings.warn(
                        f'unsupported {level.compression!r}', stacklevel=2
                    )
                    continue
                if int(level.compression) >= 1000:
                    warnings.warn(
                        f'unsupported {level.compression!r}', stacklevel=2
                    )
                    continue
                if level.pixeltype == CziPixelType.UNKNOWN:
                    warnings.warn(f'unknown {level.pixeltype!r}', stacklevel=2)
                    continue

                if level.nbytes < MAX_SIZE:
                    arr = level.asarray()
                    assert arr.nbytes == level.nbytes
                    assert arr.dtype == level.dtype
                    assert arr.shape == level.shape

                chunks = level.chunks()
                e0 = level.directory_entries[0]
                e0_dims = list(e0.dims)
                if 'Y' in e0_dims and 'X' in e0_dims:
                    y_i, x_i = e0_dims.index('Y'), e0_dims.index('X')
                    rx, ry = e0.start[x_i], e0.start[y_i]
                    rw = min(e0.shape[x_i], 512)
                    rh = min(e0.shape[y_i], 512)
                    next(iter(level(roi=(rx, ry, rw, rh)).chunks())).asarray()
                else:
                    next(iter(chunks)).asarray()


@pytest.mark.parametrize(
    'filename',
    [
        f
        for f in glob.glob('**/*.czi', root_dir=DATA, recursive=True)
        if not (DATA / f).is_dir()
    ],
)
def test_glob_vs_pylibczirw(filename):
    """Cross-validate scene count, shape, and pixel type against pylibCZIrw."""
    pytest.importorskip('pylibCZIrw')
    from pylibCZIrw import czi as pyczi

    # maps CziPixelType.name to pylibCZIrw strings
    ptype_map = {
        'GRAY8': 'Gray8',
        'GRAY16': 'Gray16',
        'GRAY32FLOAT': 'Gray32Float',
        'GRAY32': 'Gray32',
        'GRAY64': 'Gray64',
        'BGR24': 'Bgr24',
        'BGR48': 'Bgr48',
        'BGR96FLOAT': 'Bgr96Float',
        'BGRA32': 'Bgra32',
    }
    if 'defective' in filename:
        pytest.xfail(reason='file is marked defective')

    KNOWN_BAD = {
        'Zeiss-5-SlidePreview-Zstd1-HiLo.czi',
    }
    if os.path.split(filename)[-1] in KNOWN_BAD:
        pytest.xfail(reason='libCZI bug')

    filepath = DATA / filename
    try:
        czidoc_ctx = pyczi.open_czi(str(filepath))
    except Exception as exc:
        pytest.skip(f'pylibCZIrw cannot open file: {exc}')
    with czidoc_ctx as czidoc:
        ref_scenes = czidoc.scenes_bounding_rectangle_no_pyramid
        ref_bbox = czidoc.total_bounding_box_no_pyramid
        ref_ptypes = czidoc.pixel_types
        with CziFile(filepath) as czi:
            if ref_scenes:
                assert len(ref_scenes) == len(czi.scenes)
                for s_key, czi_s in czi.scenes.items():
                    ref_rect = ref_scenes[s_key]
                    assert ref_rect.x == czi_s.bbox[0]
                    assert ref_rect.y == czi_s.bbox[1]
                    assert ref_rect.w == czi_s.sizes.get('X', 0)
                    assert ref_rect.h == czi_s.sizes.get('Y', 0)
            # Non-XY shape dims from total_bounding_box_no_pyramid: compare
            # only dims where pylibCZIrw reports > 1, i.e. it actually
            # captures variation.
            # T and Z are skipped because pylibCZIrw does not report
            # non-spatial displacement counts (e.g. line scans, FCS), so
            # the values may legitimately differ.
            skip_dims = {'X', 'Y', 'T', 'Z'}
            if ref_bbox and czi.scenes:
                for dim, (_start, ref_size) in ref_bbox.items():
                    if dim in skip_dims or ref_size <= 1:
                        continue
                    czi_max = max(
                        (s.sizes.get(dim, 0) for s in czi.scenes.values()),
                        default=0,
                    )
                    if czi_max > 0:
                        assert ref_size == czi_max
            # channel count and pixel type per channel (homogeneous only)
            if ref_ptypes and czi.scenes:
                first_scene = next(iter(czi.scenes.values()))
                ref_nc = len(ref_ptypes)
                czi_nc = first_scene.sizes.get('C', 1)
                assert ref_nc == czi_nc
                czi_ptype_name = CziPixelType(first_scene.pixeltype).name
                expected_str = ptype_map.get(czi_ptype_name)
                # only compare when all channels share same pixel type
                ref_ptype_values = set(ref_ptypes.values())
                if expected_str is not None and len(ref_ptype_values) == 1:
                    ref_ptype_str = next(iter(ref_ptype_values))
                    if ref_ptype_str not in ('Invalid', '???'):
                        assert ref_ptype_str == expected_str

            # Pixel-data comparison: for each scene assert that
            # czifile.asarray() and pylibCZIrw produce identical arrays.
            # Skip:
            #   - scenes larger than MAX_SIZE
            #   - lossy compressions where decoders may legitimately differ
            #   - scenes missing spatial Y/X axes (line/point scans)
            #   - scenes with non-T/C/Z non-spatial dims (e.g. Airyscan H)
            #     that pylibCZIrw does not model
            # pylibCZIrw supports only these pixel types for read()
            _plib_ptypes = {
                'GRAY8',
                'GRAY16',
                'GRAY32FLOAT',
                'BGR24',
                'BGR48',
                'BGR96FLOAT',
            }
            # Iterate over (S-coordinate, CziImage) pairs; s_key is the
            # absolute S-coordinate stored in the file (consistent with
            # pylibCZIrw's scenes_bounding_rectangle keys).
            for s_key, image in czi.scenes.items():
                if image.nbytes > MAX_SIZE:
                    continue
                compression = int(image.compression)
                if (
                    compression == CziCompressionType.JPEG
                    or compression >= 100
                ):
                    continue
                czi_dims = set(image.dims)
                if 'Y' not in czi_dims or 'X' not in czi_dims:
                    continue  # line / point scan
                nonspatial = [
                    d for d in image.dims if d not in ('Y', 'X', 'S')
                ]
                if set(nonspatial) - {'T', 'C', 'Z'}:
                    continue  # has dims pylibCZIrw cannot index (e.g. H)
                if CziPixelType(image.pixeltype).name not in _plib_ptypes:
                    continue  # pixel type not supported by pylibCZIrw
                if any(e.is_pyramid for e in image.directory_entries):
                    continue  # super-resolution subblocks differ between libs
                if any(
                    e.pixel_type != image.pixeltype
                    for e in image.directory_entries
                ):
                    # mixed pixel types: pylibCZIrw reads per-channel types
                    continue
                czi_arr = image.asarray()
                try:
                    ref_arr = pylibczirw_read_scene(
                        czidoc, image, s_key, bool(ref_scenes)
                    )
                except RuntimeError:
                    continue  # pylibCZIrw cannot decode, missing codec, type
                assert numpy.array_equal(czi_arr, ref_arr)


def pylibczirw_read_scene(czidoc, image, scene, has_scenes):
    """Return numpy array for *image* read via pylibCZIrw.

    The returned array matches the axis order produced by
    ``CziImage.asarray()``: ``(*nonspatial_dims, Y, X)`` for gray pixel
    types and ``(*nonspatial_dims, Y, X, S)`` for BGR/BGRA pixel types.

    Parameters:
        czidoc:
            Open ``pylibCZIrw.czi.CziReader`` instance for the same file.
        image:
            CziImage representing one scene (from ``CziFile.scenes``).
        scene:
            Zero-based scene index used as ``scene`` argument of
            ``czidoc.read()``.
        has_scenes:
            ``True`` when the file has explicit scene bounding rectangles
            (``czidoc.scenes_bounding_rectangle_no_pyramid`` is non-empty).
            When ``False`` the ``scene`` filter is omitted so that
            single-scene files without a Scene dimension are read in full.

    Returns:
        Pixel array with the same shape and dtype as ``image.asarray()``.

    """
    roi = image.bbox  # (x, y, w, h) — absolute pixel coords
    scene = scene if has_scenes else None
    # Always pass the explicit roi to get the exact spatial extent matching
    # image.bbox; pylibCZIrw's auto-derived scene roi is tile-rounded and
    # may be slightly larger.
    nonspatial = [d for d in image.dims if d not in ('Y', 'X', 'S')]
    has_s = 'S' in image.dims  # True for BGR/BGRA pixel types

    if not nonspatial:
        # Pure-spatial scene: a single read covers the whole array.
        arr = czidoc.read(roi=roi, scene=scene)  # (H, W, s)
    else:
        # Multi-dim: iterate every (T, C, Z, ...) combination and read one
        # spatial plane per call, then stack back into the full shape.
        # pylibCZIrw uses 0-based indices matching czifile's convention.
        sizes = image.sizes
        combos = list(
            itertools.product(*[range(sizes[d]) for d in nonspatial])
        )
        planes = [
            czidoc.read(
                roi=roi,
                plane=dict(zip(nonspatial, combo, strict=True)),
                scene=scene,
            )
            for combo in combos
        ]
        # planes: list of (H, W, s); stack (n_combos, H, W, s)
        stacked = numpy.stack(planes)
        nonspatial_shape = tuple(sizes[d] for d in nonspatial)
        arr = stacked.reshape((*nonspatial_shape, *stacked.shape[1:]))

    if not has_s:
        arr = arr[..., 0]
    elif arr.shape[-1] == 3:
        # pylibCZIrw returns BGR; czifile converts BGR -> RGB
        arr = arr[..., ::-1].copy()
    elif arr.shape[-1] == 4:
        # pylibCZIrw returns BGRA; czifile converts BGRA -> RGBA
        arr = arr[..., [2, 1, 0, 3]].copy()
    return arr


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=czifile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))

# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False
