Read Carl Zeiss(r) Image (CZI) files
====================================

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
