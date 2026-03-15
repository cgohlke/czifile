Revisions
=========

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

- Replace deprecated imagecodecs.jxr codec.

2019.7.2.2

- Replace deprecated tifffile.stripnull function.

2019.7.2.1

- Fix broken and deprecated imports.
- Update copyright and package metadata.

2019.7.2

- Require tifffile 2019.7.2.

2019.6.18

- Add package main function to view CZI files.
- Fix BGR to RGB conversion.
- Fix czi2tif conversion on Python 2.

2019.5.22

- Fix czi2tif conversion when CZI metadata contain non-ASCII characters.
- Use imagecodecs_lite as a fallback for imagecodecs.
- Make CziFile.metadata a function (breaking).
- Make scipy an optional dependency; fallback on ndimage or fail on zoom().

2019.1.26

- Fix czi2tif console script.

2018.10.18

- Rename zisraw package to czifile.

2018.8.29

- Move czifile.py and related modules into zisraw package.
- Move usage examples to main docstring.
- Require imagecodecs package for decoding JpegXrFile, JpgFile, and LZW.

2018.6.18

- Save CZI metadata to TIFF description in czi2tif.
- Fix AttributeError using max_workers=1.
- Make Segment.SID and DimensionEntryDV1.dimension str types.
- Return metadata as XML unicode string or dict, not etree.
- Return timestamps, focus positions, events, and luts as tuple or ndarray

2017.7.21

- Use multi-threading in CziFile.asarray to decode and copy segment data.
- Always convert BGR to RGB. Remove bgr2rgb options.
- Decode JpegXR directly from byte arrays.

2017.7.13

- Add function to convert CZI file to memory-mappable TIFF file.

2017.7.11

- Add 'out' parameter to CziFile.asarray.
- Remove memmap option from CziFile.asarray (breaking).
- Change spline interpolation order to 0 (breaking).
- Make axes return a string.
- Require tifffile 2017.7.11.

2014.10.10

- Read data into a memory mapped array (optional).

2013.12.4

- Decode JpegXrFile and JpgFile via _czifle extension module.
- Attempt to reconstruct tiled mosaic images.

2013.11.20

- Initial release.
