#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# czi2tif.py

"""Convert CZI file to memory-mappable BigTIFF file.

Usage: czi2tif czi_filename [tif_filename]

"""

import sys

try:
    from .czifile import czi2tif
except ImportError:
    try:
        from czifile.czifile import czi2tif
    except ImportError:
        from czifile import czi2tif


def main(argv=None):
    """Czi2tif command line usage main function."""
    if argv is None:
        argv = sys.argv
    if len(argv) > 1:
        truncate = '--truncate' in argv
        if truncate:
            argv.remove('--truncate')
        czi2tif(argv[1], argv[2] if len(argv) > 2 else None,
                bigtiff=True, truncate=truncate)
    else:
        print()
        print(__doc__)


if __name__ == '__main__':
    sys.exit(main())
