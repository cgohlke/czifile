# czifile/tests/conftest.py

import os
import sys

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )

if os.environ.get('SKIP_CODECS', None):
    sys.modules['imagecodecs'] = None


def pytest_report_header(config):
    try:
        from numpy import __version__ as numpy
        from test_czifile import config
        from tifffile import __version__ as tifffile

        from czifile import __version__ as czifile

        try:
            from imagecodecs import __version__ as imagecodecs
        except ImportError:
            imagecodecs = 'N/A'
        return (
            f'versions: czifile-{czifile}, '
            f'tifffile-{tifffile}, '
            f'imagecodecs-{imagecodecs}, '
            f'numpy-{numpy}\n'
            f'test config: {config()}'
        )
    except Exception:
        pass


collect_ignore = ['_tmp', 'data']

# mypy: ignore-errors
