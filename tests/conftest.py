# czifile/tests/conftest.py

"""Pytest configuration."""

import os
import sys

if os.environ.get('VSCODE_CWD'):
    # work around pytest not using PYTHONPATH in VSCode
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    )

if os.environ.get('SKIP_CODECS', None):
    sys.modules['imagecodecs'] = None


def pytest_configure(config):
    """Show all warnings, including duplicates."""
    config.addinivalue_line('filterwarnings', 'always')


def pytest_report_header(config):
    """Return pytest report header."""
    try:
        from fsspec import __version__ as fsspec
        from numpy import __version__ as numpy
        from test_czifile import config
        from tifffile import __version__ as tifffile

        import czifile

        try:
            from imagecodecs import __version__ as imagecodecs
        except ImportError:
            imagecodecs = 'N/A'
        return (
            f'Python {sys.version.splitlines()[0]}\n'
            f'packagedir: {czifile.__path__[0]}\n'
            f'versions: czifile-{czifile.__version__}, '
            f'tifffile-{tifffile}, '
            f'imagecodecs-{imagecodecs}, '
            f'numpy-{numpy}, '
            f'fsspec-{fsspec}\n'
            f'test config: {config()}'
        )
    except Exception as exc:
        return f'pytest_report_header failed: {exc!s}'


collect_ignore = ['_tmp', 'data']

# mypy: ignore-errors
