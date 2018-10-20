# -*- coding: utf-8 -*-
# czifile/tests/conftest.py

collect_ignore = ['_tmp', 'data']


def pytest_report_header(config):
    try:
        import czifile
        import imagecodecs
        return 'versions: czifile-%s, imagecodecs-%s, numpy-%s' % (
            czifile.__version__, imagecodecs.__version__, numpy.__version__)
    except Exception:
        pass
