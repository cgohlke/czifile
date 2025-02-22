# -*- coding: utf-8 -*-
# czifile/__init__.py

from .czifile import *
from .czifile import __all__, __doc__, __version__


def _set_module():
    """Set __module__ attribute for all public objects."""
    globs = globals()
    module = globs['__name__']
    for item in __all__:
        obj = globs[item]
        if hasattr(obj, '__module__'):
            obj.__module__ = module


_set_module()
