# czifile/docs/conf.py

import enum
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.split(here)[0])

import czifile

project = 'czifile'
copyright = '2013-2026, Christoph Gohlke'
author = 'Christoph Gohlke'
version = czifile.__version__

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    # 'sphinxcontrib.spelling',
    # 'sphinx.ext.viewcode',
    # 'sphinx.ext.autosectionlabel',
    # 'numpydoc',
    # 'sphinx_issues',
]

templates_path = ['_templates']

html_theme = 'alabaster'
html_theme_options = {'sidebar_width': '256px'}

html_static_path = ['_static']
html_css_files = ['custom.css']
html_show_sourcelink = False

autodoc_member_order = 'bysource'  # bysource, groupwise
autodoc_default_flags = ['members']
autodoc_preserve_defaults = True
autodoc_typehints = 'description'
autodoc_type_aliases = {'ArrayLike': 'numpy.ArrayLike'}
autoclass_content = 'class'
autosectionlabel_prefix_document = True
autosummary_generate = True
toc_object_entries = False

napoleon_google_docstring = True
napoleon_numpy_docstring = False

html_sidebars = {
    '**': ['localtoc.html', 'moduletoc.html', 'searchbox.html'],
}


def add_api(app, what, name, obj, options, lines):
    if what == 'module':
        lines.extend(('API', '---'))


def _generate_moduletoc():
    """Generate _templates/moduletoc.html by inspecting the czifile module."""
    module = czifile

    _segment_keywords = ('Segment', 'Directory', 'Entry', 'Dimension')

    groups: dict[str, list[str]] = {
        'Classes': [],
        'Segment Data': [],
        'Exceptions': [],
        'Enumerations': [],
        'Functions': [],
        'Constants': [],
    }

    for name in module.__all__:  # type: ignore[attr-defined]
        obj = getattr(module, name)
        if isinstance(obj, type):
            if issubclass(obj, BaseException):
                groups['Exceptions'].append(name)
            elif issubclass(obj, enum.Enum):
                groups['Enumerations'].append(name)
            elif any(k in name for k in _segment_keywords):
                groups['Segment Data'].append(name)
            else:
                groups['Classes'].append(name)
        elif name == name.upper() or (
            name.startswith('__') and name.endswith('__')
        ):
            groups['Constants'].append(name)
        else:
            groups['Functions'].append(name)

    lines = ['<h3>Module API</h3>\n']
    for group, names in groups.items():
        if not names:
            continue
        lines.append(f'<h4>{group}</h4>')
        lines.append('<ul>')
        lines.extend(
            f'<li><a href="#czifile.{name}">{name}</a></li>' for name in names
        )
        lines.append('</ul>\n')

    outpath = os.path.join(here, '_templates', 'moduletoc.html')
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    content = '\n'.join(lines)
    with open(outpath, 'w', encoding='utf-8') as fh:
        fh.write(content)


def setup(app):
    _generate_moduletoc()
    app.connect('autodoc-process-docstring', add_api)


# mypy: allow-untyped-defs, allow-untyped-calls
