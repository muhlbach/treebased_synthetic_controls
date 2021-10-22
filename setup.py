#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Universal setup.py-file inspired by "https://github.com/kennethreitz/setup.py"
To run this, cd to folder and run $ python setup.py upload
Remeber to UPDATE the version every time
"""
# Import libraries
import io
import os
import sys
from shutil import rmtree
from setuptools import find_packages, setup, Command

## Package meta-data.
# Package
NAME = 'treebased_synthetic_controls'
DESCRIPTION = 'Tree-based Synthetic Control Group Methods (Mühlbach & Nielsen, 2020)'
URL = f'https://github.com/muhlbach/{NAME}'
LONG_DESC_TYPE = "text/markdown"
LICENSE = 'MIT License'

# Author
AUTHOR = 'Nicolaj Søndergaard Mühlbach'
AUTHOR_EMAIL = 'n.muhlbach@gmail.com'

#
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.0.3'

# What packages are required for this module to be executed?
INSTALL_REQUIRES = [
    'numpy',
    'pandas',
    'statsmodels',
    'sklearn',
]

# What packages are optional?
EXTRAS_REQUIRE = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

# Find path
here = os.path.abspath(os.path.dirname(__file__)) # Same as pathlib.Path(__file__).parent

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    # Use short description as long description
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    try:
        project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
        with open(os.path.join(here, project_slug, '__version__.py')) as f:
            exec(f.read(), about)
    except:
        about['__version__'] = VERSION
else:
    about['__version__'] = VERSION

class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    license=LICENSE,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
