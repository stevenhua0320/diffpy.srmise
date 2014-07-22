#!/usr/bin/env python

from setuptools import setup, find_packages
setup(
    name = "diffpy.srmise",
    version = "0.4a1",
    namespace_packages = ['diffpy'],
    packages = find_packages(),
    include_package_data = True,
    zip_safe = False,

    # Dependencies
    # pdfgui
    # numpy
    # scipy
    # matplotlib >= 1.1.0
    install_requires = ['diffpy.pdfgui', 'matplotlib >= 1.1.0', 'numpy', 'scipy'],

    # other arguments here...
    entry_points = {
        'console_scripts' : [
            'srmise = diffpy.srmise.applications.extract:main',
            'srmiseplot = diffpy.srmise.applications.plot:main',
            ]
    },
)
