#!/usr/bin/env python

# Installation script for diffpy.srmise

"""diffpy.srmise - Peak extraction/fitting tool for pair distribution functions

Packages:   diffpy.srmise
"""

import os

from setuptools import find_packages, setup

# versioncfgfile holds version data for git commit hash and date.
# It must reside in the same directory as version.py.
MYDIR = os.path.dirname(os.path.abspath(__file__))
versioncfgfile = os.path.join(MYDIR, "diffpy/srmise/version.cfg")


def gitinfo():
    from subprocess import PIPE, Popen

    kw = dict(stdout=PIPE, cwd=MYDIR)
    proc = Popen(["git", "describe", "--match=v[[:digit:]]*"], **kw)
    desc = proc.stdout.read()
    proc = Popen(["git", "log", "-1", "--format=%H %at %ai"], **kw)
    glog = proc.stdout.read()
    rv = {}
    rv["version"] = "-".join(desc.strip().split("-")[:2]).lstrip("v")
    rv["commit"], rv["timestamp"], rv["date"] = glog.strip().split(None, 2)
    return rv


def getversioncfg():
    #    from ConfigParser import SafeConfigParser

    cp = SafeConfigParser()
    cp.read(versioncfgfile)
    gitdir = os.path.join(MYDIR, ".git")
    if not os.path.isdir(gitdir):
        return cp
    try:
        g = gitinfo()
    except OSError:
        return cp
    d = cp.defaults()
    if g["version"] != d.get("version") or g["commit"] != d.get("commit"):
        cp.set("DEFAULT", "version", g["version"])
        cp.set("DEFAULT", "commit", g["commit"])
        cp.set("DEFAULT", "date", g["date"])
        cp.set("DEFAULT", "timestamp", g["timestamp"])
        cp.write(open(versioncfgfile, "w"))
    return cp


# versiondata = getversioncfg()

# define distribution, but make this module importable
setup_args = dict(
    name="diffpy.srmise",
    version="0.0.0",
    namespace_packages=["diffpy"],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # Dependencies
    # numpy
    # scipy
    # matplotlib >= 1.1.0
    install_requires=["matplotlib >= 1.1.0", "numpy", "scipy"],
    # other arguments here...
    entry_points={
        "console_scripts": [
            "srmise = diffpy.srmise.applications.extract:main",
            "srmiseplot = diffpy.srmise.applications.plot:main",
        ]
    },
    author="Luke Granlund",
    author_email="luke.r.granlund@gmail.com",
    description=(
        "Peak extraction and peak fitting tool for atomic "
        "pair distribution functions."
    ),
    license="BSD-style license",
    url="https://github.com/diffpy/diffpy.srmise/",
    keywords="peak extraction fitting PDF AIC multimodeling",
    classifiers=[
        # List of possible values at
        # http://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 2.6",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries",
    ],
)

if __name__ == "__main__":
    setup(**setup_args)
