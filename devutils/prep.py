#!/usr/bin/env python
# Run examples and other scripts that should be updated before release.

import io
import os
import re
import sys

__basedir__ = os.getcwdu()

from numpy.compat import unicode

# Example imports


class Test:

    def __init__(self):
        self.messages = []

    def test(self, call, *args, **kwds):
        m = sys.modules[call.__module__]
        testname = m.__name__ + "." + call.__name__
        path = os.path.dirname(m.__file__)
        os.chdir(path)
        try:
            call(*args, **kwds)
            self.messages.append("%s: success" % testname)
        except Exception as e:
            self.messages.append("%s: error, details below.\n%s" % (testname, e))
        finally:
            os.chdir(__basedir__)

    def report(self):
        print("==== Results of Tests ====")
        print("\n".join(self.messages))


def scrubeol(directory, filerestr):
    """Use unix-style endlines for files in directory matched by regex string.

    Parameters
    ----------
    directory - A directory to scrub
    filerestr - A regex string defining which files to match."""
    os.chdir(__basedir__)
    os.chdir(directory)
    files = [re.match(filerestr, f) for f in os.listdir(".")]
    files = [f.group(0) for f in files if f]

    for f in files:
        original = open(f)
        text = unicode(original.read())
        original.close()

        updated = io.open(f, "w", newline="\n")
        updated.write(text)
        updated.close()

        print("Updated %s to unix-style endlines." % f)


def rm(directory, filerestr):
    """Delete files in directory matched by regex string.

    Parameters
    ----------
    directory - A directory to scrub
    filerestr - A regex string defining which files to match."""
    os.chdir(__basedir__)
    os.chdir(directory)
    files = [re.match(filerestr, f) for f in os.listdir(".")]
    files = [f.group(0) for f in files if f]

    for f in files:
        os.remove(f)

        print("Deleted %s." % f)


if __name__ == "__main__":

    # Temporarily add examples to path
    lib_path = os.path.abspath(os.path.join("..", "doc", "examples"))
    sys.path.append(lib_path)

    # Delete existing files that don't necessarily have a fixed name.
    rm("../doc/examples/output", r"known_dG.*\.pwa")
    rm("../doc/examples/output", r"unknown_dG.*\.pwa")

    # Testing examples
    examples = Test()
    test_names = [
        "extract_single_peak",
        "parameter_summary",
        "fit_initial",
        "query_results",
        "multimodel_known_dG1",
        "multimodel_known_dG2",
        "multimodel_unknown_dG1",
        "multimodel_unknown_dG2",
    ]

    test_modules = []
    for test in test_names:
        test_modules.append(__import__(test))

    for test in test_modules:
        examples.test(test.run, plot=False)

    examples.report()

    # Convert output of example files to Unix-style endlines for sdist.
    if os.linesep != "\n":
        print("==== Scrubbing Endlines ====")
        # All *.srmise and *.pwa files in examples directory.
        scrubeol("../doc/examples/output", r".*(\.srmise|\.pwa)")
