#!/usr/bin/env python
# Run examples and other scripts that should be updated before release.

import io, os, sys
import re

__basedir__ = os.getcwdu()

# Example imports


class Test:

    def __init__(self):
        self.messages = []

    def test(self, call, *args, **kwds):
        m = sys.modules[call.__module__]
        testname = m.__name__+'.'+call.__name__
        path = os.path.dirname(m.__file__)
        os.chdir(path)
        try:
            call(*args, **kwds)
            self.messages.append("%s: success" %testname)
        except Exception, e:
            self.messages.append("%s: error, details below.\n%s" %(testname, e))
        finally:
            os.chdir(__basedir__)    
        
    def report(self):
        print '==== Results of Tests ===='
        print '\n'.join(self.messages)
        
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
        
        updated = io.open(f, 'w', newline='\n')
        updated.write(text)
        updated.close()
        
        print "Updated %s to unix-style endlines." %f
    


if __name__ == "__main__":

    # Temporarily add examples to path
    lib_path = os.path.abspath(os.path.join('..','doc','examples'))
    sys.path.append(lib_path)

    ### Testing examples
    examples = Test()
    import Ag_singlepeak, Ag_multiplepeaks, TiO2_parameterdetail, TiO2_initialpeaks, C60_multimodelextraction, C60_multimodelanalysis
    examples.test(Ag_singlepeak.run, plot=False)
    examples.test(Ag_multiplepeaks.run, plot=False)
    examples.test(TiO2_parameterdetail.run, plot=False)
    examples.test(TiO2_initialpeaks.run, plot=False)
    examples.test(C60_multimodelextraction.run, plot=False)
    examples.test(C60_multimodelanalysis.run, plot=False)
    examples.report()
    
    ### Convert output of example files to Unix-style endlines for sdist.
    if os.linesep != '\n':
        print "==== Scrubbing Endlines ===="
        # All *.srmise and *.pwa files in examples directory.
        scrubeol("../doc/examples/output", r".*(\.srmise|\.pwa)")

        
