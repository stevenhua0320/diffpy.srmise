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

    ### Testing examples
    examples = Test()
    from diffpy.srmise.applications.examples import C60, caffeine, SrTiO3_qmax26, C60multiextract, C60multimodel
    examples.test(C60.run, plot=False)
    examples.test(caffeine.run, plot=False)
    examples.test(SrTiO3_qmax26.run, plot=False)
    examples.test(C60multiextract.run, plot=False)
    examples.test(C60multimodel.main, ['--report', '--savepwa'])
    
    examples.report()
    
    ### Convert output of example files to Unix-style endlines for sdist.
    if os.linesep != '\n':
        print "==== Scrubbing Endlines ===="
        # All *.srmise and *.pwa files in examples directory.
        scrubeol("diffpy/srmise/applications/examples", r".*(\.srmise|\.pwa)")

        
