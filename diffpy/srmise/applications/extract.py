#!/usr/bin/env python
##############################################################################
#
# SrMise            by Luke Granlund
#                   (c) 2014 trustees of the Michigan State University.
#                   All rights reserved.
#
# File coded by:    Luke Granlund
#
# See LICENSE.txt for license information.
#
##############################################################################

from optparse import OptionParser, OptionGroup
import diffpy.srmise

def main():
    """Default SrMise entry-point."""

    usage = ("usage: %prog pdf_file [options]\n"
             "pdf_file can be any pdf readable by diffpy.pdfgui, or a file saved by SrMise.")

    from diffpy.srmise import __version__
    version = "diffpy.srmise "+__version__

    descr = ("The SrMise package is a tool to aid extracting and fitting peaks "
             "that comprise a pair distribution function.")

    epilog = ("The options above override those already existing in a .srmise file, as well "
              "as the usual defaults in the SrMise package summarized here.\n\n"
              "Defaults (when qmax > 0)\n"
              "Baseline - Try to estimate a linear (Polynomial(degree=1)) baseline, otherwise"
              "use no baseline at all.\n"
              "dg - The uncertainty reported in the PDF (if any), otherwise 5% of maximum range of data.\n"
              "Nyquist - True\n"
              "Range - All the data\n"
              "Resolution - The Nyquist rate (thinking about changing this, though).\n"
              "Scale - False\n\n"
              "Defaults (when qmax = 0):\n"
              "Baseline - as above\n"
              "dg - as above\n"
              "Nyquist - False (and no effect if True)\n"
              "Range - as above\n"
              "Resolution - Four times the average distance between data points\n"
              "Scale - False (and no effect if True)\n\n"
              "Known issues\n"
              "1) Automatic estimation of qmax is unreliable if the provided input is not already sampled at "
              "at least twice the Nyquist rate.\n"
              "2) Plotting has only been tested with IPython, which provides several attractive features. "
              "Some plotting functions likely require its added functionality.\n"
              "3) Liveplotting doesn't yet use an idle handler, so any interaction with its window will "
              "likely cause it to freeze, even with IPython.\n"
              "4) Peak extraction is unreliable if the data are not moderately supersampled first.  When qmax > 0 "
              "this is handled automatically when needed, but when qmax = 0 no resampling of any kind is performed.\n"
              "5) Very small uncertainties drastically increase running time, and imply a model with more "
              "peaks than the data supports.  Unfortunately, the uncertainties calculated for most experimental "
              "PDFs are examples, often by one or two orders of magnitude.  For this reason the user should almost "
              "always specify and dg to use during extraction, and survey a range of these to find promising "
              "modesl.\n"
              "6) The interface for manually finding and/or tweaking extracted peaks is inelegant.\n"
              "7) Though not frequent, occasionally parameters get stuck and cause strange behavior like "
              "terrible peaks which do not improve, or seemingly good ones which are removed.  If obvious "
              "peaks are missing, a simple thing to try is to set the PDFPeakExtraction instance's "
              "initial_peaks member to the extracted model and extract again.")

    parser = OptionParser(usage=usage, description=descr, epilog=epilog, version=version,
                          formatter=IndentedHelpFormatterWithNL())

    parser.set_defaults(plot=False, liveplot=False, wait=False, performextraction=True, verbosity="warning")
    dg_defaults = {'absolute':None, 'data':None, 'max-fraction':.05, 'ptp-fraction':.05, 'dG-fraction':1.}

    parser.add_option("--peakfunction", dest="peakfunction", metavar="PF",
                      help="Fit peak function PF defined in diffpy.srmise.peaks. "
                           "e.g. 'GaussianOverR(maxwidth=0.7)'")
    parser.add_option("--me", "-m", dest="modelevaluator", metavar="ME",
                      help="ModelEvaluator defined in diffpy.srmise.modelevaluators. "
                           "e.g. 'AICc'")

    group = OptionGroup(parser, "Baseline Options",
                        "The PDF baseline is considered fixed until final stages of peak extraction, "
                        "and a good estimate is critical to reliable results.  Automated estimation "
                        "implemented only for linear baseline, and crudely.")
    group.add_option("--baseline", dest="baseline", metavar="BL",
                      help="Estimate baseline from baseline function BL defined in diffpy.srmise.baselines. "
                           "e.g. 'Polynomial(degree=1)'.  All parameters are free.")
    group.add_option("--bcrystal", dest="bcrystal", type="string", metavar="rho0[c]",
                      help="Use linear baseline defined by crystal number density rho0.  "
                           "Append 'c' to make parameter constant.  "
                           "Equivalent to '--bpoly1 -4*pi*rho0[c] 0c'.")
    group.add_option("--bsrmise", dest="bsrmise", type="string", metavar="file",
                      help="Use baseline from specified .srmise file.")
    group.add_option("--bpoly0", dest="bpoly0", type="string", metavar="a0[c]",
                      help="Use constant baseline given by y=a0.  Append 'c' to make parameter constant.")
    group.add_option("--bpoly1", dest="bpoly1", type="string", nargs=2, metavar="a1[c] a0[c]",
                      help="Use baseline given by y=a1*x + a0.  Append 'c' to make parameter constant.")
    group.add_option("--bpoly2", dest="bpoly2", type="string", nargs=3, metavar="a2[c] a1[c] a0[c]",
                      help="Use baseline given by y=a2*x^2+a1*x + a0.  Append 'c' to make parameter constant.")
    group.add_option("--bseq", dest="bseq", type="string", metavar="file",
                      help="Use baseline interpolated from x,y values in file."
                           "This baseline has no free parameters.")
    group.add_option("--bspherical", dest="bspherical", type="string", nargs=2, metavar="s[c] r[c]",
                      help="Use spherical nanoparticle baseline with scale s and radius r.  "
                           "Append 'c' to make parameter constant.")
    parser.add_option_group(group)


    group = OptionGroup(parser, "Uncertainty Options",
                        "May specify ")
    parser.add_option("--dg-mode", dest="dg_mode", type="choice",
                      choices=['absolute', 'data', 'max-fraction', 'ptp-fraction', 'dG-fraction'],
                      help="Treat value for --dg as absolute, fraction of maximum value in range, fraction of peak-to-peak value in range, "
                           "or fraction of reported dG.  If --dg is given but mode is not, mode is absolute. "
                           "Otherwise, dG-fraction is default if the PDF reports uncertainties, and max-fraction is default if it does not.")
    parser.add_option("--dg", dest="dg", type="float",
                      help="Perform extraction assuming uncertainty dg. Defaults depend on --dg-mode as follows. "
                      "absolute=%s, max-fraction=%s, ptp-fraction=%s, dG-fraction=%s."
                      %(dg_defaults['absolute'], dg_defaults['max-fraction'], dg_defaults['ptp-fraction'], dg_defaults['dG-fraction']))
#    parser.add_option("--multimodel", nargs=3, dest="multimodel", type="float", metavar="dg_min dg_max n",
#                      help="Generate n models from dg_min to dg_max (given by --dg-mode) and perform multimodel analysis. "
#                           "This overrides any value given for --dg")
    parser.add_option_group(group)


    parser.add_option("--nyquist", action="store_true", dest="nyquist",
                      help="Use Nyquist resampling if qmax > 0.")
    parser.add_option("--no-nyquist", action="store_false", dest="nyquist",
                      help="Do not use Nyquist resampling.")
    parser.add_option("--qmax", dest="qmax", type="float", metavar="QMAX",
                      help="Model peaks with this maximum q value.")
    parser.add_option("--range", nargs=2, dest="rng", type="float", metavar="rmin rmax",
                      help="Extract over the range (rmin, rmax).")
    parser.add_option("--resolution", dest="cres", type="float", metavar="cres",
                      help="Cluster resolution.")
    parser.add_option("--scale", action="store_true", dest="scale",
                      help="Scale supersampled uncertainties by sqrt(oversampling) in intermediate steps when Nyquist sampling.")
    parser.add_option("--no-scale", action="store_false", dest="scale",
                      help="Never rescale uncertainties.")

    group = OptionGroup(parser, "Analysis Options",
                        "")
    parser.add_option("--extract", action="store_true", dest="performextraction",
                      help="[Default] Perform extraction.")
    parser.add_option("--no-extract", action="store_false", dest="performextraction",
                      help="Do not perform extraction.")
    parser.add_option_group(group)


    group = OptionGroup(parser, "Output Options",
                        "These options may have different effects for single or multimodel extraction.")
    parser.add_option("--name", dest="name", help="Name extraction, serves as basis for any saved files.")
    parser.add_option("--pwa", dest="pwafile", metavar="FILE",
                      help="Save summary of result to FILE (.pwa format).")
    parser.add_option("--save", dest="savefile", metavar="FILE",
                      help="Save result of extraction to FILE (.srmise format).")
    parser.add_option("--plot", "-p", action="store_true", dest="plot",
                      help="Plot extracted peaks.")
    parser.add_option_group(group)


    group = OptionGroup(parser, "Debug Options",
                        "Control output to console.")
    parser.add_option("--informative", "-i", action="store_const", const="info", dest="verbosity",
                      help="Summary of progress.")
    parser.add_option("--quiet", "-q", action="store_const", const="warning", dest="verbosity",
                      help="[Default] Show minimal summary.")
    parser.add_option("--silent", "-s", action="store_const", const="critical", dest="verbosity",
                      help="No non-critical output.")
    parser.add_option("--verbose", "-v", action="store_const", const="debug", dest="verbosity",
                      help="Show verbose output.")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Deprecated/Experimental Options",
                        "Not for general use.")
    parser.add_option("--liveplot", "-l", action="store_true", dest="liveplot",
                      help="Plot extracted peaks when fitting. (Experimental)")
    parser.add_option("--wait", "-w", action="store_true", dest="wait",
                      help="When using liveplot wait for user after plotting.")
    parser.add_option_group(group)

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("Exactly one argument required. \n"+usage)

    from diffpy.srmise import srmiselog
    srmiselog.setlevel(options.verbosity)

    import numpy as np
    from diffpy.srmise.pdfpeakextraction import PDFPeakExtraction
    from diffpy.srmise.srmiseerrors import SrMiseDataFormatError, SrMiseFileError

    if options.peakfunction is not None:
        from diffpy.srmise import peaks
        try:
            options.peakfunction = eval("peaks."+options.peakfunction)
        except Exception, err:
            print err
            print "Could not create peak function '%s'.  Using default type." %options.peakfunction
            options.peakfunction = None

    if options.modelevaluator is not None:
        from diffpy.srmise import modelevaluators
        try:
            options.modelevaluator = eval("modelevaluators."+options.modelevaluator)
        except Exception,  err:
            print err
            print "Could not find ModelEvaluator '%s'.  Using default type." %options.modelevaluator
            options.modelevaluator = None

    if options.bcrystal is not None:
        from diffpy.srmise.baselines import Polynomial
        bl = Polynomial(degree=1)
        options.baseline = parsepars(bl, [options.bcrystal, '0c'])
        options.baseline.pars[0] = -4*np.pi*options.baseline.pars[0]
    elif options.bsrmise is not None:
        # use baseline from existing file
        blext = PDFPeakExtraction()
        blext.read(options.bsrmise)
        options.baseline = blext.extracted.baseline
    elif options.bpoly0 is not None:
        from diffpy.srmise.baselines import Polynomial
        bl = Polynomial(degree=0)
        options.baseline = parsepars(bl, [options.bpoly0])
        #options.baseline = bl.actualize([options.bpoly0], "internal")
    elif options.bpoly1 is not None:
        from diffpy.srmise.baselines import Polynomial
        bl = Polynomial(degree=1)
        options.baseline = parsepars(bl, options.bpoly1)
    elif options.bpoly2 is not None:
        from diffpy.srmise.baselines import Polynomial
        bl = Polynomial(degree=2)
        options.baseline = parsepars(bl, options.bpoly2)
    elif options.bseq is not None:
        from diffpy.srmise.baselines import FromSequence
        bl = FromSequence(options.bseq)
        options.baseline = bl.actualize([], "internal")
    elif options.bspherical is not None:
        from diffpy.srmise.baselines import NanoSpherical
        bl = NanoSpherical()
        options.baseline = parsepars(bl, options.bspherical)
    elif options.baseline is not None:
        from diffpy.srmise import baselines
        try:
            options.baseline = eval("baselines."+options.baseline)
        except Exception, err:
            print err
            print "Could not create baseline '%s'.  No baseline will be used." %options.baseline
            options.baseline = None

    filename = args[0]

    if filename is not None:
        ext = PDFPeakExtraction()
        try:
            ext.read(filename)
        except (SrMiseDataFormatError, SrMiseFileError, Exception):
            ext.loadpdf(filename)

        pdict = {}
        if options.peakfunction is not None:
            pdict["pf"] = [options.peakfunction]
        if options.baseline is not None:
            pdict["baseline"] = options.baseline
        if options.cres is not None:
            pdict["cres"] = options.cres
        if options.dg_mode is None:
            if options.dg is not None:
                options.dg_mode = "absolute"
            elif ext.dy is None:
                options.dg_mode = "max-fraction"
            else:
                options.dg_mode = "dG-fraction"
        if options.dg is None:
            options.dg = dg_defaults[options.dg_mode]
        if options.dg_mode == "absolute":
            pdict["effective_dy"] = options.dg*np.ones(len(ext.x))
        elif options.dg_mode == "max-fraction":
            pdict["effective_dy"] = options.dg*ext.y.max()*np.ones(len(ext.x))
        elif options.dg_mode == "ptp-fraction":
            pdict["effective_dy"] = options.dg*ext.y.ptp()*np.ones(len(ext.y))
        elif options.dg_mode == "dG-fraction":
            pdict["effective_dy"] = options.dg*ext.dy            
        if options.rng is not None:
            pdict["rng"] = list(options.rng)
        if options.qmax is not None:
            pdict["qmax"] = options.qmax
        if options.nyquist is not None:
            pdict["nyquist"] = options.nyquist
        if options.scale is not None:
            pdict["scale"] = options.scale
        if options.modelevaluator is not None:
            pdict["error_method"] = options.modelevaluator

        if options.liveplot:
            from diffpy.srmise import srmiselog
            srmiselog.liveplotting(True, options.wait)

        ext.setvars(**pdict)
        if options.performextraction:
            ext.extract()
        out = ext.extracted

        if options.savefile is not None:
            try:
                ext.write(options.savefile)
            except SrMiseFileError, err:
                print err
                print "Could not save result to '%s'." %options.savefile


        if options.pwafile is not None:
            try:
                ext.writepwa(options.pwafile)
            except SrMiseFileError, err:
                print err
                print "Could not save pwa summary to '%s'." %options.pwafile

        print ext

        if options.plot:
            import matplotlib.pyplot as plt
            plt.figure(1)
            ext.plot()
            plt.figure(2)
            ext.plot(joined=True)


            plt.show()
        elif options.liveplot:
            plt.show()

def parsepars(mp, parseq):
    """Return actualized model from sequence of strings.

    Each item in parseq must be interpretable as a float, or as
    a float with the character 'c' appended.  If 'c' is appended,
    that parameter will be fixed.

    Parameters:
    mp - A ModelPart instance
    parseq - A sequence of string
    """
    pars = []
    free = []
    for p in parseq:
        if p[-1] == 'c':
            pars.append(float(p[0:-1]))
            free.append(False)
        else:
            pars.append(float(p))
            free.append(True)
    return mp.actualize(pars, "internal", free=free)


### Class to preserve newlines in optparse
# Borrowed, with minor changes, from
# http://groups.google.com/group/comp.lang.python/browse_frm/thread/6df6e6b541a15bc2/09f28e26af0699b1

from optparse import IndentedHelpFormatter
import textwrap

class IndentedHelpFormatterWithNL(IndentedHelpFormatter):
  def _format_text(self, text):
    if not text: return ""
    text_width = self.width - self.current_indent
    indent = " "*self.current_indent
# the above is still the same
    bits = text.split('\n')
    formatted_bits = [
      textwrap.fill(bit,
        text_width,
        initial_indent=indent,
        subsequent_indent=indent)
      for bit in bits]
    result = "\n".join(formatted_bits) + "\n"
    return result

  def format_option(self, option):
    # The help for each option consists of two parts:
    #   * the opt strings and metavars
    #   eg. ("-x", or "-fFILENAME, --file=FILENAME")
    #   * the user-supplied help string
    #   eg. ("turn on expert mode", "read data from FILENAME")
    #
    # If possible, we write both of these on the same line:
    #   -x    turn on expert mode
    #
    # But if the opt string list is too long, we put the help
    # string on a second line, indented to the same column it would
    # start in if it fit on the first line.
    #   -fFILENAME, --file=FILENAME
    #       read data from FILENAME
    result = []
    opts = self.option_strings[option]
    opt_width = self.help_position - self.current_indent - 2
    if len(opts) > opt_width:
      opts = "%*s%s\n" % (self.current_indent, "", opts)
      indent_first = self.help_position
    else: # start help on same line as opts
      opts = "%*s%-*s  " % (self.current_indent, "", opt_width, opts)
      indent_first = 0
    result.append(opts)
    if option.help:
      help_text = self.expand_default(option)
# Everything is the same up through here
      help_lines = []
      for para in help_text.split("\n"):
        help_lines.extend(textwrap.wrap(para, self.help_width))
# Everything is the same after here
      result.append("%*s%s\n" % (
        indent_first, "", help_lines[0]))
      result.extend(["%*s%s\n" % (self.help_position, "", line)
        for line in help_lines[1:]])
    elif opts[-1] != "\n":
      result.append("\n")
    return "".join(result)
### End class

if __name__ == "__main__":
    main()
