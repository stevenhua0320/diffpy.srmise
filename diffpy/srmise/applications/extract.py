#!/usr/bin/env python
##############################################################################
#
# SrMise            by Luke Granlund
#                   (c) 2014-2015 trustees of the Michigan State University.
#                   All rights reserved.
#
# File coded by:    Luke Granlund
#
# See LICENSE.txt for license information.
#
##############################################################################

import textwrap
from optparse import IndentedHelpFormatter, OptionGroup, OptionParser

import matplotlib.pyplot as plt
import numpy as np


def main():
    """Default SrMise entry-point."""

    usage = (
        "usage: %prog pdf_file [options]\n"
        "pdf_file is a file containing a PDF (accepts several "
        "common formats), or a .srmise file."
    )

    from diffpy.srmise import __version__

    version = "diffpy.srmise " + __version__

    descr = (
        "The SrMise package is a tool to aid extracting and fitting peaks "
        "that comprise a pair distribution function.  This script exposes "
        "basic peak extraction functionality. For many PDFs it is "
        "sufficient to specify the range, baseline, and sometimes an ad "
        "hoc uncertainty. See the discussion of these options below for "
        "further guidance."
    )

    epilog = (
        "Options set above override those from an existing .srmise "
        "file, as well as the usual defaults summarized here.\n\n"
        "Defaults (when qmax > 0)\n"
        "------------------------\n"
        "baseline - None (identically 0).\n"
        "dg - The uncertainty reported in the PDF (if any), otherwise "
        "5% of maximum value of PDF.\n"
        "nyquist - True\n"
        "range - All the data\n"
        "cres - The Nyquist rate.\n"
        "supersample - 4.0\n"
        "scale - (Deprecated) False\n\n"
        "Defaults (when qmax = 0)\n"
        "------------------------\n"
        "baseline - as above\n"
        "dg - as above\n"
        "nyquist - False (and no effect if True)\n"
        "range - as above\n"
        "cres - Four times the average distance between data points\n"
        "supersample - Parameter has no effect.\n"
        "scale - (Deprecated) False, and no effect if True\n\n"
        "Known issues\n"
        "------------\n"
        "1) Peak extraction works best when the data are moderately "
        "oversampled first.  When qmax > 0 this is handled "
        "automatically, but when qmax = 0 no resampling of any kind is "
        "performed.\n"
        "2) Peak extraction performed on a PDF file and a .srmise file "
        "derived from that data with identical extraction parameters "
        "can give different results even on the same platform.  This is "
        "because the original data may undergo some processing before it "
        "can be saved by SrMise.  For consistent results, always specify "
        "the original PDF, or always load the PDF from a .srmise file "
        "you save before performing any peak extraction on that data.\n"
        "3) Liveplotting depends on the matplotlib backend, and doesn't "
        "implement an idle handler, so interaction with its window will "
        "likely cause a freeze."
    )

    # TODO: Move to argparse (though not in 2.6 by default) to handle
    # variable-length options without callbacks.  Longterm, the major
    # value is using the same option to specify a baseline that should
    # use estimation vs. one that should use explicitly provided pars.
    parser = OptionParser(
        usage=usage,
        description=descr,
        epilog=epilog,
        version=version,
        formatter=IndentedHelpFormatterWithNL(),
    )

    parser.set_defaults(
        plot=False,
        liveplot=False,
        wait=False,
        performextraction=True,
        verbosity="warning",
    )
    dg_defaults = {
        "absolute": None,
        "data": None,
        "max-fraction": 0.05,
        "ptp-fraction": 0.05,
        "dG-fraction": 1.0,
    }

    parser.add_option(
        "--extract",
        action="store_true",
        dest="performextraction",
        help="[Default] Perform extraction.",
    )
    parser.add_option(
        "--no-extract",
        action="store_false",
        dest="performextraction",
        help="Do not perform extraction.",
    )
    parser.add_option(
        "--range",
        nargs=2,
        dest="rng",
        type="float",
        metavar="rmin rmax",
        help="Extract over the range (rmin, rmax).",
    )
    parser.add_option(
        "--qmax",
        dest="qmax",
        type="string",
        metavar="QMAX",
        help="Model peaks with this maximum q value.",
    )
    parser.add_option(
        "--nyquist",
        action="store_true",
        dest="nyquist",
        help="Use Nyquist resampling if qmax > 0.",
    )
    parser.add_option(
        "--no-nyquist",
        action="store_false",
        dest="nyquist",
        help="Do not use Nyquist resampling.",
    )
    parser.add_option(
        "--pf",
        dest="peakfunction",
        metavar="PF",
        help="Fit peak function PF defined in " "diffpy.srmise.peaks, e.g. " "'GaussianOverR(maxwidth=0.7)'",
    )
    parser.add_option(
        "--cres",
        dest="cres",
        type="float",
        metavar="cres",
        help="Clustering resolution.",
    )
    parser.add_option(
        "--supersample",
        dest="supersample",
        type="float",
        metavar="SS",
        help="Minimum initial oversampling rate as multiple of " "Nyquist rate.",
    )
    parser.add_option(
        "--me",
        "-m",
        dest="modelevaluator",
        metavar="ME",
        help="ModelEvaluator defined in " "diffpy.srmise.modelevaluators, e.g. 'AIC'",
    )

    group = OptionGroup(
        parser,
        "Baseline Options",
        "SrMise cannot determine the appropriate type of "
        "baseline (e.g. crystalline vs. some nanoparticle) "
        "solely from the data, so the user should specify the "
        "appropriate type and/or parameters. (Default is "
        "identically 0, which is unphysical.) SrMise keeps the "
        "PDF baseline fixed at its initial value until the "
        "final stages of peak extraction, so results are "
        "frequently conditioned on that choice. (See the "
        "SrMise documentation for details.)  A good estimate "
        "is therefore important for best results.  SrMise can "
        "estimate initial parameters from the data for linear "
        "baselines in some situations (all peaks are positive, "
        "and the degree of overlap in the region of extraction "
        "is not too great), but in most cases it is best to "
        "provide reasonable initial parameters.  Run 'srmise "
        "pdf_file.gr [baseline_option] --no-extract --plot' "
        "for different values of the parameters for rapid "
        "visual estimation.",
    )
    group.add_option(
        "--baseline",
        dest="baseline",
        metavar="BL",
        help="Estimate baseline from baseline function BL "
        "defined in diffpy.srmise.baselines, e.g. "
        "'Polynomial(degree=1)'.  All parameters are free. "
        "(Many POSIX shells attempt to interpret the "
        "parentheses, and on these shells the option should "
        "be surrounded by quotation marks.)",
    )
    group.add_option(
        "--bcrystal",
        dest="bcrystal",
        type="string",
        metavar="rho0[c]",
        help="Use linear baseline defined by crystal number "
        "density rho0. Append 'c' to make parameter "
        "constant. Equivalent to "
        "'--bpoly1 -4*pi*rho0[c] 0c'.",
    )
    group.add_option(
        "--bsrmise",
        dest="bsrmise",
        type="string",
        metavar="file",
        help="Use baseline from specified .srmise file.",
    )
    group.add_option(
        "--bpoly0",
        dest="bpoly0",
        type="string",
        metavar="a0[c]",
        help="Use constant baseline given by y=a0. " "Append 'c' to make parameter constant.",
    )
    group.add_option(
        "--bpoly1",
        dest="bpoly1",
        type="string",
        nargs=2,
        metavar="a1[c] a0[c]",
        help="Use baseline given by y=a1*x + a0.  Append 'c' to " "make parameter constant.",
    )
    group.add_option(
        "--bpoly2",
        dest="bpoly2",
        type="string",
        nargs=3,
        metavar="a2[c] a1[c] a0[c]",
        help="Use baseline given by y=a2*x^2+a1*x + a0.  Append " "'c' to make parameter constant.",
    )
    group.add_option(
        "--bseq",
        dest="bseq",
        type="string",
        metavar="FILE",
        help="Use baseline interpolated from x,y values in FILE. " "This baseline has no free parameters.",
    )
    group.add_option(
        "--bspherical",
        dest="bspherical",
        type="string",
        nargs=2,
        metavar="s[c] r[c]",
        help="Use spherical nanoparticle baseline with scale s "
        "and radius r. Append 'c' to make parameter "
        "constant.",
    )
    parser.add_option_group(group)

    group = OptionGroup(
        parser,
        "Uncertainty Options",
        "Ideally a PDF reports the accurate experimentally "
        "determined uncertainty.  In practice, many PDFs "
        "report none, while for others the reported values "
        "are not necessarily reliable. (If in doubt, ask your "
        "friendly neighborhood diffraction expert!) Even when "
        "uncertainties are accurate, it can be "
        "pragmatically useful to see how the results of "
        "peak extraction change when assuming a different "
        "value.  Nevertheless, the primary determinant of "
        "model complexity in SrMise is the uncertainty, so an "
        "ad hoc uncertainty yields ad hoc model complexity. "
        "See the SrMise documentation for further discussion, "
        "including methods to mitigate this issue with "
        "multimodel selection.",
    )
    group.add_option(
        "--dg-mode",
        dest="dg_mode",
        type="choice",
        choices=["absolute", "data", "max-fraction", "ptp-fraction", "dG-fraction"],
        help="Define how values passed to '--dg' are treated. "
        "Possible values are: \n"
        "'absolute' - The actual uncertainty in the PDF.\n"
        "'max-fraction' - Fraction of max value in PDF.\n"
        "'ptp-fraction' - Fraction of max minus min value "
        "in the PDF.\n"
        "'dG-fraction' - Fraction of dG reported by PDF.\n"
        "If '--dg' is specified but mode is not, then mode "
        "ia absolute.  Otherwise, 'dG-fraction' is default "
        "if the PDF reports uncertaintes, and 'max-fraction' "
        "ia default if it does not.",
    )
    group.add_option(
        "--dg",
        dest="dg",
        type="float",
        help="Perform extraction assuming uncertainty dg. "
        "Defaults depend on --dg-mode as follows:\n"
        "'absolute'=%s\n"
        "'max-fraction'=%s\n"
        "'ptp-fraction'=%s\n"
        "'dG-fraction'=%s"
        % (
            dg_defaults["absolute"],
            dg_defaults["max-fraction"],
            dg_defaults["ptp-fraction"],
            dg_defaults["dG-fraction"],
        ),
    )
    #    group.add_option("--multimodel", nargs=3, dest="multimodel", type="float",
    #                     metavar="dg_min dg_max n",
    #                     help="Generate n models from dg_min to dg_max (given by "
    #                          "--dg-mode) and perform multimodel analysis. "
    #                          "This overrides any value given for --dg")
    parser.add_option_group(group)

    group = OptionGroup(parser, "Saving and Plotting Options", "")
    group.add_option(
        "--pwa",
        dest="pwafile",
        metavar="FILE",
        help="Save summary of result to FILE (.pwa format).",
    )
    group.add_option(
        "--save",
        dest="savefile",
        metavar="FILE",
        help="Save result of extraction to FILE (.srmise " "format).",
    )
    group.add_option("--plot", "-p", action="store_true", dest="plot", help="Plot extracted peaks.")
    group.add_option(
        "--liveplot",
        "-l",
        action="store_true",
        dest="liveplot",
        help="(Experimental) Plot extracted peaks when fitting.",
    )
    group.add_option(
        "--wait",
        "-w",
        action="store_true",
        dest="wait",
        help="(Experimental) When using liveplot wait for user " "after plotting.",
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Verbosity Options", "Control detail printed to console.")
    group.add_option(
        "--informative",
        "-i",
        action="store_const",
        const="info",
        dest="verbosity",
        help="Summary of progress.",
    )
    group.add_option(
        "--quiet",
        "-q",
        action="store_const",
        const="warning",
        dest="verbosity",
        help="[Default] Show minimal summary.",
    )
    group.add_option(
        "--silent",
        "-s",
        action="store_const",
        const="critical",
        dest="verbosity",
        help="No non-critical output.",
    )
    group.add_option(
        "--verbose",
        "-v",
        action="store_const",
        const="debug",
        dest="verbosity",
        help="Show verbose output.",
    )
    parser.add_option_group(group)

    group = OptionGroup(parser, "Deprecated Options", "Not for general use.")
    group.add_option(
        "--scale",
        action="store_true",
        dest="scale",
        help="(Deprecated) Scale supersampled uncertainties by "
        "sqrt(oversampling) in intermediate steps when "
        "Nyquist sampling.",
    )
    group.add_option(
        "--no-scale",
        action="store_false",
        dest="scale",
        help="(Deprecated) Never rescale uncertainties.",
    )
    parser.add_option_group(group)

    (options, args) = parser.parse_args()

    if len(args) != 1:
        parser.error("Exactly one argument required. \n" + usage)

    from diffpy.srmise import srmiselog

    srmiselog.setlevel(options.verbosity)

    from diffpy.srmise.pdfpeakextraction import PDFPeakExtraction
    from diffpy.srmise.srmiseerrors import SrMiseDataFormatError, SrMiseFileError

    try:
        options.peakfunction = eval("peaks." + options.peakfunction)
    except Exception as err:
        print(err)
        print("Could not create peak function '%s'. Exiting." % options.peakfunction)
        return

    try:
        options.modelevaluator = eval("modelevaluators." + options.modelevaluator)
    except Exception as err:
        print(err)
        print("Could not find ModelEvaluator '%s'. Exiting." % options.modelevaluator)
        return

    if options.bcrystal is not None:
        from diffpy.srmise.baselines import Polynomial

        bl = Polynomial(degree=1)
        options.baseline = parsepars(bl, [options.bcrystal, "0c"])
        options.baseline.pars[0] = -4 * np.pi * options.baseline.pars[0]
    elif options.bsrmise is not None:
        # use baseline from existing file
        blext = PDFPeakExtraction()
        blext.read(options.bsrmise)
        options.baseline = blext.extracted.baseline
    elif options.bpoly0 is not None:
        from diffpy.srmise.baselines import Polynomial

        bl = Polynomial(degree=0)
        options.baseline = parsepars(bl, [options.bpoly0])
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

        try:
            options.baseline = eval("baselines." + options.baseline)
        except Exception as err:
            print(err)
            print("Could not create baseline '%s'. Exiting." % options.baseline)
            return

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
            pdict["effective_dy"] = options.dg * np.ones(len(ext.x))
        elif options.dg_mode == "max-fraction":
            pdict["effective_dy"] = options.dg * ext.y.max() * np.ones(len(ext.x))
        elif options.dg_mode == "ptp-fraction":
            pdict["effective_dy"] = options.dg * ext.y.ptp() * np.ones(len(ext.y))
        elif options.dg_mode == "dG-fraction":
            pdict["effective_dy"] = options.dg * ext.dy
        if options.rng is not None:
            pdict["rng"] = list(options.rng)
        if options.qmax is not None:
            pdict["qmax"] = options.qmax if options.qmax == "automatic" else float(options.qmax)
        if options.nyquist is not None:
            pdict["nyquist"] = options.nyquist
        if options.supersample is not None:
            pdict["supersample"] = options.supersample
        if options.scale is not None:
            pdict["scale"] = options.scale
        if options.modelevaluator is not None:
            pdict["error_method"] = options.modelevaluator

        if options.liveplot:
            from diffpy.srmise import srmiselog

            srmiselog.liveplotting(True, options.wait)

        ext.setvars(**pdict)
        cov = None
        if options.performextraction:
            cov = ext.extract()

        if options.savefile is not None:
            try:
                ext.write(options.savefile)
            except SrMiseFileError as err:
                print(err)
                print("Could not save result to '%s'." % options.savefile)

        if options.pwafile is not None:
            try:
                ext.writepwa(options.pwafile)
            except SrMiseFileError as err:
                print(err)
                print("Could not save pwa summary to '%s'." % options.pwafile)

        print(ext)
        if cov is not None:
            print(cov)

        if options.plot:
            from diffpy.srmise.applications.plot import makeplot

            makeplot(ext)
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
        if p[-1] == "c":
            pars.append(float(p[0:-1]))
            free.append(False)
        else:
            pars.append(float(p))
            free.append(True)
    return mp.actualize(pars, "internal", free=free)


# Class to preserve newlines in optparse
# Borrowed, with minor changes, from
# http://groups.google.com/group/comp.lang.python/browse_frm/thread/6df6e6b541a15bc2/09f28e26af0699b1


class IndentedHelpFormatterWithNL(IndentedHelpFormatter):
    def _format_text(self, text):
        if not text:
            return ""
        text_width = self.width - self.current_indent
        indent = " " * self.current_indent
        # the above is still the same
        bits = text.split("\n")
        formatted_bits = [
            textwrap.fill(bit, text_width, initial_indent=indent, subsequent_indent=indent) for bit in bits
        ]
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
        else:  # start help on same line as opts
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
            result.append("%*s%s\n" % (indent_first, "", help_lines[0]))
            result.extend(["%*s%s\n" % (self.help_position, "", line) for line in help_lines[1:]])
        elif opts[-1] != "\n":
            result.append("\n")
        return "".join(result)


# End class

if __name__ == "__main__":
    main()
