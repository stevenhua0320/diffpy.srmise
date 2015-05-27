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
"""plot extracted peaks and comparison to ideal distances (if given)"""

import sys
import optparse

from diffpy.srmise import PDFPeakExtraction, PeakStability
from diffpy.srmise.pdfpeakextraction import resample

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import mpl_toolkits.axisartist as AA

# For a given figure, returns a label of interest
labeldict = {}

default_gobs_style = {'color' : 'b', 'linestyle' : '',
        'markeredgecolor' : 'b', 'marker' : 'o',
        'markerfacecolor' : 'none', 'markersize' : 4}

default_gfit_style = {'color' : 'g'}
default_gind_style = {'facecolor' : 'green', 'alpha' : 0.2}
default_gres_style = {}
default_ep_style = {}
default_ip_style = {}

default_dg_style = {'linestyle' : 'none', 'color' : 'black',
        'marker' : 'o',  'markerfacecolor' : 'black',
        'markeredgecolor' : 'black',
        'markersize' : 1, 'antialiased': False}


def setfigformat(figsize):
    from matplotlib import rc
    rc('legend', numpoints=2)
    rc('figure', figsize=figsize)
    rc('axes', titlesize=12, labelsize=11)
    rc('xtick', labelsize=10)
    rc('ytick', labelsize=10)
    rc('lines', linewidth=0.75, markeredgewidth=0.5)
    return

def gr_formataxis(ax=None):
    if ax is None:  ax = plt.gca()
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    ax.yaxis.set_ticks_position('both')
    return

def comparepositions(ppe, ip=None, **kwds):
    ax = kwds.get("ax", plt.gca())
    base = kwds.get("base", 0.)
    yideal = kwds.get("yideal", -1.)
    yext = kwds.get("yext", 1.)
    ep_style = kwds.get("ep_style", default_ep_style)
    ip_style = kwds.get("ip_style", default_ip_style)
    yideal_label = kwds.get("yideal_label", r'ideal')
    yext_label = kwds.get("yext_label", r'found')
    pmin = kwds.get("pmin", -np.inf)
    pmax = kwds.get("pmax", np.inf)

    ep = [p["position"] for p in ppe.model]
    ep = [p for p in ep if p >= pmin and p <= pmax]

    if ip is not None:
        xi = np.NaN + np.zeros(3*len(ip))
        xi[0::3] = ip
        xi[1::3] = ip
        yi = np.zeros_like(xi) + base
        yi[1::3] += yideal
        plt.plot(xi, yi, 'b', lw=1.5, **ip_style)

    xe = np.NaN + np.zeros(3*len(ep))
    xe[0::3] = ep
    xe[1::3] = ep
    ye = np.zeros_like(xe) + base
    ye[1::3] += yext
    plt.plot(xe, ye, 'g', lw=1.5, **ep_style)

    if ip is not None:
        yb = (base, base)
        plt.axhline(base, linestyle=":", color="k" )
        ax.yaxis.set_ticks([base+.5*yideal, base+.5*yext])
        ax.yaxis.set_ticklabels([yideal_label, yext_label])
    else:
        ax.yaxis.set_ticks([base+.5*yext])
        ax.yaxis.set_ticklabels([yext_label])

    # Set ylim explicitly, for case where yext is empty.
    if ip is not None:
        plt.ylim(base+yideal, base+yext)
    else:
        plt.ylim(base+yideal, base)

    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_markersize(0)
        tick.tick2line.set_markersize(0)
        tick.label1.set_verticalalignment('center')
        tick.label1.set_fontsize(8)
    ticks = ax.yaxis.get_major_ticks()
    ticks[-1].label1.set_color("green")
    if ip is not None:
        ticks[0].label1.set_color("blue")
    return

def dgseries(stability, **kwds):
    ax = kwds.get("ax", plt.gca())
    dg_style = kwds.get("dg_style", default_dg_style)

    scale = kwds.get("scale", 1.)

    dgmin = kwds.get("dgmin", stability.results[0][0])*scale
    dgmax = kwds.get("dgmax", stability.results[-1][0])*scale

    pmin = kwds.get("pmin", 0.)
    pmax = kwds.get("pmax", np.inf)

    x = []
    y = []
    for dg, peaks, bl, dr in stability.results:
        if dg*scale < dgmin or dg*scale > dgmax:
            continue
        peakpos = [p["position"] for p in peaks]
        peakpos = [p for p in peakpos if p >= pmin and p <= pmax]
        x.extend(peakpos)
        y.extend(np.zeros_like(peakpos) + dg*scale)
    plt.plot(x, y, **dg_style)

def labelallsubplots():
    rv = []
    for i, c in enumerate('abcd'):
        plt.subplot(221 + i)
        s = "(%s)" % c
        ht = plt.text(0.04, 0.95, s,
                horizontalalignment='left', verticalalignment='top',
                transform=gca().transAxes, weight='bold')
        rv.append(ht)
    return rv


def makeplot(ppe_or_stability, ip=None, **kwds):
    """Plot stuff"""
    if isinstance(ppe_or_stability, PeakStability):
        stability = ppe_or_stability
        ppe = stability.ppe
    else:
        stability = None
        ppe = ppe_or_stability
    ext = ppe.extracted

    figdict = {}

    # Range along x-axis
    xlo = kwds.get("xlo", ext.r_cluster[0])
    xhi = kwds.get("xhi", ext.r_cluster[-1])

    # Range of PDF to display
    # This is deferred until the defaults can be calculated
    # min_gr
    # max_gr

    # Define heights and interstitial offsets
    # All values in percent of main axis.
    top_offset = kwds.get("top_offset", 0.)
    dg_height = kwds.get("dg_height", 15. if stability is not None else 0.)
    cmp_height = kwds.get("cmp_height", 15. if ip is not None else 7.5)
    datatop_offset = kwds.get("datatop_offset", 3.)
    # <- Data appears here ->
    databottom_offset = kwds.get("databottom_offset", 3.)
    # <- Residual appears here ->
    bottom_offset = kwds.get("bottom_offset", 3.)

    # Style options
    dg_style = kwds.get("dg_style", default_dg_style)
    gobs_style = kwds.get("gobs_style", default_gobs_style)
    gfit_style = kwds.get("gfit_style", default_gfit_style)
    gind_style = kwds.get("gind_style", default_gind_style)
    gres_style = kwds.get("gres_style", default_gres_style)
    ep_style = kwds.get("ep_style", default_ep_style)
    ip_style = kwds.get("ip_style", default_ip_style)

    # Label options
    userxlabel = kwds.get("xlabel", r'r ($\mathrm{\AA}$)')
    userylabel = kwds.get("ylabel", r'G ($\mathrm{\AA^{-2}}$)')
    datalabelx = kwds.get("datalabelx", .04)
    yideal_label = kwds.get("yideal_label", r'ideal')
    yext_label = kwds.get("yext_label", r'found')

    # Other options
    datalabel = kwds.get("datalabel", None)
    dgformatstr = kwds.get("dgformatstr", r'$\delta$g=%s')
    dgformatpost = kwds.get("dgformatpost", None) #->userfunction(string)
    show_fit = kwds.get("show_fit", True)
    show_individual = kwds.get("show_individual", True)
    fill_individual = kwds.get("fill_individual", True)
    show_observed = kwds.get("show_observed", True)
    show_residual = kwds.get("show_residual", True)
    mask_residual = kwds.get("mask_residual", False) #-> number
    show_annotation = kwds.get("show_annotation", True)
    scale = kwds.get("scale", 1.) # Apply a global scaling factor to the data



    # Define the various data which will be plotted
    r = ext.r_cluster
    dr = (r[-1]-r[0])/len(r)
    rexpand = np.concatenate((np.arange(r[0]-dr, xlo, -dr)[::-1], r, np.arange(r[-1]+dr, xhi+dr, dr)))
    rfine = np.arange(r[0], r[-1], .1*dr)
    gr_obs = np.array(resample(ppe.x, ppe.y, rexpand))*scale
    #gr_fit = resample(r, ext.value(), rfine)
    gr_fit = np.array(ext.value(rfine))*scale
    gr_fit_baseline = np.array(ext.valuebl(rfine))*scale
    gr_fit_ind = [gr_fit_baseline + np.array(p.value(rfine))*scale for p in ext.model]
    gr_res = np.array(ext.residual())*scale

    if mask_residual:
        gr_res = np.ma.masked_outside(gr_res, -mask_residual, mask_residual)

    all_gr = []
    if show_fit: all_gr.append(gr_fit)
    #if show_individual: all_gr.extend([gr_fit_baseline, gr_fit_ind])
    if show_individual:
        all_gr.append(gr_fit_baseline)
        if len(gr_fit_ind) > 0:
            all_gr.extend(gr_fit_ind)
    if show_observed: all_gr.append(gr_obs)

    # gr_fit_ind is a list of lists, so use np.min/max
    # The funky bit with scale makes sure that a user-specified value
    # has scale applied to it, without messing up the default values,
    # which are calculated from already scaled quantities.
    min_gr = kwds.get("min_gr", np.min([np.min(gr) for gr in all_gr])/scale)*scale
    max_gr = kwds.get("max_gr", np.max([np.max(gr) for gr in all_gr])/scale)*scale


    if show_residual:
        min_res = np.min(gr_res)
        max_res = np.max(gr_res)
    else:
        min_res = 0.
        max_res = 0.

    # Derive various y limits based on all the offsets
    rel_height = 100. - top_offset - dg_height - cmp_height - datatop_offset - databottom_offset - bottom_offset
    abs_height = 100*((max_gr - min_gr) + (max_res - min_res))/rel_height

    yhi = max_gr + (top_offset + dg_height + cmp_height + datatop_offset)*abs_height/100
    ylo = yhi - abs_height

    yhi = kwds.get("yhi", yhi)
    ylo = kwds.get("ylo", ylo)

    datatop = yhi - (yhi-ylo)*.01*(top_offset + dg_height + cmp_height)
    datalabeltop = 1 - .01*(top_offset + dg_height + cmp_height + datatop_offset)
    resbase = ylo + bottom_offset*abs_height/100 - min_res

    resbase = kwds.get("resbase", resbase)


    fig = kwds.get("figure", plt.gcf())
    ax_data = AA.Subplot(fig, 111)
    fig.add_subplot(ax_data)
    figdict["fig"] = fig
    figdict["data"] = ax_data

    # Plot the data, fit, and residual
    if show_observed:
        plt.plot(rexpand, gr_obs, **gobs_style)
    if show_fit:
        plt.plot(rfine, gr_fit, **gfit_style)
    if fill_individual:
        for peak in gr_fit_ind:
            plt.fill_between(rfine, gr_fit_baseline, peak, **gind_style)
    if show_residual:
        plt.plot(r, gr_res + resbase, 'r-', **gres_style)
        plt.plot((xlo, xhi), 2*[resbase], 'k:')

    # Format ax_data
    plt.xlim(xlo, xhi)
    plt.ylim(ylo, yhi)
    plt.xlabel(userxlabel)
    plt.ylabel(userylabel)
    ax_data.xaxis.set_minor_locator(plt.MultipleLocator(1))
    #ax_data.yaxis.set_minor_locator(plt.MultipleLocator(np.max([1,int((yhi-ylo)/20)])))
    ax_data.yaxis.set_label_position('left')
    ax_data.yaxis.tick_left()
    ax_data.yaxis.set_ticks_position('both')

    # Remove labels above where insets begin
    #ax_data.yaxis.set_ticklabels([str(int(loc)) for loc in ax_data.yaxis.get_majorticklocs() if loc < datatop])
    ax_data.yaxis.set_ticks([loc for loc in ax_data.yaxis.get_majorticklocs() if (loc < datatop and loc >= ylo)])


    # Dataset label
    if datalabel is not None:
        dl = plt.text(datalabelx, datalabeltop, datalabel, ha='left', va='top',
             transform=ax_data.transAxes, weight='bold')
    else:
        dl = None
    figdict["datalabel"] = dl

    # Create new x axis at bottom edge of compare inset
    ax_data.axis["top"].set_visible(False)
    ax_data.axis["newtop"] = ax_data.new_floating_axis(0, datatop, axis_direction="bottom") # "top" bugged?
    ax_data.axis["newtop"].toggle(all=False, ticks=True)
    ax_data.axis["newtop"].major_ticks.set_tick_out(True)
    ax_data.axis["newtop"].minor_ticks.set_tick_out(True)

    # New y-axis label, since AxisLabel positions cannot be set manually.
    # The original label is invisible, but we use its (dynamic) x position
    # to update the new label, which we define have the correct y position.
    # A bit of a tradeoff for the nice insets and ability to define new axes.
    newylabel = plt.text(-.1, .5*(datatop-ylo)/(yhi-ylo), userylabel,
         ha='center', va='center', rotation='vertical', transform=ax_data.transAxes)
    labeldict[fig] = newylabel # so we can find the correct text object
    fig.canvas.mpl_connect('draw_event', on_draw) # original label invisibility and updating

    # Compare extracted (and ideal, if provided) peak positions clearly.
    if cmp_height > 0:
        ax_cmp = inset_axes(ax_data,
                            width="100%",
                            height="%s%%" %cmp_height,
                            loc=2,
                            bbox_to_anchor=(0., -.01*(top_offset+dg_height), 1, 1),
                            bbox_transform=ax_data.transAxes,
                            borderpad=0)

        figdict["cmp"] = ax_cmp
        plt.axes(ax_cmp)
        comparepositions(ext, ip, ep_style=ep_style, ip_style=ip_style, yideal_label=yideal_label, yext_label=yext_label)
        plt.xlim(xlo, xhi)
        ax_cmp.set_xticks([])

    # Show how extracted peak positions change as dg is changed
    if dg_height > 0:

        ax_dg = inset_axes(ax_data,
                           width="100%",
                           height="%s%%" %dg_height,
                           loc=2,
                           bbox_to_anchor=(0, -.01*top_offset, 1, 1),
                           bbox_transform=ax_data.transAxes,
                           borderpad=0)

        figdict["dg"] = ax_dg
        plt.axes(ax_dg)
        dgkwds = {}
        if "scale" in kwds:
            dgkwds["scale"] = kwds["scale"]
        if "dgmin" in kwds:
            dgkwds["dgmin"] = kwds["dgmin"]
        if "dgmax" in kwds:
            dgkwds["dgmax"] = kwds["dgmax"]
        dgseries(stability, base=0, pmin=r[0], pmax=r[-1], **dgkwds)
        plt.xlim(xlo, xhi)
        ax_dg.xaxis.set_major_locator(plt.NullLocator())
        ax_dg.yaxis.set_major_locator(plt.MaxNLocator(3))
        plt.ylabel(r'$\delta$g')

    # Annotate the actual dg shown
    if show_annotation:
        dg = np.mean(ext.error_cluster)*scale
        dgstr = dgformatstr %dg
        if "dgformatpost" in kwds: #post-processing on dg annotation
            dgstr = kwds["dgformatpost"](dgstr)

        if len(ext.model) > 0:
            xpos = np.mean([xlo, ext.model[0]["position"]]) # OK for now.
        else:
            xpos = xlo
        if dg_height > 0 and cmp_height > 0:
            # Arrow, text in compare distances line
            ylo2, yhi2 = ax_dg.get_ylim()
            if ip is not None:
                ypos = ylo2 - .25*cmp_height/dg_height*(yhi2-ylo2)
            else:
                ypos = ylo2 - .5*cmp_height/dg_height*(yhi2-ylo2)
            plt.annotate(dgstr, xy=(xlo, dg), xycoords='data', va='center', ha='center',
                        xytext=(xpos,ypos), textcoords='data', size=8, color='green',
                        arrowprops=dict(arrowstyle="->",
                                        connectionstyle="angle,angleA=90,angleB=0,rad=10",
                                        color="green"))

        elif dg_height > 0:
            # Arrow, and text located somewhere in main plot region
            # Must change axes
            pass
        elif cmp_height > 0:
            # No arrow, text in compare distances line
            # Must change axes
            plt.axes(ax_cmp)
            ylo2, yhi2 = ax_cmp.get_ylim()
            ypos = yhi2/2.
            plt.text(xpos, ypos, dgstr, va='center', ha='center', size=8, color='green')
        else:
            # Text only in main plot region
            # Must change axes
            pass

    plt.draw()

    return figdict


# Bit of a kluge to make sure the label on the y-axis
# is placed correctly.  The "invisiblelabel" has correct
# x-position, but it's y-position cannot be manually set.
# The visiblelabel has correct y-position, and we update
# its x-position based on invisiblelabel.  Of course,
# invisiblelabel must be temporarily made visible to update
# its values.
_lastxpos = 0
def on_draw(event):
    global _lastxpos
    ax_main = plt.gcf().get_axes()[0]
    invisiblelabel = ax_main.axis["left"].label
    invisiblelabel.set_visible(True)
    visiblelabel = labeldict[plt.gcf()]
    bbox = invisiblelabel.get_window_extent(invisiblelabel._renderer)
    bbox = bbox.inverse_transformed(ax_main.transAxes)
    bbox = bbox.get_points()
    xpos = np.mean(np.transpose(bbox)[0])

    # This, and the whole lastxpos business, is so label is properly
    # updated when using the Agg backend to create a .png. (at least in
    # matplotlib 1.1.0)  For some reason the invisible label is not set
    # correctly when drawn with that backend unless redrawn at least once.
    # If it is kept visible the whole time this problem doesn't occur.
    # This problem doesn't occur onscreen (TkAgg) or printing PDFs, and
    # didn't occur in matplotlib 1.0.0.
    if abs(xpos - _lastxpos) > .001:
        _lastxpos = xpos
        plt.draw()
    else:
        _lastxpos = xpos

    invisiblelabel.set_visible(False)
    xpos_old = visiblelabel.get_position()[0]
    if abs(xpos - xpos_old) > .001:
        labeldict[plt.gcf()].set_x(xpos)
        plt.draw()
    return

def readcompare(filename):
    """Returns a list of distances read from filename, otherwise None."""
    from diffpy.srmise.srmiseerrors import SrMiseDataFormatError, SrMiseFileError

    # TODO: Make this safer
    try:
        datastring = open(filename,'rb').read()
    except Exception, err:
        raise err

    import re
    res = re.search(r'^[^#]', datastring, re.M)
    if res:
        datastring = datastring[res.end():].strip()

    distances = []

    try:
        for line in datastring.split("\n"):
            distances.append(float(line))
    except (ValueError, IndexError), err:
        print "Could not read distances from '%s'. Ignoring file." %filename

    if len(distances) == 0:
        return None
    else:
        return distances


def main():
    # configure options parsing
    usage = ("%prog srmise_file [options]\n"
            "srmise_file can be an extraction file saved by SrMise, "
            "or a data file saved by PeakStability.")
    descr = ("A very basic tool for somewhat prettier plotting than provided by "
             "the basic SrMise classes.  Can be used to compare peak positions "
             "with those from a list.\n"
             "NOTE: At this time the utility only works with peaks extracted using diffpy.srmise.PDFPeakExtraction.")

    parser = optparse.OptionParser(usage=usage, description=descr)
    parser.add_option("--compare", type="string",
            help="Compare extracted distances to distances listed (1/line) in this file.")
    parser.add_option("--model", type="int",
            help="Plot given model from set.  Ignored if srmise_file is not a PeakStability file.")
    parser.add_option("--show", action="store_true",
            help="execute pylab.show() blocking call")
    parser.add_option("-o", "--output", type="string",
            help="save plot to the specified file")
    parser.add_option("--format", type="string", default="eps",
            help="output format for plot saving")
    parser.allow_interspersed_args = True
    opts, args = parser.parse_args(sys.argv[1:])


    if len(args) != 1:
        parser.error("Exactly one argument required. \n"+usage)

    filename = args[0]

    if filename is not None:
        toplot = PDFPeakExtraction()
        try:
            toplot.read(filename)
        except (Exception):
            toplot = PeakStability()
            try:
                toplot.load(filename)
            except Exception:
                print "File '%s' is not a .srmise or PeakStability data file." %filename
                return

    if opts.model is not None:
        try:
            toplot.setcurrent(opts.model)
        except (Exception):
            print "Ignoring model, %s is not a PeakStability file." %filename

    distances = None
    if opts.compare is not None:
        # use baseline from existing file
        distances = readcompare(opts.compare)

    setfigformat(figsize=(6., 4.0))
    figdict = makeplot(toplot, distances)
    if opts.output:
        plt.savefig(opts.output, format=opts.format, dpi=600)
    if opts.show:
        plt.show()
    else:
        plt.draw()
    return


if __name__ == '__main__':
    main()
