# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 13:21:19 2016

@author: craigm

various useful functions

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt


#%%
def fix_eps(fpath):
    """Fix carriage returns in EPS files caused by Arial font."""
    txt = b""
    with open(fpath, "rb") as f:
        for line in f:
            if b"\r\rHebrew" in line:
                line = line.replace(b"\r\rHebrew", b"Hebrew")
            txt += line
    with open(fpath, "wb") as f:
        f.write(txt)
#%%             

from matplotlib.patches import Circle, Ellipse, Rectangle
from matplotlib.collections import PatchCollection

__all__ = ['circles', 'ellipses', 'rectangles']


def circles(x, y, s, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of circles. 
    Similar to plt.scatter, but the size of circles are in data scale.

    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, ) 
        Radius of circles.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence 
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)  
        `c` can be a 2-D array in which the rows are RGB or RGBA, however. 
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls), 
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, s=a*0.2, c=a, alpha=0.5, ec='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    zipped = np.broadcast(x, y, s)
    patches = [Circle((x_, y_), s_)
               for x_, y_, s_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def ellipses(x, y, w, h=None, rot=0.0, c='b', vmin=None, vmax=None, **kwargs):
    """
    Make a scatter plot of ellipses. 
    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of ellipses.
    w, h : scalar or array_like, shape (n, )
        Total length (diameter) of horizontal/vertical axis.
        `h` is set to be equal to `w` by default, ie. circle.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    ellipses(a, a, w=4, h=a, rot=a*30, c=a, alpha=0.5, ec='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.

    if h is None:
        h = w

    zipped = np.broadcast(x, y, w, h, rot)
    patches = [Ellipse((x_, y_), w_, h_, rot_)
               for x_, y_, w_, h_, rot_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


def rectangles(x, y, w, h=None, rot=0.0, c='b', vmin=None, vmax=None, ax=None, facecolor='none',**kwargs):
    """
    Make a scatter plot of rectangles.

    Parameters
    ----------
    x, y : scalar or array_like, shape (n, )
        Center of rectangles.
    w, h : scalar or array_like, shape (n, )
        Width, Height.
        `h` is set to be equal to `w` by default, ie. squares.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (anti-clockwise).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    rectangles(a, a, w=5, h=6, rot=a*30, c=a, alpha=0.5, ec='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None

    if 'fc' in kwargs:
        kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs:
        kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs:
        kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs:
        kwargs.setdefault('linewidth', kwargs.pop('lw'))
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.
    if ax is None:
        ax = plt.gca() 
        
    if h is None:
        h = w
    d = np.sqrt(np.square(w) + np.square(h)) / 2.
    t = np.deg2rad(rot) + np.arctan2(h, w)
    x, y = x - d * np.cos(t), y - d * np.sin(t)

    zipped = np.broadcast(x, y, w, h, rot)
    patches = [Rectangle((x_, y_), w_, h_, rot_)
               for x_, y_, w_, h_, rot_ in zipped]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        c = np.broadcast_to(c, zipped.shape).ravel()
        collection.set_array(c)
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    return collection


#%%
#Class to normalise colors to center around a value
from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


#%%
#GRID X,Y,Z points for using in plt.contourf
def gridder(x, y, z, xint, yint, epsilon=None):
    """
    Where x, y are east and north in metres, z is the value to be gridded.
    
    xint and yint are the x and y grid spacing (in metres).
    
    Returns x_grid, y_grid, zi.
    
    Uses scipy.interpolate Rbf which fills the whole mesh.
        
    Use as x_grid, y_grid, zi = gridder(x,y,z, xint, yint).

    Then plt.contourf(x_grid,y_grid, zi) etc    
    """
    import numpy as np
    from scipy.interpolate import Rbf
    #Eastings grid points
    xmin = np.int(x.min())
    #xmin = xmin.round(-2)
    xmax = np.int(x.max())
    #xmax = xmax.round(-2).astype(int)
    xi = range(xmin-xint, xmax+xint, xint)
    
    #Northings grid points
    ymin = np.int(y.min())
    #ymin = ymin.round(-2).astype(int)
    ymax = np.int(y.max())
   # ymax = ymax.round(-2).astype(int)
    yi = range (ymin-yint, ymax+yint, yint)
    
    #make meshgrid
    x_grid,y_grid = np.meshgrid(xi,yi)
    
    #use RBF to grid
    if epsilon is None:
        rbf =Rbf(x.values, y.values, z.values, epsilon=1)
    else:
        rbf =Rbf(x.values, y.values, z.values, epsilon=epsilon)
    zi = rbf(x_grid, y_grid)
    return x_grid, y_grid, zi

#%%
def gridder_griddata(x, y, z, xint, yint, method=None):
    """
    Where x, y are east and north in metres, z is the value to be gridded.
    
    xint and yint are the x and y grid spacing (in metres).
    
    Returns x_grid, y_grid, zi.
    
    Uses scipy.interpolate griddata.
    
    griddata wont grid outside the extent of the points
    (unless you choose nearest, but that is ugly), use utils.gridder to fill the
    whole mesh if that is what you want as that uses rbf.
        
    Use as x_grid, y_grid, zi = gridder(x,y,z, xint, yint).

    Then plt.contourf(x_grid,y_grid, zi) etc    
    """
    import numpy as np
    from scipy.interpolate import griddata
    #Eastings grid points
    xmin = np.int(x.min())
    #xmin = xmin.round(-2)
    xmax = np.int(x.max())
    #xmax = xmax.round(-2).astype(int)
    xi = range(xmin-xint, xmax+xint, xint)
    
    #Northings grid points
    ymin = np.int(y.min())
    #ymin = ymin.round(-2).astype(int)
    ymax = np.int(y.max())
   # ymax = ymax.round(-2).astype(int)
    yi = range (ymin-yint, ymax+yint, yint)
    
    #make meshgrid
    x_grid,y_grid = np.meshgrid(xi,yi)
    
    #use gridddata to grid
    if method is None:
        zi = griddata((x.values, y.values), z.values, (x_grid, y_grid),
                      method='cubic')
    else:
        zi = griddata((x.values, y.values), z.values, (x_grid, y_grid),
                      method=method)
            
    #zi = grid(x_grid, y_grid)
    return x_grid, y_grid, zi
#%%
def mask_outside_polygon(poly_verts, ax=None):
    """
    Plots a mask on the specified axis ("ax", defaults to plt.gca()) such that
    all areas outside of the polygon specified by "poly_verts" are masked.  

    "poly_verts" must be a list of tuples of the verticies in the polygon in
    counter-clockwise order.

    Returns the matplotlib.patches.PathPatch instance plotted on the figure.
    
    poly_verts = zip(grid_mask.x,grid_mask.y)
    
    where grid_mask.x and grid_mask.y are x and y coords of the polygon vertices    
    
    then use as:
    
    utils.mask_outside_polygon(poly_verts)    
    
    """
    import matplotlib.patches as mpatches
    import matplotlib.path as mpath

    if ax is None:
        ax = plt.gca()

    # Get current plot limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Verticies of the plot boundaries in clockwise order
    bound_verts = [(xlim[0], ylim[0]), (xlim[0], ylim[1]), 
                   (xlim[1], ylim[1]), (xlim[1], ylim[0]), 
                   (xlim[0], ylim[0])]

    # A series of codes (1 and 2) to tell matplotlib whether to draw a line or 
    # move the "pen" (So that there's no connecting line)
    bound_codes = [mpath.Path.MOVETO] + (len(bound_verts) - 1) * [mpath.Path.LINETO]
    poly_codes = [mpath.Path.MOVETO] + (len(poly_verts) - 1) * [mpath.Path.LINETO]

    # Plot the masking patch
    path = mpath.Path(bound_verts + poly_verts, bound_codes + poly_codes)
    patch = mpatches.PathPatch(path, facecolor='white', edgecolor='none',zorder=4)
    patch = ax.add_patch(patch)

    # Reset the plot limits to their original extents
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(direction='inout')
    return patch
#%% 
import shapefile
def plot_shapefile(filename, color = None, linewidth = None, linestyle = None,
                   ax=None, label=None, zorder=None):
    """
    give file name
    
    optional to specify linestyle, color and linewidth and which axes to plot on (for subplots)
    """
    if linestyle is None:
        linestyle = 'solid'
    
    if color is None:
        color = 'k'

    if linewidth is None:
        linewidth = 1
       
    if linestyle is None:
        linestyle = 'solid'    
        
    if ax is None:
        ax = plt.gca()    
    
    if label is None:
        label = ''
        
    if zorder is None:
        zorder = 1
        
    r = shapefile.Reader(filename)
    shapes = r.shapes()
    records = r.records()
    for record, shape in zip(records,shapes):
        easts,norths = zip(*shape.points)
        out = ax.plot(easts, norths, color = color, linewidth = linewidth,
                      linestyle = linestyle, label = label, zorder=1)       
    return out

############################################################################################
#PLOT SQUARES
def squares(x, y, w, h=None, rot=0.0, c='b', vmin=None, vmax=None, ax=None, **kwargs):
    """
    Plot a set of rectangles.
    
    x, y : scalar or array_like, shape (n, )
        Center of rectangles.
    w, h : scalar or array_like, shape (n, )
        Width, Height.
        `h` is set to be equal to `w` by default, ie. squares.
    rot : scalar or array_like, shape (n, )
        Rotation in degrees (it plots anti-clockwise from X axis).
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.
    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`
    Examples
    --------
    a = np.arange(11)
    squares(a, a, w=5, h=6, rot=a*30, c=a, alpha=0.5, ec='none')
    plt.colorbar()
    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    if np.isscalar(c):
        kwargs.setdefault('color', c)
        c = None
            
    if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
    if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
    if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
    if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))
    
    # You can set `facecolor` with an array for each patch,
    # while you can only set `facecolors` with a value for all.
 
    if h is None: 
        h = w
    d = np.sqrt(np.square(w) + np.square(h))/2.
    t = np.deg2rad(rot) + np.arctan2(h, w)
    x, y = x - d*np.cos(t), y - d*np.sin(t)
    patches = [Rectangle((x_, y_), w_, h_, rot_) for x_, y_, w_, h_, rot_ in 
               np.broadcast(x, y, w, h, rot)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)
    if ax is None:
        ax = plt.gca()

    ax.add_collection(collection)
    ax.autoscale_view()
    plt.draw_if_interactive()
    if c is not None:
        plt.sci(collection)
    
    return collection
    

#%%
def m2km(axis=None):
    import matplotlib.pyplot as plt
    """
    Stolen from fatiando
    
    Convert the x and y tick labels from meters to kilometers.

    Parameters:

    * axis : matplotlib axis instance
        The plot.

    .. tip:: Use ``fatiando.vis.gca()`` to get the current axis. Or the value
        returned by ``fatiando.vis.subplot`` or ``matplotlib.pyplot.subplot``.

    """
    if axis is None:
        axis = plt.gca()
    axis.set_xticklabels(['%g' % (0.001 * l) for l in axis.get_xticks()])
    axis.set_yticklabels(['%g' % (0.001 * l) for l in axis.get_yticks()])
	
	
# %%
def smooth(x, window_len=151, window='hanning'):
    """
    smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    
    output:
        the smoothed signal
        
    example:
    
    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
     
    TODO: the window parameter could be the window itself if an array instead of a string   
    """
    #print x
    #print 'last x',x.count()-1
    
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    
    if window_len < 3:
        return x
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    
    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    print(len(s))
    
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]