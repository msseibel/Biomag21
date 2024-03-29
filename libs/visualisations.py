import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import matplotlib as mpl
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import markers
import matplotlib.colors as mcolors
import pandas as pd
from scipy import signal,fftpack,linalg
from matplotlib import cm,colors
from matplotlib.widgets import RectangleSelector
from copy import deepcopy
import warnings
import itertools
from mne.decoding import CSP
import mne
from mne.channels import make_standard_montage
from libs.utils import MEGArray,_check_option,_setup_vmin_vmax,import_or_install
from libs import preprocessing
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

#######################################
#######################################
#######################################
#######################################
# Start: Graph Visualisation
#######################################
#######################################
#######################################
#######################################

# This import registers the 3D projection, but is otherwise unused.
import pylab
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from mpl_toolkits.mplot3d import proj3d
from scipy import spatial
from matplotlib.widgets import Button
from operator import itemgetter

class SharedSensorGraph():

    def __init__(self,from_graph,**kwargs):

        self.fig,self.axes = plt.subplots(1,2,figsize=(15,10),
                        subplot_kw={'projection':'3d'},num='joint')
        self.axes = self.axes.flatten()
        
        if from_graph:
            self.plot_fun = from_graph_plotInteractiveSensors
        else:
            self.plot_fun = plotInteractiveSensors
            

        
    def set_subplot_content(self,axIdx,**kwargs):
        self.plot_fun(ax=self.axes[axIdx],**kwargs)
        
    def finish(self):
        ax = self.axes[0]
        ax2 = self.axes[1]
        def on_move(event):
            if event.inaxes == self.axes[0]:
                if self.axes[0].button_pressed in self.axes[0]._rotate_btn:
                    self.axes[1].view_init(elev=self.axes[0].elev, azim=self.axes[0].azim)
                elif self.axes[0].button_pressed in self.axes[0]._zoom_btn:
                    self.axes[1].set_xlim3d(self.axes[0].get_xlim3d())
                    self.axes[1].set_ylim3d(self.axes[0].get_ylim3d())
                    self.axes[1].set_zlim3d(self.axes[0].get_zlim3d())
            elif event.inaxes == self.axes[1]:
                if self.axes[1].button_pressed in self.axes[1]._rotate_btn:
                    self.axes[0].view_init(elev=self.axes[1].elev, azim=self.axes[1].azim)
                elif self.axes[1].button_pressed in self.axes[1]._zoom_btn:
                    self.axes[0].set_xlim3d(self.axes[1].get_xlim3d())
                    self.axes[0].set_ylim3d(self.axes[1].get_ylim3d())
                    self.axes[0].set_zlim3d(self.axes[1].get_zlim3d())
            else:
                return
        c1 = self.fig.canvas.mpl_connect('motion_notify_event', on_move)

def get_convex_simplices(pts):
    cHullB = spatial.ConvexHull(pts,incremental=True)
    return cHullB.simplices

def UI_Interface(fig):
    fig.canvas.draw()
    axupdate = plt.axes([0.81, 0.05, 0.1, 0.075])
    bxupdate = Button(axupdate, 'Update Label Names')   
    axupdate._button = bxupdate
    return axupdate
    
def addButtonCallback(button_axis,pts,ax,annotations,sitename):
    # sitename can be removed later
    class LabelUpdater(object):
        def __init__(self,pts,ax,annotations,sitename):
            self.pts=pts
            self.ax=ax
            self.annotations=annotations
            self.sitename=sitename
        def update_annotation(self, event):
            updateAnnotations(self.pts,self.ax,self.annotations,self.sitename) 
            
    updater = LabelUpdater(pts,ax,annotations,sitename)
    connectID = button_axis._button.on_clicked(updater.update_annotation)    
    print('ID',connectID)
    
def updateAnnotations(pts,ax,annotations,site=None):  
    fig = ax.get_figure()
    for pidx in np.arange(len(pts)):   
        x2, y2, _ = proj3d.proj_transform(pts[pidx,0],pts[pidx,1],pts[pidx,2], ax.get_proj())
        annotations[pidx].xy = x2,y2
        annotations[pidx].update_positions(fig.canvas.renderer)
    # improve with https://stackoverflow.com/questions/8955869/why-is-plotting-with-matplotlib-so-slow
    fig.canvas.blit(ax.bbox)
    #fig.canvas.draw()    

def from_graph_plotInteractiveSensors(graph,title,ax=None):
    mpl.use('Qt5Agg')

    ax = _from_graph_plotInteractiveSensors(graph,title,ax)
    fig = ax.get_figure()
    pts = np.array(itemgetter(*np.arange(160))(graph.nodes('sensloc')))
    annotations = setLabel(pts,ax,title)
    make_button=True
    for axfig in fig.axes:
        if hasattr(axfig,'_button'):
            make_button=False
            button_axis = axfig
    if make_button:
        button_axis = UI_Interface(fig)
    addButtonCallback(button_axis,pts,ax,annotations,title)
    
def plotInteractiveSensors(pts,triangles,sitename,ax):
    mpl.use('Qt5Agg')
    ax = _plotInteractiveSensors(pts,sitename,triangles,ax)
    annotations = setLabel(pts,ax,sitename)
    UI_Interface(pts,ax,sitename)
    
    
def _plotInteractiveSensors(pts,site,simplices=[],labels=[],ax=None):
    #mpl.use('Qt5Agg')
    if len(simplices)==0:
        simplices = get_convex_simplices(pts)
    if not ax:
        fig = plt.figure(site,figsize=(15,15))
        ax = fig.add_subplot(111, projection="3d")

    # Plot defining corner points
    ax.scatter(pts.T[0], pts.T[1], pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    for s in simplices:
        s = np.append(s, s[0])  # Here we cycle back to the first coordinate
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
    ax.set_title('site: '+site)
    return ax

def _from_graph_plotInteractiveSensors(graph,site,ax=None):
    #mpl.use('Qt5Agg')
    pts = np.array(itemgetter(*np.arange(160))(graph.nodes('sensloc')))
    if not ax:
        fig = plt.figure(site,figsize=(15,15))
        ax = fig.add_subplot(111, projection="3d")
    
    # Plot defining corner points
    ax.scatter(pts.T[0], pts.T[1], pts.T[2], "ko")

    # 12 = 2 * 6 faces are the simplices (2 simplices per square face)
    
    for s in list(graph.edges):
        s = np.array(s)
        ax.plot(pts[s, 0], pts[s, 1], pts[s, 2], "r-")
        
    ax.set_title('site: '+site)
    return ax


def setLabel(pts,ax,site,labels=[]):
    #fig = plt.gcf()
    if len(labels)==0:
        labels = np.arange(len(pts))
    #fig = ax.get_figure()
    annotations = []
    for pidx,p in enumerate(labels): 
        x2, y2, _ = proj3d.proj_transform(pts[pidx,0],pts[pidx,1],pts[pidx,2], ax.get_proj())
        annotations += [ax.annotate(
            str(p), 
            xy = (x2, y2), xytext = (20, -20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            #bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'),
        )]
    return annotations

  
    
#######################################
#######################################
#######################################
#######################################
# END: Graph Visualisation
#######################################
#######################################
#######################################
#######################################




def sliding_stat_plot(signal,stat_func,title,*args,**kwargs):    
    """
    signal, array (channels,time_steps)
    *args are passed to stat_func
    """
    n_channels = signal.shape[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [20, 1]},figsize=(12,int(n_channels//3)))
    
    sliding_stats,taxis = stat_func(*args)(signal)
    visualize_eeg_stack(sliding_stats.T,taxis,title=title,ax=ax1)
    #ax1.set_ylim([50000, 100000])
    ax2.plot(np.squeeze(taxis),np.squeeze(np.mean(sliding_stats,axis=0)))
    fig.tight_layout()
    return fig
    
def _hide_frame(ax):
    """Hide axis frame for topomaps."""
    ax.get_yticks()
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.set_frame_on(False)

def _prepare_topomap(pos, ax, check_nonzero=True):
    """Prepare the topomap axis and check positions.
    Hides axis frame and check that position information is present.
    """
    _hide_frame(ax)
    if check_nonzero and not pos.any():
        raise RuntimeError('No position information found, cannot compute '
                           'geometries for topomap.')

def _handle_default(k, v=None):
    """Avoid dicts as default keyword arguments.
    Use this function instead to resolve default dict values. Example usage::
        scalings = _handle_default('scalings', scalings)
    """
    this_mapping = deepcopy(DEFAULTS[k])
    if v is not None:
        if isinstance(v, dict):
            this_mapping.update(v)
        else:
            for key in this_mapping.keys():
                this_mapping[key] = v
    return this_mapping

def _draw_outlines(ax, outlines):
    """Draw the outlines for a topomap."""
    outlines_ = {k: v for k, v in outlines.items()
                 if k not in ['patch']}
    for key, (x_coord, y_coord) in outlines_.items():
        if 'mask' in key or key in ('clip_radius', 'clip_origin'):
            continue
        ax.plot(x_coord, y_coord, color='k', linewidth=1, clip_on=False)
    return outlines_

def _validate_type(item, types=None, item_name=None, type_name=None):
    """Validate that `item` is an instance of `types`.
    Parameters
    ----------
    item : object
        The thing to be checked.
    types : type | str | tuple of types | tuple of str
         The types to be checked against.
         If str, must be one of {'int', 'str', 'numeric', 'info', 'path-like'}.
    """
    if types == "int":
        _ensure_int(item, name=item_name)
        return  # terminate prematurely
    elif types == "info":
        from mne.io import Info as types

    if not isinstance(types, (list, tuple)):
        types = [types]

    check_types = sum(((type(None),) if type_ is None else (type_,)
                       if not isinstance(type_, str) else _multi[type_]
                       for type_ in types), ())
    if not isinstance(item, check_types):
        if type_name is None:
            type_name = ['None' if cls_ is None else cls_.__name__
                         if not isinstance(cls_, str) else cls_
                         for cls_ in types]
            if len(type_name) == 1:
                type_name = type_name[0]
            elif len(type_name) == 2:
                type_name = ' or '.join(type_name)
            else:
                type_name[-1] = 'or ' + type_name[-1]
                type_name = ', '.join(type_name)
        raise TypeError('%s must be an instance of %s, got %s instead'
                        % (item_name, type_name, type(item),))

def _get_extra_points(pos, extrapolate, origin, radii):
    """Get coordinates of additinal interpolation points."""
    from scipy.spatial.qhull import Delaunay
    radii = np.array(radii, float)
    assert radii.shape == (2,)
    x, y = origin
    # auto should be gone by now
    _check_option('extrapolate', extrapolate, ('head', 'box', 'local'))

    # the old method of placement - large box
    mask_pos = None
    if extrapolate == 'box':
        extremes = np.array([pos.min(axis=0), pos.max(axis=0)])
        diffs = extremes[1] - extremes[0]
        extremes[0] -= diffs
        extremes[1] += diffs
        eidx = np.array(list(itertools.product(
            *([[0] * (pos.shape[1] - 1) + [1]] * pos.shape[1]))))
        pidx = np.tile(np.arange(pos.shape[1])[np.newaxis], (len(eidx), 1))
        outer_pts = extremes[eidx, pidx]
        return outer_pts, mask_pos, Delaunay(np.concatenate((pos, outer_pts)))
   # check if positions are colinear:
    diffs = np.diff(pos, axis=0)
    with np.errstate(divide='ignore'):
        slopes = diffs[:, 1] / diffs[:, 0]
    colinear = ((slopes == slopes[0]).all() or np.isinf(slopes).all())

    # compute median inter-electrode distance
    if colinear or pos.shape[0] < 4:
        dim = 1 if diffs[:, 1].sum() > diffs[:, 0].sum() else 0
        sorting = np.argsort(pos[:, dim])
        pos_sorted = pos[sorting, :]
        diffs = np.diff(pos_sorted, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        distance = np.median(distances)
    else:
        tri = Delaunay(pos, incremental=True)
        idx1, idx2, idx3 = tri.simplices.T
        distances = np.concatenate(
            [np.linalg.norm(pos[i1, :] - pos[i2, :], axis=1)
             for i1, i2 in zip([idx1, idx2], [idx2, idx3])])
        distance = np.median(distances)
    if extrapolate == 'local':
        if colinear or pos.shape[0] < 4:
            # special case for colinear points and when there is too
            # little points for Delaunay (needs at least 3)
            edge_points = sorting[[0, -1]]
            line_len = np.diff(pos[edge_points, :], axis=0)
            unit_vec = line_len / np.linalg.norm(line_len) * distance
            unit_vec_par = unit_vec[:, ::-1] * [[-1, 1]]

            edge_pos = (pos[edge_points, :] +
                        np.concatenate([-unit_vec, unit_vec], axis=0))
            new_pos = np.concatenate([pos + unit_vec_par,
                                      pos - unit_vec_par, edge_pos], axis=0)

            if pos.shape[0] == 3:
                # there may be some new_pos points that are too close
                # to the original points
                new_pos_diff = pos[..., np.newaxis] - new_pos.T[np.newaxis, :]
                new_pos_diff = np.linalg.norm(new_pos_diff, axis=1)
                good_extra = (new_pos_diff > 0.5 * distance).all(axis=0)
                new_pos = new_pos[good_extra]

            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, new_pos, tri

        # get the convex hull of data points from triangulation
        hull_pos = pos[tri.convex_hull]

        # extend the convex hull limits outwards a bit
        channels_center = pos.mean(axis=0)
        radial_dir = hull_pos - channels_center
        unit_radial_dir = radial_dir / np.linalg.norm(radial_dir, axis=-1,
                                                      keepdims=True)
        hull_extended = hull_pos + unit_radial_dir * distance
        mask_pos = hull_pos + unit_radial_dir * distance * 0.5
        hull_diff = np.diff(hull_pos, axis=1)[:, 0]
        hull_distances = np.linalg.norm(hull_diff, axis=-1)
        del channels_center

        # Construct a mask
        mask_pos = np.unique(mask_pos.reshape(-1, 2), axis=0)
        mask_center = np.mean(mask_pos, axis=0)
        mask_pos -= mask_center
        mask_pos = mask_pos[
            np.argsort(np.arctan2(mask_pos[:, 1], mask_pos[:, 0]))]
        mask_pos += mask_center

        # add points along hull edges so that the distance between points
        # is around that of average distance between channels
        add_points = list()
        eps = np.finfo('float').eps
        n_times_dist = np.round(0.25 * hull_distances / distance).astype('int')
        for n in range(2, n_times_dist.max() + 1):
            mask = n_times_dist == n
            mult = np.arange(1 / n, 1 - eps, 1 / n)[:, np.newaxis, np.newaxis]
            steps = hull_diff[mask][np.newaxis, ...] * mult
            add_points.append((hull_extended[mask, 0][np.newaxis, ...] +
                               steps).reshape((-1, 2)))

        # remove duplicates from hull_extended
        hull_extended = np.unique(hull_extended.reshape((-1, 2)), axis=0)
        new_pos = np.concatenate([hull_extended] + add_points)
    else:
        assert extrapolate == 'head'
        # return points on the head circle
        angle = np.arcsin(distance / 2 / np.mean(radii))
        points_l = np.arange(0, 2 * np.pi, angle)
        use_radii = radii * 1.1
        points_x = np.cos(points_l) * use_radii[0] + x
        points_y = np.sin(points_l) * use_radii[1] + y
        new_pos = np.stack([points_x, points_y], axis=1)
        if colinear or pos.shape[0] == 3:
            tri = Delaunay(np.concatenate([pos, new_pos], axis=0))
            return new_pos, mask_pos, tri
    tri.add_points(new_pos)
    return new_pos, mask_pos, tri


class _GridData(object):
    """Unstructured (x,y) data interpolator.
    This class allows optimized interpolation by computing parameters
    for a fixed set of true points, and allowing the values at those points
    to be set independently.
    """

    def __init__(self, pos, extrapolate, origin, radii, border):
        # in principle this works in N dimensions, not just 2
        assert pos.ndim == 2 and pos.shape[1] == 2, pos.shape
        _validate_type(border, ('numeric', str), 'border')

        # Adding points outside the extremes helps the interpolators
        outer_pts, mask_pts, tri = _get_extra_points(
            pos, extrapolate, origin, radii)
        self.n_extra = outer_pts.shape[0]
        self.mask_pts = mask_pts
        self.border = border
        self.tri = tri

    def set_values(self, v):
        """Set the values at interpolation points."""
        # Rbf with thin-plate is what we used to use, but it's slower and
        # looks about the same:
        #
        #     zi = Rbf(x, y, v, function='multiquadric', smooth=0)(xi, yi)
        #
        # Eventually we could also do set_values with this class if we want,
        # see scipy/interpolate/rbf.py, especially the self.nodes one-liner.
        from scipy.interpolate import CloughTocher2DInterpolator

        if isinstance(self.border, str):
            if self.border != 'mean':
                msg = 'border must be numeric or "mean", got {!r}'
                raise ValueError(msg.format(self.border))
            # border = 'mean'
            n_points = v.shape[0]
            v_extra = np.zeros(self.n_extra)
            indices, indptr = self.tri.vertex_neighbor_vertices
            rng = range(n_points, n_points + self.n_extra)
            used = np.zeros(len(rng), bool)
            for idx, extra_idx in enumerate(rng):
                ngb = indptr[indices[extra_idx]:indices[extra_idx + 1]]
                ngb = ngb[ngb < n_points]
                if len(ngb) > 0:
                    used[idx] = True
                    v_extra[idx] = v[ngb].mean()
            if not used.all() and used.any():
                # Eventually we might want to use the value of the nearest
                # point or something, but this case should hopefully be
                # rare so for now just use the average value of all extras
                v_extra[~used] = np.mean(v_extra[used])
        else:
            v_extra = np.full(self.n_extra, self.border, dtype=float)

        v = np.concatenate((v, v_extra))
        self.interpolator = CloughTocher2DInterpolator(self.tri, v)
        return self

    def set_locations(self, Xi, Yi):
        """Set locations for easier (delayed) calling."""
        self.Xi = Xi
        self.Yi = Yi
        return self

    def __call__(self, *args):
        """Evaluate the interpolator."""
        if len(args) == 0:
            args = [self.Xi, self.Yi]
        return self.interpolator(*args)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def _setup_interp(pos, res, extrapolate, sphere, outlines, border):
    xlim = np.inf, -np.inf,
    ylim = np.inf, -np.inf,
    mask_ = np.c_[outlines['mask_pos']]
    clip_radius = outlines['clip_radius']
    clip_origin = outlines.get('clip_origin', (0., 0.))
    xmin, xmax = (np.min(np.r_[xlim[0],
                               mask_[:, 0],
                               clip_origin[0] - clip_radius[0]]),
                  np.max(np.r_[xlim[1],
                               mask_[:, 0],
                               clip_origin[0] + clip_radius[0]]))
    ymin, ymax = (np.min(np.r_[ylim[0],
                               mask_[:, 1],
                               clip_origin[1] - clip_radius[1]]),
                  np.max(np.r_[ylim[1],
                               mask_[:, 1],
                               clip_origin[1] + clip_radius[1]]))
    xi = np.linspace(xmin, xmax, res)
    yi = np.linspace(ymin, ymax, res)
    Xi, Yi = np.meshgrid(xi, yi)
    interp = _GridData(pos, extrapolate, clip_origin, clip_radius, border)
    extent = (xmin, xmax, ymin, ymax)
    return extent, Xi, Yi, interp

def _make_head_outlines(sphere, pos, outlines, clip_origin):
    """Check or create outlines for topoplot."""
    assert isinstance(sphere, np.ndarray)
    x, y, _, radius = sphere
    del sphere

    if outlines in ('head', 'skirt', None):
        ll = np.linspace(0, 2 * np.pi, 101)
        head_x = np.cos(ll) * radius + x
        head_y = np.sin(ll) * radius + y
        dx = np.exp(np.arccos(np.deg2rad(12)) * 1j)
        dx, dy = dx.real, dx.imag
        nose_x = np.array([-dx, 0, dx]) * radius + x
        nose_y = np.array([dy, 1.15, dy]) * radius + y
        ear_x = np.array([.497, .510, .518, .5299, .5419, .54, .547,
                          .532, .510, .489]) * (radius * 2)
        ear_y = np.array([.0555, .0775, .0783, .0746, .0555, -.0055, -.0932,
                          -.1313, -.1384, -.1199]) * (radius * 2) + y

        if outlines is not None:
            # Define the outline of the head, ears and nose
            outlines_dict = dict(head=(head_x, head_y), nose=(nose_x, nose_y),
                                 ear_left=(ear_x + x, ear_y),
                                 ear_right=(-ear_x + x, ear_y))
        else:
            outlines_dict = dict()

        # Make the figure encompass slightly more than all points
        mask_scale = 1.25 if outlines == 'skirt' else 1.
        # We probably want to ensure it always contains our most
        # extremely positioned channels, so we do:
        mask_scale = max(
            mask_scale, np.linalg.norm(pos, axis=1).max() * 1.01 / radius)
        outlines_dict['mask_pos'] = (mask_scale * head_x, mask_scale * head_y)
        clip_radius = radius * mask_scale
        outlines_dict['clip_radius'] = (clip_radius,) * 2
        outlines_dict['clip_origin'] = clip_origin
        outlines = outlines_dict

    elif isinstance(outlines, dict):
        if 'mask_pos' not in outlines:
            raise ValueError('You must specify the coordinates of the image '
                             'mask.')
    else:
        raise ValueError('Invalid value for `outlines`.')

    return outlines

def _topomap_plot_sensors(pos_x, pos_y, sensors, ax):
    """Plot sensors."""
    if sensors is True:
        ax.scatter(pos_x, pos_y, s=0.25, marker='o',
                   edgecolor=['k'] * len(pos_x), facecolor='none')
    else:
        ax.plot(pos_x, pos_y, sensors)
#outer_pts, mask_pts, tri = _get_extra_points(
#    pos, extrapolate, origin, radii)

def guess_sphere(pos,sphere_units='m'):
    """
    1. Estimate Center of sensors
    2. Average Distance should be a good estimate of head radius.
    :param pos: sensor positions
    :return: head radius
    """
    center = np.mean(pos,axis=0)
    sphere = np.mean(np.linalg.norm(pos-center,axis=1))

    # Boilercode taken from mne
    sphere = np.array(sphere, dtype=float)
    if sphere.shape == ():
        sphere = np.concatenate([[0.] * 3, [sphere]])
    if sphere.shape != (4,):
        raise ValueError('sphere must be float or 1D array of shape (4,), got '
                         'array-like of shape %s' % (sphere.shape,))
    # 0.21 deprecation can just remove this conversion
    if sphere_units is None:
        sphere_units = 'mm'
    _check_option('sphere_units', sphere_units, ('m', 'mm'))
    if sphere_units == 'mm':
        sphere /= 1000.

    sphere = np.array(sphere, float)
    return sphere

def plot_topomap(data,
                 pos, # array(nchan,2)
                 vmin=None, vmax=None, cmap=None,
                  sensors=True,
                  res=64, axes=None, names=None, show_names=False, mask=None,
                  mask_params=None, outlines='head',
                  contours=6, image_interp='bilinear', show=True,
                  onselect=None, extrapolate=None,
                  sphere=None, #x, y, _, radius = sphere # nd.array.shape (4,)
                  border=None, ch_type='eeg'):
                  
    border = border if border else _BORDER_DEFAULT              
    extrapolate = extrapolate if extrapolate else _EXTRAPOLATE_DEFAULT 
    sphere = sphere if sphere else guess_sphere(pos)
    data = np.asarray(data)
    #print('Plotting topomap for data shape %s' % (data.shape,))
    """
    if isinstance(pos, Info):  # infer pos from Info object
        picks = _pick_data_channels(pos, exclude=())  # pick only data channels
        pos = pick_info(pos, picks)

        # check if there is only 1 channel type, and n_chans matches the data
        ch_type = _get_channel_types(pos, unique=True)
        info_help = ("Pick Info with e.g. mne.pick_info and "
                     "mne.io.pick.channel_indices_by_type.")
        if len(ch_type) > 1:
            raise ValueError("Multiple channel types in Info structure. " +
                             info_help)
        elif len(pos["chs"]) != data.shape[0]:
            raise ValueError("Number of channels in the Info object (%s) and "
                             "the data array (%s) do not match. "
                             % (len(pos['chs']), data.shape[0]) + info_help)
        else:
            ch_type = ch_type.pop()

        if any(type_ in ch_type for type_ in ('planar', 'grad')):
            # deal with grad pairs
            picks = _pair_grad_sensors(pos, topomap_coords=False)
            pos = _find_topomap_coords(pos, picks=picks[::2], sphere=sphere)
            data, _ = _merge_ch_data(data, ch_type, [])
            data = data.reshape(-1)
        else:
            picks = list(range(data.shape[0]))
            pos = _find_topomap_coords(pos, picks=picks, sphere=sphere)
    """


    _check_option('extrapolate', extrapolate, ('box', 'local', 'head', 'auto'))
    #if extrapolate == 'auto':
    #    extrapolate = 'local' if ch_type in _MEG_CH_TYPES_SPLIT else 'head'

    if data.ndim > 1:
        raise ValueError("Data needs to be array of shape (n_sensors,); got "
                         "shape %s." % str(data.shape))

    # Give a helpful error message for common mistakes regarding the position
    # matrix.
    pos_help = ("Electrode positions should be specified as a 2D array with "
                "shape (n_channels, 2). Each row in this matrix contains the "
                "(x, y) position of an electrode.")
    if pos.ndim != 2:
        error = ("{ndim}D array supplied as electrode positions, where a 2D "
                 "array was expected").format(ndim=pos.ndim)
        raise ValueError(error + " " + pos_help)
    elif pos.shape[1] == 3:
        error = ("The supplied electrode positions matrix contains 3 columns. "
                 "Are you trying to specify XYZ coordinates? Perhaps the "
                 "mne.channels.create_eeg_layout function is useful for you.")
        raise ValueError(error + " " + pos_help)
    # No error is raised in case of pos.shape[1] == 4. In this case, it is
    # assumed the position matrix contains both (x, y) and (width, height)
    # values, such as Layout.pos.
    elif pos.shape[1] == 1 or pos.shape[1] > 4:
        raise ValueError(pos_help)
    pos = pos[:, :2]

    if len(data) != len(pos):
        raise ValueError("Data and pos need to be of same length. Got data of "
                         "length %s, pos of length %s" % (len(data), len(pos)))

    norm = min(data) >= 0
    vmin, vmax = _setup_vmin_vmax(data, vmin, vmax, norm)
    if cmap is None:
        cmap = 'Reds' if norm else 'RdBu_r'

    outlines = _make_head_outlines(sphere, pos, outlines, (0., 0.))
    assert isinstance(outlines, dict)

    ax = axes if axes else plt.gca()
    _prepare_topomap(pos, ax)

    _use_default_outlines = any(k.startswith('head') for k in outlines)
    mask_params = _handle_default('mask_params', mask_params)

    # find mask limits
    clip_radius = outlines['clip_radius']
    clip_origin = outlines.get('clip_origin', (0., 0.))
    extent, Xi, Yi, interp = _setup_interp(
        pos, res, extrapolate, sphere, outlines, border)
    interp.set_values(data)
    Zi = interp.set_locations(Xi, Yi)()

    # plot outline
    patch_ = None
    if 'patch' in outlines:
        patch_ = outlines['patch']
        patch_ = patch_() if callable(patch_) else patch_
        patch_.set_clip_on(False)
        ax.add_patch(patch_)
        ax.set_transform(ax.transAxes)
        ax.set_clip_path(patch_)
    if _use_default_outlines:
        from matplotlib import patches
        if extrapolate == 'local':
            patch_ = patches.Polygon(
                interp.mask_pts, clip_on=True, transform=ax.transData)
        else:
            patch_ = patches.Ellipse(
                clip_origin, 2 * clip_radius[0], 2 * clip_radius[1],
                clip_on=True, transform=ax.transData)

    # plot interpolated map
    im = ax.imshow(Zi, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                   aspect='equal', extent=extent,
                   interpolation=image_interp)

    # gh-1432 had a workaround for no contours here, but we'll remove it
    # because mpl has probably fixed it
    linewidth = mask_params['markeredgewidth']
    cont = True
    if isinstance(contours, (np.ndarray, list)):
        pass
    elif contours == 0 or ((Zi == Zi[0, 0]) | np.isnan(Zi)).all():
        cont = None  # can't make contours for constant-valued functions
    if cont:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('ignore')
            cont = ax.contour(Xi, Yi, Zi, contours, colors='k',
                              linewidths=linewidth / 2.)

    if patch_ is not None:
        im.set_clip_path(patch_)
        if cont is not None:
            for col in cont.collections:
                col.set_clip_path(patch_)

    pos_x, pos_y = pos.T

    if sensors is not False and mask is None:
        _topomap_plot_sensors(pos_x, pos_y, sensors=sensors, ax=ax)
    elif sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)
        idx = np.where(~mask)[0]
        _topomap_plot_sensors(pos_x[idx], pos_y[idx], sensors=sensors, ax=ax)
    elif not sensors and mask is not None:
        idx = np.where(mask)[0]
        ax.plot(pos_x[idx], pos_y[idx], **mask_params)


    if isinstance(outlines, dict):
        _draw_outlines(ax, outlines)

    if show_names:
        if names is None:
            raise ValueError("To show names, a list of names must be provided"
                             " (see `names` keyword).")
        if show_names is True:
            def _show_names(x):
                return x
        else:
            _show_names = show_names
        show_idx = np.arange(len(names)) if mask is None else np.where(mask)[0]
        for ii, (p, ch_id) in enumerate(zip(pos, names)):
            if ii not in show_idx:
                continue
            ch_id = _show_names(ch_id)
            ax.text(p[0], p[1], ch_id, horizontalalignment='center',
                    verticalalignment='center', size='x-small')

    plt.subplots_adjust(top=.95)

    if onselect is not None:
        ax.RS = RectangleSelector(ax, onselect=onselect)
    #plt_show(show)
    return im, cont, interp


def visualize_butterfly(time_axis,eegdata,title='',
    ax=None,showPlot=False,xlabel='',ylabel=''):
    """
    eegdata (time_points,channels)
    time_axis (time_points)
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
        showPlot=True
    else:
        fig = ax.get_figure()
    ax.plot(time_axis,eegdata)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.axvline(0, linestyle='--', color='k', label='Onset')
    if showPlot:
        fig.tight_layout() 
        fig.show()

def visualize_grid(points3D,trilist):
    fig = plt.figure()
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    #ax = fig.gca(projection='3d')
    ax = Axes3D(fig) 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=60., azim=45.)
    ax.set_aspect('auto')
    ax.plot_trisurf(points3D[:, 0], points3D[:, 1], points3D[:, 2],
                    triangles=trilist, color='lightblue', edgecolor='black',
                    linewidth=1)
    plt.show()

def get_basis(trilist, points3D):
    mesh_eeg = tm.TriMesh(trilist, points3D)
    sphara_basis = sb.SpharaBasis(mesh_eeg, 'fem')
    basis_functions, natural_frequencies = sphara_basis.basis()
    return basis_functions, natural_frequencies, mesh_eeg

def visualize_basis(points3D,trilist,basis_functions=None):
    """
    basis_functions is ignored, exists for old code
    """
    # sphinx_gallery_thumbnail_number = 2
    basis_functions, natural_frequencies, mesh_eeg=get_basis(trilist, points3D)
    figsb1, axes1 = plt.subplots(nrows=5, ncols=3, figsize=(8, 12),
                                 subplot_kw={'projection': '3d'})
    for i in range(np.size(axes1)):
        colors = np.mean(basis_functions[trilist, i + 0], axis=1)
        ax = axes1.flat[i]
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=70., azim=15.)
        ax.set_aspect('auto')
        trisurfplot = ax.plot_trisurf(points3D[:, 0], points3D[:, 1],
                                      points3D[:, 2], triangles=trilist,
                                      cmap=plt.cm.bwr,
                                      edgecolor='white', linewidth=0.)
        trisurfplot.set_array(colors)
        #trisurfplot.set_clim(-1, 1)

    cbar = figsb1.colorbar(trisurfplot, ax=axes1.ravel().tolist(), shrink=0.75,
                           orientation='horizontal', fraction=0.05, pad=0.05,
                           anchor=(0.5, -4.0))

    plt.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=1.0)
    plt.show(block=False)


def visualize_spharagram(sphara_trans_eegdata,time_axis=None,eegdata=None,ysel=None,dt=None):
    """
    param: ysel number of basis functions for transformation
    param: sphara_trans_eegdata [timesteps, N_bases]
    """
    if dt and not time_axis:
        time_axis = np.arange(0,len(sphara_trans_eegdata)*dt,dt)
    if not ysel:
        ysel = sphara_trans_eegdata.shape[-1]
    y = np.arange(0, ysel)
    
    figsteeg, (axsteeg1, axsteeg2) = plt.subplots(nrows=2)
    axsteeg1.plot(time_axis, eegdata[:, :].transpose())
    axsteeg1.set_ylabel('V/µV')
    axsteeg1.set_title('EEG data, {} channels'.format(eegdata.shape[0]))
    #axsteeg1.set_ylim(-2.5, 2.5)
    #axsteeg1.set_xlim(-50, 130)
    axsteeg1.grid(True)

    pcm = axsteeg2.pcolormesh(time_axis, y,
                          np.square(np.abs(sphara_trans_eegdata.transpose()
                                           [0:ysel, :])))
    axsteeg2.set_xlabel('t/ms')
    axsteeg2.set_ylabel('# BF')
    axsteeg2.set_title('Power contribution of SPHARA basis functions')
    axsteeg2.grid(True)
    figsteeg.colorbar(pcm, ax=[axsteeg1,axsteeg2], shrink=0.45,
                      anchor=(0.85, 0.0), label='power / a.u.')

    plt.subplots_adjust(left=0.1, right=0.85, bottom=0.1, top=0.95, hspace=0.35)
    #plt.show(block=False)
    
def visualize_1d_filter(b,a,fs):
    w, h = freqz(b, a, worN=2000)
    plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)


def scatterHist(Data_Nclasses_N_xy,UseChannelMarkers,binwidth_x=[],binwidth_y=[],xLabel='',yLabel=''):
    #Data_Nclasses_N_xy: N_classes,features,N_samples
    # assuming features=2
    x = Data_Nclasses_N_xy[:, 0, :]
    y = Data_Nclasses_N_xy[:, 1, :]
    if binwidth_x:
        # now determine nice limits by hand:
        x_max = np.max(x)
        y_max = np.max(y)
        x_min = np.min(x)
        y_min = np.min(y)

        bins_x = np.arange(x_min, x_max + binwidth_x, binwidth_x)
        bins_y = np.arange(y_min, y_max + binwidth_y, binwidth_y)
    else:
        bins_x = np.histogram_bin_edges(x, bins='auto')
        bins_y = np.histogram_bin_edges(y, bins='auto')

    fig, axScatter = plt.subplots(figsize=(5.5, 5.5))

    # the scatter plot:
    colorList = list(mcolors.BASE_COLORS)
    for class_i in range(len(Data_Nclasses_N_xy)):
        if UseChannelMarkers:
            if Data_Nclasses_N_xy.shape[-1]==4:
                markersList = ['v', 'o', '*', 'x']
            else:
                markersList = list(markers.MarkerStyle.filled_markers)
            for ch in range(Data_Nclasses_N_xy.shape[-1]):
                axScatter.scatter(Data_Nclasses_N_xy[class_i,0,:,ch], Data_Nclasses_N_xy[class_i,1,:,ch],c=colorList[class_i],
                                  marker=markersList[ch])
        else:
            axScatter.scatter(Data_Nclasses_N_xy[class_i, 0].flatten(), Data_Nclasses_N_xy[class_i, 1].flatten(), c=colorList[class_i])
        #axScatter.set_aspect(1.)
    axScatter.set_xlabel(xLabel)
    axScatter.set_ylabel(yLabel)

    # create new axes on the right and on the top of the current axes
    # The first argument of the new_vertical(new_horizontal) method is
    # the height (width) of the axes to be created in inches.
    divider = make_axes_locatable(axScatter)
    axHistx = divider.append_axes("top", 1.2, pad=0.1, sharex=axScatter)
    axHisty = divider.append_axes("right", 1.2, pad=0.1, sharey=axScatter)

    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)




    axHistx.hist(x.flatten(), bins=bins_x)
    axHisty.hist(y.flatten(), bins=bins_y, orientation='horizontal')

    # the xaxis of axHistx and yaxis of axHisty are shared with axScatter,
    # thus there is no need to manually adjust the xlim and ylim of these
    # axis.

    #axHistx.set_yticks([0, 50, 100])

    #axHisty.set_xticks([0, 50, 100])

    plt.draw()
    plt.show()
    
    

#from sklearn.discriminant_analysis import QDA

###############################################################################
# colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)


###############################################################################
# generate datasets
def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


def dataset_cov():
    '''Generate 2 Gaussians samples with different covariance matrices'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y


###############################################################################
# plot functions
def plot_data(X, y, y_pred, fig_index,lda = None):
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    if lda is None:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA()
    splot = plt.subplot(2, 2, fig_index)
    if fig_index == 1:
        plt.title('Linear Discriminant Analysis')
        plt.ylabel('Data with fixed covariance')
    elif fig_index == 2:
        plt.title('Quadratic Discriminant Analysis')
    elif fig_index == 3:
        plt.ylabel('Data with varying covariances')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    ymin, ymax = X[:, 1].min(), X[:, 1].max()

    # class 0: dots
    plt.plot(X0_tp[:, 0], X0_tp[:, 1], 'o', color='red')
    plt.plot(X0_fp[:, 0], X0_fp[:, 1], '.', color='#990000')  # dark red

    # class 1: dots
    plt.plot(X1_tp[:, 0], X1_tp[:, 1], 'o', color='blue')
    plt.plot(X1_fp[:, 0], X1_fp[:, 1], '.', color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.))
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='k')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             'o', color='black', markersize=10)
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             'o', color='black', markersize=10)

    return splot


def plot_accuracy_over_time_train_vs_test(starttime, train_accuracies, test_accuracies,time_before_action):
    """
    Todo: Create subfunction to call it with just one data set (either test or train)
    This method here, should only arange both plots in subplots.
    :param starttime: (Ntimepoints)
    :param train_accuracies: (Ntimepoints,numPatients)
    :param test_accuracies: (Ntimepoints,numPatients)
    :return:
    """
    plt.figure()
    plt.subplot(121)
    plt.plot(starttime, np.mean(train_accuracies, axis=1))
    plt.axvline(time_before_action, linestyle='--', color='k', label='Onset')
    plt.axvline(time_before_action+3*512, linestyle='--', color='k', label='End')
    
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.title('Train Accuracies (start_time)')
    plt.subplot(122)
    plt.plot(starttime, np.mean(test_accuracies, axis=1))
    plt.axvline(time_before_action, linestyle='--', color='k', label='Onset')
    plt.axvline(time_before_action+3*512, linestyle='--', color='k', label='End')
    plt.axhline(0.5, linestyle='-', color='k', label='Chance')
    plt.title('Test Accuracies (start_time)')
    cf = plt.gcf()
    plt.legend()
    plt.show()
    return cf

def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled Gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                              180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.5)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())

def visualize_eeg_stack(eeg_data,time_axis,title,ax=None,showPlot=False,show_onset=False,gamma=1.1):
    """
    eeg_data: MEGArray or np.ndarray: (time_steps,channels)
    """
    n_channels = eeg_data.shape[-1]
    if not ax:
        fig = plt.figure(figsize=(12,int(n_channels//3)))
        ax = fig.gca()
        showPlot=True
    else:
        fig = ax.get_figure()
    maxch = 0
   # print('ta in eegstack',time_axis.shape)
    ticks=[]
    
    eeg_data = eeg_data - np.mean(eeg_data,axis=0,keepdims=True)
    chmin = np.min(eeg_data,axis=0)
    chmax = np.max(eeg_data,axis=0)
    spans = chmax-chmin
    gamma = 1.
    span = 0
    for ch in range(n_channels):
        #min_ch = np.min(eeg_data[:,ch])
        #max_ch = np.max(eeg_data[:,ch])
        #eeg_data[:,ch] = eeg_data[:,ch] / (chmax[ch]-chmin[ch])**gamma
        span = span + np.max([chmax[ch]-chmin[ch],np.mean(spans)]) + spans[ch-1]#np.abs(np.min(eeg_data[:,ch]))
        data_line = eeg_data[:,ch]+span
        ax.plot(np.squeeze(time_axis),np.squeeze(data_line))#,color='k'
        ticks+=[np.mean(data_line)]
        #plt.axvline(3*fs, linestyle='--', color='k', label='End')
        #span  = np.abs(np.max(eeg_data[:,ch])-np.min(eeg_data[:,ch]))
        #maxch = maxch+span
        #print(maxch,span)
        
    ax.set_title(title,fontsize=20)
    if type(eeg_data)==MEGArray:
        plt.yticks(ticks,eeg_data.ch_names)
    else:
        #print('plot yticks',ticks)
        ax.set_yticks(ticks)
        ax.set_yticklabels(np.arange(eeg_data.shape[1]))
        #plt.yticks(ticks,np.arange(eeg_data.shape[1]))
        #print(ax.get_yticks())
    if show_onset:
        plt.axvline(0, linestyle='--', color='k', label='Onset')      
    if showPlot:
        fig.tight_layout() 
        fig.legend()
        fig.show()
        
visualize_stack = visualize_eeg_stack


# MNE channel orientation format:
# ch_pos = rawcon.info['chs'][i]['loc'][:3]*1000
# local_ori_coord = rawcon.info['chs'][i]['loc'][3:].reshape(3,3) 
#     -> local coordinate system, going 1 step in each direction gives the vector that defines
#      the coil plane orientation.

def compare_matching(chanpos,chpos_reference,matching,greedy_match):
    plt.figure(figsize=(10,6))
    plt.subplot(121)
    plt.scatter(np.arange(160),np.linalg.norm(chpos_reference[greedy_match]-chanpos,axis=-1),color='r',label='Greedy')
    plt.scatter(np.arange(160),np.linalg.norm(chpos_reference[matching]-chanpos,axis=-1),color='b',marker='x',label='Hungarian')
    plt.title('Greedy vs Linear Sum Assignment')
    plt.legend(loc='upper left')
    plt.subplot(122)
    plt.scatter(np.arange(160),np.linalg.norm(chpos_reference[greedy_match]-chanpos,axis=-1),color='r',label='Greedy')
    plt.scatter(np.arange(160),np.linalg.norm(chpos_reference[matching]-chanpos,axis=-1),color='b',marker='x',label='Hungarian')
    plt.title('Greedy vs Linear Sum Assignment')
    plt.legend(loc='upper left')
    plt.ylim([-1,60])



from mpl_toolkits.mplot3d import Axes3D
def plot_with_reference_sytem(chanpos,chpos_reference,
        site1='MNE KIT reference',
        site2='site'):
    """
    chanpos: site A or site B
    chpos_reference: another MEG system e.g. mne test systems from KIT
    """
    fig = plt.figure(figsize=(8,8))
    ax = Axes3D(fig)
    ax.scatter(chpos_reference[:,0],chpos_reference[:,1],
        chpos_reference[:,2],color='k',label=site1)
    ax.scatter(chanpos [:,0],chanpos [:,1],chanpos[:,2],label=site2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init( 16,93)
    ax.legend()
    
    
def vector_dot(p2,p3):
    return (p2@p3)/np.linalg.norm(p2,axis=-1)/np.linalg.norm(p3,axis=-1)
    
def winkel(p2,p3):
    p2 = np.array(p2)
    p3 = np.array(p3)
    return np.degrees(np.arccos(vector_dot(p2,p3)))

def plot_ortho_to(ortho_to,ax=None):
    if not ax:
        fig = plt.figure(figsize=(8,8))
        ax = Axes3D(fig)
    else:
        fig = ax.get_figure()    
    pos_xorth= np.stack(np.array(ortho_to['x'])[:,1])
    pos_yorth= np.stack(np.array(ortho_to['y'])[:,1])
    pos_zorth= np.stack(np.array(ortho_to['z'])[:,1])

    ax.scatter(pos_xorth[:,0],pos_xorth[:,1],pos_xorth[:,2],marker='x',label='Orthogonal to x-axis')
    ax.scatter(pos_yorth[:,0],pos_yorth[:,1],pos_yorth[:,2],marker='x',color='y',label='Orthogonal to y-axis')
    ax.scatter(pos_zorth[:,0],pos_zorth[:,1],pos_zorth[:,2],marker='x',color='k',label='Orthogonal to z-axis')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    fig.suptitle('Local sensor coordinate system is orthogonal to: ')
    fig.legend()
    
def ortho_from_index(ortho_to,i):
    try:
        return pd.DataFrame(np.array(ortho_to['x'])[:,1],np.array(ortho_to['x'])[:,0],columns=['x']).loc[i]
    except:
        try:
            return pd.DataFrame(np.array(ortho_to['y'])[:,1],np.array(ortho_to['y'])[:,0],columns=['y']).loc[i]
        except:
            return pd.DataFrame(np.array(ortho_to['z'])[:,1],np.array(ortho_to['z'])[:,0],columns=['z']).loc[i]    
    
def plotIgel(rawData,ax=None,ortho_to=None,**kwargs):
    """
    kwargs are oassed ti quiver
    """
    
    if type(rawData)==mne.io.array.array.RawArray or type(rawData)==mne.io.kit.kit.RawKIT:
        orientations = np.array([rawData.info['chs'][i]['loc'][3:].reshape(3,3)*2 for i in range(160)])
        positions = np.array([rawData.info['chs'][i]['loc'][:3]*1000 for i in range(160)])
    elif type(rawData)==list:
        assert len(rawData)==2
        positions =rawData[0]
        orientations =rawData[1]
    else:
        raise ValueError
    if not ax:
        fig = plt.figure()
        ax = Axes3D(fig)
    else:
        fig = ax.get_figure()
    
    for i in range(160):
        local_ori_coord = orientations[i]
        ch_pos = positions[i]
        
        ori_ex = local_ori_coord[0]*10
        ori_ey = local_ori_coord[1]*10
        ori_ez = local_ori_coord[2]*10                   
        
        faden_sum = (local_ori_coord[0]+local_ori_coord[1]+local_ori_coord[2])
        
        faden =faden_sum/(np.linalg.norm(faden_sum))
        faden = faden+ch_pos
        faden = faden/(np.linalg.norm(faden))*10

        ax.plot(ch_pos[0],ch_pos[1],zs=ch_pos[2],marker='x')
        if not ortho_to:
            ax.quiver(ch_pos[0],ch_pos[1],ch_pos[2],ori_ex[0],ori_ex[1],ori_ex[2],**kwargs)
            ax.quiver(ch_pos[0],ch_pos[1],ch_pos[2],ori_ey[0],ori_ey[1],ori_ey[2],**kwargs)
            ax.quiver(ch_pos[0],ch_pos[1],ch_pos[2],ori_ez[0],ori_ez[1],ori_ez[2],**kwargs)
        else:
            ch_pos = ortho_from_index(ortho_to,i)
            orth  = ch_pos.index[0]
            ch_pos = ch_pos[0]*1000          
            color = {'x':'b','y':'y','z':'k'}
            qu1 = ax.quiver(ch_pos[0],ch_pos[1],ch_pos[2],ori_ex[0],ori_ex[1],ori_ex[2],color=color[orth],label='Orthogonal to {}-axis'.format(orth))
            qu2 = ax.quiver(ch_pos[0],ch_pos[1],ch_pos[2],ori_ey[0],ori_ey[1],ori_ey[2],color=color[orth],label='Orthogonal to {}-axis'.format(orth))
            qu3 = ax.quiver(ch_pos[0],ch_pos[1],ch_pos[2],ori_ez[0],ori_ez[1],ori_ez[2],color=color[orth],label='Orthogonal to {}-axis'.format(orth))
    
        #ax.quiver(ch_pos[0],ch_pos[1],ch_pos[2],faden[0],faden[1],faden[2],color='r')
        #print('winkel ex-orientation: ',winkel(ori_ex,faden_sum))
        #print('winkel ey-orientation: ',winkel(ori_ey,faden_sum))
        #print('winkel ez-orientation: ',winkel(ori_ez,faden_sum))
        #print('\n')
    fig.legend(handles=[qu1,qu2,qu3],labels=('orthogonal to x','orthogonal to y','orthogonal to z'))
    return ax

def plot_sensor_coord_system(coord_system,title=' '):
    e1_sens,e2_sens,e3_sens = coord_system
    ex_device = np.array([1,0,0])
    ey_device = np.array([0,1,0])
    ez_device = np.array([0,0,1])
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.quiver(0,0,0,*ex_device,color='k')
    ax.quiver(0,0,0,*ey_device,color='k')
    ax.quiver(0,0,0,*ez_device,color='k')

    ax.quiver(0,0,0,*e1_sens,color='r',label='e1')
    ax.quiver(0,0,0,*e2_sens,color='g',label='e2')
    ax.quiver(0,0,0,*e3_sens,color='b',label='e3')
    
    ax.quiver(0,0,0,*np.sum(coord_system,0),color='k',label='coil orientation')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)
    fig.legend()


def visualize_stft(eeg_data,fs,title,ax,showPlot=False):
    """
    eeg_data: (time_steps,channels)
    averages over all frequency spectrums
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
        showPlot=True
    else:
        fig = ax.get_figure()
    eeg_fft = fftpack.fft(eeg_data,axis=0)
    Tmess = eeg_data.shape[0]/fs
    print(Tmess,type(fs))
    frequency_axis = np.arange(0,fs/2,1/Tmess)
    print('frq ax:',1/Tmess,frequency_axis.shape,np.abs(eeg_data.mean(axis=-1)).shape,np.arange(-fs/2,fs/2,1/Tmess))
    #frequency_axis/=fs
    ax.plot(frequency_axis,2/eeg_data.shape[0]*np.abs(eeg_fft.mean(axis=-1)[:eeg_data.shape[0]//2]))
    ax.set_title(title)
    if showPlot:
        fig.tight_layout() 
        fig.show()


def visualize_fft(eeg_data,fs,title,ax,showPlot=False):
    """
    eeg_data: (time_steps,channels)
    
    averages over all frequency spectrums
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
        showPlot=True
    else:
        fig = ax.get_figure()
    eeg_fft = fftpack.fft(eeg_data,axis=0)
    Tmess = eeg_data.shape[0]/fs
    print(Tmess,type(fs))
    
    frequency_axis = np.arange(0,int(fs//2),1/Tmess)
    print('frq ax:',1/Tmess,frequency_axis.shape,np.abs(eeg_data.mean(axis=-1)).shape)
    #frequency_axis/=fs
    ax.plot(frequency_axis,2/eeg_data.shape[0]*np.abs(eeg_fft.mean(axis=-1)[:eeg_data.shape[0]//2]))
    ax.set_xlabel('Frequency in [Hz]')
    ax.set_title(title)
    if showPlot:
        fig.tight_layout() 
        fig.show()

def debug_filter(eegdata_raw,eegdata_filtered,fs,f_cut,filt_type,channel):
    #if not fs:
    #    w, h = signal.freqs(b, a)
    #    xlabel = 'Frequency [radians / second]'
    #else:
    #    w, h = signal.freqz(b, a)
    #    # f = 
    #    xlabel = r'Normalized Frequency [$\times \pi$ rad / samples]'
    #    print('-3dB frequency',fs/2*np.pi*np.median(w[(20 * np.log10(abs(h))> -3.2)*(20 * np.log10(abs(h))< -2.8) ]))#* (20 * np.log10(abs(h)) < -3.05)
    t = np.linspace(2-0.25, 5+0.25, int(3.5*fs), False)  # 2 second
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.plot(t, eegdata_raw[1024-128:5*fs+128])
    ax1.set_title('Before filter')
    #ax1.axis([0, 1, -2, 2])
    ax2.plot(t, eegdata_filtered[1024-128:5*fs+128])
    ax2.set_title('{} After {} Hz {}-pass filter'.format(channel,f_cut,filt_type))
    #ax2.axis([0, 1, -2, 2])
    ax2.set_xlabel('Time [seconds]')
    plt.tight_layout()
    plt.show(block=False)

def visualize_time_correlation(eeg_data,fs,title,ax=None,showPlot=False,centered=True):
    """
    Calculates the mutual correlation of each channel: <x_j,x_i>, with i,j being channel coefficients.
    
    eeg_data: (time_steps,channels)
    centered: if True (default) applies mean centering of each channel.
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
        showPlot=True
    else:
        fig = ax.get_figure()
    num_time_steps = eeg_data.shape[0]
    numChannels = eeg_data.shape[1]
    
    # time_steps x channels
    if centered:
        mean_ch = np.mean(eeg_data,axis=0)
        diff = eeg_data-mean_ch
    else:
        diff = eeg_data    
    correlation = np.matmul(diff.T,diff)
    norm = np.sum(diff**2,axis=0).reshape(eeg_data.shape[1],1)
    norm = np.matmul(norm,norm.T)
    correlation = correlation/np.sqrt(norm)
    #correlation = correlation/np.sqrt(np.sum((eeg_fft-mean_ch)**2,axis=0))
    print(correlation.shape)
    
    im = ax.matshow(correlation)
    plt.colorbar(im)
    ax.set_title(title)
    
    if showPlot:
        fig.tight_layout() 
        fig.show()


def visualize_fft_correlation(eeg_data,fs,title,ax=None,showPlot=False):
    """
    eeg_data: (time_steps,channels)
    
    using fft, because the fft enables a delay invariant representation -> larger correlation 
    """
    if not ax:
        fig = plt.figure()
        ax = fig.gca()
        showPlot=True
    else:
        fig = ax.get_figure()
    num_time_steps = eeg_data.shape[0]
    numChannels = eeg_data.shape[1]
    eeg_fft = np.abs(fftpack.fft(eeg_data,axis=0)[:int(num_time_steps/2)])

    
    mean_ch = np.mean(eeg_fft,axis=0)
    diff = eeg_fft-mean_ch
    correlation = np.matmul(diff.T,diff)
    norm = np.sum(diff**2,axis=0).reshape(eeg_data.shape[1],1)
    norm = np.matmul(norm,norm.T)
    correlation = correlation/np.sqrt(norm)
    
    
    #correlation = correlation/np.sqrt(np.sum((eeg_fft-mean_ch)**2,axis=0))
    print(correlation.shape)
    print(np.sum(correlation.imag))
    print(np.sum(np.real(correlation)))
    
    im = ax.matshow(correlation)
    plt.colorbar(im)
    
    if showPlot:
        fig.tight_layout() 
        fig.show()

def plot_csp_feature(eeg_data,labels,components,fs,montage='biosemi64'):
    """
    eeg_data: (trials,channels,time_steps)
    tmin: in [seconds]
    
    """
    if type(eeg_data)==np.ndarray:
        import default_data_params_gix034 as gix034
        info = mne.create_info(gix034.ch_names,fs,'eeg')
    elif type(eeg_data)==MEGArray:
         info = mne.create_info(eeg_data.ch_names,fs,'eeg')
    
    if type(montage)==str:
        #gix034 is standard_1010, but that does not exist in mne. 
        #But I assume, that 1010 is a subset of 1005.
        # Todo: What is the difference to: 'biosemi64'
        montage = make_standard_montage(montage)
        info.set_montage(montage,match_case=False)
    elif type(montage)==mne.channels.montage.DigMontage:
        info.set_montage(montage,match_case=False)
    else:
        raise ValueError("Montage must be of one of str or DigMontage. \n You can create a montage with mne.channels.make_dig_montage(ch_pos:dict(ch_names:position),coord_frame='head')")

    print(eeg_data.shape,labels.shape)
    #raw = mne.EpochsArray(eeg_data,info,labels,tmin,{'left':0,'right':1},reject=None)    
    csp = CSP(n_components=4, reg='ledoit_wolf', log=True, norm_trace=False,cov_est='epoch')
    #csp = CSP(n_components=components, reg='ledoit_wolf', log=True,
    #    norm_trace=False,cov_est='epoch')
    # plot CSP patterns estimated on full data for visualization
    csp.fit_transform(eeg_data, labels)
    plt.ion()
    csp.plot_patterns(info, ch_type='eeg', units='Patterns (AU)', size=1.5)
    fig=plt.gcf()
    plt.show(block=False)
    
def plot_lda_cov(lda, splot):
    plot_ellipse(splot, lda.means_[0], lda.covariance_, 'red')
    plot_ellipse(splot, lda.means_[1], lda.covariance_, 'blue')


def plot_confusion_matrix(y_true,y_pred,title='',class_labels=[],**kwargs):
    """
    class_labels: name of the classes
    """
    from sklearn.metrics import confusion_matrix
    num_classes = len(np.unique(y_true))
    conf_matrix = confusion_matrix(y_true,y_pred)
    if 'fontsize' in kwargs.keys():
        fontsize = kwargs['fontsize']
    else:
        fontsize=12    
    
    fig, ax = plt.subplots(figsize=(10,10))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    for i in np.arange(num_classes):
        for j in np.arange(num_classes):
            c = conf_matrix[j,i]
            ax.text(i, j, str(c), va='center', ha='center',fontsize=fontsize)
    plt.xlabel('Predicted Label',fontsize=12)
    plt.ylabel('True Label',fontsize=12)
    if class_labels:
        ax.set_xticklabels(['']+class_labels,fontsize=fontsize)
        ax.set_yticklabels(['']+class_labels,fontsize=fontsize)
    plt.title(title, y=1.08,fontsize=14)
    plt.show(block=False)
    
def plot_qda_cov(qda, splot):
    plot_ellipse(splot, qda.means_[0], qda.covariances_[0], 'red')
    plot_ellipse(splot, qda.means_[1], qda.covariances_[1], 'blue')

def plot_predicitions(predictions,title):
    assert len(predictions.shape)==1,"Predicitions must be categorical{0,..,num_classes-1} - not one-hot"
    classes = np.unique(predicitions)
    num_classes = len(classes)
    if len(num_classes)==2:
        plot_binary_predicitions(predictions,title)
    else:
        plot_multi_predictions(predicitions,title)
def plot_multi_predictions(predictions,title):
    #bars = [np.sum(predicitions==c) for c in classes]
    plt.figure()
    plt.hist(predictions)
    plt.title(title)
    plt.xlabel('classes')
    plt.ylabel('# predicted class')
    plt.show()

def ch_name_as_marker(ax,locs,labels):
    c = 0
    for x, y in zip(locs[0], locs[1]):
        ax.text(x, y, str(labels[c]), color="k", fontsize=12)
        c+=1

def plot_2d_feature(boolean_picks,control_feature=None,dementia_feature=None,mci_feature=None,use_log=False,
                    xlabel=r'$S_{xx}|_{f=2}^{6 Hz}  \,\, [ (\frac{fT}{m})^2 ]$',
                    ylabel=r'$S_{xx}|_{f=6}^{10 Hz} \,\, [ (\frac{fT}{m})^2 ]$',
                   ch_names=None,ax=None,plot_confidence=True):
    """
    control_feature: (subjects,num_features=2,num_channels)
    """
    picks = np.arange(control_feature.shape[2])[boolean_picks]
    if ax is None:
        fig = plt.figure(figsize=(12,12))    
        ax = fig.gca()
    else:
        fig = ax.get_figure()
    if control_feature is not None:
        avg_feature_control   = control_feature.mean(axis=0)
        ax.scatter(avg_feature_control[0,boolean_picks],avg_feature_control[1,boolean_picks],color='g',label='Healthy')
        if plot_confidence:
            for p in picks:
                confidence_ellipse(control_feature[:,0,p],control_feature[:,1,p],ax,n_std=.1,facecolor='g',alpha=0.5)
        
    #ch_name_as_marker(ax,[avg_bandscontrol[0],avg_bandscontrol[1]],np.arange(160))
    if dementia_feature is not None:
        avg_feature_dementia  = dementia_feature.mean(axis=0)
        ax.scatter(avg_feature_dementia[0,boolean_picks],avg_feature_dementia[1,boolean_picks],color='r',label='AD')
        if plot_confidence:
            for p in picks:
                confidence_ellipse(dementia_feature[:,0,p],dementia_feature[:,1,p],ax,n_std=.1,facecolor='r',alpha=0.5)

    if mci_feature is not None:
        avg_feature_mci       = mci_feature.mean(axis=0)
        ax.scatter(avg_feature_mci[0,boolean_picks],avg_feature_mci[1,boolean_picks],color='b',label='MCI')
        if plot_confidence:
            for p in picks:
                confidence_ellipse(mci_feature[:,0,p],mci_feature[:,1,p],ax,n_std=.1,facecolor='b',alpha=0.5)
    
    if ch_names is not None:
        ch_name_as_marker(ax,[avg_feature_dementia[0],avg_feature_dementia[1]],ch_names)
    
    ax.set_xlabel(xlabel,fontsize=18)
    ax.set_ylabel(ylabel,fontsize=18)
    if use_log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    ax.legend()
    return fig

def plot_binary_predicitions(predictions,title):
                plt.figure();plt.hist(predictions.flatten());plt.title(title);plt.show()
            
def plot_topomap_series(data,pos,time_samples,fs,t_onset,
    sphere=None,title='',ch_names=None,fig=None,row=1,vmin=None,vmax=None):
    """
    time_points: iterable, time points in seconds
    """
    #time_points = np.array(time_points)
    #min_time = np.min(time_points)
    #if min_time<0:
    #    time_points+=min_time
    #time_points_discrete = (np.array(time_points)*fs).astype('int')
    if type(fig)==plt.Figure:
        subplot_spec = fig._gridspecs[0]
    time_samples = np.array(time_samples)
    time_points=(time_samples-t_onset)/fs
    if not fig:
        fig = plt.figure(figsize=(15,10))
        subplot_spec = gridspec.GridSpec(ncols=row, nrows=len(time_samples), figure=fig2)
            
    for j,t in enumerate(time_samples):
        fig.add_subplot(subplot_spec[row-1,j])            
        if j==0:
            plt.ylabel(title,fontsize=16)
            #ax0 = plt.gca()
            #ax0.text(-0.1,0.5,title, va='center',rotation='vertical',fontsize=16)
        plot_topomap(data[t],pos,sphere=sphere,names=ch_names,vmin=vmin,vmax=vmax)
        plt.xlabel(str(np.round((time_points[j])*1000,decimals=2))+'ms')
    #fig.text(0.04, 0.5, task, va='center', rotation='vertical')
    #fig.suptitle(title)
    
def get_Frequenzgang_of_temporal_layer(model,fs,layer_name,channel_name):
    """
    model = models.Model(full_model.layers[0].input,full_model.get_layer(layer_name).output)
    """
    channel_idx = channelname2idx(channel_name)
    print(channel_idx)
    t = np.arange(0,1,1/fs)
    amplitude64 = []
    for k in np.arange(64):
        s = np.repeat(1*np.sin(np.pi*2*k*t).reshape(128,1),64,axis=1)
        filtered = m2.predict(np.expand_dims(s,axis=0))
        amplitude64 += [abs(np.fft.fft(np.squeeze(filtered[0,channel_idx]).T)[:,k])]
    amplitude64 = np.array(amplitude64)
    fig = plt.figure()
    for x in np.arange(8):
        for y in np.arange(8):
            ax=fig.add_subplot(8,8,x*8+y+1)
            ax.plot(np.arange(64),amplitude64.reshape(8,8,64)[y,x])
    plt.show()    

class EEG_Inspector():
    
    def __init__(self,fs,time_before_action):
        """
        time_before_action: in samples
        """
        self.fs = fs
        
        self.time_points = []
        self.time_before_action = time_before_action
        self.tmin = 0
        self.tmax = None
        
    def set_tmin(self,tmin):
        self.tmin = tmin
    def set_tmax(self,tmax):
        self.tmax = tmax
        
    def make_n_set_time_axis(self,signal_length):
        """
        signal_length in units [samples]
        """
        time_axis = (np.arange(signal_length)-self.time_before_action)/self.fs
        self.set_time_axis(time_axis)
        
    def set_time_axis(self,time_axis):
        self.time_axis = time_axis
        
    def set_sensorPositionsXY(self,sensorPositionXY):
        self.sensorPositionXY = sensorPositionXY
        
    def get_signal_envelope(self,signal):
        analytic_signal = hilbert(signal)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi) * self.fs)
        return amplitude_envelope
    
    def select_frequency_band(self,eeg_data,f_low,f_high):
        if f_low<=0:
            f_low=0.1
        eeg_data = preprocessing.filtering(eeg_data,f_cut=f_low,fs=self.fs,filt_type='high',use0phase=True) 
        # Schirrmeister,2017 und Lawhern,2018 verwenden oberen f_cut von 38 bzw 40 Hz                     
        eeg_data = preprocessing.filtering(eeg_data,f_cut=f_high,fs=self.fs,filt_type='low',use0phase=True)              
        return eeg_data
    
    def get_power_in_channel(self,eeg_data):
        """
        eeg_data: (numTrials,time_steps,numChannels)
        or:
        eeg_data: (time_steps, numChannels)
        """
        eeg_data = self.apply_time_crop(eeg_data)
        
        if len(eeg_data.shape)==3:
            return np.mean(eeg_data**2,axis=1)
        else:
            return np.mean(eeg_data**2,axis=0)
    
    def get_instantenous_power(self,eeg_data):
        return eeg_data**2
        
    def get_subject_average(self,eeg_data,axis=0):
        """
        eeg_data: np.ndarray or utils.MEGArray
        """
        eeg_avg  = np.mean(eeg_data,axis=axis)
        return eeg_avg
    
    def plot_bad_channels(self,feedbackLeft,feedbackRight):
        plt.figure()
        plt.subplot(121)
        plt.bar(np.arange(len(feedbackLeft['dropped_channels'])),np.array(feedbackLeft['dropped_channels']).sum(axis=-1))
        plt.title('Dropped Channels Left, bad subjects already removed')
        plt.subplot(122)
        plt.bar(np.arange(len(feedbackRight['dropped_channels'])),np.array(feedbackRight['dropped_channels']).sum(axis=-1))
        plt.title('Dropped Channels Right, bad subjects already removed')
        plt.show(block=False)
    
    def apply_time_crop(self,subject_data):
        if len(subject_data.shape)==2:
            return subject_data[self.tmin:self.tmax]
        elif len(subject_data.shape)==3:
            return subject_data[:,self.tmin:self.tmax]
    
    def get_pfurtscheller_frequency_bands(self,eeg_data,band_width,f_overlap,f_max=40):
        """
        Don't use averaged data. (Because ERD/ERS are not completly phase locked.)
        pfurtscheller analyses small frequency bands
 
        Parameters
        ..........
        eeg_data:(time_steps,channels)
        Returns
        -------
        bands: ndarray (num_freq_bands,time_step,channels)
        """
        num_bands = int(np.floor(f_max/(band_width-f_overlap))-1)
        f_low = 0
        bands = []
        for k in range(num_bands):
            bands+=[self.select_frequency_band(eeg_data,f_low,f_low+band_width)]
            f_low = f_low+f_overlap
        if type(eeg_data)==MEGArray:
            return MEGArray(bands,eeg_data.ch_names)
        else:
            return np.array(bands)
        
    def plot_Left_Right_FT(self,subject_left_avg,subject_right_avg,subject):
        subject_left_avg = self.apply_time_crop(subject_left_avg)
        subject_right_avg = self.apply_time_crop(subject_right_avg)
        
        fig = plt.figure(figsize=(20,12.5))
        ax1 = plt.subplot(121)
        visualize_fft(subject_left_avg,fs=self.fs,
            title='FFT subject: {} Left Hand Movement'.format(subject),ax=ax1)
        ax2 = plt.subplot(122)
        visualize_fft(subject_right_avg,fs=self.fs,
            title='FFT subject: {} Right Hand Movement'.format(subject),ax=ax2)
        fig.show()

    def plot_Left_Right_butterfly(self,subject_left_avg,subject_right_avg,subjectID):
        subject_left_avg = self.apply_time_crop(subject_left_avg)
        subject_right_avg = self.apply_time_crop(subject_right_avg)
        
        
        fig = plt.figure(figsize=(20,12.5))
        ax1 = plt.subplot(121)
        visualize_butterfly(self.fs,self.time_axis,subject_left_avg,
            title='subject {} Butterly Left Hand Movement'.format(subjectID),ax=ax1)
        ax2 = plt.subplot(122)
        visualize_butterfly(self.fs,self.time_axis,subject_right_avg,
            title='subject {} Butterly Right Hand Movement'.format(subjectID),ax=ax2)
        fig.show()

    def plot_Left_Right_eeg_stack(self,subject_left_avg,subject_right_avg,subject):
        subject_left_avg = self.apply_time_crop(subject_left_avg)
        subject_right_avg = self.apply_time_crop(subject_right_avg)
        
        
        fig = plt.figure(figsize=(20,12.5))
        ax1 = plt.subplot(121)
        visualize_eeg_stack(subject_left_avg,self.time_axis,
            title='subject: {} Left Hand '.format(subject),
            ax=ax1)
        ax2 = plt.subplot(122)
        visualize_eeg_stack(subject_right_avg,self.time_axis,
            title='subject: {} Right Hand '.format(subject),
            ax=ax2)
        fig.show()
        
    def plot_Left_Right_FFT_correlation(self,subject_left_avg,subject_right_avg,subject):
        subject_left_avg = self.apply_time_crop(subject_left_avg)
        subject_right_avg = self.apply_time_crop(subject_right_avg)
        
        fig = plt.figure(figsize=(20,12.5))
        ax1 = plt.subplot(121)
        visualize_fft_correlation(subject_left_avg,self.fs,
            'subject: {} Left Hand. Channel-Correlation'.format(subject),ax1)
        ax2 = plt.subplot(122)
        visualize_fft_correlation(subject_right_avg,self.fs,
            title='subject: {} Right Hand. Channel-Correlation'.format(subject),
            ax=ax2)
        fig.show()

    def plot_Left_Right_time_correlation(self,subject_left_avg,subject_right_avg,subject):
        subject_left_avg = self.apply_time_crop(subject_left_avg)
        subject_right_avg = self.apply_time_crop(subject_right_avg)
        
        fig = plt.figure(figsize=(20,12.5))
        ax1 = plt.subplot(121)
        visualize_time_correlation(subject_left_avg,self.fs,
            'subject: {} Left Hand. Channel-Correlation'.format(subject),ax1)
        ax2 = plt.subplot(122)
        visualize_time_correlation(subject_right_avg,self.fs,
            title='subject: {} Right Hand. Channel-Correlation'.format(subject),
            ax=ax2)
        fig.show()         
    
    def get_positions_from_montage(self,montage):
        """
        """
        if type(montage)==str:
            montage = make_standard_montage(montage)
        elif type(mne.channels.montage.DigMontage):
            pass
        else:
            raise ValueError("montage must be str or instance of DigMontage")
        return np.array(list(montage._get_ch_pos().values()))
            
    def plot_Left_Right_topo_plots(self,subject_left_avg,subject_right_avg,
            time_points,montage,title=''):
        """
        time_points : list,
        """
        #get_discrete_time_points
        discrete_tp = []
        for tp in time_points:
            discrete_tp+=[np.argmin(np.abs(self.time_axis-tp))]
        #print(discrete_tp)    
        sensorPositionsXY = self.get_positions_from_montage(montage)[:,:2]
        
        subject_left_avg = self.apply_time_crop(subject_left_avg)
        subject_right_avg = self.apply_time_crop(subject_right_avg)
        vmin = np.percentile([subject_left_avg,subject_right_avg],0.05)
        vmax = np.percentile(([subject_left_avg,subject_right_avg]),99.5)
        fig = plt.figure(figsize=(20,12.5))
        gs = fig.add_gridspec(2,len(time_points))
        plot_topomap_series(subject_left_avg,sensorPositionsXY,time_samples=discrete_tp,
            fs=self.fs,t_onset=self.time_before_action,title='class 1',fig=fig,row=1,vmin=vmin,vmax=vmax)
             
        plot_topomap_series(subject_right_avg,sensorPositionsXY,time_samples=discrete_tp,
            fs=self.fs,t_onset=self.time_before_action,title='class 2',fig=fig,row=2,vmin=vmin,vmax=vmax)
        fig.suptitle(title)
        
        # [rect]: [left, bottom, width, height]
        cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
        fig.colorbar(cm.ScalarMappable(cmap='RdBu_r'), cax=cbar_ax)
        
        plt.show(block=False)
        print('finished')
        
    def set_subject(self,subject,patientIDs):
        assert subject<len(patientIDs)
        self.subject = subject
        self.subjectID = patientIDs[subject]
        
    def set_time_points(self,time_points):
        self.time_points = time_points
        
    def inspect_eeg_data(self,eeg_data,numTrials,time_before_action,
            feedbackLeft,feedbackRight,sensorPos):
        subject_left  = self.get_subject_average(eeg_data[sub_vis,:numTrials])
        subject_right = self.get_subject_average(eeg_data[sub_vis,numTrials:])
        
    
"""
###############################################################################
for i, (X, y) in enumerate([dataset_fixed_cov(), dataset_cov()]):
    # LDA
    lda = LDA(solver="svd", store_covariance=True)
    y_pred = lda.fit(X, y).predict(X)
    splot = plot_data(lda, X, y, y_pred, fig_index=2 * i + 1)
    plot_lda_cov(lda, splot)
    plt.axis('tight')

    # QDA
    qda = QDA()
    y_pred = qda.fit(X, y, store_covariances=True).predict(X)
    splot = plot_data(qda, X, y, y_pred, fig_index=2 * i + 2)
    plot_qda_cov(qda, splot)
    plt.axis('tight')
plt.suptitle('LDA vs QDA')
plt.show()
"""