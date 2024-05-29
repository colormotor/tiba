'''
o-O-o o-O-o o--o    O
  |     |   |   |  / \
  |     |   O--o  o---o
  |     |   |   | |   |
  o   o-O-o o--o  o   o

Painting robot utils.
© Daniel Berio (@colormotor) 2023 - ...

matplotlyb - Matplotlib and Plotly wrapper, for easy plotting and visualization
and not having to deal too much with the Plotly API. The interface is similar to Matplotlib
'''

import matplotlib.pyplot as plt
import plotly
import plotly.express as px
from collections import defaultdict
import matplotlib.colors as mcolors
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import numbers
import matplotlib

from . import trajectory_utils as tu

import plotly.io as pio
pio.renderers.default = 'browser'

glob = lambda: None
glob.fig = None

def is_number(x):
    return isinstance(x, numbers.Number)

def plotly_color(clr):
    if clr is None: # N
        return None
    if type(clr) == str:
        clr = mcolors.to_rgb(clr)
    elif not is_number(clr[0]):
        return [plotly_color(c) for c in clr]
    return 'rgb(%.3f, %.3f, %.3f)'%(clr[0]*255, clr[1]*255, clr[2]*255)

class Axes:
    def __init__(self, colspan=1, rowspan=1, title=''):
        self.plots = []
        self.equal = False
        self.flip_y = False
        self.box = None
        self.rowspan = rowspan
        self.colspan = colspan
        self.title = title
        self.equal = False

class Axes3d(Axes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class Figure:
    def __init__(self, figsize=(5, 5), title=''):
        self.figsize = figsize
        self.plots = defaultdict(dict)
        self.cur_ax = None
        self.figtitle = title

    @property
    def num_rows(self):
        maxr = 0
        for k, row in self.plots.items():
            rowspans = [0] + [ax.rowspan for ax in row.values()]
            maxr = max(maxr, max(*rowspans) + k)
        return maxr
        #return max(int(k) for k in self.plots.keys())
        #return len(self.plots)

    @property
    def num_cols(self):
        maxc = 0
        for _, row in self.plots.items():
            maxc = max(maxc, max(k+ax.colspan for k, ax in row.items()))
        return maxc

        #return max(len(v) for v in self.plots.values())

    def subplot(self, row, col, rowspan=1, colspan=1, title=''):
        if (row-1) not in self.plots or (col-1) not in self.plots[row-1]:
            self.cur_ax = Axes(colspan=colspan, rowspan=rowspan, title=title)
            self.plots[row-1][col-1] = self.cur_ax
        else:
            self.cur_ax = self.plots[row-1][col-1]

    def subplot3d(self, row, col, rowspan=1, colspan=1, title=''):
        self.cur_ax = Axes3d(colspan=colspan, rowspan=rowspan, title=title)
        self.plots[row-1][col-1] = self.cur_ax

    def axis_equal(self, flag=True):
        self.cur_ax.equal = flag

    def flip_y(self, flag=True):
        self.cur_ax.flip_y = flag

    def setup_2d(self):
        self.axis_equal()
        self.flip_y()

    def title(self, title):
        self.cur_ax.title = title

    def add_plot(self, plot):
        if self.cur_ax is None:
            self.subplot(1, 1)

        self.cur_ax.plots.append(plot)

    def show(self, impl='plotly'):
        if impl == 'plotly':
            show_plotly(self)
        elif impl == 'plt':
            show_plt(self)
        else:
            print("Unknown implementation")

def from_kwargs(kwargs, labels, default):
    for label in labels:
        if label in kwargs:
            return kwargs[label]
    return default

class Plot:
    def __init__(self, **kwargs):
        # Optional args
        self.alpha = from_kwargs(kwargs, ['alpha', 'a'], 1.0)
        self.lw =    from_kwargs(kwargs, ['linewidth', 'lw'], 2.0)
        self.ls =    from_kwargs(kwargs, ['linestyle', 'ls'], '-')
        self.label = from_kwargs(kwargs, ['label'], None)
        self.clr = from_kwargs(kwargs, ['clr', 'color', 'c'], None)
        self.fill = from_kwargs(kwargs, ['fill'], None)

class Image:
    def __init__(self, *args, **kwargs):
        self.cmap = from_kwargs(kwargs, ['cmap'], 'gray')
        if len(args)==1:
            self.im = args[0]
        elif len(args)==2:
            self.im = args[0]
            self.cmap = args[1]


class Plot2d(Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # Standard matplotlib like argument parsing
        if len(args) == 1: # Only y values
            self.y = args[0]
            self.x = np.linspace(0, len(self.y)-1, len(self.y))
        elif len(args) == 2: # x and y values or y and string color values
            if type(args[1]) == str:
                self.y = args[0]
                self.x = np.linspace(0, len(self.y)-1, len(self.y))
                self.clr = args[1] # mcolors.to_rgb(args[1])
            else:
                self.x = args[0]
                self.y = args[1]
        elif len(args) == 3: # x, y, and color values
            self.x = args[0]
            self.y = args[1]
            self.clr = args[2] # mcolors.to_rgb(args[2])


class Plot3d(Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        # Standard matplotlib like argument parsing
        if len(args) == 1: # xyz array
            X = np.array(args[0])
            self.X = X
        elif len(args) == 2: # x and y values or y and color values
            self.X = np.array(args[0])
            self.clr = args[1] #plotly_color(args[1])
        elif len(args) == 3: # x, y, z values
            self.X = np.vstack(args).T
        elif len(args) == 4:
            self.X = np.vstack(args[:3]).T
            self.clr = args[3] # plotly_color(args[3]) #mcolors.to_rgb(args[3])


class Bar(Plot):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        if len(args) == 1:
            self.vals = args[0]
        if len(args) == 2:
            self.vals = args[0]
            self.clr = args[1]
        self.limits = from_kwargs(kwargs, ['limits'], None)


## Immediate mode

def figure(figsize=(5, 5), title=''):
    glob.fig = Figure(figsize=figsize, title=title)
    return glob.fig

def subplot(*args, **kwargs):
    glob.fig.subplot(*args, **kwargs)

def plot(*args, **kwargs):
    glob.fig.add_plot(Plot2d(*args, **kwargs))

def line(a, b, *args, **kwargs):
    P = np.array([a, b])
    glob.fig.add_plot(Plot2d(P[:,0], P[:,1], *args, **kwargs))

def bar(*args, **kwargs):
    glob.fig.add_plot(Bar(*args, **kwargs))

def plot3d(*args, **kwargs):
    glob.fig.add_plot(Plot3d(*args, **kwargs))

def pose(pose, size=0.1, colors=['r', 'g', 'b'], **kwargs):
    pose = np.array(tu.to_pose(pose))
    p = pose[:3, 3]
    x = pose[:3, 0]
    y = pose[:3, 1]
    z = pose[:3, 2]

    stroke3d([p, p + x*size], colors[0], **kwargs)
    stroke3d([p, p + y*size], colors[1], **kwargs)
    stroke3d([p, p + z*size], colors[2], **kwargs)

def stroke(X, color=None, **kwargs):
    if type(X) == list and X and not is_number(X[0]):
        for Xi in X:
            stroke(Xi, color, **kwargs)
        return
    if not len(X):
        return
    X = np.array(X)
    #if color is not None:
    #    color = mcolors.to_rgb(color)
    glob.fig.add_plot(Plot2d(X[:,0], X[:,1], color=color, **kwargs))

def axis_3d(A, size, **kwargs):
    glob.fig.add_plot(Plot3d([A[:3,3], A[:3,3] + A[:3,0]*size], 'r', **kwargs))
    glob.fig.add_plot(Plot3d([A[:3,3], A[:3,3] + A[:3,1]*size], 'g', **kwargs))
    glob.fig.add_plot(Plot3d([A[:3,3], A[:3,3] + A[:3,2]*size], 'b', **kwargs))

def stroke3d(X, color=None, **kwargs):
    if type(X) == list and X and not is_number(X[0][0]):
        for Xi in X:
            stroke3d(Xi, color, **kwargs)
        return
    if not len(X):
        return
    X = np.array(X)
    #if color is not None:
    #    color = mcolors.to_rgb(color)
    glob.fig.add_plot(Plot3d(X, color=color, **kwargs))

def rectangle(*args, color=None, fill=False, **kwargs):
    if len(args)==1:
        [l, t], [r, b] = args[0]
    if len(args)==2:
        [l, t], [r, b] = args
    elif len(args)==4:
        l, t = args[:2]
        w, h = args[2:]
        r, b = l+w, t+h
    else:
        print("Invalid arguments", len(args), args)
    X = np.array([[l, t], [r, t], [r, b], [l, b], [l, t]])
    if fill:
        fill = 'toself'
    else:
        fill = None
    stroke(X, color, fill=fill, **kwargs)

def imshow(*args, **kwargs):
    glob.fig.add_plot(Image(*args, **kwargs))

def axis_equal(flag=True):
    glob.fig.axis_equal(flag)

def flip_y(flag=True):
    glob.fig.flip_y(flag)

def setup_2d():
    glob.fig.setup_2d()

def title(title):
    glob.fig.title(title)

def show(impl='plotly'):
    glob.fig.show(impl=impl)


## Matploltlib-like API

def show_plt(fig):
    '''
    Matplotlib implementation for figure and plots
    '''
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    mode_map = {':': 'lines', '.':'markers', '-': 'lines', '--': 'lines', 'o.': 'lines+markers', 'o-': 'lines+markers'}
    mode_map_dash = {':': 'dot', '.': 'dot', '-': 'solid', '--': 'dash', 'o.': 'dot', 'o-': 'solid'}

    # Plotly implementation
    # Retrieve number of rows and columns and create specs and titles
    num_rows = fig.num_rows
    num_cols = fig.num_cols

    gs = GridSpec(num_rows, num_cols)
    pfig = plt.figure(figsize=fig.figsize)

    ax_ind = 1
    for i in range(num_rows):
        for j in range(num_cols):

            if i in fig.plots and j in fig.plots[i]:
                ax = fig.plots[i][j]
                rowspan = ax.rowspan
                colspan = ax.colspan
                is_3d = np.any([type(plot) == Plot3d for plot in ax.plots])
                if is_3d:
                    print("3d axis")
                    axp = pfig.add_subplot(gs[i:i+rowspan, j:j+colspan], projection='3d')
                else:
                    axp = pfig.add_subplot(gs[i:i+rowspan, j:j+colspan])
                if ax.title:
                    plt.title(ax.title)
                for k, plot in enumerate(ax.plots):
                    if type(plot) == Plot2d:
                        # Set categorical color for each plot in axis if color is not set
                        if plot.clr is None:
                            clr = categorical_color(k)
                        else:
                            clr = plot.clr
                        plt.plot(plot.x, plot.y, color=clr, lw=plot.lw, ls=plot.ls, alpha=plot.alpha, label=plot.label)
                    elif type(plot) == Image:
                        plt.imshow(plot.im, cmap=plot.cmap)
                    elif type(plot) == Bar:
                        y = np.array(plot.vals)
                        if len(y.shape) == 1:
                            n = len(y)
                        else:
                            n = y.shape[1]
                        x = np.arange(n)

                        # If not color set, set categorical color for each bar
                        if plot.clr is None:
                            clr = [categorical_color(iv) for iv in range(n)]
                        else:
                            clr = [plot.clr for _ in range(n)]
                        w = 1.0
                        if plot.limits is not None:
                            min_vals = np.ones(n) * plot.limits[0]
                            max_vals = np.ones(n) * plot.limits[1]
                            w = 0.4
                            heights = max_vals - min_vals
                            centers = (max_vals + min_vals) / 2
                            plt.bar(x, heights,
                                    bottom=min_vals,
                                    color=clr,
                                    width=w,
                                    alpha=0.3)

                        if len(y.shape) == 1:
                            plt.bar(x, heights,
                                    bottom=min_vals,
                                    color=clr,
                                    width=w,
                                    alpha=0.9)

                            for ind in range(n):
                                plt.vlines(xmin=x[ind]-w, xmax=x[ind]+w,
                                           ymin=0,
                                           ymax=y[ind],
                                           color=clr[ind])
                        else:
                            plt.bar(x, y[1] - y[0],
                                    bottom=y[0],
                                    color=clr,
                                    width=w,
                                    alpha=0.9)

                    elif type(plot) == Plot3d:
                        # Set categorical color for each plot in axis if color is not set
                        if plot.clr is None:
                            clr = categorical_color(k)
                        else:
                            clr = plot.clr
                        x=plot.X[:,0]
                        y=plot.X[:,1]
                        z=plot.X[:,2]

                        if type(clr) != str and not is_number(clr[0]):
                            plt.scatter(x, y, z, c=clr,
                                    alpha=plot.alpha,
                                    label=plot.label)
                        else:
                            plt.plot(x, y, z, color=clr, lw=plot.lw,
                                    ls=plot.ls, alpha=plot.alpha,
                                    label=plot.label)
                if ax.equal:
                    plt.axis('equal')
                if ax.flip_y:
                    plt.gca().invert_yaxis()
    plt.show()


def show_plotly(fig):
    '''
    Plotly implementation for figure and plots
    '''
    typemap = {type(Plot2d): 'scatter',
                type(Plot3d): 'surface'}
    mode_map = {':': 'lines', '.':'markers', '-': 'lines', '--': 'lines', 'o.': 'lines+markers', 'o-': 'lines+markers'}
    mode_map_dash = {':': 'dot', '.': 'dot', '-': 'solid', '--': 'dash', 'o.': 'dot', 'o-': 'solid'}

    # Plotly implementation
    # Retrieve number of rows and columns and create specs and titles
    num_rows = fig.num_rows
    num_cols = fig.num_cols
    specs = []
    titles = []
    # Fill in specs for subplots
    for i in range(num_rows):
        row = []
        for j in range(num_cols):
            if i in fig.plots and j in fig.plots[i]:
                ax = fig.plots[i][j]
                row.append(dict(rowspan=ax.rowspan, colspan=ax.colspan))
                titles.append(ax.title)
                is_3d = np.any([type(plot) == Plot3d for plot in ax.plots])
                if is_3d:
                    row[-1]['type'] = 'scene'
                is_bar = np.any([type(plot) == Bar for plot in ax.plots])
                if is_bar:
                    row[-1]['type'] = 'bar'
            else:
                row.append(None)
        specs.append(row)
    print('Plot grid:', num_rows, num_cols)
    pfig = make_subplots(rows=num_rows, cols=num_cols,
                         specs=specs,
                         subplot_titles=titles,
                         horizontal_spacing = 0.05,
                         vertical_spacing = 0.05)

    ax_ind = 1
    for i in range(num_rows):
        for j in range(num_cols):
            if i in fig.plots and j in fig.plots[i]:
                ax = fig.plots[i][j]

                for k, plot in enumerate(ax.plots):
                    if type(plot) == Plot2d:
                        # Set categorical color for each plot in axis if color is not set
                        if plot.clr is None:
                            clr = categorical_color(k)
                        else:
                            clr = plot.clr
                        pfig.add_trace(go.Scatter(x=plot.x, y=plot.y,
                                                    mode=mode_map[plot.ls],
                                                    line=dict(color=plotly_color(clr),
                                                            width=plot.lw,
                                                            dash=mode_map_dash[plot.ls]),
                                                    opacity=plot.alpha,
                                                    fill=plot.fill, #plotly_color(plot.fill),
                                                    showlegend=plot.label is not None,
                                                    name=plot.label if plot.label else None
                                                    ),
                                                    row=i+1, col=j+1
                                                    )
                    elif type(plot) == Image:
                        pfig.add_trace(go.Image(z=plot.img))
                        #px.imshow(plot.img, color_continuous_scale=plot.cmap)
                    elif type(plot) == Bar:
                        y = np.array(plot.vals)
                        if len(y.shape) == 1:
                            n = len(y)
                        else:
                            n = y.shape[1]
                        x = np.arange(n)

                        # If not color set, set categorical color for each bar
                        if plot.clr is None:
                            clr = [plotly_color(categorical_color(iv)) for iv in range(n)]
                        else:
                            clr = plotly_color(plot.clr)
                        if plot.limits is not None:
                            min_vals = np.ones(n) * plot.limits[0]
                            max_vals = np.ones(n) * plot.limits[1]
                            pfig.add_trace(go.Bar(x=x, y=max_vals - min_vals,
                                                  base=min_vals,
                                                  marker_color=clr,
                                                  opacity=0.3,
                                                  showlegend=False),
                                           row=i+1, col=j+1)

                        if len(y.shape) == 1:
                            pfig.add_trace(go.Bar(x=x, y=y,
                                                marker_color=clr,
                                                opacity=plot.alpha,
                                                showlegend=plot.label is not None,
                                                name=plot.label if plot.label else None),
                                        row=i+1, col=j+1)
                        else:
                            pfig.add_trace(go.Bar(x=x, y=y[1] - y[0],
                                                  base = y[0],
                                                marker_color=clr,
                                                opacity=plot.alpha,
                                                showlegend=plot.label is not None,
                                                name=plot.label if plot.label else None),
                                        row=i+1, col=j+1)


                    elif type(plot) == Plot3d:
                        # Set categorical color for each plot in axis if color is not set
                        if plot.clr is None:
                            clr = categorical_color(k)
                        else:
                            clr = plot.clr

                        pfig.add_trace(go.Scatter3d(
                            x=plot.X[:,0],
                            y=plot.X[:,1],
                            z=plot.X[:,2],
                            mode=mode_map[plot.ls],
                            line=dict(color=plotly_color(clr),
                                      width=plot.lw,
                                      dash=mode_map_dash[plot.ls]),
                            opacity=plot.alpha,
                            showlegend=plot.label is not None,
                            name=plot.label if plot.label else None),
                                       row=i+1, col=j+1)


                if ax.equal:
                    # Ugly hack to make axis for a given subplot equal with plotly
                    pfig.update_xaxes(row=i+1, col=j+1, scaleanchor="y%d"%ax_ind, scaleratio=1)
                if ax.flip_y:
                    pfig.update_yaxes(row=i+1, col=j+1, autorange='reversed')
                ax_ind += 1

    pfig.update_layout(barmode='relative',
                       uirevision=True,
                       title=dict(text=fig.figtitle),
                       #margin=dict(l=0, r=0, b=0, t=0), # Tight layout
                       scene = dict(
                           aspectmode='data' # Force aspect ratio to be 1:1:1 for 3d
                       ),)

    pio.show(pfig)


## Categorical colors

categorical_palettes = {
    'Tabular':[
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
(1.0, 0.4980392156862745, 0.054901960784313725),
(1.0, 0.7333333333333333, 0.47058823529411764),
(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
(0.596078431372549, 0.8745098039215686, 0.5411764705882353),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(1.0, 0.596078431372549, 0.5882352941176471),
(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
(0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
(0.7686274509803922, 0.611764705882353, 0.5803921568627451),
(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
(0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
(0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
(0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
(0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
(0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
    ],
'Dark2_8':[
(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
(0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
(0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
(0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
(0.4, 0.6509803921568628, 0.11764705882352941),
(0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
(0.6509803921568628, 0.4627450980392157, 0.11372549019607843),
(0.4, 0.4, 0.4)
],
'Custom':[
[104.0/255, 175.0/255, 252.0/255],
[66.0/255, 47.0/255, 174.0/255],
[71.0/255, 240.0/255, 163.0/255],
[29.0/255, 104.0/255, 110.0/255],
[52.0/255, 218.0/255, 234.0/255],
[45.0/255, 93.0/255, 168.0/255],
[219.0/255, 119.0/255, 230.0/255],
[165.0/255, 46.0/255, 120.0/255],
[171.0/255, 213.0/255, 51.0/255],
[29.0/255, 109.0/255, 31.0/255],
[143.0/255, 199.0/255, 137.0/255],
[226.0/255, 50.0/255, 9.0/255],
[93.0/255, 242.0/255, 62.0/255],
[94.0/255, 64.0/255, 40.0/255],
[247.0/255, 147.0/255, 2.0/255],
[255.0/255, 0.0/255, 135.0/255],
[226.0/255, 150.0/255, 163.0/255],
[216.0/255, 197.0/255, 152.0/255],
[97.0/255, 8.0/255, 232.0/255],
[243.0/255, 212.0/255, 38.0/255]
],
'Paired_12':[ # Okeish
(0.6509803921568628, 0.807843137254902, 0.8901960784313725),
(0.12156862745098039, 0.47058823529411764, 0.7058823529411765),
(0.6980392156862745, 0.8745098039215686, 0.5411764705882353),
(0.2, 0.6274509803921569, 0.17254901960784313),
(0.984313725490196, 0.6039215686274509, 0.6),
(0.8901960784313725, 0.10196078431372549, 0.10980392156862745),
(0.9921568627450981, 0.7490196078431373, 0.43529411764705883),
(1.0, 0.4980392156862745, 0.0),
(0.792156862745098, 0.6980392156862745, 0.8392156862745098),
(0.41568627450980394, 0.23921568627450981, 0.6039215686274509),
(1.0, 1.0, 0.6),
(0.6941176470588235, 0.34901960784313724, 0.1568627450980392)
],
'Tableau_20':[
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(0.6823529411764706, 0.7803921568627451, 0.9098039215686274),
(1.0, 0.4980392156862745, 0.054901960784313725),
(1.0, 0.7333333333333333, 0.47058823529411764),
(0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
(0.596078431372549, 0.8745098039215686, 0.5411764705882353),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(1.0, 0.596078431372549, 0.5882352941176471),
(0.5803921568627451, 0.403921568627451, 0.7411764705882353),
(0.7725490196078432, 0.6901960784313725, 0.8352941176470589),
(0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
(0.7686274509803922, 0.611764705882353, 0.5803921568627451),
(0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
(0.9686274509803922, 0.7137254901960784, 0.8235294117647058),
(0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
(0.7803921568627451, 0.7803921568627451, 0.7803921568627451),
(0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
(0.8588235294117647, 0.8588235294117647, 0.5529411764705883),
(0.09019607843137255, 0.7450980392156863, 0.8117647058823529),
(0.6196078431372549, 0.8549019607843137, 0.8980392156862745)
],
'Bold_10':[
(0.4980392156862745, 0.23529411764705882, 0.5529411764705883),
(0.06666666666666667, 0.6470588235294118, 0.4745098039215686),
(0.2235294117647059, 0.4117647058823529, 0.6745098039215687),
(0.9490196078431372, 0.7176470588235294, 0.00392156862745098),
(0.9058823529411765, 0.24705882352941178, 0.4549019607843137),
(0.5019607843137255, 0.7294117647058823, 0.35294117647058826),
(0.9019607843137255, 0.5137254901960784, 0.06274509803921569),
(0.0, 0.5254901960784314, 0.5843137254901961),
(0.8117647058823529, 0.10980392156862745, 0.5647058823529412),
(0.9764705882352941, 0.4823529411764706, 0.4470588235294118)
],
    'Prism_10':[
(0.37254901960784315, 0.27450980392156865, 0.5647058823529412),
(0.11372549019607843, 0.4117647058823529, 0.5882352941176471),
(0.2196078431372549, 0.6509803921568628, 0.6470588235294118),
(0.058823529411764705, 0.5215686274509804, 0.32941176470588235),
(0.45098039215686275, 0.6862745098039216, 0.2823529411764706),
(0.9294117647058824, 0.6784313725490196, 0.03137254901960784),
(0.8823529411764706, 0.48627450980392156, 0.0196078431372549),
(0.8, 0.3137254901960784, 0.24313725490196078),
(0.5803921568627451, 0.20392156862745098, 0.43137254901960786),
(0.43529411764705883, 0.25098039215686274, 0.4392156862745098)
    ],
'ColorBlind_10':[
(0.0, 0.4196078431372549, 0.6431372549019608),
(1.0, 0.5019607843137255, 0.054901960784313725),
(0.6705882352941176, 0.6705882352941176, 0.6705882352941176),
(0.34901960784313724, 0.34901960784313724, 0.34901960784313724),
(0.37254901960784315, 0.6196078431372549, 0.8196078431372549),
(0.7843137254901961, 0.3215686274509804, 0.0),
(0.5372549019607843, 0.5372549019607843, 0.5372549019607843),
(0.6352941176470588, 0.7843137254901961, 0.9254901960784314),
(1.0, 0.7372549019607844, 0.4745098039215686),
(0.8117647058823529, 0.8117647058823529, 0.8117647058823529)
],
    'BlueRed_12':[
(0.17254901960784313, 0.4117647058823529, 0.6901960784313725),
(0.7098039215686275, 0.7843137254901961, 0.8862745098039215),
(0.9411764705882353, 0.15294117647058825, 0.12549019607843137),
(1.0, 0.7137254901960784, 0.6901960784313725),
(0.6745098039215687, 0.3803921568627451, 0.23529411764705882),
(0.9137254901960784, 0.7647058823529411, 0.6078431372549019),
(0.4196078431372549, 0.6392156862745098, 0.8392156862745098),
(0.7098039215686275, 0.8745098039215686, 0.9921568627450981),
(0.6745098039215687, 0.5294117647058824, 0.38823529411764707),
(0.8666666666666667, 0.788235294117647, 0.7058823529411765),
(0.7411764705882353, 0.0392156862745098, 0.21176470588235294),
(0.9568627450980393, 0.45098039215686275, 0.47843137254901963)
    ],
    'plut_categorical_12':[
(1.0, 0.42745098039215684, 0.6823529411764706),
(0.8313725490196079, 0.792156862745098, 0.22745098039215686),
(0.0, 0.7450980392156863, 1.0),
(0.9215686274509803, 0.6745098039215687, 0.9803921568627451),
(0.6196078431372549, 0.6196078431372549, 0.6196078431372549),
(0.403921568627451, 0.8823529411764706, 0.7098039215686275),
(0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
(0.6039215686274509, 0.8156862745098039, 1.0),
(0.8862745098039215, 0.5215686274509804, 0.26666666666666666),
(0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
(1.0, 0.4980392156862745, 0.054901960784313725),
(0.36470588235294116, 0.6941176470588235, 0.35294117647058826)
]

}


default_categorical_name = 'plut_categorical_12' #Tableau_20' #'Tabular' #'BlueRed_12' #'Tabular' #'BlueRed_12' #'Tabular' #'BlueRed_12'# 'Paired_12' #OK #'ColorBlind_10' #OK # 'Dark2_8'# OKeish 'Bold_10' #OK #
def categorical_color(i):
    return categorical_palettes[default_categorical_name][i % len(categorical_palettes[default_categorical_name])]
default_color = categorical_color

def cmap(v, colormap='turbo'): #turbo'): #'PuRd'):
    c = matplotlib.cm.get_cmap(colormap)
    return c(v)


