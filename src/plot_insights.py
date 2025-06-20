import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.transforms import Affine2D
from matplotlib.spines import Spine
import matplotlib.colors as mcolors
import math
import seaborn as sns

def radar_factory(num_vars, frame='polygon', horizontal_scale=0.3, is_diamond=False):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'polygon' and is_diamond:
                diamond_verts = np.array([
                    [0.5, 1.0],
                    [0.5 + 0.5*horizontal_scale, 0.5],
                    [0.5, 0.0],
                    [0.5 - 0.5*horizontal_scale, 0.5],
                ])
                return Polygon(diamond_verts, edgecolor="k", linewidth=2, closed=True)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k", linewidth=2)
            elif frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'polygon' and is_diamond:
                verts = np.array([
                    [0, 1],
                    [horizontal_scale, 0],
                    [0, -1],
                    [-horizontal_scale, 0],
                ])
                path = Path(np.vstack([verts, verts[0]]),
                           [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])
                spine = Spine(axes=self, spine_type='circle', path=path)
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            elif frame == 'circle':
                return super()._gen_axes_spines()
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def create_radar_chart(categories, data, category_colors, title, horizontal_scale=0.3, ax=None):
    num_vars = len(categories)
    min_y = 0
    max_y = 1.1

    if len(category_colors) != num_vars:
        raise ValueError(f"Number of colors ({len(category_colors)}) must match number of categories ({num_vars})")

    diamond_mode = False
    if num_vars == 2:
        diamond_mode = True
        num_vars = 4
        theta = radar_factory(num_vars, horizontal_scale=horizontal_scale, is_diamond=True)
        original_colors = category_colors.copy()
        categories = [categories[0], '', categories[1], '']
        category_colors = [original_colors[0], '', original_colors[1], '']
        padded_data = []
        for d in data:
            if len(d) != 2:
                raise ValueError("2-class data requires exactly 2 values per dataset")
            padded_data.append([d[0], 0, d[1], 0])
        data = padded_data
        num_vars = 4
    else:
        theta = radar_factory(num_vars, is_diamond=False)

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection='radar'))
    else:
        fig = ax.figure

    normalized_data = []
    for d in data:
        row_sum = sum(d)
        if not diamond_mode:
            norm_d = [val / row_sum if row_sum != 0 else 0 for val in d]
        else:
            norm_d = [
                d[0]/(d[0]+d[2]) if (d[0]+d[2]) != 0 else 0,
                0,
                d[2]/(d[0]+d[2]) if (d[0]+d[2]) != 0 else 0,
                0
            ]
        normalized_data.append(norm_d)

    ax.grid(False)
    grid_values = np.linspace(min_y, max_y, 6)

    for g in grid_values:
        if diamond_mode:
            r = [g, g*horizontal_scale, g, g*horizontal_scale, g]
            ax.plot(np.append(theta, theta[0]), r, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        else:
            polygon_vertices = [(t, g) for t in theta]
            polygon_vertices.append((theta[0], g))
            ax.plot(*zip(*polygon_vertices), color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    for t in theta:
        ax.plot([t, t], [min_y, max_y], color='black', linestyle='--', linewidth=0.5, alpha=0.7)

    for i, d in enumerate(normalized_data):
        max_value_index = np.argmax(d)
        category_color = (
            original_colors[0] if diamond_mode and max_value_index == 0 else
            original_colors[1] if diamond_mode and max_value_index == 2 else
            'gray' if diamond_mode else category_colors[max_value_index]
        )
        ax.scatter(theta[max_value_index], d[max_value_index], s=100, color=category_color, marker='x')

    ax.set_varlabels(categories)
    ax.set_ylim(min_y, max_y)
    ax.set_yticks([])
    ax.set_title(title, y=1.1)
    return fig, ax

def auto_subplot_grid(n):
    """Finds a near-square grid for `n` subplots."""
    rows = math.isqrt(n)
    cols = math.ceil(n / rows)
    return rows, cols


def plot_radar_charts(df, categories=None, colors=None, titles=None, correct=None, fig_title=None):
    """
    df: DataFrame containing specimen level predictions
    categories: List of radar chart categories
    colors: List of category colors
    titles: List of chart titles
    correct: Dictionary mapping lot_id to boolean for background coloring
    fig_title: Title for the entire figure
    """
    if categories is None:
        categories = df.columns.tolist()
    num_vars = len(categories)

    if fig_title is None:
        fig_title = "Insights, Green: Correct, Red: Incorrect"

    # Set default color scheme if none is provided
    if colors is None:
        colors = sns.color_palette("husl", len(categories))

    # Register projection
    radar_factory(4 if num_vars == 2 else num_vars, 
                is_diamond=True if num_vars == 2 else False)
    
    grouped = df.groupby(df.index)
    num_charts = len(grouped)
    rows, cols = auto_subplot_grid(num_charts)
    
    fig, axs = plt.subplots(rows, cols, 
                           subplot_kw=dict(projection='radar'), 
                           figsize=(cols * 4, rows * 4))
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for i, (index, group) in enumerate(grouped):
        title = index if titles is None else titles[i]
        data = group.values.tolist()
        create_radar_chart(categories, data, colors, title, ax=axs[i])
        
        # Add background color based on correctness
        if correct is not None:
            is_correct = correct.get(index, False)
            bg_color = mcolors.to_rgba('lightgreen', 0.2) if is_correct else mcolors.to_rgba('lightcoral', 0.2)
            axs[i].set_facecolor(bg_color)

    # Hide unused subplots
    for j in range(num_charts, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # or another value < 1
    fig.suptitle(fig_title, fontsize=16)

    return fig
    plt.show()