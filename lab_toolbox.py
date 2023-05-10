#!/usr/bin/env python3
'''
This file contains the functions that support data transformations
during the processes in the SOIL lab.
'''
from scipy.interpolate import splrep, splev
import numpy as np
from constants import constants
import plotly.graph_objs as go
from typing import Dict
from plotly.subplots import make_subplots
from scipy.integrate import simpson
from numpy import trapz


class Measurement:
    def __init__(self, name, raw_data):

        self.name = name
        self.raw_data = raw_data

        self._baseline_selected_points = []
        self._integral_selected_points = []

    def add_baseline_point(self, x_coord):
        self._baseline_selected_points.append(x_coord)

    def remove_baseline_point(self, x_coord):
        if x_coord not in self._baseline_selected_points:
            raise ValueError(
                f"{x_coord} not in {self._baseline_selected_points}"
            )

        self._baseline_selected_points.remove(x_coord)

    def add_integral_point(self, x_coord):
        ''' Adds a point to a list which are used to calculate an integral '''

        self._integral_selected_points.append(x_coord)

    def remove_integral_point(self, x_coord):
        ''' Remove a point from the list of points used to gen. an integral '''

        if x_coord not in self._integral_selected_points:
            raise ValueError(
                f"{x_coord} not in {self._integral_selected_points}"
            )

        self._integral_selected_points.remove(x_coord)

    @property
    def baseline_selected_points(self):
        # Return the points, but sorted!
        return np.sort(np.array(self._baseline_selected_points))

    @property
    def integral_selected_points(self):
        # Return the points, but sorted!
        return np.sort(np.array(self._integral_selected_points))

    @property
    def x(self):
        return np.sort(np.array(self.raw_data.index))

    @property
    def y(self):
        return np.array(self.raw_data)

    @property
    def spline(self):
        # Use the sorted baseline points here
        try:
            tck = splrep(self.x, self.y, t=self.baseline_selected_points, k=3)
            return splev(self.x, tck)
        except ValueError:
            print(self.x)

    @property
    def filtered_baseline(self):
        try:
            return self.y - self.spline
        except TypeError:
            print(f"Spline error on {self.name} with last calculation, "
                  "returning raw data")
            return self.y

    def calculate_integrals(self, samples):
        # Calculate the area under the curve
        # start and end are the x values
        # Use the filtered baseline

        # Get list of tuples of start and end points
        ranges_all = [
            self.integral_selected_points[x:x+2]
            for x in range(0, len(self.integral_selected_points), 2)
        ]
        # Remove any ranges that are not complete (groups of 2)
        ranges = [x for x in ranges_all if len(x) == 2]

        integral_pairs = []
        for start, end in ranges:
            # Get Y values between start and end
            y_values = self.filtered_baseline[
                np.where((self.x >= start) & (self.x <= end))
            ]

            # Get X values between start and end
            x_values = self.x[
                np.where((self.x >= start) & (self.x <= end))
            ]

            # Calculate the area
            # area = np.trapz(y_values, x_values)
            area = simpson(y_values, x_values)
            integral_pairs.append((start, end, area))

        # Buffer output list with NaNs to match the number of samples
        while len(integral_pairs) < samples:
            integral_pairs.append((np.nan, np.nan, np.nan))

        return integral_pairs


def filter_baseline_interactively(
    measurements: Dict[str, Measurement]
) -> go.Figure:
    ''' Generates a subplot given the dictionary of measurements

    Allows user interaction to define the baseline points
    '''

    def cb_update_baseline_filter_plot(trace, points, selector):
        # Must iterate through each plot in the subplot as everytime on_click
        # is actuated, it returns a list
        for i, subplot in enumerate(fig.data):
            if (
                i == points.trace_index
                and len(points.point_inds)
                and "filtered" not in points.trace_name
            ):  # Make sure we're working on the plot with changes

                point_color = list(subplot.marker.color)
                point_size = list(subplot.marker.size)
                measurement = measurements[points.trace_name]
                for j in points.point_inds:
                    # Point is given in a list, but we only click one at a
                    # time, unwrap it
                    clicked_point = points.xs[0]

                    # If a user clicks on the same point twice
                    if clicked_point in measurement.baseline_selected_points:
                        point_color[j] = constants.DEFAULT_POINT_COLOUR
                        point_size[j] = constants.DEFAULT_POINT_SIZE
                        measurement.remove_baseline_point(clicked_point)
                    else:
                        point_color[j] = '#2F4F4F'
                        point_size[j] = 20
                        measurement.add_baseline_point(clicked_point)

                    with fig.batch_update():
                        # Update the color and size of un/clicked point
                        subplot.marker.color = point_color
                        subplot.marker.size = point_size

                        # Update the associated spline plots. Assume plot
                        # name is spline_<columnname>
                        for splineplot in fig.data:
                            if splineplot.name == (
                                f"spline_{points.trace_name}"
                            ):
                                splineplot.y = measurement.spline
                            if splineplot.name in [
                                f"corrected_{points.trace_name}",
                                f"filtered_{points.trace_name}"
                            ]:
                                splineplot.y = measurement.filtered_baseline

    # Add the subplots for each column
    rows = 9
    cols = 1
    plot_titles = list(measurements.keys()) + ['All measurements']
    fig = go.FigureWidget(make_subplots(
        rows=rows, cols=cols,
        subplot_titles=plot_titles,
        vertical_spacing=0.02)
    )

    for x, measurement in enumerate(measurements.items()):
        name, data = measurement  # Unpack dictionary tuple

        fig.add_trace(
            go.Scattergl(
                x=data.x,
                y=data.y,
                name=name,
                line=dict(color="#636EFA"),  # Blue
                mode='lines+markers',
            ),
            row=x+1,
            col=1
        )

        # Add generated spline to same plot
        fig.add_trace(
            go.Scatter(
                x=data.x,
                y=data.spline,
                line=dict(color='#FF6692'),  # Red
                mode='lines',
                name=f"spline_{name}",
                hoverinfo='skip',  # Disable clicking/hovering on this line
            ),
            row=x+1,
            col=1
        )

        # Add subtracted baseline
        fig.add_trace(
            go.Scatter(
                x=data.x,
                y=data.filtered_baseline,
                # Make the line grey
                line=dict(color='#2F4F4F'),
                mode='lines',
                name=f"corrected_{name}",
                opacity=0.2,
                hoverinfo='skip',  # Disable clicking/hovering on this line
            ),
            row=x+1,
            col=1
        )

    # Finally add a plot with all measurements together
    for name, props in measurements.items():
        fig.add_trace(
            go.Scatter(
                x=props.x,
                y=props.filtered_baseline,
                name=f"filtered_{name}",
            ),
            row=9,
            col=1
        )

    DEFAULT_POINT_SIZE = 0
    DEFAULT_POINT_COLOUR = '#a3a7e4'

    for subfig in fig.data:
        # Callback for on click
        subfig.on_click(cb_update_baseline_filter_plot)

        # Change colours of selected points (only on the raw data)
        if 'spline' not in subfig.name:
            # Define the colour and size of points on first load
            subfig.marker.color = [DEFAULT_POINT_COLOUR] * len(subfig.y)
            subfig.marker.size = [DEFAULT_POINT_SIZE] * len(subfig.y)

            # Change scatter marker border colour/width (to make it visible)
            subfig.marker.line.color = 'DarkSlateGrey'
            subfig.marker.line.width = 0

    # Format and show fig
    fig.update_layout(
        height=2000, width=900,
        margin=dict(l=1, r=1, t=25, b=1),
        )

    return fig


def integrate_peaks_interactively(
    measurements: Dict[str, Measurement],
    samples: int
) -> go.Figure:
    ''' Generates a subplot given the dictionary of measurements

    Allows user to select start and end of peaks to integrate
    '''

    def cb_update_integral_filter_plot(trace, points, selector):
        # Must iterate through each plot in the subplot as everytime on_click
        # is actuated, it returns a list
        for i, subplot in enumerate(fig.data):
            if (
                i == points.trace_index
                and len(points.point_inds)
            ):  # Make sure we're working on the plot with changes

                point_color = list(subplot.marker.color)
                point_size = list(subplot.marker.size)
                measurement = measurements[points.trace_name]
                for j in points.point_inds:
                    # Point is given in a list, but we only click one at a
                    # time, unwrap it
                    clicked_point = points.xs[0]

                    # If a user clicks on the same point twice
                    if (
                        clicked_point in measurement.integral_selected_points
                    ):
                        point_color[j] = constants.DEFAULT_POINT_COLOUR
                        point_size[j] = constants.DEFAULT_POINT_SIZE
                        measurement.remove_integral_point(clicked_point)
                    elif (
                        len(measurement.integral_selected_points)
                        <= (samples * 2) - 1
                    ):
                        point_color[j] = '#2F4F4F'
                        point_size[j] = 20
                        measurement.add_integral_point(clicked_point)
                        print(f"{measurement.name}: {measurement.calculate_integrals(samples)}")

                    with fig.batch_update():
                        # Update the color and size of un/clicked point
                        subplot.marker.color = point_color
                        subplot.marker.size = point_size

    # Add the subplots for each column
    rows = 8
    cols = 1
    plot_titles = list(measurements.keys()) + ['All measurements']
    fig = go.FigureWidget(make_subplots(
        rows=rows, cols=cols,
        subplot_titles=plot_titles,
        vertical_spacing=0.02)
    )

    for x, measurement in enumerate(measurements.items()):
        name, data = measurement  # Unpack dictionary tuple

        fig.add_trace(
            go.Scattergl(
                x=data.x,
                y=data.filtered_baseline,
                name=name,
                mode='lines+markers',
            ),
            row=x+1,
            col=1
        )

    DEFAULT_POINT_SIZE = 0
    DEFAULT_POINT_COLOUR = '#a3a7e4'

    for subfig in fig.data:
        # Callback for on click
        subfig.on_click(cb_update_integral_filter_plot)

        # Change colours of selected points (only on the raw data)
        if 'spline' not in subfig.name:
            # Define the colour and size of points on first load
            subfig.marker.color = [DEFAULT_POINT_COLOUR] * len(subfig.y)
            subfig.marker.size = [DEFAULT_POINT_SIZE] * len(subfig.y)

            # Change scatter marker border colour/width (to make it visible)
            subfig.marker.line.color = 'DarkSlateGrey'
            subfig.marker.line.width = 0

    # Format and show fig
    fig.update_layout(
        height=2000, width=900,
        margin=dict(l=1, r=1, t=25, b=1),
        )

    return fig
