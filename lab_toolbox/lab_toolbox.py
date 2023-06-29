#!/usr/bin/env python3
'''
This file contains the functions that support data transformations
during the processes in the SOIL lab.
'''
import numpy as np
import pandas as pd
from lab_toolbox.constants import constants
import plotly.graph_objs as go
from typing import Dict, Tuple, List, Any
from plotly.subplots import make_subplots
from scipy.integrate import simpson
import os
import csv
import pybaselines
from scipy.constants import physical_constants
import datetime


class Measurement:
    ''' A class to hold a single measurement and its associated data  '''
    def __init__(self, name, raw_data):

        self.name = name
        self.raw_data = raw_data

        self._baseline_selected_points = [self.x[0], self.x[-1]]
        self._spline = np.array([])
        self.update_spline(interpolation_method='linear')

        self._integral_selected_points = []
        self._integration_properties = []

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
        return self._spline

    def update_spline(self, interpolation_method):
        # Use the sorted baseline points here
        try:
            fitter = pybaselines.Baseline(self.x, check_finite=False)
            # For all of the select points in self.base_selected_points
            # which is a subset of self.x, create an array of the
            # associated y value in self.y and update the spline with those
            # points
            pairs = np.array([
                (x, self.y[np.where(self.x == x)][0])
                for x in self.baseline_selected_points
            ])

            if len(self.baseline_selected_points) < 4:
                # If less than two points have been chosen, fit with polyline
                self._spline = fitter.interp_pts(
                    self.x.reshape(-1, 1),
                    baseline_points=pairs,
                    interp_method='linear')[0]
            else:
                self._spline = fitter.interp_pts(
                    self.x.reshape(-1, 1),
                    baseline_points=pairs,
                    interp_method=interpolation_method
                )[0]

        except ValueError as e:
            print("Error defining spline", e)

    @property
    def filtered_baseline(self):
        try:
            return self.y - self.spline
        except TypeError:
            print(f"Spline error on {self.name} with last calculation, "
                  "returning raw data")
            return self.y

    @property
    def integral_peaks(self):
        ''' Returns a list of x values for the peaks '''
        peaks = []
        for prop in self._integration_properties:
            if prop['peak_x'] is not None and not np.isnan(prop['peak_x']):
                peaks.append(prop['peak_x'])

        return peaks

    def save_plot(self, output_dir):
        ''' Outputs a plot of the measurement to the specified path '''

        fig = go.Figure(
            go.Scatter(
                x=self.x/60,
                y=self.filtered_baseline*10**6,
                mode='lines',
                name=self.name,
            )
        )

        fig.update_layout(
            title=self.name,
            xaxis_title="Time (min)",
            yaxis_title="Current (μA)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="RebeccaPurple"
            )
        )
        # Remove the / from the filename if exists
        filename = os.path.join(output_dir, f"{self.name.split('/')[0]}.png")

        fig.write_image(filename, width=1980, height=1080)
        print(f"Saved plot to {filename}")

    def find_peak(
            self,
            x_start,
            x_end
    ) -> Tuple[float, float | int]:
        ''' Finds the peak value between the x_start and x_end values

        Uses the filtered baseline data to find the peak.

        Parameters
        ----------
        x_start : float
            The x value to start looking for the peak
        x_end : float
            The x value to stop looking for the peak

        Returns
        -------
        Tuple[float | int, float]
            The x and y values of the peak, the data type of the x value
            depends on the data type of the x values in the measurement
        '''

        # Get the filtered baseline data between the start and end points
        y = self.filtered_baseline[
            np.where((self.x >= x_start) & (self.x <= x_end))
        ]

        # Get the x values between the start and end points
        x = self.x[
            np.where((self.x >= x_start) & (self.x <= x_end))
        ]

        # Find the peak absolute value from the filtered baseline
        peak_y = np.max(np.abs(y))

        # Get the x value of the peak
        peak_x = x[np.where(np.abs(y) == peak_y)][0]

        return peak_x, peak_y

    def _calculate_integrals(
        self,
        samples,
        integration_method: str = 'simpson',
    ) -> List[Dict[str, int | float]]:
        ''' Calculates the area under the curve between the selected points

        Uses the filtered baseline for data and the selected points to
        calculate the area under the curve between the points.

        The selected points reside in the integral_selected_points list
        that have been added in this Measurement class.

        Parameters
        ----------
        samples : int
            The number of samples to use when calculating the integral, this
            should be input by the user as they will know how many samples
            they had taken during the measurement
        integration_method : str, optional
            The method to use when calculating the integral, by default
            'simpson', to use the trapezoidal rule use 'trapz'

        Returns
        -------
        List[Tuple[float, float, float]]
            A list of tuples containing the start and end points of the
            integral and the area under the curve between those points
        '''

        # Get list of tuples of start and end points
        ranges_all = [
            self.integral_selected_points[x:x+2]
            for x in range(0, len(self.integral_selected_points), 2)
        ]
        # Remove any ranges that are not complete (groups of 2)
        ranges = [x for x in ranges_all if len(x) == 2]

        self._integration_properties = []
        for start, end in ranges:
            # Get Y values between start and end
            y_values = self.filtered_baseline[
                np.where((self.x >= start) & (self.x <= end))
            ]

            # Get X values between start and end
            x_values = self.x[
                np.where((self.x >= start) & (self.x <= end))
            ]

            # Calculate the integral
            area = integrate_coulomb_as_mole(
                y=y_values, x=x_values, method=integration_method
            )

            peak_x, peak_y = self.find_peak(start, end)

            self._integration_properties.append({
                "start": start,
                "end": end,
                "area": area,
                "peak_x": peak_x,
                "peak_y": peak_y
            })

        # Buffer output list with NaNs to match the number of samples
        while len(self._integration_properties) < samples:
            self._integration_properties.append({
                "start": np.nan,
                "end": np.nan,
                "area": np.nan,
                "peak_x": np.nan,
                "peak_y": np.nan
            })

        return self._integration_properties


def integrate_coulomb_as_mole(
    y: np.ndarray,
    x: np.ndarray,
    method: str = 'trapz'
) -> float:
    ''' Calculate the integral of the x and y values and convert to moles

    The Faraday constant is used to convert the integral to moles.

    Parameters
    ----------
    y : np.ndarray
        The y values as electrical current to integrate
    x : np.ndarray
        The x values to integrate
    method : str, optional
        The method to use when calculating the integral, by default
        'trapz', to use the Simpson's rule use 'simpson'

    Returns
    -------
    float
        The integral of the x and y values converted to moles
    '''

    if method == 'trapz':
        area = np.trapz(y, x)
    elif method == 'simpson':
        area = simpson(y, x)
    else:
        raise ValueError(
            f"Integration method '{method}' not "
            "supported, use 'trapz' or 'simpson'"
        )

    # Convert coulombs to moles using Faraday constant
    area /= physical_constants["Faraday constant"][0]

    return area


def filter_baseline_interactively(
    measurements: Dict[str, Measurement],
    interpolation_method: str = 'linear',
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
                        measurement.remove_baseline_point(clicked_point)
                    else:
                        measurement.add_baseline_point(clicked_point)

                    measurement.update_spline(interpolation_method)

                    # Go through all the selected points and update their
                    # color and size. If not in list, return them to default
                    # Get the mask of the selected points against the x array
                    mask = np.isin(
                        measurement.x,
                        measurement.baseline_selected_points
                    )
                    for k, point in enumerate(measurement.x):
                        if mask[k]:  # If true, point is selected
                            point_color[k] = constants.SELECTED_POINT_COLOUR
                            point_size[k] = constants.SELECTED_POINT_SIZE
                        else:  # If false, point is not selected
                            point_color[k] = constants.DEFAULT_POINT_COLOUR
                            point_size[k] = constants.DEFAULT_POINT_SIZE

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

    for subfig in fig.data:
        # Callback for on click
        subfig.on_click(cb_update_baseline_filter_plot)

        # Change colours of selected points (only on the raw data)
        if 'spline' not in subfig.name:
            # Define the colour and size of points on first load
            # Set the start and end points to be selected. It would be better
            # to do this in the loop above, or actuate the callback on load,
            # however the start and end are always considered in the spline.
            marker_colour_list = [constants.DEFAULT_POINT_COLOUR] * len(
                subfig.y
            )
            marker_colour_list[0] = constants.SELECTED_POINT_COLOUR
            marker_colour_list[-1] = constants.SELECTED_POINT_COLOUR
            subfig.marker.color = marker_colour_list

            marker_size_list = [constants.DEFAULT_POINT_SIZE] * len(subfig.y)
            marker_size_list[0] = constants.SELECTED_POINT_SIZE
            marker_size_list[-1] = constants.SELECTED_POINT_SIZE
            subfig.marker.size = marker_size_list

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
    samples: int,
    integration_method: str = 'trapz',
) -> go.Figure:
    ''' Generates a subplot given the dictionary of measurements

    Allows user to select start and end of peaks to integrate
    '''

    # Insure the integration data arrays are empty before starting otherwise
    # the plots interaction logic will break with the old data
    for measurement in measurements.values():
        measurement._integral_selected_points = []
        measurement._integration_properties = []

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
                        measurement._calculate_integrals(
                            samples, integration_method
                        )

                    for i, point in enumerate(point_color):
                        # Reset integral max points before setting new ones
                        if point == constants.INTEGRAL_PEAK_COLOUR:
                            point_color[i] = constants.DEFAULT_POINT_COLOUR
                            point_size[i] = constants.DEFAULT_POINT_SIZE

                    for point in measurement.integral_peaks:
                        val = np.where(subplot.x == point)[0]
                        if val and len(val) and not np.isnan(val):
                            peak_index = val.item()
                            point_color[peak_index] = (
                                constants.INTEGRAL_PEAK_COLOUR
                            )
                            point_size[peak_index] = (
                                constants.INTEGRAL_PEAK_SIZE
                            )

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


def output_data(
    measurements: Dict[str, Measurement],
    sample_count: int,
    output_dir: str | None = None,
) -> None:
    # Create output dir based on UTC time, with absolute path
    if output_dir is None:
        output_dir = os.path.join(
            os.getcwd(), f'output-{datetime.datetime.utcnow()}'
        )
    else:
        # Add absolute path to output dir
        output_dir = os.path.join(os.getcwd(), output_dir)

    # Create folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Form dataframe from baseline data
    baseline_data: Dict[str, Any] = {"Time/s": []}
    raw_data: Dict[str, Any] = {"Time/s": []}
    for name, data in measurements.items():
        # Create baseline data structure
        baseline_data[name] = data.filtered_baseline
        baseline_data["Time/s"] = data.x  # Any of the x will do, use the last

        # Create raw data structure
        raw_data[name] = data.y
        raw_data["Time/s"] = data.x

        # Save plots
        data.save_plot(output_dir)

    # Save baseline filtered data
    df = pd.DataFrame.from_dict(baseline_data)
    df.set_index('Time/s', inplace=True, drop=True)
    baseline_filename = os.path.join(
        output_dir, constants.FILENAME_BASELINE_SUBTRACTED_DATA
    )
    df.to_csv(baseline_filename)
    print(f"Saved baseline subtracted data to {baseline_filename}")

    # Do the same for the raw data
    df = pd.DataFrame.from_dict(baseline_data)
    df.set_index('Time/s', inplace=True, drop=True)
    rawdata_filename = os.path.join(
        output_dir, constants.FILENAME_RAW_DATA
    )
    df.to_csv(rawdata_filename)
    print(f"Saved raw data to {rawdata_filename}")

    header = ['measurement']
    for i in range(sample_count):
        for name in ["start", "end", "area", "peak_time", "peak_value"]:
            header.append(f"sample{i+1}_{name}")

    # Save summary data
    summary_filename = os.path.join(
        output_dir, constants.FILENAME_SUMMARY_DATA
    )
    with open(summary_filename, 'w') as csvfile:
        outputwriter = csv.writer(csvfile, delimiter=',')
        outputwriter.writerow(header)
        for name, measurement in measurements.items():
            row = []
            row.append(name)
            for i, sample in enumerate(measurement._integration_properties):
                row += [sample['start'], sample['end'], sample['area'],
                        sample['peak_x'], sample['peak_y']]
            outputwriter.writerow(row)
    print(f"Saved summary data to {summary_filename}")


def find_header_start(
    filename_path: str,
    header_text: str = constants.FILE_HEADER
) -> int:
    ''' Find the start of the header in the file

    Iterates through the file until it finds the header start. The header is
    defined as a constant in the constants module.

    Parameters
    ----------
    filename_path : str
        The path to the file to find the header start of
    header_text : str
        The text to find in the file to indicate the start of the header

    Returns
    -------
    int
        The line number of the header start
    '''

    qty_empty_rows = 0

    with open(filename_path, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == header_text:
                return i - qty_empty_rows
            elif len(line.strip()) == 0:
                ''' Pandas does not count empty rows as part of the header '''
                qty_empty_rows += 1

    raise ValueError('Could not find header start')


def import_data(
    filename_path: str,
    header_start: int | None = None
) -> Dict[str, Measurement]:
    ''' Import data from a file

    The header start variable is given to pandas, which is the value of the
    line number of the header start. If this is not given, the function will
    attempt to find the header start. If this is given manually it is important
    to not include empty lines in this count.

    For example, if the header starts on line 5 (starting at 0), but there are
    2 empty lines before the header, the header start should be 3.


    Parameters
    ----------
    filename_path : str
        The path to the file to import
    header_start : int, optional
        The line number of the header start, by default None

    Returns
    -------
    Dict[str, Measurement]
        A dictionary of measurements
    '''

    # Find the header start
    if header_start is None:
        header_start = find_header_start(filename_path)

    # Import data
    df = pd.read_csv(filename_path, header=header_start)
    df.set_index('Time/s', inplace=True)  # Set index to the time column
    df.columns = df.columns.str.strip()  # Strip whitespace from the columns

    # Create a structure to hold all of the measurements (i1/A, i2/A, ...)
    measurements = {}
    for col in df.columns:
        x = Measurement(col, df[col])
        measurements[x.name] = x

    return measurements
