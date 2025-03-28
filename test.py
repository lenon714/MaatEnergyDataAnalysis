#@title Run This Cell
# Current version 6.0

from google.colab import output
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
mplstyle.use(['fast'])

from mpl_toolkits.mplot3d import Axes3D
import os
import time
import re
import sys
import cv2
import pandas as pd
import numpy as np
import gdown

from google.colab import drive
from pathlib import Path
from ipyfilechooser import FileChooser
import ipywidgets as widgets
from IPython.display import display, clear_output
import plotly.graph_objects as go
import plotly.express as px

import scipy
from scipy import signal as signal
from scipy.stats import boxcox, linregress
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

# Define global variables for UI dropdowns and column mappings
global col_names, addons, addons_list, func_list, experiments, sigma
global column_groups, column_keys, column_groups_A, column_keys_A, column_groups_B, column_keys_B, lab, linestyle
global axis_dct

# Column names corresponding to different experimental parameters
col_names = ["Flow Rate (slpm)", "Power (kW)", "Temperature (deg C)",
             "Pressure (psig)", "Gas Composition (%)", "Stub Position"]

# Define column groupings based on experimental dataset structure

# Dataset A: Column indices grouped by variable type
column_groups_A = [
    [1, 5, 3],  # Flow Rate
    [12, 13],  # Power
    [20, 21, 22, 23],  # Temperature
    [24, 25],  # Pressure
    [26, 27, 28, 29, 30, 31, 32],  # Gas Composition
    [18, 19]  # Stub Position
]

# Column names corresponding to Dataset A
column_keys_A = [
    'Time', 'MFC1', 'MFC2_Actual', 'MFC3', 'MFC1_Dem', 'MFC2', 'MFC3_Dem',
    'Nozzle #', 'Nozzle_ID (mm)', 'Nozzle_Velocity (m/s)',
    'SP_Forwarded_Power_(kW)', 'Lim_Reflected_Power_(kW)',
    'FP', 'RP', 'Reflected_Power_(%)', 'Absorbed_Power_(kW)',
    'X_Pos', 'Y_Pos', 'X Pos', 'Y Pos',
    'T1', 'T2', 'T3', 'T4', 'P1', 'P2',
    'CO', 'CO2', 'CH4', 'H2', 'CnHm', 'O2', 'N2',
    'CH4_Conv_%', 'Total_kWh/kq_H2', 'Carbon_Prod._(g/min)',
    'Tuner Temp', 'Sliding Short', 'Magnetron RP Voltage',
    'DC Forward Voltage', 'DC Reverse Voltage', 'Magnetron RP',
    'DC Forward Power', 'DC Reflected Power'
]

# Dataset B: Alternative grouping of column indices
column_groups_B = [
    [1, 2, 3],  # Flow Rate
    [4, 5],  # Power
    [6, 7, 8, 9],  # Temperature
    [24, 25, 10],  # Pressure
    [11, 12, 13, 14, 15, 16, 17],  # Gas Composition
    []  # Additional variables (empty for now)
]

# Column names corresponding to Dataset B
column_keys_B = [
    "Time", "St_CO2_Actual", "CO2_Actual", "CH4_Actual",
    "FP", "RP", "T1", "T2", "T3", "T4", "Pressure",
    "CO", "CO2", "CH4", "CnHm", "H2", "O2", "LHV", "Others",
    "Velocity", "X_Pos", "Y_Pos", "X_Sig", "Y_Sig",
    "Before_CS", "After_CS", "Delta_P", "Ignition",
    "St_CO2_Demand", "CO2_Demand", "CH4_Demand",
    "Retract_Demand", "Ignitor_Extended", "Ignitor_Retracted"
]

# Set initial dataset selection to Dataset A
column_groups = column_groups_A
column_keys = column_keys_A
lab = "A"  # Label to identify the dataset version

# List of available line styles for plotting
linestyle = ['solid', 'dashed', 'dotted', 'dashdot']

# Default standard deviation value (used for error bars, smoothing, etc.)
sigma = 10

# Dictionary to store axis properties (custom settings for plotting)
axis_dct = {}

# Define the initial directory path (Modify as needed)
initial_path = '/content/drive/Shareddrives/MERL - Experimental/'  # Change this to your desired starting directory

# Cache to store previously accessed paths
cached_paths = []

# Create a FileChooser widget to allow users to select a directory
chooser = FileChooser(initial_path)
chooser.title = '<b>Select a Folder</b>'  # Set the title of the widget
chooser.show_only_dirs = True  # Ensure only directories are shown (not files)

# Create a button for confirming folder selection
confirm_button = widgets.Button(
    description='Confirm Selection',
    button_style='success',  # Green success-style button
    tooltip='Click to confirm folder selection',
    icon='check'  # Checkmark icon for better UI experience
)

# Variable to store the selected directory
selected_directory = None

# Function to handle button click event when user selects a folder
def on_confirm_button_click(button):
    global selected_directory
    if chooser.selected_path:  # Ensure a folder is selected
        selected_directory = chooser.selected_path
        print(f"Selected folder: {selected_directory}")  # Display selected folder path
        cached_paths.append(selected_directory)  # Store selected folder in cache

        # Populate dropdown with subdirectories from the selected folder
        exp_dropdown.options = sorted([
            f for f in os.listdir(selected_directory) if os.path.isdir(os.path.join(selected_directory, f))
        ])

        # Make UI elements visible after selection
        lab_toggle.layout.display = "block"
        plot_tab.layout.display = "block"

# Attach the button click event to the handler function
confirm_button.on_click(on_confirm_button_click)

# Display the file chooser widget and confirmation button in the notebook
display(chooser)
display(confirm_button)

# Define column indices for specific data fields (modify as needed)
Time_colnum = 0  # Column index for Time data
FP_colnum = 12   # Column index for FP (some relevant data field)
RP_colnum = 13   # Column index for RP (some relevant data field)
RPlim_colnum = 11  # Column index for RP limit (some relevant data field)

def trimMwOn(fullData):
    """
    Trims the dataset to only include rows between the microwave turning on
    and the RP limit being triggered. Also, it resets the time column so that
    t=0 corresponds to the microwave turning on.
    """
    try:
        # Find the index where FP column first exceeds 0.1 (microwave turns on)
        startIndex = fullData.loc[fullData.iloc[:, FP_colnum] > 0.1].index[1]

        # Find the index where FP column last exceeds 0.1 (microwave turns off)
        endIndex_1 = fullData.iloc[::-1].loc[fullData.iloc[::-1, FP_colnum] > 0.1].index[1]
    except:
        # Default values if an error occurs in locating indices
        startIndex = 5
        endIndex_1 = len(fullData) - 5

    # Remove data outside the desired range (trimming with a buffer of ±5 rows)
    MwOnData = fullData.truncate(startIndex - 5, endIndex_1 + 5, 0, True)

    # Reset time so that t=0 is when the microwave turns on
    time_at_start = MwOnData.iloc[0, Time_colnum]
    MwOnData.iloc[:, Time_colnum] -= time_at_start
    fullData.iloc[:, Time_colnum] -= time_at_start
    fullData.iloc[:, Time_colnum] -= fullData.iloc[0, Time_colnum] - 1


def FindFolder(parts):
    """
    Finds the folder that corresponds to the given input parts.
    The function reformats the first part to include a space between
    letters and digits, then searches for a matching folder.
    """
    first_part = parts[0]

    # Insert a space between letters and digits in the first part of the folder name
    for i, char in enumerate(first_part):
        if char.isdigit():
            formatted_first_part = first_part[:i] + " " + first_part[i:]
            break
    else:
        formatted_first_part = first_part  # No digits found, keep as is

    formatted_second_part = "".join(parts[1:])  # Keep the second part unchanged

    base_path = Path(selected_directory)
    unique_start = formatted_first_part  # Use formatted first part to match folders

    # Find folders that match the unique starting pattern
    matching_folders = list(base_path.glob(f"{unique_start}*"))

    # If no matching folders are found, check cached paths
    if not matching_folders:
        for folder in cached_paths:
            matching_folders = list(base_path.glob(f"{unique_start}*"))
            base_path = Path(folder)
            if matching_folders:
                break

    return matching_folders, formatted_second_part


def ReadFolder(matching_folders, folder_name, formatted_second_part, header):
    """
    Reads a CSV file from a matched folder and returns the data.
    """
    if matching_folders:
        # Assume the first match is the correct folder
        target_folder = matching_folders[0]
        print(f"Opening folder: {target_folder}")

        data_folder = Path(target_folder) / folder_name
        print(formatted_second_part)
        data_path = Path(data_folder) / formatted_second_part  # Construct full file path

        # Attempt to read CSV file
        try:
            data_orig = pd.read_csv(data_path, header=header)
            data = data_orig
            # trimMwOn(data)  # Uncomment if trimming is needed
        except:
            # If the file is not found, print an error message
            print("CSV File was not found")
            return None
        return data
    else:
        # If no matching folder is found, print an error message
        print("Experiment folder was not found")
        return None


def ReadData(input_string, header=0):
    """
    Processes an input string in the format 'experiment_data' to find and read
    the corresponding CSV file.
    """
    # Split the input string at "_"
    parts = input_string.split("_")

    # Ensure there are exactly two parts
    if len(parts) >= 2:
        folder_name = "Data"
        matching_folders, formatted_second_part = FindFolder(parts)

        # Read the dataset from the folder
        data = ReadFolder(matching_folders, folder_name, formatted_second_part, header)
        if data is None:
            return None
    else:
        # Print an error message if the format is incorrect
        print("Input string", input_string, "does not contain exactly one underscore.")
        return None

    return data


def VerticalLine(breakpoints):
    """
    Draws vertical dashed red lines at the specified breakpoints.
    The input breakpoints should be a comma-separated string of numbers.
    """
    breakpoints = breakpoints.split(", ")  # Split input string into a list of breakpoints
    print(breakpoints)  # Debugging output to verify breakpoints
    for line in breakpoints:
        plt.axvline(x=int(line) / 60, color='r', linestyle='--')  # Convert to minutes and plot


def HorizontalLine(breakpoints):
    """
    Draws horizontal dashed red lines at the specified breakpoints.
    The input breakpoints should be a comma-separated string of numbers.
    """
    breakpoints = breakpoints.split(", ")  # Split input string into a list of breakpoints
    print(breakpoints)  # Debugging output to verify breakpoints
    for line in breakpoints:
        plt.axhline(y=int(line), color='r', linestyle='--')  # Plot horizontal line at given y-value

# Dictionary to manage additional features (e.g., adding lines to plots)
addons = {
    "Vertical Line": {"active": False, "value": ""},
    "Horizontal Line": {"active": False, "value": ""}
}
addons_list = ["Vertical Line", "Horizontal Line"]  # List of available addons
func_list = [VerticalLine, HorizontalLine]  # Corresponding functions for each addon

experiments = []  # List to store experiments

def addon_checkbox(description):
    """
    Creates a checkbox widget for enabling or disabling an addon.

    Args:
        description (str): The label for the checkbox.

    Returns:
        widgets.Checkbox: A checkbox widget that updates the `addons` dictionary when toggled.
    """
    checkbox = widgets.Checkbox(description=description)  # Create a checkbox with a given label

    def on_change(b):
        """
        Callback function to update the `addons` dictionary when checkbox state changes.
        """
        addons[b['owner'].description]['active'] = b["new"]  # Update the addon state (True/False)

    checkbox.observe(on_change, names='value')  # Attach observer for value changes
    return checkbox  # Return the checkbox widget


def addon_text(description):
    """
    Creates a text input widget for user-defined addon parameters.

    Args:
        description (str): The label for the text input field.

    Returns:
        widgets.Text: A text widget that updates the `addons` dictionary when modified.
    """
    textbox = widgets.Text(description=description, style={'description_width': '0px'})  # Create a text box

    def on_change(b):
        """
        Callback function to update the `addons` dictionary when text input changes.
        """
        addons[b['owner'].description]['text'] = b["new"]  # Store the input text in the addon dictionary

    textbox.observe(on_change, names='value')  # Attach observer for text changes
    return textbox  # Return the text widget


def TrimData(data, fp_column):
    """
    Trims data to include only the relevant portion where the `fp_column` values are above a threshold.

    Args:
        data (pd.DataFrame): The dataset to be trimmed.
        fp_column (int): The column index used to determine the trimming range.

    Returns:
        pd.DataFrame: The trimmed dataset with adjusted time values.
    """
    fp = data.iloc[:, fp_column]  # Extract the column used for filtering

    try:
      # Define upper and lower index bounds where the column value is greater than or equal to 1
      u_range = fp.loc[fp >= 1].index.max() + 10
      l_range = fp.loc[fp >= 1].index.min() - 10
      
      # Adjust the time column so that t=0 starts at the lower range
      data.iloc[:, 0] = data.iloc[:, 0] - data.iloc[l_range, 0]
      trimmed_data = data.loc[l_range:u_range]
    except:
      trimmed_data = data

    # Return only the relevant range of data
    return trimmed_data

def GeneratePlot(axs, data, num_runs, count, input_string=""):
    """
    Generates plots for different data columns and assigns labels, styles, and legends.

    Args:
        axs (array): Array of subplot axes.
        data (pd.DataFrame): The dataset to plot.
        num_runs (int): Number of experiments being plotted.
        count (int): Counter for differentiating line styles.
        input_string (str, optional): Identifier for the experiment (used for labeling). Defaults to "".
    """
    global column_groups, col_names, linestyle  # Access global variables

    # Iterate through the defined column groups
    for i, group in enumerate(column_groups):
        for col_index in group:
            # Create a unique legend label depending on the number of runs
            if num_runs > 1:
                legend_label = data.columns[col_index] + "_" + input_string
            else:
                legend_label = data.columns[col_index]

            # Plot the selected data column against time (converted to minutes)
            axs[i].plot(data.iloc[:, 0] / 60, data.iloc[:, col_index], label=legend_label,
                        linestyle=linestyle[count])  # Apply different line styles

        # Configure legends
        if input_string == "":
            axs[i].legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)  # Place legend outside the plot
        else:
            axs[i].legend()  # Standard legend placement

    # Set axis labels for all subplots
    for j in range(len(column_groups)):
        axs[j].set_ylabel(col_names[j])
        axs[j].set_xlabel("Time (min)")

    plt.tight_layout()  # Adjust layout to prevent overlap


def PlotRun(exp):
    """
    Reads experiment data and plots it across multiple subplots.

    Args:
        exp (list): List of experiment identifiers to load and plot.
    """
    global lab  # Access the global variable to determine additional plotting conditions

    # Create a figure with a 2x3 grid of subplots (2 rows, 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    count = 0  # Track line styles for different experiments

    for input_string in exp:
        data = ReadData(input_string)  # Load experiment data
        if data is None:
            break  # Stop if no data is found

        axs = axs.flatten()  # Flatten the 2D subplot array for easy iteration
        
        trimmed_data = TrimData(data, 12)

        # Generate plots for this dataset
        GeneratePlot(axs, trimmed_data, len(exp), count, input_string)
        count += 1  # Increment count for different line styles

    if data is not None:
        print(exp)  # Print experiment identifiers
        plt.show()  # Display the plots
    else:
        plt.close(fig)  # Close figure if no valid data was found

    # Additional plotting for single-experiment cases
    if len(exp) == 1 and data is not None and lab == "A":
        plt.subplots_adjust(right=0.95)  # Add space for legends

        # Create a new figure for metrics plot
        fig2 = plt.figure(figsize=(8, 6))
        second_column_group = [33, 34, 35]  # Select additional data columns

        # Rename columns for better readability
        data.rename(columns={'CH4_Conv_%': 'CH4 Conv. (%)', 'Total_kWh/kq_H2': 'Total kWh/kg H2',
                             'Carbon_Prod._(g/min)': 'Carbon Prod. (g/min)'}, inplace=True)

        # Plot the additional metrics
        for col_index in second_column_group:
            plt.plot(data.iloc[:, 0] / 60, data.iloc[:, col_index], label=data.columns[col_index])

        plt.ylim(0, 100)  # Set y-axis limits
        plt.legend()
        plt.xlabel('Time (min)')
        plt.ylabel('Metrics')

        plt.show()  # Display the second plot


def PlotGroup(exp):
    """
    Plots a group of related variables from the dataset for multiple experiment runs.

    Args:
        exp (list): List of experiment identifiers.
    """
    global col_names, addons, addons_list, func_list, column_groups
    global linestyle, axis_dct

    group = grp_dropdown.value
    for input_string in exp:
        # Read experiment data
        data = ReadData(input_string)
        if data is None:
            break  # Exit loop if no valid data

        data = TrimData(data, 12)  # Trim the dataset
        axis_values = axis_dct.get(input_string, {})  # Get axis limits safely

        try:
            variable_group = column_groups[col_names.index(group)]
            if len(exp) > 1:
                legend_label = f"{data.columns[variable_group]}_{input_string}"
            else:
                legend_label = data.columns[variable_group]
            # Plot data
            plt.plot(data.iloc[:, 0] / 60, data.iloc[:, variable_group],
                     label=legend_label, linestyle=linestyle[exp.index(input_string)])
        except KeyError:
            # If a custom option is used
            plt.plot(data.iloc[:, 0] / 60, data[group],
                     linestyle=linestyle[exp.index(input_string)])

    # Apply additional customizations from addons
    for i in range(len(addons)):
        if addons[addons_list[i]]["active"]:
            func = func_list[i]
            func(addons[addons_list[i]]["text"])

    # Apply graph settings if data exists
    if data is not None:
        plt.legend()
        plt.xlabel('Time (min)')
        plt.ylabel(group)

        # Set axis limits safely
        if axis_values.get('xlow') and axis_values.get('xhigh'):
            plt.xlim(float(axis_values['xlow']) * 2000, float(axis_values['xhigh']) * 2000)
        if axis_values.get('ylow') and axis_values.get('yhigh'):
            plt.ylim(float(axis_values['ylow']) * 2000, float(axis_values['yhigh']) * 2000)

        print(group, exp)
        plt.show()


def PlotSingleVar(exp):
    """
    Plots a single variable across multiple experiment runs.

    Args:
        exp (list): List of experiment identifiers.
    """
    global col_names, addons, addons_list, func_list, group
    global column_groups, column_keys, linestyle, axis_dct

    for input_string in exp:
        # Read experiment data
        data = ReadData(input_string)
        if data is None:
            break  # Exit loop if no valid data

        data = TrimData(data, 12)  # Trim dataset
        axis_values = axis_dct.get(input_string, {})  # Get axis limits safely
        legend_label = input_string  # Default legend label

        try:
            variable_index = column_keys.index(one_var_dropdown.value)
            # Plot data
            plt.plot(data.iloc[:, 0] / 60, data.iloc[:, variable_index],
                     label=legend_label, linestyle=linestyle[exp.index(input_string)])
        except KeyError:
            # If a custom option is used
            plt.plot(data.iloc[:, 0] / 60, data[single_variable],
                     label=legend_label, linestyle=linestyle[exp.index(input_string)])

    # Apply additional customizations from addons
    for i in range(len(addons)):
        if addons[addons_list[i]]["active"]:
            func = func_list[i]
            func(addons[addons_list[i]]["text"])

    # Apply graph settings if data exists
    if data is not None:
        if len(exp) > 1:
            plt.legend()
        plt.xlabel('Time (min)')
        plt.ylabel(one_var_dropdown.value)

        # Set axis limits safely
        if axis_values.get('xlow') and axis_values.get('xhigh'):
            plt.xlim(float(axis_values['xlow']) * 2000, float(axis_values['xhigh']) * 2000)
        if axis_values.get('ylow') and axis_values.get('yhigh'):
            plt.ylim(float(axis_values['ylow']) * 2000, float(axis_values['yhigh']) * 2000)

        print(one_var_dropdown.value, exp)
        plt.show()


def PlotInteractive(exp):
  global col_names, addons, addons_list, func_list, group, single_variable, column_groups, column_keys
  global selected_points

  input_string = exp[0]  # Only taking the first experiment for now
  data = ReadData(input_string)  # Read data
  data = TrimData(data, 12)  # Trim initial points

  variable_index = column_keys.index(single_variable)  # Get variable index

  # Convert x (time) and y (data values) while handling NaNs
  x = np.nan_to_num(data.iloc[:, 0] / 60, nan=0)
  y = np.nan_to_num(data.iloc[:, variable_index], nan=0)

  # Create an interactive Plotly figure
  fig = go.FigureWidget([go.Scatter(x=x, y=y, mode='lines+markers',
                                    marker=dict(opacity=0), name=single_variable,
                                    showlegend=True, legendrank=0)],
                        layout=go.Layout(height=500, width=800,
                                         yaxis={'title':group},
                                         xaxis={'title':'Time'}))

  # UI elements for interactivity
  reset_button = widgets.Button(description="Reset Selected")
  remove_lines_button = widgets.Button(description="Remove Lines")
  smooth_button = widgets.Button(description="Smooth", button_style="Danger")
  derivative_button = widgets.Button(description="Derivative", button_style="Info")
  integral_button = widgets.Button(description="Integral", button_style="Success")
  integral_text = widgets.Label(value="Integral:")
  linearize_button = widgets.Button(description="Linearize", button_style="Warning", disabled=True)
  sigma_text = widgets.Text(description="Sigma")

  avg_display = widgets.Label(value="Average:")
  std_display = widgets.Label(value="Standard Deviation:")
  value_display = widgets.VBox([smooth_button, derivative_button,
                                widgets.HBox([integral_button, integral_text]),
                                linearize_button, sigma_text, avg_display, std_display])

  selected_points = np.array([])  # Store user-selected points

  # Function to capture selected points on plot
  def select_points(trace, points, selector):
    global selected_points
    selected_points = np.array([(x[i], y[i]) for i in points.point_inds])
    calculate_average()
    calculate_std()

  # Attach selection event to Plotly figure
  fig.data[0].on_selection(select_points)

  # Calculate and display average
  def calculate_average():
    global selected_points
    avg_y = np.mean(selected_points[:, 1]) if np.any(selected_points) else np.mean(data.iloc[:, variable_index])
    avg_display.value = f"Average: {avg_y:.2f}"

  # Calculate and display standard deviation
  def calculate_std():
    global selected_points
    std_y = np.std(selected_points[:, 1], ddof=1) if np.any(selected_points) else np.std(data.iloc[:, variable_index], ddof=1)
    std_display.value = f"Standard Deviation: {std_y:.2f}"

  # Apply Gaussian smoothing
  def gaussian_smooth(x, y, sigma=10):
    y_smoothed = gaussian_filter1d(y, sigma=sigma)
    fig.data = tuple(trace for trace in fig.data if trace.name != 'Smoothed')
    fig.add_trace(go.Scatter(x=x, y=y_smoothed, mode='lines', name='Smoothed', legendrank=1))
    return x, y_smoothed

  # Compute numerical derivative
  def calc_derivative(x, y):
    x_data, y_data = gaussian_smooth(x, y, sigma=10)
    dy_dx = np.gradient(y_data, x_data)
    fig.data = tuple(trace for trace in fig.data if trace.name != 'Derivative')
    fig.add_trace(go.Scatter(x=x_data, y=dy_dx, mode='lines', name='Derivative', legendrank=2))

  # Reset selected points
  def reset_selected_points(b):
    global selected_points
    selected_points = []
  reset_button.on_click(reset_selected_points)

  # Remove additional lines from the plot
  def remove_lines(b):
    fig.data = tuple(trace for trace in fig.data if trace.name not in ['Smoothed', 'Derivative', 'Integral'])
  remove_lines_button.on_click(remove_lines)

  # Display figure and controls
  figure = widgets.HBox([fig, value_display])
  display(figure, widgets.HBox([reset_button, remove_lines_button]))


def PlotDATAQ(exp):
  """
  DATAQ NOTES:
  COL0: Relative Time
  COL1: Forward Power
  COL2: Reverse Power
  COL3: Event Marker Type (2 - DATAQ Button, 3 - Start/Stop)
  COL4: Event Marker Number
  COL5: Unknown
  COL6: Date
  COL7: ABS Time
  """
  global linestyle, axis_dct
  textbox_lst = []
  power_data = []
  output = []
  cond_time_lst = []
  fig, axs = plt.subplots(3, 1, figsize=(16, 10))
  for input_string in exp:
    # For multiple plots
    data = ReadData(input_string, header=None)
    if data is None:
      break

    # data = TrimData(data, 1)
    try:
      axis_values = axis_dct[input_string]
    except:
      axis_values = axis_dct[input_string.replace('DATAQ', "")]

    time = data[0]/60
    fp = 1.4062*data[1]**1.9174
    rp = 1.3322*data[2]**1.8911
    em = data[3]

    l_range = fp.loc[fp >= 1].index.min()-4000 if fp.loc[fp >= 1].index.min() > 4000 else 0
    u_range = fp.loc[fp >= 1].index.max()+4000

    if type(em.loc[0]) == np.int64:
      try:
        cond_time = em.loc[em == 2].index[1]/120000 - time[l_range]
      except:
        cond_time = 0
    else:
      cond_time = 0
    cond_time_lst.append(cond_time)

    if axis_values['xlow'] != "" and axis_values['xhigh'] != "":
      u_range_fr = float(axis_values['xhigh'])*2000 + l_range
      l_range_fr = float(axis_values['xlow'])*2000 + l_range
    else:
      u_range_fr = u_range
      l_range_fr = l_range

    realline = True
    for run in axis_dct.keys():
      if axis_dct[run]['xlow'] != "" or axis_dct[run]['xhigh'] != "":
        realline = False
        break
    if realline:
      time_fr = time - time[l_range] - cond_time + cond_time_lst[0]
      time1 = time_fr
    else:
      time_fr = time - time[l_range_fr]
      time1 = time - time[l_range]

    if axis_values['ylow'] != "" and axis_values['yhigh'] != "":
      ylimFP = [float(axis_values['ylow']), float(axis_values['yhigh'])]
    else:
      ylimFP = (3, 6)
    if axis_values['y2low'] != "" and axis_values['y2high'] != "":
      ylimRP = [float(axis_values['y2low']), float(axis_values['y2high'])]
    else:
      ylimRP = (0, 0.5)

    axs[0].plot(time1.loc[l_range:u_range], fp.loc[l_range:u_range],
                label=input_string + ' FP')
    axs[0].plot(time1.loc[l_range:u_range], rp.loc[l_range:u_range],
                label=input_string + ' RP')
    axs[1].plot(time_fr.loc[l_range_fr:u_range_fr], fp.loc[l_range_fr:u_range_fr],
                label=input_string + ' FP')
    axs[2].plot(time_fr.loc[l_range_fr:u_range_fr], rp.loc[l_range_fr:u_range_fr],
                label=input_string + ' RP')

    start_text = widgets.Text()
    end_text = widgets.Text()
    textbox_lst.append([start_text, end_text])
    output.append(widgets.Label())

  if plot_events.value and type(em.loc[0]) == np.int64:
    breakpoints = em.loc[em > 0].index[2:]/120000
    for line in breakpoints:
      if line <= u_range/120000 and line >= l_range/120000:
        axs[0].axvline(x=line, color='orange', linestyle='--')
      if line <= u_range_fr/120000 and line >= l_range_fr/120000:
        axs[1].axvline(x=line, color='orange', linestyle='--')
        axs[2].axvline(x=line, color='orange', linestyle='--')

  if cond_time_lst[0] != 0 and axis_values['xlow'] == "" and axis_values['xhigh'] == "":
    if cond_time_lst[0] <= u_range/120000 and cond_time_lst[0] >= l_range/120000:
      axs[0].axvline(cond_time_lst[0], color='r', linestyle='--')
    if cond_time_lst[0] <= u_range_fr/120000 and cond_time_lst[0] >= l_range_fr/120000:
      axs[1].axvline(cond_time_lst[0], color='r', linestyle='--')
      axs[2].axvline(cond_time_lst[0], color='r', linestyle='--')
  axs[0].legend(loc='upper right')
  axs[0].set_xlabel('Time (min)')
  axs[0].set_ylabel('Power (kW)')
  # axs[0].set_xlim(0, (u_range-l_range)/120000)

  axs[1].legend(loc='upper right')
  axs[1].set_ylim(ylimFP[0], ylimFP[1])
  axs[1].set_xlabel('Time (min)')
  axs[1].set_ylabel('Power (kW)')
  # axs[1].set_xlim(l_range_fr/120000, u_range_fr/120000)

  axs[2].legend(loc='upper right')
  axs[2].set_ylim(ylimRP[0], ylimRP[1])
  axs[2].set_xlabel('Time (min)')
  axs[2].set_ylabel('Power (kW)')
  # axs[2].set_xlim(l_range_fr/120000, u_range_fr/120000)
  plt.show()

  calc_button = widgets.Button(description='Calculate')
  def calc_avg_std(b):
    for i in range(len(textbox_lst)):
      start_index = int(textbox_lst[i][0].value)*2000
      end_index = int(textbox_lst[i][1].value)*2000
      fp_mean = np.mean(power_data[i][0][start_index:end_index])
      rp_mean = np.mean(power_data[i][1][start_index:end_index])
      fp_std = np.std(power_data[i][0][start_index:end_index])
      rp_std = np.std(power_data[i][1][start_index:end_index])
      output[i].value = f'Average: {fp_mean}, {rp_mean} Standard Deviation: {fp_std}, {rp_std}'
  calc_button.on_click(calc_avg_std)

  for i in range(len(textbox_lst)):
    display(widgets.VBox([widgets.Label(value=str(i)), widgets.VBox(textbox_lst[i])]))
    display(output[i])
  display(calc_button)

def PlotRun_W_DATAQ(exp):
  PlotRun(exp)
  DATAQ_exp = []
  for experiment in exp:
    exp_split = experiment.split("_")
    DATAQ_exp.append(exp_split[0]+'_DATAQ'+exp_split[1])
  print(DATAQ_exp)
  PlotDATAQ(DATAQ_exp)

def PlotCustom(exp):
  """
  Choose any variable from any of the data files to plot against time (for now)
  Y axis 1 or Y axis 2
  Choose time range for each variable and run individually
  Choose y axis range for each axis individually
  """
  fig, axs = plt.subplots(2, 1, figsize=(16, 10))

  data = ReadData(exp[0])

  trimmed_data = TrimData(data, 12)

  if axis == 1:
    axs[0].plot()
  elif axis == 2:
    axs[1].plot()

  plt.show()

def reset_ui():
  clear_output(wait=True)
  display(chooser)
  display(confirm_button)
  display(box)

def format_string(s):
  """Formats the string to ensure:
  - The prefix (letters) is uppercase.
  - The underscore and 'Run' part remain properly formatted.
  - Removes extra spaces while keeping comma separation.
  - Retains any trailing letter at the end.
  """
  parts = [part.strip() for part in s.split(",")]  # Split by comma and strip spaces
  formatted_parts = []
  for part in parts:
    match = re.match(r"([a-zA-Z]+)(\d+)_([a-zA-Z]+)(\d+)([a-zA-Z]?)", part)  # Match pattern with optional trailing letter
    if match:
      formatted_parts.append(f"{match.group(1).upper()}{match.group(2)}_{match.group(3).capitalize()}{match.group(4)}{match.group(5).upper()}")
    else:
      formatted_parts.append(part)  # Keep unchanged if it doesn't match
  return ", ".join(formatted_parts)  # Join back with a comma and space

def run():
    # Toggles between Lab A and Lab B
    lab_toggle = widgets.ToggleButton(description="Lab A", button_style="danger")
    def update_lab_text(b):
        global column_groups, column_keys, column_groups_A, column_keys_A, column_groups_B, column_keys_B, lab
        lab_toggle.description = "Lab B" if b['new'] else "Lab A"
        lab_toggle.button_style = "info" if b['new'] else "danger"
        lab = "B" if b['new'] else "A"
        column_groups = column_groups_B if b['new'] else column_groups_A
        column_keys = column_keys_B if b['new'] else column_keys_A

        if lab == "A":
            grp_dropdown.options = col_names
        elif lab == "B":
            grp_dropdown.options = col_names[:5]
        get_one_var_options()
    lab_toggle.observe(update_lab_text, names='value')

    # Dropdown for selecting plotting function
    plot_select = widgets.Dropdown(options=["PlotRun_W_DATAQ", "PlotRun",
                                            "PlotGroup", "PlotSingleVar",
                                            "PlotInteractive", "PlotDATAQ",
                                            "PlotCustom"],
                                        description="Function",
                                        style={'description_width': 'initial'})
    def on_plot_select_change(b):
        # Updates which widgets to display based on plotting function
        if b["new"] == "PlotRun_W_DATAQ":
            grp_dropdown.layout.display = "none"
            one_var_dropdown.layout.display = "none"
            addon_tab.layout.display = "block"
            xaxis_label.layout.display = "block"
            xaxis_low.layout.display = "block"
            xaxis_high.layout.display = "block"
            yaxis_label.layout.display = "block"
            yaxis_label.value = "Y Axis"
            yaxis_low.layout.display = "block"
            yaxis_high.layout.display = "block"
            yaxis2_label.value = "RP Y Axis"
            yaxis2_label.layout.display = "block"
            yaxis2_low.layout.display = "block"
            yaxis2_high.layout.display = "block"
            custom_tab.layout.display = "none"
        elif b["new"] == "PlotRun":
            grp_dropdown.layout.display = "none"
            one_var_dropdown.layout.display = "none"
            addon_tab.layout.display = "none"
            custom_tab.layout.display = "none"
        elif b["new"] == "PlotGroup":
            grp_dropdown.layout.display = "block"
            one_var_dropdown.layout.display = "none"
            addon_tab.layout.display = "block"
            xaxis_label.layout.display = "block"
            xaxis_low.layout.display = "block"
            xaxis_high.layout.display = "block"
            yaxis_label.layout.display = "block"
            yaxis_label.value = "Y Axis"
            yaxis_low.layout.display = "block"
            yaxis_high.layout.display = "block"
            yaxis2_label.layout.display = "none"
            yaxis2_low.layout.display = "none"
            yaxis2_high.layout.display = "none"
            custom_tab.layout.display = "none"
        elif b["new"] == "PlotSingleVar":
            grp_dropdown.layout.display = "block"
            one_var_dropdown.layout.display = "block"
            addon_tab.layout.display = "block"
            xaxis_label.layout.display = "block"
            xaxis_low.layout.display = "block"
            xaxis_high.layout.display = "block"
            yaxis_label.layout.display = "block"
            yaxis_label.value = "Y Axis"
            yaxis_low.layout.display = "block"
            yaxis_high.layout.display = "block"
            yaxis2_label.layout.display = "none"
            yaxis2_low.layout.display = "none"
            yaxis2_high.layout.display = "none"
            custom_tab.layout.display = "none"
        elif b["new"] == "PlotInteractive":
            grp_dropdown.layout.display = "block"
            one_var_dropdown.layout.display = "block"
            addon_tab.layout.display = "block"
            xaxis_label.layout.display = "block"
            xaxis_low.layout.display = "block"
            xaxis_high.layout.display = "block"
            yaxis_label.layout.display = "block"
            yaxis_label.value = "Y Axis"
            yaxis_low.layout.display = "block"
            yaxis_high.layout.display = "block"
            yaxis2_label.layout.display = "none"
            yaxis2_low.layout.display = "none"
            yaxis2_high.layout.display = "none"
            custom_tab.layout.display = "none"
        elif b["new"] == "PlotDATAQ":
            grp_dropdown.layout.display = "none"
            one_var_dropdown.layout.display = "none"
            addon_tab.layout.display = "block"
            xaxis_label.layout.display = "block"
            xaxis_low.layout.display = "block"
            xaxis_high.layout.display = "block"
            yaxis_label.layout.display = "block"
            yaxis_label.value = "FP Y Axis"
            yaxis_low.layout.display = "block"
            yaxis_high.layout.display = "block"
            yaxis2_label.layout.display = "block"
            yaxis2_low.layout.display = "block"
            yaxis2_high.layout.display = "block"
            custom_tab.layout.display = "none"
        elif b["new"] == "PlotCustom":
            grp_dropdown.layout.display = "none"
            one_var_dropdown.layout.display = "none"
            addon_tab.layout.display = "none"
            extra_tab.layout.display = "block"
            custom_tab.layout.display = "block"
        try:
            update_run_options(exp_dropdown.value)
        except:
            pass
    plot_select.observe(on_plot_select_change, names='value')

    # Combobox for selecting experiment
    exp_dropdown = widgets.Combobox(options=[], description='Experiment',
                                    style={'description_width': 'initial'})
    def on_exp_dropdown_change(b):
    # Updates run dropdown options when a new experiment is selected
        update_run_options(b['new'])
    exp_dropdown.observe(on_exp_dropdown_change, names='value')

    # Dropdown for selecting group when using PlotGroup, PlotSingleVar, or PlotInteractive
    grp_dropdown = widgets.Dropdown(options=col_names, description='Group')
    def on_grp_dropdown_change(b):
        get_one_var_options()
    grp_dropdown.observe(on_grp_dropdown_change, names='value')

    # Dropdown for selecting group when using PlotSingleVar or PlotInteractive
    one_var_dropdown = widgets.Dropdown(description="Variable")

    def get_one_var_options():
        # Updates options single variable options when group is changed
        global column_groups, column_keys, col_names, one_var_col_options, one_var_options
        one_var_col_options = column_groups[col_names.index(grp_dropdown.value)]
        one_var_options = []
        for option in one_var_col_options:
            one_var_options.append(column_keys[option])
        one_var_dropdown.options = one_var_options
    get_one_var_options()

    separate_graphs = widgets.Checkbox(description="Separate Graphs")

    run_dropdown = widgets.Dropdown(options=[], description='Run',
                                    style={'description_width': 'initial'})

    add_run_button = widgets.Button(description="Add Run", layout=widgets.Layout(width='12%'))
    def on_add_run_button_click(b):
        global experiments, axis_dct
        extra_tab.layout.display = "block"
        exp = exp_dropdown.value.split(" ")
        run = exp[0] + exp[1] + "_" + run_dropdown.value
        if run not in experiments:
            axis_dct[run] = {'xlow':"", 'xhigh':"",
                            'ylow':"", 'yhigh':"",
                            'y2low':"", 'y2high':""}
            run_tags.options = list(run_tags.options) + [run]
            experiments.append(run)
    add_run_button.on_click(on_add_run_button_click)
    run_tags = widgets.ToggleButtons(options=[], style={"button_width": "auto"},
                                    button_style='warning')
    def on_run_tag_change(b):
        global axis_dct, run_tag_change
        run_tag_change = True
        try:
            run = b['new']
            xaxis_low.value = axis_dct[run]['xlow']
            xaxis_high.value = axis_dct[run]['xhigh']
            yaxis_low.value = axis_dct[run]['ylow']
            yaxis_high.value = axis_dct[run]['yhigh']
            yaxis2_low.value = axis_dct[run]['y2low']
            yaxis2_high.value = axis_dct[run]['y2high']
        except:
            xaxis_low.value = ""
            xaxis_high.value = ""
            yaxis_low.value = ""
            yaxis_high.value = ""
            yaxis2_low.value = ""
            yaxis2_high.value = ""
        run_tag_change = False
    run_tags.observe(on_run_tag_change, names='value')

    shift_left_button = widgets.Button(description="<--", layout=widgets.Layout(width='60px'))
    def move_tag_left(b):
        lst = list(run_tags.options)
        index = run_tags.options.index(run_tags.value)
        new_index = index - 1
        new_index = max(0, min(new_index, len(lst) - 1))  # Ensure within bounds
        
        item = lst.pop(index)  # Remove item from current position
        lst.insert(new_index, item)  # Insert item at new position
        run_tags.options = lst
        run_tags.value = item
    shift_left_button.on_click(move_tag_left)

    shift_right_button = widgets.Button(description="-->", layout=widgets.Layout(width='60px'))
    def move_tag_right(b):
        lst = list(run_tags.options)
        index = run_tags.options.index(run_tags.value)
        new_index = index + 1
        new_index = max(0, min(new_index, len(lst) - 1))  # Ensure within bounds
        
        item = lst.pop(index)  # Remove item from current position
        lst.insert(new_index, item)  # Insert item at new position
        run_tags.options = lst
        run_tags.value = item
    shift_right_button.on_click(move_tag_right)

    remove_run_tag_button = widgets.Button(description="Remove Run", layout=widgets.Layout(width='120px'))
    def remove_run_tag(b):
        global experiments, axis_dct
        if run_tags.options != ():
            experiments.remove(run_tags.value)
            axis_dct.pop(run_tags.value)
            run_tags.options = [option for option in run_tags.options if option != run_tags.value]
        if run_tags.options == ():
            extra_tab.layout.display = "none"
    remove_run_tag_button.on_click(remove_run_tag)

    def update_run_options(experiment):
        run_dropdown.disabled = False
        try:
            directory = selected_directory + "/" + experiment + "/Data"
            if plot_select.value == "PlotDATAQ":
                runs = sorted([f for f in os.listdir(directory) if f.endswith('.csv') and f.startswith('DATAQ')])
            else:
                runs = sorted([f for f in os.listdir(directory) if f.endswith('.csv') and not f.startswith('DATAQ')])
        except:
            # Displays message when there are no csv files in folder
            runs = ["No CSV files found"]
            run_dropdown.disabled = True
        run_dropdown.options = runs

    vert_line_box = addon_checkbox("Vertical Line")
    vert_line_text = addon_text("Vertical Line")
    horz_line_box = addon_checkbox("Horizontal Line")
    horz_line_text = addon_text("Horizontal Line")

    line_tab = widgets.VBox([
        widgets.HBox([vert_line_box, vert_line_text]),
        widgets.HBox([horz_line_box, horz_line_text])
        ])

    textbox_size = "200px"
    xaxis_label = widgets.Label(value="X Axis")
    xaxis_low = widgets.Text(placeholder="Lower Limit",
                            layout=widgets.Layout(width=textbox_size))
    xaxis_high = widgets.Text(placeholder="Upper Limit",
                            layout=widgets.Layout(width=textbox_size))
    yaxis_label = widgets.Label(value="FP Y Axis")
    yaxis_low = widgets.Text(placeholder="Lower Limit",
                            layout=widgets.Layout(width=textbox_size))
    yaxis_high = widgets.Text(placeholder="Upper Limit",
                            layout=widgets.Layout(width=textbox_size))
    yaxis2_label = widgets.Label(value="RP Y Axis")
    yaxis2_low = widgets.Text(placeholder="Lower Limit",
                            layout=widgets.Layout(width=textbox_size))
    yaxis2_high = widgets.Text(placeholder="Upper Limit",
                            layout=widgets.Layout(width=textbox_size))
    def on_axis_text_change(b):
        global axis_dct, run_tag_change
        if not run_tag_change:
            run = run_tags.value
            axis_dct[run] = {'xlow':xaxis_low.value, 'xhigh':xaxis_high.value,
                                        'ylow':yaxis_low.value, 'yhigh':yaxis_high.value,
                                        'y2low':yaxis2_low.value, 'y2high':yaxis2_high.value}
    xaxis_low.observe(on_axis_text_change, names='value')
    xaxis_high.observe(on_axis_text_change, names='value')
    yaxis_low.observe(on_axis_text_change, names='value')
    yaxis_high.observe(on_axis_text_change, names='value')
    yaxis2_low.observe(on_axis_text_change, names='value')
    yaxis2_high.observe(on_axis_text_change, names='value')

    axis_tab = widgets.GridBox(children=[xaxis_label, xaxis_low, xaxis_high,
                                        yaxis_label, yaxis_low, yaxis_high,
                                        yaxis2_label, yaxis2_low, yaxis2_high],
                                layout=widgets.Layout(width='80%',
                                grid_template_rows='auto auto auto',
                                grid_template_columns='15% 40% 40%',
                                grid_template_areas='''
                                "xaxis_label xaxis_low, xaxis_high"
                                "yaxis_label yaxis_low, yaxis_high"
                                "yaxis2_label yaxis2_low, yaxis2_high"
                                '''))

    plot_events = widgets.Checkbox(description='Plot Events')

    addon_tab = widgets.VBox([axis_tab, plot_events])
    trace_var_dropdowns = []
    trace_buttons = widgets.ToggleButtons(options=[])
    add_trace = widgets.Button(description='Add Trace')
    def on_add_trace_button_click(b):
        exp = exp_dropdown.value.split(" ")
        run = exp[0] + exp[1] + "_" + run_dropdown.value
        if run not in experiments:
            trace_buttons.options = list(run_tags.options) + [run]

        folder_name = "Data"
        matching_folders, formatted_second_part = FindFolder([exp[0]+exp[1], run_dropdown.value])

        if matching_folders:
            # Assume the first match is the correct folder
            target_folder = matching_folders[0]
            print(f"Opening folder: {target_folder}")

            data_folder = Path(target_folder) / folder_name
            print(formatted_second_part)
            data_path = Path(data_folder) / formatted_second_part  # Construct full file path
            lab_view_options = pd.read_csv(data_path, nrows=0).columns.tolist()
        
        dataq_options = ['DATAQ FP', 'DATAQ RP']
        trace_var_options = lab_view_options + dataq_options
        trace_var_dropdowns.append(widgets.Dropdown(options=trace_var_options))
    widgets.HBox(trace_var_dropdowns).layout.display = "block"

    add_trace.on_click(on_add_trace_button_click)

    custom_tab = widgets.VBox([trace_buttons, widgets.HBox(trace_var_dropdowns)])

    plot_confirm_button = widgets.Button(description="Confirm", button_style="success")
    def on_plot_confirm_button_click(b):
        global experiments
        if separate_graphs.value:
            for experiment in experiments:
                exp = [experiment]
                eval(f"{plot_select.value}(exp)")
        elif experiments == []:
            exp = exp_dropdown.value.split(" ")
            run = [exp[0] + exp[1] + "_" + run_dropdown.value]
            eval(f"{plot_select.value}(run)")
        else:
            eval(f"{plot_select.value}(experiments)")
    plot_confirm_button.on_click(on_plot_confirm_button_click)

    clear_output_button = widgets.Button(description="Clear Output", button_style="danger")
    def on_clear_output_button_click(b):
        reset_ui()
    clear_output_button.on_click(on_clear_output_button_click)

    # Display Section
    plot_tab = widgets.VBox([
        lab_toggle,
        widgets.HBox([plot_select, separate_graphs]),
        widgets.HBox([exp_dropdown]),
        widgets.HBox([run_dropdown, add_run_button, add_trace]),
        widgets.HBox([grp_dropdown]),
        widgets.HBox([one_var_dropdown]),
        widgets.HBox([plot_confirm_button, clear_output_button])
        ])

    extra_tab = widgets.VBox([run_tags, addon_tab, 
                            widgets.HBox([remove_run_tag_button, shift_left_button, shift_right_button]),])
                            # custom_tab])

    # super_tab = widgets.VBox([lab_toggle, widgets.HBox([plot_tab, extra_tab])])

    box = widgets.GridBox(children=[plot_tab, extra_tab],
                        layout=widgets.Layout(
                        width='100%',
                        grid_template_rows='auto',
                        grid_template_columns='50% 50%',
                        grid_template_areas='''
                        "plot_tab extra_tab"
                        '''))

    # Set all widget displays to none
    plot_tab.layout.display = "none"
    lab_toggle.layout.display = "none"
    grp_dropdown.layout.display = "none"
    one_var_dropdown.layout.display = "none"
    addon_tab.layout.display = "block"
    extra_tab.layout.display = "none"
    # custom_tab.layout.display = "block"

    # Reset textbox UI to allign
    xaxis_label.layout.display = "none"
    xaxis_low.layout.display = "none"
    xaxis_high.layout.display = "none"
    yaxis_label.layout.display = "none"
    yaxis_low.layout.display = "none"
    yaxis_high.layout.display = "none"
    yaxis2_label.layout.display = "none"
    yaxis2_low.layout.display = "none"
    yaxis2_high.layout.display = "none"
    xaxis_label.layout.display = "block"
    xaxis_low.layout.display = "block"
    xaxis_high.layout.display = "block"
    yaxis_label.layout.display = "block"
    yaxis_low.layout.display = "block"
    yaxis_high.layout.display = "block"
    yaxis2_label.layout.display = "block"
    yaxis2_low.layout.display = "block"
    yaxis2_high.layout.display = "block"

    display(box)