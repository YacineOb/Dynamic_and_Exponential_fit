#########################################################################
# Import Libraries ######################################################
#########################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from matplotlib import rcParams, cycler

#########################################################################
# Aesthetic matplotlib parameters #######################################
#########################################################################
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.size'] = 30
rcParams['font.weight'] = 'bold'
rcParams['axes.linewidth'] = 3
rcParams['axes.labelpad'] = 10.0
rcParams["axes.labelweight"] = "bold"
plot_color_cycle = cycler('color', ['000000', '0000FE', 'FE0000', '008001', 'FD8000', '8c564b',
                                    'e377c2', '7f7f7f', 'bcbd22', '17becf'])
rcParams['axes.prop_cycle'] = plot_color_cycle
rcParams['axes.xmargin'] = 0
rcParams['axes.ymargin'] = 0
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams.update({"figure.figsize": (6.4, 4.8),
                 "figure.subplot.left": 0.177, "figure.subplot.right": 0.946,
                 "figure.subplot.bottom": 0.156, "figure.subplot.top": 0.965,
                 "axes.autolimit_mode": "round_numbers",
                 "xtick.major.size": 7,
                 "xtick.minor.size": 3.5,
                 "xtick.major.width": 1.1,
                 "xtick.minor.width": 1.1,
                 "xtick.major.pad": 5,
                 "xtick.minor.visible": True,
                 "ytick.major.size": 7,
                 "ytick.minor.size": 3.5,
                 "ytick.major.width": 1.1,
                 "ytick.minor.width": 1.1,
                 "ytick.major.pad": 5,
                 "ytick.minor.visible": True,
                 "lines.markersize": 5,
                 "lines.markerfacecolor": "none",
                 "lines.markeredgewidth": 0.8})

#########################################################################
# Read document and get data ############################################
#########################################################################

# Choose the document(s) you want to read ###############################
Document = 'Young.xlsx'  # Path to your file. You can use data from different excel workbook

# Read the first document ###############################################
df = pd.read_excel(Document, sheet_name='young', index_col=0)  # First is the path, second is the sheet name
Rawdata = df.fillna(0)  # replace missing values with "NaN". Prevent errors or wrong results
data_array = Rawdata.to_numpy()  # Transform dataframe into array for later manipulations
data = data_array.mean(axis=0)  # Take the mean from each column
datastd = np.std(df, axis=0)  # Calculate the standard deviation (SD) from each column
dataSEM = stats.sem(df, axis=0)  # Calculate the standard error of mean (SEM) from each column

# Read the second document ############################################
df1 = pd.read_excel(Document, sheet_name='old', index_col=0)  # Same as above, for second document
Rawdata1 = df1.fillna(0)
data_array1 = Rawdata1.to_numpy()
data1 = data_array1.mean(axis=0)
datastd1 = np.std(df1, axis=0)
dataSEM1 = stats.sem(df1, axis=0)

# # Read the third document #############################################
# df2 = pd.read_excel(Document, sheet_name='20-24', index_col=0)  # Same as above
# Rawdata2 = df2.fillna(0)
# data_array2 = Rawdata2.to_numpy()
# data2 = data_array2.mean(axis=0)
# datastd2 = np.std(df2, axis=0)
# dataSEM2 = stats.sem(df2, axis=0)
#
# # Read the fourth document ############################################
# df3 = pd.read_excel(Document, sheet_name='Old', index_col=0)  # Same as above
# Rawdata3 = df3.fillna(0)
# data_array3 = Rawdata3.to_numpy()
# data3 = data_array3.mean(axis=0)
# datastd3 = np.std(df3, axis=0)
# dataSEM3 = stats.sem(df3, axis=0)

#########################################################################
# Plot data #############################################################
#########################################################################

# Define the color of the markers #######################################
marker_param_1 = {'ecolor': 'blue'}  # blue
marker_param_2 = {'ecolor': 'green'}  # green
marker_param_3 = {'ecolor': 'magenta'}  # magenta
marker_param_4 = {'ecolor': 'red'}  # red

# Create a figure
plt.figure(figsize=(10, 8))

# Plot first set of data
plt.errorbar(np.arange(0, len(data), 1), data, yerr=dataSEM, color='b',
             marker='o', markeredgecolor='b', markerfacecolor='none', capsize=3, elinewidth=1,
             markeredgewidth=1.3, **marker_param_1, label='young')  # The label is here

# Plot second set of data
# plt.errorbar(np.arange(0, len(data1), 1), data1, yerr=dataSEM1, color='g',
#              marker='o', markeredgecolor='g', markerfacecolor='none', capsize=3, elinewidth=1,
#              markeredgewidth=1.3, **marker_param_2, label='old')  # The label is here

# # Plot third set of data
# plt.errorbar(np.arange(0, len(data2), 1), data2, yerr=dataSEM2, color='m',
#              marker='o', markeredgecolor='m', markerfacecolor='none', capsize=3, elinewidth=1,
#              markeredgewidth=1.3, **marker_param_3, label='20-24')  # The label is here
#
# Plot fourth set of data
plt.errorbar(np.arange(0, len(data1), 1), data1, yerr=dataSEM1, color='r',
             marker='o', markeredgecolor='r', markerfacecolor='none', capsize=3, elinewidth=1,
             markeredgewidth=1.3, **marker_param_4, label='Aged')  # The label is here

# Additional details #####################################################
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)  # draw a dashed line at y=0
plt.legend(fontsize=20, frameon=False)
plt.ylabel('Depolarization (mV)', fontsize=40)
plt.xlabel('Time (s)', fontsize=40)
plt.axis([-1, 45, 0, 40])
plt.tight_layout()
plt.savefig(os.path.join(os.environ["HOMEPATH"], "Desktop") + '\\YourFigureName.png', dpi=300)
plt.show()


#########################################################################
# Exponential fitting plot ##############################################
#########################################################################

# defining the exponential equation for fitting
def func(x, a, b, c):
    """Basic exponential equation. You might have to adapt it if you want to look at specific aspects.
    ONE IMPORTANT POINT, you have to provide a "guess" to the algorithm.
    The exponential may have multiple solutions. By providing a guess you just tell the algorithm
    to start looking around your guessed value. It saves time and gives more accurate/reproducible results.
    Guesses are based on values from papers, e.g. we know that the CA1 membrane time constant should be around 15 ms.
    If you have no idea, set your guesses to '1'.
    The exponential fitting might (will) display many warning messages. No worries, they are mostly informative.
    If your equation is unsolvable with the algorithm, a warning message will tell you that the "maxfev" reached the
    maximum allocated iterations. You can increase the number of iteration (maxfev, the easiest) or change your
    strategy e.g. linearize your data.
    a is the initial condition,
    x is the abscissa, usually time,
    c is the constant,
    b is what we are interested in: the time constant or the growth rate"""

    return a * np.exp(-b * x) + c


# Choose which part of the data you would like to fit
"""If your data is not very noisy and doesn't have multiple components, you may succeed to fit everything. 
Otherwise try to fit the exponential part of the data you are interested in."""
start = 0  # From where you want to fit. From the first bin = 0
end = 20  # int(len(data)) is the last bin

# Create a second figure
plt.figure(figsize=(10, 8))

# Generate X-axis values based on number of values
xdata = np.arange(start, end, 1)

# First curve and fit
popt, pcov = curve_fit(func, xdata, data[start:end], p0=(1, 1, 1), maxfev=1000)  # p0 = initial guess
plt.plot(xdata, func(xdata, *popt), 'b--')
plt.errorbar(np.arange(0, int(len(data)), 1), data[:int(len(data))], yerr=dataSEM,
             color='blue', fmt='o', markeredgecolor='blue', markerfacecolor='none', capsize=3,
             elinewidth=1, markeredgewidth=1.3, **marker_param_1, label="young, " + r'$\tau_{14-15}$' + ' = '
                                                                        + str(round(1 / popt[1], 3)) + ' s')

# Second curve and fit
# popt1, pcov1 = curve_fit(func, xdata, data1[start:end], p0=(1, 1, 1), maxfev=1000)  # p0 = initial guess
# plt.plot(xdata, func(xdata, *popt1), 'green', linestyle="--")
# plt.errorbar(np.arange(0, int(len(data1)), 1), data1[:int(len(data1))], yerr=dataSEM1,
#              color='green', fmt='o', markeredgecolor='green', markerfacecolor='none', capsize=3,
#              elinewidth=1, markeredgewidth=1.3, **marker_param_2, label='old, ' + r'$\tau_{16-19}$' + ' = '
#                                                                         + str(round(1 / popt1[1], 3)) + ' s')

# # Third curve and fit
# popt2, pcov2 = curve_fit(func, xdata, data2[start:end], p0=(1, 1, 1), maxfev=1000)  # p0 = initial guess
# plt.plot(xdata, func(xdata, *popt2), 'm--')
# plt.errorbar(np.arange(0, int(len(data2)), 1), data2[:int(len(data2))], yerr=dataSEM2,
#              color='k', fmt='o', markeredgecolor='magenta', markerfacecolor='none', capsize=3,
#              elinewidth=1, markeredgewidth=1.3, **marker_param_3, label='20-24, ' + r'$\tau_{20-24}$' + ' = '
#                                                                         + str(round(1 / popt2[1], 3)) + ' s')
#
# Fourth curve and fit
popt3, pcov3 = curve_fit(func, xdata, data1[start:end], p0=(1, 1, 1), maxfev=1000)  # p0 = initial guess
plt.plot(xdata, func(xdata, *popt3), 'r--')
plt.errorbar(np.arange(0, int(len(data1)), 1), data1[:int(len(data1))], yerr=dataSEM1,
             color='k', fmt='o', markeredgecolor='red', markerfacecolor='none', capsize=3,
             elinewidth=1, markeredgewidth=1.3, **marker_param_4, label='Aged, ' + r'$\tau_{Old}$' + ' = '
                                                                        + str(round(1 / popt3[1], 3)) + ' s')

# Additional details for the plot ##########################################################
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)  # draw a dashed line at y=0
plt.legend(fontsize=20, frameon=False)
plt.ylabel('Depolarization (mV)', fontsize=40)
plt.xlabel('Time (s)', fontsize=40)
plt.axis([start - 1, end - 0.5, 0, 40])
plt.tight_layout()
plt.savefig(os.path.join(os.environ["HOMEPATH"], "Desktop") + '\\YourFigureName_exponential.png', dpi=300)
plt.show()

#########################################################################
print('Done.')
