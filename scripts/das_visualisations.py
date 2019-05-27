#!/usr/bin/env python3

import datetime as dt
import numpy as np
import matplotlib.pylab as plt
from matplotlib.dates import date2num
import matplotlib.ticker as ticker
import seaborn as sns
import os


def pubs_over_time(pub_date_data_list,
                   color_list=None,
                   label_list=None,
                   legend_title=None,
                   year_range=(2000, 2019),
                   hist_type='barstacked',
                   output_fname=None,
                   required_date=None,
                   encouraged_date=None):
    """Plot histogram of publications over time for groups of articles.

    Parameters
    ----------
    pub_date_data_list : list of pandas.DataFrame
        One or more pandas data frames containing the publications to be
        counted.
    color_list : list of str, optional
        One color for each group in `pub_date_data_list`, by default None.
    label_list : list of str, optional
        One label for each group in `pub_date_data_list`, by default None.
    legend_title : str, optional
        Title for legend, by default None
    year_range : tuple of two ints, optional
        Minimum and maximum years to be plotted on x axis,
        by default (2000, 2019).
    hist_type : {'bar', 'barstacked', 'step'  or 'stepfilled'}, optional
        Histogram style, by default 'barstacked'.
    output_fname : str, optional
        Filename to save output file as .png image. Passing this argument will
        make a new directory if necessary and will overwrite a file that
        already exists there without warning, by default None (no figure
        saved).
    required_date : datetime, optional
        Date at which to add a solid vertical line to indicate when the journal
        or publisher required a data availability statement to be included in
        the submitted article, by default None (no line added).
    encouraged_date : datetime, optional
        Date at which to add a dashed vertical line to indicate when the
        journal or publisher required a data availability statement to be
        included in the submitted article, by default None (no line added).
    """

    # Make the year list
    year_list = np.arange(year_range[0], year_range[1])
    year_list = [dt.date(year, 6, 15) for year in year_list]
    year_list = date2num(year_list)

    # Create histogram across time
    fig, ax = plt.subplots(figsize=(10, 6))

    # Make the stacked histogram
    ax.hist(pub_date_data_list,
            year_list,
            histtype=hist_type,
            color=color_list,
            label=label_list)

    # Add the required date line (if applicable)
    if required_date:
        ax.axvline(required_date, color='k', linestyle='solid', linewidth=3)

    # Add the encouraged date line (if applicable)
    if encouraged_date:
        ax.axvline(required_date, color='k', linestyle='dash', linewidth=3)

    # Add the legend
    if label_list:
        legend = ax.legend()
        legend.set_title(legend_title, prop={'size': 14})
        legend._legend_box.align = "left"

    # Adjust the plot to make it pretty
    ax.set_xlabel('Publication date')
    ax.set_xticks(year_list[::3])
    ax.set_ylabel('Number of articles')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    sns.despine()

    # Save the figure
    if output_fname:
        d = os.path.dirname(output_fname)
        if not os.path.isdir(d):
            os.makedirs(d)
        fig.savefig(output_fname, dpi=100, bbox_inches='tight')

    return fig, ax


def get_mandate_dates(df_policies, publisher='All', journal='All'):
    """Get the required and encouraged dates at which the specific publisher
    or journal mandated data availability statements appear in the submitted
    articles.

    Parameters
    ----------
    df_policies : pandas dataframe
        Know policy dates for journals and publishers
    publisher : str
        Publisher name, by default "All"
    journal : str
        Journal name, by default "All"
    """
    try:
        required_date = df_policies.loc[(df_policies['Group'] == publisher) &
                                        (df_policies['Journal'] == journal),
                                        'Required'].values[0]
    except IndexError:
        required_date = None

    try:
        encouraged_date = df_policies.loc[(df_policies['Group'] == publisher) &
                                          (df_policies['Journal'] == journal),
                                          'Encouraged'].values[0]

        # Replace encouraged date with None if it doesn't exist
        if np.isnat(encouraged_date):
            encouraged_date = None

    except IndexError:
        encouraged_date = None

    return required_date, encouraged_date
