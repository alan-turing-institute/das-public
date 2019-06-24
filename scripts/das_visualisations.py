#!/usr/bin/env python3

import datetime as dt
import itertools as it
import numpy as np
import matplotlib.pylab as plt
import matplotlib.dates as mdates
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
    year_list = np.arange(year_range[0], year_range[1]+1)
    bins_list = [dt.date(year, 1, 1) for year in year_list]
    bins_list = mdates.date2num(bins_list)
    xticks_list = [dt.date(year, 6, 15) for year in year_list[:-1]]
    xticks_label_list = year_list[:-1]
    if len(year_list) > 8:
        xticks_list = [dt.date(year, 6, 15) for year in year_list[:-1:3]]
        xticks_list = mdates.date2num(xticks_list)
        xticks_label_list = year_list[:-1:3]

    # Create histogram across time
    fig, ax = plt.subplots(figsize=(10, 6))

    # Make the stacked histogram
    ax.hist(pub_date_data_list,
            bins=bins_list,
            histtype=hist_type,
            color=color_list,
            label=label_list)

    # Add the required date line (if applicable)
    if required_date:
        ax.axvline(mdates.date2num(required_date),
                   color='k',
                   linestyle='solid',
                   linewidth=3)

    # Add the encouraged date line (if applicable)
    if encouraged_date:
        ax.axvline(mdates.date2num(encouraged_date),
                   color='k',
                   linestyle='dashed',
                   linewidth=3)

    # Add the legend
    if label_list:
        legend = ax.legend()
        legend.set_title(legend_title, prop={'size': 14})
        legend._legend_box.align = "left"

    # Adjust the plot to make it pretty
    ax.set_xlabel('Publication date')
    ax.set_xticks(xticks_list)
    ax.set_xticklabels(xticks_label_list)
    ax.set_ylabel('Number of articles')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    sns.despine()

    # Tight layout to look really pretty
    plt.tight_layout()

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
    # Make journal and publisher lowercase to make it easier to search
    publisher = publisher.lower()
    journal = journal.lower()

    try:
        required_date = df_policies.loc[(df_policies['Group'] == publisher) &
                                        (df_policies['Journal'] == journal),
                                        'Required'].values[0]

    except IndexError:
        try:
            required_date = df_policies.loc[(df_policies['Group'] ==
                                             publisher) &
                                            (df_policies['Journal'] == 'all'),
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
        try:
            encouraged_date = df_policies.loc[(df_policies['Group'] ==
                                              publisher) &
                                              (df_policies['Journal'] ==
                                              'all'),
                                              'Encouraged'].values[0]

            # Replace encouraged date with None if it doesn't exist
            if np.isnat(encouraged_date):
                encouraged_date = None

        except IndexError:
            encouraged_date = None

    return required_date, encouraged_date


def make_lots_of_plots(df,
                       publisher_journal_dict,
                       palette_extended,
                       df_policies):

    # Define a few masks:
    # - the three das classes and no DAS
    das1_mask = df['das_class'] == 1
    das2_mask = df['das_class'] == 2
    das3_mask = df['das_class'] == 3
    nodas_mask = df['has_das'] == False

    # Plot the data over two year ranges
    year_dict = {'Dates_2000to2019': (2000, 2019),
                 'Dates_2012to2019': (2012, 2019)}

    for ((article_selection_label,
         (publisher_name,
          journal_name,
          article_selection_mask,
          color_counter)),
         (year_str, year_range)) in it.product(publisher_journal_dict.items(),
                                               year_dict.items()):

        # Stack the data you want to visualise
        pub_date_data = [df.loc[(article_selection_mask) &
                                (nodas_mask), 'p_date'],
                         df.loc[(article_selection_mask) &
                                (das1_mask), 'p_date'],
                         df.loc[(article_selection_mask) &
                                (das2_mask), 'p_date'],
                         df.loc[(article_selection_mask) &
                                (das3_mask), 'p_date']]

        # Label the data frame
        label_list = ['no DAS', 'Class 1', 'Class 2', 'Class 3']

        # Get the right colours
        color_list = [palette_extended[(color_counter*6) + 2],
                      palette_extended[(color_counter*6) + 3],
                      palette_extended[(color_counter*6) + 4],
                      palette_extended[(color_counter*6) + 5]]

        # Set up the legend
        legend_title = article_selection_label

        # Get the required and encouraged dates
        (required_date,
         encouraged_date) = get_mandate_dates(df_policies,
                                              publisher=publisher_name,
                                              journal=journal_name)

        date_line_dict = {'NoDateLine': (None, None),
                          'DateLine': (required_date, encouraged_date)}

        # Lets make one stacked and one regular bar histogram
        # and one version with and one without the datelines
        for (hist_type,
             (date_line_str,
              (required_date,
               encouraged_date))) in it.product(['bar', 'barstacked'],
                                                date_line_dict.items()):

            output_fname = os.path.join('..',
                                        'figures',
                                        year_str,
                                        date_line_str,
                                        hist_type,
                                        ('PubsOverTime_{}_ByDas.png').format(article_selection_label.replace(" ", "_")))

            # Make the figure
            fig, ax = pubs_over_time(pub_date_data,
                                     color_list=color_list,
                                     label_list=label_list,
                                     legend_title=legend_title,
                                     year_range=year_range,
                                     hist_type=hist_type,
                                     output_fname=output_fname,
                                     required_date=required_date,
                                     encouraged_date=encouraged_date)

            if ((hist_type == 'barstacked') and
               (date_line_str == 'DateLine')):
                plt.show()

            plt.close()
