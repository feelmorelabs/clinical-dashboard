from clinical_data import *
import os
from bokeh.layouts import widgetbox, layout, column, row
from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, HoverTool, TapTool, Arrow, OpenHead, NormalHead, VeeHead, Title, Span, Label, \
    Slider, NumeralTickFormatter, BasicTicker
from bokeh.models.widgets import CheckboxButtonGroup, Select, DateRangeSlider, Div, DataTable, DateFormatter, \
    TableColumn, Panel, Tabs, CheckboxGroup, TextInput, Button, Dropdown, PasswordInput, FileInput, DatePicker, Panel, \
    Tabs
from bokeh.palettes import Greys256, linear_palette
import pandas as pd
from constants import *
from log import LogMixin
from widget_helper import Spinner

import itertools

def partitions(items, n):
    if n == 1:
        return [set([e]) for e in items]
    results = partitions(items, n - 1)
    for i, j in itertools.combinations(range(len(results)), 2):
        newresult = results[i] | results[j]
        if newresult not in results:
            results.append(newresult)
    return results

log = LogMixin()
Greys256.reverse()

LOADER = Spinner.SPINNER

head = Div(text=ST_HEADER, style={'font-size': '30px'}, width=500, height=50)
head_2 = Div(text=ST_HEADER_2, style={'font-size': '30px'}, width=500, height=50)

logo = Div(text=ST_LOGO, style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)

space = Div(text='', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
space_2 = Div(text=f"""<hr style="color:{BLUE_1}" />""", width=SPACE_WIDTH)

title_1 = Div(text='pre vs post metrics', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
title_2 = Div(text='visit by visit metrics', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
title_3 = Div(text='Overall population metrics', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
title_4 = Div(text='p values for active vs sham tests', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
title_5 = Div(text='p values for pre vs post tests', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
title_6 = Div(text='Pre identified subgroups', style={'font-size': '30px'}, width=2 * SPACE_WIDTH, height=BOX_HEIGHT)


def make_figure(tools, height, width, title, xlabel, ylabel, yrange=None, yticker=None, xticker=None, xlabeloverride=None, ylabeloverride=None,
                xlabelorientation=None):
    log.logger.info(f'making figure of title {title}')
    _fig = figure(tools=tools, title=title, plot_height=height, plot_width=width)

    _fig.title.align = 'center'
    _fig.xaxis.axis_label = xlabel
    _fig.yaxis.axis_label = ylabel

    if yrange is not None:
        _fig.y_range.bounds = yrange

    if yticker is not None:
        _fig.yaxis.ticker = yticker

    if xticker is not None:
        _fig.xaxis.ticker = xticker

    if xlabeloverride is not None:
        _fig.xaxis.major_label_overrides = xlabeloverride

    if ylabeloverride is not None:
        _fig.yaxis.major_label_overrides = ylabeloverride

    if xlabelorientation is not None:
        _fig.xaxis.major_label_orientation = xlabelorientation

    return _fig


def initial_run():
    global df
    global data_table_1, data_table_2, data_table_3, data_table_4, data_table_5, data_table_6
    global gender_select, med_select, location_select, numdev_select, race_select, noisy_select, age_select, comp_select
    global compute_button, auto_button
    global source_1, source_2, source_3, source_4, source_5, source_6
    global all_noisy, all_genders, all_med, all_locs, all_ages, all_comps, all_devs, all_races

    compute_button = Button(label="Recompute stats", button_type='success', width=200)

    auto_button = Button(label="Auto identify subgroups", button_type='success', width=200)

    comp_select = Select(title="Compliance rate selection", value="all",
                         options=["all", "more than 33%", "not found or less than 33%"], width=200)

    noisy_select = Select(title="Noisy device selection", value="all", options=["all", "noisy", "not noisy"], width=200)

    gender_select = Select(title="gender selection", value="all", options=["all", "F", "M"], width=200)

    med_select = Select(title="medication status selection", value="all", options=["all", "Med", "Unmed"], width=200)

    numdev_select = Select(title="Number of devices selection", value="all", options=["all", "1", "2", "3", "4"],
                           width=200)

    location_select = Select(title="Location selection", value="all",
                             options=["all", "Charlottesville", "Charlotte", "Woodlands", "Sugarland"], width=200)

    race_select = Select(title="Race selection", value="all",
                         options=['all', 'Caucasian', 'African American', 'Did not fill', 'Hispanic', 'Asian',
                                  'Caucasian & African American'], width=200)

    age_select = Select(title="Age selection", value="all",
                        options=['all', '40+', 'between 20 and 30', 'between 30 and 40'], width=200)

    all_noisy = [True, False]

    all_genders = ["M", "F"]

    all_med = ["Med", "Unmed"]

    all_locs = ["Charlottesville", "Charlotte", "Woodlands", "Sugarland"]

    all_ages = ['40+', 'between 20 and 30', 'between 30 and 40']

    all_comps = ["more than 33%", "not found or less than 33%"]

    all_devs = [1, 2, 3, 4]

    all_races = ['Caucasian', 'African American', 'Did not fill', 'Hispanic', 'Asian', 'Caucasian & African American']


    df = pd.read_csv('s3://science-box/clinical_trial/trial_v3.csv')

    df = pre_transform_df(df)

    results = stat_summary(df, stat_tests=True)


    result_1 = results[0].reset_index()

    result_2 = results[1].reset_index()

    result_3 = results[2].reset_index()

    result_4 = results[3].reset_index()

    result_5 = results[4].reset_index()



    columns_1 = [TableColumn(field=Ci, title=Ci) for Ci in result_1.columns]

    source_1 = ColumnDataSource(result_1)
    data_table_1 = DataTable(columns=columns_1, source=source_1, height=200,
                                  width=1600)

    columns_2 = [TableColumn(field=Ci, title=Ci) for Ci in result_2.columns]

    source_2 = ColumnDataSource(result_2)
    data_table_2 = DataTable(columns=columns_2, source=source_2, height=200,
                             width=1600)

    columns_3 = [TableColumn(field=Ci, title=Ci) for Ci in result_3.columns]

    source_3 = ColumnDataSource(result_3)
    data_table_3 = DataTable(columns=columns_3, source=source_3, height=200,
                             width=1600)

    columns_4 = [TableColumn(field=Ci, title=Ci) for Ci in result_4.columns]

    source_4 = ColumnDataSource(result_4)
    data_table_4 = DataTable(columns=columns_4, source=source_4, height=200,
                             width=1600)

    columns_5 = [TableColumn(field=Ci, title=Ci) for Ci in result_5.columns]

    source_5 = ColumnDataSource(result_5)
    data_table_5 = DataTable(columns=columns_5, source=source_5, height=200,
                             width=1600)


    columns_6 = [TableColumn(field=Ci, title=Ci) for Ci in ['gender', 'location', 'noisy', 'compliance rate',  'num devices']]
    source_6 = ColumnDataSource(pd.DataFrame())

    data_table_6 = DataTable(columns=columns_6, source=source_6, height=200,
                             width=1600)


    compute_button.on_click(stat_recompute)

    auto_button.on_click(auto_gp_identification)



def auto_gp_identification():
    space.text = LOADER
    curdoc().add_next_tick_callback(auto_gp_identification_computation)


def auto_gp_identification_computation():
    space.text = LOADER
    best_combs = []
    gend_comb = []
    loc_comb = []
    noise_comb = []
    comp_comb = []
    numdev_comb = []
    _count = 0

    for _gend in partitions(all_genders, len(all_genders)):
        for _noise in partitions(all_noisy, len(all_noisy)):
                for _loc in partitions(all_locs, len(all_locs)):
                        for _comp in partitions(all_comps, len(all_comps)):
                            for _dev in [{1}, {1,2,3,4}, {2,3,4}]:
                                    tmp_df = df[(df['# Devices'].isin(_dev)) & (df['comp_group'].isin(_comp)) & (df['Location'].isin(_loc)) & (df['noisy'].isin(_noise)) & (df['Gender'].isin(_gend))]
                                    if tmp_df.shape[0] > 30:
                                        _count += 1
                                        tmpstat = stat_summary(tmp_df, stat_tests=True)[-2]
                                        tmpstat = tmpstat.replace('na', np.nan)

                                        pval = tmpstat[tmpstat.astype(float) < 0.05].count().sum()
                                        if pval > 2:
                                            gend_comb.append(', '.join(str(key) for key in _gend))
                                            loc_comb.append(', '.join(str(key) for key in _loc))
                                            noise_comb.append(', '.join(str(key) for key in _noise))
                                            comp_comb.append(', '.join(str(key) for key in _comp))
                                            numdev_comb.append(', '.join(str(key) for key in _dev))

                                            #tmp_comb = {'gender': _gend, 'noise': _noise, 'loc': _loc, 'comp rate': _comp, 'devs': _dev}
                                            #best_combs.append(tmp_comb)


    title_6.text = f'Pre identified subgroups: found {len(gend_comb)} groups with at least 3 p values <= 0.05 and size >=30 among {_count}'
    comb_df = pd.DataFrame({'gender': gend_comb, 'location': loc_comb, 'noisy': noise_comb, 'compliance rate': comp_comb, 'num devices': numdev_comb})

    new_source_6 = ColumnDataSource(comb_df)
    source_6.data = new_source_6.data
    space.text = ''





def stat_recompute():
    selected_gender = str(gender_select.value)
    selected_med = str(med_select.value)
    selected_loc = str(location_select.value)
    selected_dev = numdev_select.value
    selected_race = str(race_select.value)
    selected_noisy = str(noisy_select.value)
    selected_age = str(age_select.value)
    selected_comp = str(comp_select.value)



    if selected_noisy == 'noisy':
        noisy = [True]
    elif selected_noisy == 'not noisy':
        noisy = [False]
    else:
        noisy = all_noisy

    if selected_gender == 'all':
        genders = all_genders

    else:
        genders = [selected_gender]

    if selected_med == 'all':
        med = all_med

    else:
        med = [selected_med]

    if selected_loc == 'all':
        loc = all_locs

    else:
        loc = [selected_loc]

    if selected_age == 'all':
        age = all_ages

    else:
        age = [selected_age]

    if selected_comp == 'all':
        comp = all_comps

    else:
        comp = [selected_comp]

    if selected_dev == 'all':
        dev = all_devs

    else:
        dev = [int(selected_dev)]


    if selected_race == 'all':
        race = all_races

    else:
        race = [selected_race]


    tmp_df = df[(df.Gender.isin(genders)) & (df['Medication Status'].isin(med)) & (df['Location'].isin(loc)) & (df['# Devices'].isin(dev)) & (df['Race'].isin(race)) & (df['noisy'].isin(noisy)) & (df['age_group'].isin(age)) & (df['comp_group'].isin(comp))]

    results = stat_summary(tmp_df, stat_tests=True)

    result_1 = results[0].reset_index()

    result_2 = results[1].reset_index()

    result_3 = results[2].reset_index()

    result_4 = results[3].reset_index()

    result_5 = results[4].reset_index()

    new_source_1 = ColumnDataSource(result_1)
    source_1.data = new_source_1.data

    new_source_2 = ColumnDataSource(result_2)
    source_2.data = new_source_2.data

    new_source_3 = ColumnDataSource(result_3)
    source_3.data = new_source_3.data

    new_source_4 = ColumnDataSource(result_4)
    source_4.data = new_source_4.data

    new_source_5 = ColumnDataSource(result_5)
    source_5.data = new_source_5.data


def button_callback():
    space.text = ''
    if (psd_input.value == os.environ['APP_RESEARCH_PASS']) and (auth_input.value == os.environ['APP_RESEARCH_USER']):
        log.logger.info(f'logging for user {os.environ["APP_RESEARCH_USER"]}')
        curdoc().clear()
        initial_run()
        l = layout([[head], [space_2], [space], column(row(gender_select, med_select, location_select, numdev_select, noisy_select, race_select, age_select, comp_select), compute_button, auto_button), title_1, data_table_1, title_2, data_table_2, title_3, data_table_3, title_4, data_table_4, title_5, data_table_5, title_6, data_table_6])

        curdoc().add_root(l)
    else:
        log.logger.warning('incorrect auth')
        space.text = 'auth incorrect - please enter right username and password'


auth_input = TextInput(title='USERNAME')
psd_input = PasswordInput(title='PASSWORD')
button = Button(label="Connect", button_type='success')

button.on_click(button_callback)

l_0 = layout([[head], [space_2], [column(auth_input, psd_input, button)], [space]])
# put the button and plot in a layout and add to the document
curdoc().add_root(l_0)
