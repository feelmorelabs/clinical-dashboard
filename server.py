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

log = LogMixin()
Greys256.reverse()


head = Div(text=ST_HEADER, style={'font-size': '30px'}, width=500, height=50)
head_2 = Div(text=ST_HEADER_2, style={'font-size': '30px'}, width=500, height=50)

logo = Div(text=ST_LOGO, style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)

space = Div(text='', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
space_2 = Div(text=f"""<hr style="color:{BLUE_1}" />""", width=SPACE_WIDTH)

title_1 = Div(text='pre vs post metrics', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
title_2 = Div(text='visit by visit metrics', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)
title_3 = Div(text='Overall population metrics', style={'font-size': '30px'}, width=SPACE_WIDTH, height=BOX_HEIGHT)


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
    global data_table_1
    global data_table_2
    global data_table_3
    global gender_select, med_select, location_select, numdev_select, race_select
    global compute_button
    global source_1
    global source_2
    global source_3

    compute_button = Button(label="Recompute stats", button_type='success')

    gender_select = Select(title="gender selection", value="all", options=["all", "F", "M"], width=250)

    med_select = Select(title="medication status selection", value="all", options=["all", "Med", "Unmed"], width=250)

    numdev_select = Select(title="Number of devices selection", value="all", options=["all", "1", "2", "3", "4"], width=250)

    location_select = Select(title="Location selection", value="all", options=["all", "Charlottesville", "Charlotte", "Woodlands", "Sugarland"], width=250)

    race_select = Select(title="Race selection", value="all", options=['all', 'Caucasian', 'African American', 'Did not fill', 'Hispanic', 'Asian', 'Caucasian & African American'], width=250)

    df = pd.read_csv('s3://science-box/clinical_trial/trial_v2.csv')

    df = pre_transform_df(df)


    results = stat_summary(df, stat_tests=True)

    result_1 = results[0].reset_index()

    result_2 = results[1].reset_index()

    result_3 = results[2].reset_index()

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

    compute_button.on_click(stat_recompute)


def stat_recompute():
    selected_gender = str(gender_select.value)
    selected_med = str(med_select.value)
    selected_loc = str(location_select.value)
    selected_dev = numdev_select.value
    selected_race = str(race_select.value)


    if selected_gender == 'all':
        genders = ["M", "F"]

    else:
        genders = [selected_gender]

    if selected_med == 'all':
        med = ["Med", "Unmed"]

    else:
        med = [selected_med]

    if selected_loc == 'all':
        loc = ["Charlottesville", "Charlotte", "Woodlands", "Sugarland"]

    else:
        loc = [selected_loc]

    if selected_dev == 'all':
        dev = [1, 2, 3, 4]

    else:
        dev = [int(selected_dev)]


    if selected_race == 'all':
        race = ['Caucasian', 'African American', 'Did not fill', 'Hispanic', 'Asian', 'Caucasian & African American']

    else:
        race = [selected_race]

    tmp_df = df[(df.Gender.isin(genders)) & (df['Medication Status'].isin(med)) & (df['Location'].isin(loc)) & (df['# Devices'].isin(dev)) & (df['Race'].isin(race))]

    results = stat_summary(tmp_df, stat_tests=True)

    result_1 = results[0].reset_index()

    result_2 = results[1].reset_index()

    result_3 = results[2].reset_index()

    new_source_1 = ColumnDataSource(result_1)
    source_1.data = new_source_1.data

    new_source_2 = ColumnDataSource(result_2)
    source_2.data = new_source_2.data

    new_source_3 = ColumnDataSource(result_3)
    source_3.data = new_source_3.data


def button_callback():
    space.text = ''
    if (psd_input.value == os.environ['APP_RESEARCH_PASS']) and (auth_input.value == os.environ['APP_RESEARCH_USER']):
        log.logger.info(f'logging for user {os.environ["APP_RESEARCH_USER"]}')
        curdoc().clear()
        initial_run()
        l = layout([[head], [space_2], column(row(gender_select, med_select, location_select, numdev_select, race_select), compute_button), title_1, data_table_1, title_2, data_table_2, title_3, data_table_3])

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
