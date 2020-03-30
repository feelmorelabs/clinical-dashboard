import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway, wilcoxon, mannwhitneyu, chi2_contingency


def cohen_d(x,y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def age_group(row):
    if row< 18:
        return '<18'
    if row >= 18 and row <= 30:
        return 'between 20 and 30'
    if row >= 30 and row <= 40:
        return 'between 30 and 40'
    return '40+'

def make_groups(row):
    if row < 5:
        return '0-4'
    if row >= 5 and row < 10:
        return '5-9'
    if row >= 10 and row < 15:
        return '10-14'
    if row >= 15:
        return '>=15'


def pre_transform_df(df):
    for i in ['CGIe-V2', 'CGIe-V3', 'CGIe-V4']:
        df[i] = df[i].replace(
            {'Moderate no side effects': 5, 'Minimal no side effects': 9, 'Unchanged or worse with no side effects': 13
                , 'Moderate with side effects that did not significantly interfere with functioning': 6,
             'Marked no side effects': 1,
             'Marked with side effects that did not significantly interfere with functioning': 2,
             'Minimal with side effects that did not significantly interfere with functioning': 10,
             'Minimal with side effects that significantly interfere with functioning': 11,
             'Unchanged or worse with side effects that did not significantly interfere with functioning': 14})

    df['State'] = df.State.replace({1: 'active', 2: 'sham'})

    df['age_group'] = df.Age.apply(age_group)
    df['ham_pct_change'] = 100 * (df['HAM-A-V4'] - df['HAM-A-V1']) / df['HAM-A-V1']
    df['gad_pct_change'] = 100 * (df['GAD-V4'] - df['GAD-V1']) / df['GAD-V1']
    df['cgis_pct_change'] = 100 * (df['CGIs-V4'] - df['CGIs-V1']) / df['CGIs-V1']
    df['cgig_pct_change'] = 100 * (df['CGIg-V4'] - df['CGIg-V2']) / df['CGIs-V2']

    df['Medication Status'] = df['Medication Status'].replace("UnMed", "Unmed")

    df['Race'] = df['Race'].replace('African american', 'African American')
    df['Race'] = df['Race'].replace('African-American', 'African American')
    df['Race'] = df['Race'].fillna('Did not fill')

    return df


def stat_summary(df, stat_tests=False):
    n_active = df[df.State == 'active'].shape[0]
    n_sham = df[df.State == 'sham'].shape[0]

    df_hama_active_v1_mean = df[df.State == 'active']['HAM-A-V1'].mean()
    df_hama_active_v1_se = df[df.State == 'active']['HAM-A-V1'].mean() / np.sqrt(n_active)

    df_hama_sham_v1_mean = df[df.State == 'sham']['HAM-A-V1'].mean()
    df_hama_sham_v1_se = df[df.State == 'sham']['HAM-A-V1'].mean() / np.sqrt(n_sham)

    df_hama_active_v2_mean = df[df.State == 'active']['HAM-A-V2'].mean()
    df_hama_active_v2_se = df[df.State == 'active']['HAM-A-V2'].mean() / np.sqrt(n_active)

    df_hama_sham_v2_mean = df[df.State == 'sham']['HAM-A-V2'].mean()
    df_hama_sham_v2_se = df[df.State == 'sham']['HAM-A-V2'].mean() / np.sqrt(n_sham)

    df_hama_active_v3_mean = df[df.State == 'active']['HAM-A-V3'].mean()
    df_hama_active_v3_se = df[df.State == 'active']['HAM-A-V3'].mean() / np.sqrt(n_active)

    df_hama_sham_v3_mean = df[df.State == 'sham']['HAM-A-V3'].mean()
    df_hama_sham_v3_se = df[df.State == 'sham']['HAM-A-V3'].mean() / np.sqrt(n_sham)

    df_hama_active_v4_mean = df[df.State == 'active']['HAM-A-V4'].mean()
    df_hama_active_v4_se = df[df.State == 'active']['HAM-A-V4'].mean() / np.sqrt(n_active)

    df_hama_sham_v4_mean = df[df.State == 'sham']['HAM-A-V4'].mean()
    df_hama_sham_v4_se = df[df.State == 'sham']['HAM-A-V4'].mean() / np.sqrt(n_sham)

    change_hama_active = (df[df.State == 'active']['HAM-A-V4'] - df[df.State == 'active']['HAM-A-V1']).mean()
    change_hama_sham = (df[df.State == 'sham']['HAM-A-V4'] - df[df.State == 'sham']['HAM-A-V1']).mean()

    change_hama_active_se = (df[df.State == 'active']['HAM-A-V4'] - df[df.State == 'active'][
        'HAM-A-V1']).std() / np.sqrt(n_active)
    change_hama_sham_se = (df[df.State == 'sham']['HAM-A-V4'] - df[df.State == 'sham']['HAM-A-V1']).std() / np.sqrt(
        n_sham)

    pct_change_hama_active = 100 * ((df[df.State == 'active']['HAM-A-V4'] - df[df.State == 'active']['HAM-A-V1']) / (
    df[df.State == 'active']['HAM-A-V1'])).mean()
    pct_change_hama_sham = 100 * ((df[df.State == 'sham']['HAM-A-V4'] - df[df.State == 'sham']['HAM-A-V1']) / (
    df[df.State == 'sham']['HAM-A-V1'])).mean()

    pct_change_hama_active_se = 100 * ((df[df.State == 'active']['HAM-A-V4'] - df[df.State == 'active']['HAM-A-V1']) / (
    df[df.State == 'active']['HAM-A-V1'])).std() / np.sqrt(n_active)
    pct_change_hama_sham_se = 100 * ((df[df.State == 'sham']['HAM-A-V4'] - df[df.State == 'sham']['HAM-A-V1']) / (
    df[df.State == 'sham']['HAM-A-V1'])).std() / np.sqrt(n_sham)

    # pct_change_hama_active_3 = 100 * ((df[df.State == 'active']['HAM-A-V3'] - df[df.State == 'active']['HAM-A-V1']) / (df[df.State == 'active']['HAM-A-V1'])).mean()
    # pct_change_hama_sham_3 = 100 * ((df[df.State == 'sham']['HAM-A-V3'] - df[df.State == 'sham']['HAM-A-V1']) / (df[df.State == 'sham']['HAM-A-V1'])).mean()

    # pct_change_hama_active_se_3 = 100 * ((df[df.State == 'active']['HAM-A-V3'] - df[df.State == 'active']['HAM-A-V1']) / (df[df.State == 'active']['HAM-A-V1'])).std() / np.sqrt(n_active)
    # pct_change_hama_sham_se_3 = 100 * ((df[df.State == 'sham']['HAM-A-V3'] - df[df.State == 'sham']['HAM-A-V1']) / (df[df.State == 'sham']['HAM-A-V1'])).std() / np.sqrt(n_sham)

    # pct_change_hama_active_2 = 100 * ((df[df.State == 'active']['HAM-A-V2'] - df[df.State == 'active']['HAM-A-V1']) / (df[df.State == 'active']['HAM-A-V1'])).mean()
    # pct_change_hama_sham_2 = 100 * ((df[df.State == 'sham']['HAM-A-V2'] - df[df.State == 'sham']['HAM-A-V1']) / (df[df.State == 'sham']['HAM-A-V1'])).mean()

    # pct_change_hama_active_se_2 = 100 * ((df[df.State == 'active']['HAM-A-V2'] - df[df.State == 'active']['HAM-A-V1']) / (df[df.State == 'active']['HAM-A-V1'])).std() / np.sqrt(n_active)
    # pct_change_hama_sham_se_2 = 100 * ((df[df.State == 'sham']['HAM-A-V2'] - df[df.State == 'sham']['HAM-A-V1']) / (df[df.State == 'sham']['HAM-A-V1'])).std() / np.sqrt(n_sham)

    df_gad_active_v1_mean = df[df.State == 'active']['GAD-V1'].mean()
    df_gad_active_v1_se = df[df.State == 'active']['GAD-V1'].mean() / np.sqrt(n_active)

    df_gad_sham_v1_mean = df[df.State == 'sham']['GAD-V1'].mean()
    df_gad_sham_v1_se = df[df.State == 'sham']['GAD-V1'].mean() / np.sqrt(n_sham)

    df_gad_active_v2_mean = df[df.State == 'active']['GAD-V2'].mean()
    df_gad_active_v2_se = df[df.State == 'active']['GAD-V2'].mean() / np.sqrt(n_active)

    df_gad_sham_v2_mean = df[df.State == 'sham']['GAD-V2'].mean()
    df_gad_sham_v2_se = df[df.State == 'sham']['GAD-V2'].mean() / np.sqrt(n_sham)

    df_gad_active_v3_mean = df[df.State == 'active']['GAD-V3'].mean()
    df_gad_active_v3_se = df[df.State == 'active']['GAD-V3'].mean() / np.sqrt(n_active)

    df_gad_sham_v3_mean = df[df.State == 'sham']['GAD-V3'].mean()
    df_gad_sham_v3_se = df[df.State == 'sham']['GAD-V3'].mean() / np.sqrt(n_sham)

    df_gad_active_v4_mean = df[df.State == 'active']['GAD-V4'].mean()
    df_gad_active_v4_se = df[df.State == 'active']['GAD-V4'].mean() / np.sqrt(n_active)

    df_gad_sham_v4_mean = df[df.State == 'sham']['GAD-V4'].mean()
    df_gad_sham_v4_se = df[df.State == 'sham']['GAD-V4'].mean() / np.sqrt(n_sham)

    change_gad_active = (df[df.State == 'active']['GAD-V4'] - df[df.State == 'active']['GAD-V1']).mean()
    change_gad_sham = (df[df.State == 'sham']['GAD-V4'] - df[df.State == 'sham']['GAD-V1']).mean()

    change_gad_active_se = (df[df.State == 'active']['GAD-V4'] - df[df.State == 'active']['GAD-V1']).std() / np.sqrt(
        n_active)
    change_gad_sham_se = (df[df.State == 'sham']['GAD-V4'] - df[df.State == 'sham']['GAD-V1']).std() / np.sqrt(n_sham)

    pct_change_gad_active = 100 * ((df[df.State == 'active']['GAD-V4'] - df[df.State == 'active']['GAD-V1']) / (
    df[df.State == 'active']['GAD-V1'])).mean()
    pct_change_gad_sham = 100 * ((df[df.State == 'sham']['GAD-V4'] - df[df.State == 'sham']['GAD-V1']) / (
    df[df.State == 'sham']['GAD-V1'])).mean()

    pct_change_gad_active_se = 100 * ((df[df.State == 'active']['GAD-V4'] - df[df.State == 'active']['GAD-V1']) / (
    df[df.State == 'active']['GAD-V1'])).std() / np.sqrt(n_active)
    pct_change_gad_sham_se = 100 * ((df[df.State == 'sham']['GAD-V4'] - df[df.State == 'sham']['GAD-V1']) / (
    df[df.State == 'sham']['GAD-V1'])).std() / np.sqrt(n_sham)

    # pct_change_gad_active_3 = 100 * ((df[df.State == 'active']['GAD-V3'] - df[df.State == 'active']['GAD-V1']) / (df[df.State == 'active']['GAD-V1'])).mean()
    # pct_change_gad_sham_3 = 100 * ((df[df.State == 'sham']['GAD-V3'] - df[df.State == 'sham']['GAD-V1']) / (df[df.State == 'sham']['GAD-V1'])).mean()

    # pct_change_gad_active_se_3 = 100 * ((df[df.State == 'active']['GAD-V3'] - df[df.State == 'active']['GAD-V1']) / (df[df.State == 'active']['GAD-V1'])).std() / np.sqrt(n_active)
    # pct_change_gad_sham_se_3 = 100 * ((df[df.State == 'sham']['GAD-V3'] - df[df.State == 'sham']['GAD-V1']) / (df[df.State == 'sham']['GAD-V1'])).std() / np.sqrt(n_sham)

    # pct_change_gad_active_2 = 100 * ((df[df.State == 'active']['GAD-V2'] - df[df.State == 'active']['GAD-V1']) / (df[df.State == 'active']['GAD-V1'])).mean()
    # pct_change_gad_sham_2 = 100 * ((df[df.State == 'sham']['GAD-V2'] - df[df.State == 'sham']['GAD-V1']) / (df[df.State == 'sham']['GAD-V1'])).mean()

    # pct_change_gad_active_se_2 = 100 * ((df[df.State == 'active']['GAD-V2'] - df[df.State == 'active']['GAD-V1']) / (df[df.State == 'active']['GAD-V1'])).std() / np.sqrt(n_active)
    # pct_change_gad_sham_se_2 = 100 * ((df[df.State == 'sham']['GAD-V2'] - df[df.State == 'sham']['GAD-V1']) / (df[df.State == 'sham']['GAD-V1'])).std() / np.sqrt(n_sham)

    df_cgis_active_v1_mean = df[df.State == 'active']['CGIs-V1'].mean()
    df_cgis_active_v1_se = df[df.State == 'active']['CGIs-V1'].mean() / np.sqrt(n_active)

    try:
        df_cgis_active_v1_mode = df[df.State == 'active']['CGIs-V1'].mode()[0]
    except:
        df_cgis_active_v1_mode = 'na'
    df_cgis_active_v1_mode_se = 'na'

    df_cgis_sham_v1_mean = df[df.State == 'sham']['CGIs-V1'].mean()
    df_cgis_sham_v1_se = df[df.State == 'sham']['CGIs-V1'].mean() / np.sqrt(n_sham)

    try:
        df_cgis_sham_v1_mode = df[df.State == 'sham']['CGIs-V1'].mode()[0]
    except:
        df_cgis_sham_v1_mode = 'na'
    df_cgis_sham_v1_mode_se = 'na'

    df_cgis_active_v2_mean = df[df.State == 'active']['CGIs-V2'].mean()
    df_cgis_active_v2_se = df[df.State == 'active']['CGIs-V2'].mean() / np.sqrt(n_active)

    try:
        df_cgis_active_v2_mode = df[df.State == 'active']['CGIs-V2'].mode()[0]
    except:
        df_cgis_active_v2_mode = 'na'
    df_cgis_active_v2_mode_se = 'na'

    df_cgis_sham_v2_mean = df[df.State == 'sham']['CGIs-V2'].mean()
    df_cgis_sham_v2_se = df[df.State == 'sham']['CGIs-V2'].mean() / np.sqrt(n_sham)

    try:
        df_cgis_sham_v2_mode = df[df.State == 'sham']['CGIs-V2'].mode()[0]
    except:
        df_cgis_sham_v2_mode = 'na'
    df_cgis_sham_v2_mode_se = 'na'

    df_cgis_active_v3_mean = df[df.State == 'active']['CGIs-V3'].mean()
    df_cgis_active_v3_se = df[df.State == 'active']['CGIs-V3'].mean() / np.sqrt(n_active)

    try:
        df_cgis_active_v3_mode = df[df.State == 'active']['CGIs-V3'].mode()[0]
    except:
        df_cgis_active_v3_mode = 'na'
    df_cgis_active_v3_mode_se = 'na'

    df_cgis_sham_v3_mean = df[df.State == 'sham']['CGIs-V3'].mean()
    df_cgis_sham_v3_se = df[df.State == 'sham']['CGIs-V3'].mean() / np.sqrt(n_sham)

    try:
        df_cgis_sham_v3_mode = df[df.State == 'sham']['CGIs-V3'].mode()[0]
    except:
        df_cgis_sham_v3_mode = 'na'
    df_cgis_sham_v3_mode_se = 'na'

    df_cgis_active_v4_mean = df[df.State == 'active']['CGIs-V4'].mean()
    df_cgis_active_v4_se = df[df.State == 'active']['CGIs-V4'].mean() / np.sqrt(n_active)

    try:
        df_cgis_active_v4_mode = df[df.State == 'active']['CGIs-V4'].mode()[0]
    except:
        df_cgis_active_v4_mode = 'na'
    df_cgis_active_v4_mode_se = 'na'

    df_cgis_sham_v4_mean = df[df.State == 'sham']['CGIs-V4'].mean()
    df_cgis_sham_v4_se = df[df.State == 'sham']['CGIs-V4'].mean() / np.sqrt(n_sham)

    try:
        df_cgis_sham_v4_mode = df[df.State == 'sham']['CGIs-V4'].mode()[0]
    except:
        df_cgis_sham_v4_mode = 'na'
    df_cgis_sham_v4_mode_se = 'na'

    change_cgis_active = (df[df.State == 'active']['CGIs-V4'] - df[df.State == 'active']['CGIs-V1']).mean()
    change_cgis_sham = (df[df.State == 'sham']['CGIs-V4'] - df[df.State == 'sham']['CGIs-V1']).mean()

    change_cgis_active_se = (df[df.State == 'active']['CGIs-V4'] - df[df.State == 'active']['CGIs-V1']).std() / np.sqrt(
        n_active)
    change_cgis_sham_se = (df[df.State == 'sham']['CGIs-V4'] - df[df.State == 'sham']['CGIs-V1']).std() / np.sqrt(
        n_sham)

    pct_change_cgis_active = 100 * ((df[df.State == 'active']['CGIs-V4'] - df[df.State == 'active']['CGIs-V1']) / (
    df[df.State == 'active']['CGIs-V1'])).mean()
    pct_change_cgis_sham = 100 * ((df[df.State == 'sham']['CGIs-V4'] - df[df.State == 'sham']['CGIs-V1']) / (
    df[df.State == 'sham']['CGIs-V1'])).mean()

    pct_change_cgis_active_se = 100 * ((df[df.State == 'active']['CGIs-V4'] - df[df.State == 'active']['CGIs-V1']) / (
    df[df.State == 'active']['CGIs-V1'])).std() / np.sqrt(n_active)
    pct_change_cgis_sham_se = 100 * ((df[df.State == 'sham']['CGIs-V4'] - df[df.State == 'sham']['CGIs-V1']) / (
    df[df.State == 'sham']['CGIs-V1'])).std() / np.sqrt(n_sham)

    # pct_change_cgis_active_3 = 100 * ((df[df.State == 'active']['CGIs-V3'] - df[df.State == 'active']['CGIs-V1']) / (df[df.State == 'active']['CGIs-V1'])).mean()
    # pct_change_cgis_sham_3 = 100 * ((df[df.State == 'sham']['CGIs-V3'] - df[df.State == 'sham']['CGIs-V1']) / (df[df.State == 'sham']['CGIs-V1'])).mean()

    # pct_change_cgis_active_se_3 = 100 * ((df[df.State == 'active']['CGIs-V3'] - df[df.State == 'active']['CGIs-V1']) / (df[df.State == 'active']['CGIs-V1'])).std() / np.sqrt(n_active)
    # pct_change_cgis_sham_se_3 = 100 * ((df[df.State == 'sham']['CGIs-V3'] - df[df.State == 'sham']['CGIs-V1']) / (df[df.State == 'sham']['CGIs-V1'])).std() / np.sqrt(n_sham)

    # pct_change_cgis_active_2 = 100 * ((df[df.State == 'active']['CGIs-V2'] - df[df.State == 'active']['CGIs-V1']) / (df[df.State == 'active']['CGIs-V1'])).mean()
    # pct_change_cgis_sham_2 = 100 * ((df[df.State == 'sham']['CGIs-V2'] - df[df.State == 'sham']['CGIs-V1']) / (df[df.State == 'sham']['CGIs-V1'])).mean()

    # pct_change_cgis_active_se_2 = 100 * ((df[df.State == 'active']['CGIs-V2'] - df[df.State == 'active']['CGIs-V1']) / (df[df.State == 'active']['CGIs-V1'])).std() / np.sqrt(n_active)
    # pct_change_cgis_sham_se_2 = 100 * ((df[df.State == 'sham']['CGIs-V2'] - df[df.State == 'sham']['CGIs-V1']) / (df[df.State == 'sham']['CGIs-V1'])).std() / np.sqrt(n_sham)

    df_cgig_active_v2_mean = df[df.State == 'active']['CGIg-V2'].mean()
    df_cgig_active_v2_se = df[df.State == 'active']['CGIg-V2'].mean() / np.sqrt(n_active)

    df_cgig_active_v1_mode = 'na'
    df_cgig_active_v1_mode_se = 'na'
    df_cgig_sham_v1_mode = 'na'
    df_cgig_sham_v1_mode_se = 'na'

    try:
        df_cgig_active_v2_mode = df[df.State == 'active']['CGIg-V2'].mode()[0]
    except:
        df_cgig_active_v2_mode = 'na'
    df_cgig_active_v2_mode_se = 'na'

    df_cgig_sham_v2_mean = df[df.State == 'sham']['CGIg-V2'].mean()
    df_cgig_sham_v2_se = df[df.State == 'sham']['CGIg-V2'].mean() / np.sqrt(n_sham)

    try:
        df_cgig_sham_v2_mode = df[df.State == 'sham']['CGIg-V2'].mode()[0]
    except:
        df_cgig_sham_v2_mode = 'na'
    df_cgig_sham_v2_mode_se = 'na'

    df_cgig_active_v3_mean = df[df.State == 'active']['CGIg-V3'].mean()
    df_cgig_active_v3_se = df[df.State == 'active']['CGIg-V3'].mean() / np.sqrt(n_active)

    try:
        df_cgig_active_v3_mode = df[df.State == 'active']['CGIg-V3'].mode()[0]
    except:
        df_cgig_active_v3_mode = 'na'
    df_cgig_active_v3_mode_se = 'na'

    df_cgig_sham_v3_mean = df[df.State == 'sham']['CGIg-V3'].mean()
    df_cgig_sham_v3_se = df[df.State == 'sham']['CGIg-V3'].mean() / np.sqrt(n_sham)

    try:
        df_cgig_sham_v3_mode = df[df.State == 'sham']['CGIg-V3'].mode()[0]
    except:
        df_cgig_sham_v3_mode = 'na'
    df_cgig_sham_v3_mode_se = 'na'

    df_cgig_active_v4_mean = df[df.State == 'active']['CGIg-V4'].mean()
    df_cgig_active_v4_se = df[df.State == 'active']['CGIg-V4'].mean() / np.sqrt(n_active)

    try:
        df_cgig_active_v4_mode = df[df.State == 'active']['CGIg-V4'].mode()[0]
    except:
        df_cgig_active_v4_mode = 'na'
    df_cgig_active_v4_mode_se = 'na'

    df_cgig_sham_v4_mean = df[df.State == 'sham']['CGIg-V4'].mean()
    df_cgig_sham_v4_se = df[df.State == 'sham']['CGIg-V4'].mean() / np.sqrt(n_sham)

    try:
        df_cgig_sham_v4_mode = df[df.State == 'sham']['CGIg-V4'].mode()[0]
    except:
        df_cgig_sham_v4_mode = 'na'
    df_cgig_sham_v4_mode_se = 'na'

    df_cgie_active_v1_mode = 'na'
    df_cgie_active_v1_mode_se = 'na'
    df_cgie_sham_v1_mode = 'na'
    df_cgie_sham_v1_mode_se = 'na'
    try:
        df_cgie_active_v2_mean = df[df.State == 'active']['CGIe-V2'].mode()[0]
    except:
        df_cgie_active_v2_mean = 'na'
    df_cgie_active_v2_se = 'na'

    try:
        df_cgie_sham_v2_mean = df[df.State == 'sham']['CGIe-V2'].mode()[0]
    except:
        df_cgie_sham_v2_mean = 'na'
    df_cgie_sham_v2_se = 'na'

    try:
        df_cgie_active_v3_mean = df[df.State == 'active']['CGIe-V3'].mode()[0]
    except:
        df_cgie_active_v3_mean = 'na'
    df_cgie_active_v3_se = 'na'

    try:
        df_cgie_sham_v3_mean = df[df.State == 'sham']['CGIe-V3'].mode()[0]
    except:
        df_cgie_sham_v3_mean = 'na'
    df_cgie_sham_v3_se = 'na'

    try:
        df_cgie_active_v4_mean = df[df.State == 'active']['CGIe-V4'].mode()[0]
    except:
        df_cgie_active_v4_mean = 'na'

    df_cgie_active_v4_se = 'na'

    try:
        df_cgie_sham_v4_mean = df[df.State == 'sham']['CGIe-V4'].mode()[0]
    except:
        df_cgie_sham_v4_mean = 'na'

    df_cgie_sham_v4_se = 'na'

    tmp_result = pd.DataFrame({'n_active': [n_active, n_active, n_active, n_active, n_active, n_active],
                               'n_sham': [n_sham, n_sham, n_sham, n_sham, n_sham, n_sham],
                               'active_diff': [change_gad_active, change_gad_active_se, change_hama_active,
                                               change_hama_active_se, change_cgis_active, change_cgis_active_se],
                               'active_pct_change': [pct_change_gad_active, pct_change_gad_active_se,
                                                     pct_change_hama_active, pct_change_hama_active_se,
                                                     pct_change_cgis_active, pct_change_cgis_active_se],
                               'sham_diff': [change_gad_sham, change_gad_sham_se, change_hama_sham, change_hama_sham_se,
                                             change_cgis_sham, change_cgis_sham_se],
                               'sham_pct_change': [pct_change_gad_sham, pct_change_gad_sham_se, pct_change_hama_sham,
                                                   pct_change_hama_sham_se, pct_change_cgis_sham,
                                                   pct_change_cgis_sham_se]})

    tmp_result.index = ['GAD-mean', 'GAD-se', 'HAMA-mean', 'HAMA-se', 'CGIs-mean', 'CGIs-se']

    tmp_result_2 = pd.DataFrame({'active_v1': [df_gad_active_v1_mean, df_gad_active_v1_se, df_hama_active_v1_mean,
                                               df_hama_active_v1_se, df_cgis_active_v1_mean, df_cgis_active_v1_se, 'na',
                                               'na', df_cgis_active_v1_mode, df_cgis_active_v1_mode_se,
                                               df_cgig_active_v1_mode, df_cgig_active_v1_mode_se, 'na', 'na'],
                                 'active_v2': [df_gad_active_v2_mean, df_gad_active_v2_se, df_hama_active_v2_mean,
                                               df_hama_active_v2_se, df_cgis_active_v2_mean, df_cgis_active_v2_se,
                                               df_cgig_active_v2_mean, df_cgig_active_v2_se, df_cgis_active_v2_mode,
                                               df_cgis_active_v2_mode_se, df_cgig_active_v2_mode,
                                               df_cgig_active_v2_mode_se, df_cgie_active_v2_mean, df_cgie_active_v2_se]
                                    , 'active_v3': [df_gad_active_v3_mean, df_gad_active_v3_se, df_hama_active_v3_mean,
                                                    df_hama_active_v3_se, df_cgis_active_v3_mean, df_cgis_active_v3_se,
                                                    df_cgig_active_v3_mean, df_cgig_active_v3_se,
                                                    df_cgis_active_v3_mode, df_cgis_active_v3_mode_se,
                                                    df_cgig_active_v3_mode, df_cgig_active_v3_mode_se,
                                                    df_cgie_active_v3_mean, df_cgie_active_v3_se]
                                    , 'active_v4': [df_gad_active_v4_mean, df_gad_active_v4_se, df_hama_active_v4_mean,
                                                    df_hama_active_v4_se, df_cgis_active_v4_mean, df_cgis_active_v4_se,
                                                    df_cgig_active_v4_mean, df_cgig_active_v4_se,
                                                    df_cgis_active_v4_mode, df_cgis_active_v4_mode_se,
                                                    df_cgig_active_v4_mode, df_cgig_active_v4_mode_se,
                                                    df_cgie_active_v4_mean, df_cgie_active_v4_se]
                                    , 'sham_v1': [df_gad_sham_v1_mean, df_gad_sham_v1_se, df_hama_sham_v1_mean,
                                                  df_hama_sham_v1_se, df_cgis_sham_v1_mean, df_cgis_sham_v1_se, 'na',
                                                  'na', df_cgis_sham_v1_mode, df_cgis_sham_v1_mode_se,
                                                  df_cgig_sham_v1_mode, df_cgig_sham_v1_mode_se, 'na', 'na'],
                                 'sham_v2': [df_gad_sham_v2_mean, df_gad_sham_v2_se, df_hama_sham_v2_mean,
                                             df_hama_sham_v2_se, df_cgis_sham_v2_mean, df_cgis_sham_v2_se,
                                             df_cgig_sham_v2_mean, df_cgig_sham_v2_se, df_cgis_sham_v2_mode,
                                             df_cgis_sham_v2_mode_se, df_cgig_sham_v2_mode, df_cgig_sham_v2_mode_se,
                                             df_cgie_sham_v2_mean, df_cgie_sham_v2_se]
                                    , 'sham_v3': [df_gad_sham_v3_mean, df_gad_sham_v3_se, df_hama_sham_v3_mean,
                                                  df_hama_sham_v3_se, df_cgis_sham_v3_mean, df_cgis_sham_v3_se,
                                                  df_cgig_sham_v3_mean, df_cgig_sham_v3_se, df_cgis_sham_v3_mode,
                                                  df_cgis_sham_v3_mode_se, df_cgig_sham_v3_mode,
                                                  df_cgig_sham_v3_mode_se, df_cgie_sham_v3_mean, df_cgie_sham_v3_se]
                                    , 'sham_v4': [df_gad_sham_v4_mean, df_gad_sham_v4_se, df_hama_sham_v4_mean,
                                                  df_hama_sham_v4_se, df_cgis_sham_v4_mean, df_cgis_sham_v4_se,
                                                  df_cgig_sham_v4_mean, df_cgig_sham_v4_se, df_cgis_sham_v4_mode,
                                                  df_cgis_sham_v4_mode_se, df_cgig_sham_v4_mode,
                                                  df_cgig_sham_v4_mode_se, df_cgie_sham_v4_mean, df_cgie_sham_v4_se]})

    tmp_result_2.index = ['GAD-mean', 'GAD-se', 'HAMA-mean', 'HAMA-se', 'CGIs-mean', 'CGIs-se', 'CGIg-mean', 'CGIg-se',
                          'CGIs-mode', 'CGIs-mode_se', 'CGIg-mode', 'CGIg-mode-se', 'CGIe-mean', 'CGIe-se']

    age_mean = df.Age.mean()
    age_se = df.Age.std() / np.sqrt(df.shape[0])

    age_mean_active = df[df.State == 'active'].Age.mean()
    age_se_active = df[df.State == 'active'].Age.std() / np.sqrt(n_active)

    age_mean_sham = df[df.State == 'sham'].Age.mean()
    age_se_sham = df[df.State == 'sham'].Age.std() / np.sqrt(n_sham)

    med_count = df[df['Medication Status'] == 'Med'].shape[0]
    med_count_active = df[(df['Medication Status'] == 'Med') & (df.State == 'active')].shape[0]
    med_count_sham = df[(df['Medication Status'] == 'Med') & (df.State == 'sham')].shape[0]

    unmed_count = df[df['Medication Status'] == 'Unmed'].shape[0]
    unmed_count_active = df[(df['Medication Status'] == 'Unmed') & (df.State == 'active')].shape[0]
    unmed_count_sham = df[(df['Medication Status'] == 'Unmed') & (df.State == 'sham')].shape[0]

    f_count = df[df['Gender'] == 'F'].shape[0]
    f_count_active = df[(df['Gender'] == 'F') & (df.State == 'active')].shape[0]
    f_count_sham = df[(df['Gender'] == 'F') & (df.State == 'sham')].shape[0]

    m_count = df[df['Gender'] == 'M'].shape[0]
    m_count_active = df[(df['Gender'] == 'M') & (df.State == 'active')].shape[0]
    m_count_sham = df[(df['Gender'] == 'M') & (df.State == 'sham')].shape[0]

    med_m_count = df[(df['Medication Status'] == 'Med') & (df['Gender'] == 'M')].shape[0]
    med_m_count_active = df[(df['Medication Status'] == 'Med') & (df.State == 'active') & (df['Gender'] == 'M')].shape[
        0]
    med_m_count_sham = df[(df['Medication Status'] == 'Med') & (df.State == 'sham') & (df['Gender'] == 'M')].shape[0]

    med_f_count = df[(df['Medication Status'] == 'Med') & (df['Gender'] == 'F')].shape[0]
    med_f_count_active = df[(df['Medication Status'] == 'Med') & (df.State == 'active') & (df['Gender'] == 'F')].shape[
        0]
    med_f_count_sham = df[(df['Medication Status'] == 'Med') & (df.State == 'sham') & (df['Gender'] == 'F')].shape[0]

    unmed_m_count = df[(df['Medication Status'] == 'Unmed') & (df['Gender'] == 'M')].shape[0]
    unmed_m_count_active = \
    df[(df['Medication Status'] == 'Unmed') & (df.State == 'active') & (df['Gender'] == 'M')].shape[0]
    unmed_m_count_sham = df[(df['Medication Status'] == 'Unmed') & (df.State == 'sham') & (df['Gender'] == 'M')].shape[
        0]

    unmed_f_count = df[(df['Medication Status'] == 'Unmed') & (df['Gender'] == 'F')].shape[0]
    unmed_f_count_active = \
    df[(df['Medication Status'] == 'Unmed') & (df.State == 'active') & (df['Gender'] == 'F')].shape[0]
    unmed_f_count_sham = df[(df['Medication Status'] == 'Unmed') & (df.State == 'sham') & (df['Gender'] == 'F')].shape[
        0]

    cauca_count = df[df['Race'] == 'Caucasian'].shape[0]
    american_count = df[(df['Race'] == 'African American') | (df['Race'] == 'African american')].shape[0]
    asian_count = df[df['Race'] == 'Asian'].shape[0]
    hispanic_count = df[df['Race'] == 'Hispanic'].shape[0]

    cauca_count_active = df[(df['Race'] == 'Caucasian') & (df.State == 'active')].shape[0]
    american_count_active = \
    df[((df['Race'] == 'African American') | (df['Race'] == 'African american')) & (df.State == 'active')].shape[0]
    asian_count_active = df[(df['Race'] == 'Asian') & (df.State == 'active')].shape[0]
    hispanic_count_active = df[(df['Race'] == 'Hispanic') & (df.State == 'active')].shape[0]

    cauca_count_sham = df[(df['Race'] == 'Caucasian') & (df.State == 'sham')].shape[0]
    american_count_sham = \
    df[((df['Race'] == 'African American') | (df['Race'] == 'African american')) & (df.State == 'sham')].shape[0]
    asian_count_sham = df[(df['Race'] == 'Asian') & (df.State == 'sham')].shape[0]
    hispanic_count_sham = df[(df['Race'] == 'Hispanic') & (df.State == 'sham')].shape[0]

    tmp_result_3 = pd.DataFrame({'Overall': [age_mean, age_se, med_count, unmed_count, f_count, m_count, med_m_count,
                                             med_f_count, unmed_m_count, unmed_f_count, cauca_count, asian_count,
                                             american_count, hispanic_count],
                                 'Active': [age_mean_active, age_se_active, med_count_active, unmed_count_active,
                                            f_count_active, m_count_active, med_m_count_active, med_f_count_active,
                                            unmed_m_count_active, unmed_f_count_active, cauca_count_active,
                                            asian_count_active, american_count_active, hispanic_count_active],
                                 'Sham': [age_mean_sham, age_se_sham, med_count_sham, unmed_count_sham, f_count_sham,
                                          m_count_sham, med_m_count_sham, med_f_count_sham, unmed_m_count_sham,
                                          unmed_f_count_sham, cauca_count_sham, asian_count_sham, american_count_sham,
                                          hispanic_count_sham]})  # , 'pval': []})

    tmp_result_3.index = ['age-mean', 'age-se', 'med', 'unmed', 'F', 'M', 'med M', 'med F', 'unmed M', 'unmed F',
                          'caucasian', 'asian', 'african amarican', 'hispanic']

    if stat_tests:
        ######## stat tests ########

        ### unbias ###

        try:
            obs = np.array([[f_count_active, m_count_active], [f_count_sham, m_count_sham]])
            _, p_val_gender, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_gender = 'na'

        try:
            obs = np.array([[med_count_active, unmed_count_active], [med_count_sham, unmed_count_sham]])
            _, p_val_med, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_med = 'na'

        try:
            obs = np.array([[cauca_count_active, american_count_active, asian_count_active, hispanic_count_active],
                            [cauca_count_sham, american_count_sham, asian_count_sham, hispanic_count_sham]])
            _, p_val_race, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_race = 'na'

        df['GAD-v1-group'] = df['GAD-V1'].apply(make_groups)
        # df['GAD_GROUPS'] = df.apply(make_groups)

        gad_v1_small_active = np.maximum(df[(df.State == 'active') & (df['GAD-v1-group'] == '0-4')].shape[0], 0.001)
        gad_v1_medium1_active = np.maximum(df[(df.State == 'active') & (df['GAD-v1-group'] == '5-9')].shape[0], 0.001)
        gad_v1_medium2_active = np.maximum(df[(df.State == 'active') & (df['GAD-v1-group'] == '10-14')].shape[0], 0.001)
        gad_v1_high_active = np.maximum(df[(df.State == 'active') & (df['GAD-v1-group'] == '>=15')].shape[0], 0.001)

        gad_v1_small_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v1-group'] == '0-4')].shape[0], 0.001)
        gad_v1_medium1_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v1-group'] == '5-9')].shape[0], 0.001)
        gad_v1_medium2_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v1-group'] == '10-14')].shape[0], 0.001)
        gad_v1_high_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v1-group'] == '>=15')].shape[0], 0.001)

        try:
            obs = np.array([[gad_v1_small_active, gad_v1_medium1_active, gad_v1_medium2_active, gad_v1_high_active],
                            [gad_v1_small_sham, gad_v1_medium1_sham, gad_v1_medium2_sham, gad_v1_high_sham]])
            _, p_val_gad_v1, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_gad_v1 = 'na'

        df['HAM-v1-group'] = df['HAM-A-V1'].apply(make_groups)

        ham_v1_small_active = np.maximum(df[(df.State == 'active') & (df['HAM-v1-group'] == '0-4')].shape[0], 0.001)
        ham_v1_medium1_active = np.maximum(df[(df.State == 'active') & (df['HAM-v1-group'] == '5-9')].shape[0], 0.001)
        ham_v1_medium2_active = np.maximum(df[(df.State == 'active') & (df['HAM-v1-group'] == '10-14')].shape[0], 0.001)
        ham_v1_high_active = np.maximum(df[(df.State == 'active') & (df['HAM-v1-group'] == '>=15')].shape[0], 0.001)

        ham_v1_small_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v1-group'] == '0-4')].shape[0], 0.001)
        ham_v1_medium1_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v1-group'] == '5-9')].shape[0], 0.001)
        ham_v1_medium2_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v1-group'] == '10-14')].shape[0], 0.001)
        ham_v1_high_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v1-group'] == '>=15')].shape[0], 0.001)

        try:
            obs = np.array([[ham_v1_small_active, ham_v1_medium1_active, ham_v1_medium2_active, ham_v1_high_active],
                            [ham_v1_small_sham, ham_v1_medium1_sham, ham_v1_medium2_sham, ham_v1_high_sham]])
            _, p_val_ham_v1, _, _ = chi2_contingency(obs, correction=True)

        except:
            p_val_ham_v1 = {
                'na': [[ham_v1_small_active, ham_v1_medium1_active, ham_v1_medium2_active, ham_v1_high_active],
                       [ham_v1_small_sham, ham_v1_medium1_sham, ham_v1_medium2_sham, ham_v1_high_sham]]}

        cgis_v1_1_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 1)].shape[0], 0.001)
        cgis_v1_2_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 2)].shape[0], 0.001)
        cgis_v1_3_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 3)].shape[0], 0.001)
        cgis_v1_4_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 4)].shape[0], 0.001)
        cgis_v1_5_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 5)].shape[0], 0.001)
        cgis_v1_6_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 6)].shape[0], 0.001)

        cgis_v1_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V1'] == 1)].shape[0], 0.001)
        cgis_v1_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V1'] == 2)].shape[0], 0.001)
        cgis_v1_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V1'] == 3)].shape[0], 0.001)
        cgis_v1_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V1'] == 4)].shape[0], 0.001)
        cgis_v1_5_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V1'] == 5)].shape[0], 0.001)
        cgis_v1_6_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V1'] == 6)].shape[0], 0.001)

        cgie_v2_1_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V2'] == 1)].shape[0], 0.001)
        cgie_v2_2_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V2'] == 5)].shape[0], 0.001)
        cgie_v2_3_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V2'] == 9)].shape[0], 0.001)
        cgie_v2_4_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V2'] == 13)].shape[0], 0.001)

        cgie_v2_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V2'] == 1)].shape[0], 0.001)
        cgie_v2_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V2'] == 5)].shape[0], 0.001)
        cgie_v2_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V2'] == 9)].shape[0], 0.001)
        cgie_v2_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V2'] == 13)].shape[0], 0.001)

        cgie_v3_1_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V3'] == 1)].shape[0], 0.001)
        cgie_v3_2_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V3'] == 5)].shape[0], 0.001)
        cgie_v3_3_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V3'] == 9)].shape[0], 0.001)
        cgie_v3_4_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V3'] == 13)].shape[0], 0.001)

        cgie_v3_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V3'] == 1)].shape[0], 0.001)
        cgie_v3_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V3'] == 5)].shape[0], 0.001)
        cgie_v3_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V3'] == 9)].shape[0], 0.001)
        cgie_v3_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V3'] == 13)].shape[0], 0.001)

        cgie_v4_1_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V4'] == 1)].shape[0], 0.001)
        cgie_v4_2_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V4'] == 5)].shape[0], 0.001)
        cgie_v4_3_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V4'] == 9)].shape[0], 0.001)
        cgie_v4_4_active = np.maximum(df[(df.State == 'active') & (df['CGIe-V4'] == 13)].shape[0], 0.001)

        cgie_v4_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V4'] == 1)].shape[0], 0.001)
        cgie_v4_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V4'] == 5)].shape[0], 0.001)
        cgie_v4_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V4'] == 9)].shape[0], 0.001)
        cgie_v4_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIe-V4'] == 13)].shape[0], 0.001)

        try:
            obs = np.array([[cgie_v2_1_active, cgie_v2_2_active, cgie_v2_3_active, cgie_v2_4_active],
                            [cgie_v2_1_sham, cgie_v2_2_sham, cgie_v2_3_sham, cgie_v2_4_sham]])
            _, p_val_cgie_v2, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgie_v2 = 'na'

        try:
            obs = np.array([[cgie_v3_1_active, cgie_v3_2_active, cgie_v3_3_active, cgie_v3_4_active],
                            [cgie_v3_1_sham, cgie_v3_2_sham, cgie_v3_3_sham, cgie_v3_4_sham]])
            _, p_val_cgie_v3, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgie_v3 = 'na'

        try:
            obs = np.array([[cgie_v4_1_active, cgie_v4_2_active, cgie_v4_3_active, cgie_v4_4_active],
                            [cgie_v4_1_sham, cgie_v4_2_sham, cgie_v4_3_sham, cgie_v4_4_sham]])
            _, p_val_cgie_v4, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgie_v4 = 'na'

        cgis_v2_1_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V2'] == 1)].shape[0], 0.001)
        cgis_v2_2_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V2'] == 2)].shape[0], 0.001)
        cgis_v2_3_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V2'] == 3)].shape[0], 0.001)
        cgis_v2_4_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V2'] == 4)].shape[0], 0.001)
        cgis_v2_5_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V2'] == 5)].shape[0], 0.001)
        cgis_v2_6_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V2'] == 6)].shape[0], 0.001)

        cgis_v2_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V2'] == 1)].shape[0], 0.001)
        cgis_v2_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V2'] == 2)].shape[0], 0.001)
        cgis_v2_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V2'] == 3)].shape[0], 0.001)
        cgis_v2_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V2'] == 4)].shape[0], 0.001)
        cgis_v2_5_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V2'] == 5)].shape[0], 0.001)
        cgis_v2_6_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V2'] == 6)].shape[0], 0.001)

        cgis_v3_1_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 1)].shape[0], 0.001)
        cgis_v3_2_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 2)].shape[0], 0.001)
        cgis_v3_3_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 3)].shape[0], 0.001)
        cgis_v3_4_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 4)].shape[0], 0.001)
        cgis_v3_5_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 5)].shape[0], 0.001)
        cgis_v3_6_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V1'] == 6)].shape[0], 0.001)

        cgis_v3_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V3'] == 1)].shape[0], 0.001)
        cgis_v3_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V3'] == 2)].shape[0], 0.001)
        cgis_v3_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V3'] == 3)].shape[0], 0.001)
        cgis_v3_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V3'] == 4)].shape[0], 0.001)
        cgis_v3_5_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V3'] == 5)].shape[0], 0.001)
        cgis_v3_6_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V3'] == 6)].shape[0], 0.001)

        try:
            obs = np.array([[cgis_v1_1_active, cgis_v1_2_active, cgis_v1_3_active, cgis_v1_4_active, cgis_v1_5_active,
                             cgis_v1_6_active],
                            [cgis_v1_1_sham, cgis_v1_2_sham, cgis_v1_3_sham, cgis_v1_4_sham, cgis_v1_5_sham,
                             cgis_v1_6_sham]])
            _, p_val_cgis_v1, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgis_v1 = 'na'

        try:
            obs = np.array([[cgis_v2_1_active, cgis_v2_2_active, cgis_v2_3_active, cgis_v2_4_active, cgis_v2_5_active,
                             cgis_v2_6_active],
                            [cgis_v2_1_sham, cgis_v2_2_sham, cgis_v2_3_sham, cgis_v2_4_sham, cgis_v2_5_sham,
                             cgis_v2_6_sham]])
            _, p_val_cgis_v2, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgis_v2 = 'na'

        try:

            obs = np.array([[cgis_v3_1_active, cgis_v3_2_active, cgis_v3_3_active, cgis_v3_4_active, cgis_v3_5_active,
                             cgis_v3_6_active],
                            [cgis_v3_1_sham, cgis_v3_2_sham, cgis_v3_3_sham, cgis_v3_4_sham, cgis_v3_5_sham,
                             cgis_v3_6_sham]])
            _, p_val_cgis_v3, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgis_v3 = 'na'

        df['GAD-v4-group'] = df['GAD-V4'].apply(make_groups)

        gad_v4_small_active = np.maximum(df[(df.State == 'active') & (df['GAD-v4-group'] == '0-4')].shape[0], 0.001)
        gad_v4_medium1_active = np.maximum(df[(df.State == 'active') & (df['GAD-v4-group'] == '5-9')].shape[0], 0.001)
        gad_v4_medium2_active = np.maximum(df[(df.State == 'active') & (df['GAD-v4-group'] == '10-14')].shape[0], 0.001)
        gad_v4_high_active = np.maximum(df[(df.State == 'active') & (df['GAD-v4-group'] == '>=15')].shape[0], 0.001)

        gad_v4_small_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v4-group'] == '0-4')].shape[0], 0.001)
        gad_v4_medium1_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v4-group'] == '5-9')].shape[0], 0.001)
        gad_v4_medium2_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v4-group'] == '10-14')].shape[0], 0.001)
        gad_v4_high_sham = np.maximum(df[(df.State == 'sham') & (df['GAD-v4-group'] == '>=15')].shape[0], 0.001)

        try:
            obs = np.array([[gad_v4_small_active, gad_v4_medium1_active, gad_v4_medium2_active, gad_v4_high_active],
                            [gad_v4_small_sham, gad_v4_medium1_sham, gad_v4_medium2_sham, gad_v4_high_sham]])
            _, p_val_gad_v4, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_gad_v4 = 'na'

        df['HAM-v4-group'] = df['HAM-A-V4'].apply(make_groups)

        ham_v4_small_active = np.maximum(df[(df.State == 'active') & (df['HAM-v4-group'] == '0-4')].shape[0], 0.001)
        ham_v4_medium1_active = np.maximum(df[(df.State == 'active') & (df['HAM-v4-group'] == '5-9')].shape[0], 0.001)
        ham_v4_medium2_active = np.maximum(df[(df.State == 'active') & (df['HAM-v4-group'] == '10-14')].shape[0], 0.001)
        ham_v4_high_active = np.maximum(df[(df.State == 'active') & (df['HAM-v4-group'] == '>=15')].shape[0], 0.001)

        ham_v4_small_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v4-group'] == '0-4')].shape[0], 0.001)
        ham_v4_medium1_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v4-group'] == '5-9')].shape[0], 0.001)
        ham_v4_medium2_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v4-group'] == '10-14')].shape[0], 0.001)
        ham_v4_high_sham = np.maximum(df[(df.State == 'sham') & (df['HAM-v4-group'] == '>=15')].shape[0], 0.001)

        try:
            obs = np.array([[ham_v4_small_active, ham_v4_medium1_active, ham_v4_medium2_active, ham_v4_high_active],
                            [ham_v4_small_sham, ham_v4_medium1_sham, ham_v4_medium2_sham, ham_v4_high_sham]])
            _, p_val_ham_v4, _, _ = chi2_contingency(obs, correction=True)

        except:
            p_val_ham_v4 = 'na'

        cgis_v4_1_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V4'] == 1)].shape[0], 0.001)
        cgis_v4_2_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V4'] == 2)].shape[0], 0.001)
        cgis_v4_3_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V4'] == 3)].shape[0], 0.001)
        cgis_v4_4_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V4'] == 4)].shape[0], 0.001)
        cgis_v4_5_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V4'] == 5)].shape[0], 0.001)
        cgis_v4_6_active = np.maximum(df[(df.State == 'active') & (df['CGIs-V4'] == 6)].shape[0], 0.001)

        cgis_v4_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V4'] == 1)].shape[0], 0.001)
        cgis_v4_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V4'] == 2)].shape[0], 0.001)
        cgis_v4_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V4'] == 3)].shape[0], 0.001)
        cgis_v4_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V4'] == 4)].shape[0], 0.001)
        cgis_v4_5_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V4'] == 5)].shape[0], 0.001)
        cgis_v4_6_sham = np.maximum(df[(df.State == 'sham') & (df['CGIs-V4'] == 6)].shape[0], 0.001)

        try:
            obs = np.array([[cgis_v4_1_active, cgis_v4_2_active, cgis_v4_3_active, cgis_v4_4_active, cgis_v4_5_active,
                             cgis_v4_6_active],
                            [cgis_v4_1_sham, cgis_v4_2_sham, cgis_v4_3_sham, cgis_v4_4_sham, cgis_v4_5_sham,
                             cgis_v4_6_sham]])
            _, p_val_cgis_v4, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgis_v4 = 'na'

        cgig_v4_1_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 1)].shape[0], 0.001)
        cgig_v4_2_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 2)].shape[0], 0.001)
        cgig_v4_3_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 3)].shape[0], 0.001)
        cgig_v4_4_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 4)].shape[0], 0.001)
        cgig_v4_5_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 5)].shape[0], 0.001)
        cgig_v4_6_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 6)].shape[0], 0.001)

        cgig_v4_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V4'] == 1)].shape[0], 0.001)
        cgig_v4_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V4'] == 2)].shape[0], 0.001)
        cgig_v4_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V4'] == 3)].shape[0], 0.001)
        cgig_v4_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V4'] == 4)].shape[0], 0.001)
        cgig_v4_5_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V4'] == 5)].shape[0], 0.001)
        cgig_v4_6_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V4'] == 6)].shape[0], 0.001)

        cgig_v2_1_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V2'] == 1)].shape[0], 0.001)
        cgig_v2_2_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V2'] == 2)].shape[0], 0.001)
        cgig_v2_3_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V2'] == 3)].shape[0], 0.001)
        cgig_v2_4_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V2'] == 4)].shape[0], 0.001)
        cgig_v2_5_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V2'] == 5)].shape[0], 0.001)
        cgig_v2_6_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V2'] == 6)].shape[0], 0.001)

        cgig_v2_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V2'] == 1)].shape[0], 0.001)
        cgig_v2_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V2'] == 2)].shape[0], 0.001)
        cgig_v2_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V2'] == 3)].shape[0], 0.001)
        cgig_v2_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V2'] == 4)].shape[0], 0.001)
        cgig_v2_5_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V2'] == 5)].shape[0], 0.001)
        cgig_v2_6_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V2'] == 6)].shape[0], 0.001)

        cgig_v3_1_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 1)].shape[0], 0.001)
        cgig_v3_2_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 2)].shape[0], 0.001)
        cgig_v3_3_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 3)].shape[0], 0.001)
        cgig_v3_4_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 4)].shape[0], 0.001)
        cgig_v3_5_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 5)].shape[0], 0.001)
        cgig_v3_6_active = np.maximum(df[(df.State == 'active') & (df['CGIg-V4'] == 6)].shape[0], 0.001)

        cgig_v3_1_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V3'] == 1)].shape[0], 0.001)
        cgig_v3_2_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V3'] == 2)].shape[0], 0.001)
        cgig_v3_3_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V3'] == 3)].shape[0], 0.001)
        cgig_v3_4_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V3'] == 4)].shape[0], 0.001)
        cgig_v3_5_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V3'] == 5)].shape[0], 0.001)
        cgig_v3_6_sham = np.maximum(df[(df.State == 'sham') & (df['CGIg-V3'] == 6)].shape[0], 0.001)

        try:
            obs = np.array([[cgig_v4_1_active, cgig_v4_2_active, cgig_v4_3_active, cgig_v4_4_active, cgig_v4_5_active,
                             cgig_v4_6_active],
                            [cgig_v4_1_sham, cgig_v4_2_sham, cgig_v4_3_sham, cgig_v4_4_sham, cgig_v4_5_sham,
                             cgig_v4_6_sham]])
            _, p_val_cgig_v4, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgig_v4 = 'na'

        try:
            obs = np.array([[cgig_v2_1_active, cgig_v2_2_active, cgig_v2_3_active, cgig_v2_4_active, cgig_v2_5_active,
                             cgig_v2_6_active],
                            [cgig_v2_1_sham, cgig_v2_2_sham, cgig_v2_3_sham, cgig_v2_4_sham, cgig_v2_5_sham,
                             cgig_v2_6_sham]])
            _, p_val_cgig_v2, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgig_v2 = 'na'

        try:
            obs = np.array([[cgig_v3_1_active, cgig_v3_2_active, cgig_v3_3_active, cgig_v3_4_active, cgig_v3_5_active,
                             cgig_v3_6_active],
                            [cgig_v3_1_sham, cgig_v3_2_sham, cgig_v3_3_sham, cgig_v3_4_sham, cgig_v3_5_sham,
                             cgig_v3_6_sham]])
            _, p_val_cgig_v3, _, _ = chi2_contingency(obs, correction=True)
        except:
            p_val_cgig_v3 = 'na'

        results = tmp_result.round(2), tmp_result_2.round(2), tmp_result_3.round(2), {'gender': p_val_gender,
                                                                                      'medication status': p_val_med,
                                                                                      'race': p_val_race,
                                                                                      'gad-v1': p_val_gad_v1,
                                                                                      'gad-v4': p_val_gad_v4,
                                                                                      'ham-v1': p_val_ham_v1,
                                                                                      'ham-v4': p_val_ham_v4,
                                                                                      'cgis-v1': p_val_cgis_v1,
                                                                                      'cgis-v2': p_val_cgis_v2,
                                                                                      'cgis-v3': p_val_cgis_v3,
                                                                                      'cgis-v4': p_val_cgis_v4,
                                                                                      'cgig-v2': p_val_cgig_v2,
                                                                                      'cgig-v3': p_val_cgig_v3,
                                                                                      'cgig-v4': p_val_cgig_v4,
                                                                                      'cgie-v2': p_val_cgie_v2,
                                                                                      'cgie-v3': p_val_cgie_v3,
                                                                                      'cgie-v4': p_val_cgie_v4}

    else:
        results = tmp_result.round(2), tmp_result_2.round(2), tmp_result_3.round(2)

    return results
