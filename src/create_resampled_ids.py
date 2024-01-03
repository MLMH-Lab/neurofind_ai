#!/usr/bin/env python3
"""Script to create the ids lists for the further analysis.

This script undersamples UK Biobank Scanner 01 to ensure a more even age distribution.

After undersampling, subjects with too many extreme values in the regional volumes are removed (using sigma clipping).
"""
from itertools import cycle
from pathlib import Path
import tempfile

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns

from src import definitions as vars
from src.utils import LogRun, PythonLiteralOption, combine_dataset_files

colors_fullset = cycle(["#FF7070", "#00999e", "#6c6c6c", "#f7ef69", "#ff9800"])


def remove_small_scanners(participants, ids, set_name, min_size=10):
    """Remove scanners with very small size sample after matching.

    Parameters
    ----------
    participants : NDFrame of shape (n_subjects, n_features)
        DataFrame with the demographic data for all participants (before matching).
    ids : NDFrame  of shape (n_subjects)
        DataFrame with a set of ids
    set_name : str
        Set on which the function will be applied to: 'train' or 'test'
    min_size : int, default: 10
        Minimum size sample permitted.

    Returns
    -------
    ids: NDFrame of shape (n_subjects)
        Dataframe with updated IDs.
    """
    ids_df = pd.merge(participants, ids, on="id")

    scanner_size = ids_df.groupby(["Dataset"]).size()
    scanner_remove = scanner_size[scanner_size.values <= min_size].index.tolist()
    ids = ids_df[~ids_df["Dataset"].isin(scanner_remove)][["id"]]

    if not scanner_remove:
        print("No scanners from the %s set were removed due to sample size too small" % set_name)
    else:
        print("%s scanners from the %s set were removed due to sample size too small" % (len(scanner_remove), set_name))

    return ids


def sigma_clipping_exclude_step(dataframe, features=vars.COLUMNS_NAME, n_sigmas=3, n_extreme=10):
    """Exclude subjects with too many extreme measurements.

    Parameters
    ----------
    dataset : NDFrame of shape [n_subjects, n_features]
        DataFrame with the subject data.
    features : list, default=vars.COLUMNS_NAME
        List of the features to be considered on the effect size test. Default are the 101 FreeSurfer regions.
    n_sigmas : float, default=3
        Threshold for excluding subsets in standard deviations from the mean.
    n_extreme : int, default=10
        Number of features out of the `n_sigmas` range.

    Returns
    -------
    dataframe : NDFrame of shape [n_subjects, n_features]
        DataFrame with the cleaned dataset.
    """
    delta_n = 1
    while delta_n != 0:
        n_0 = len(dataframe)
        hc_filt = dataframe["Diagn"] == 1
        mean = dataframe[hc_filt][features].mean(axis=0)
        std = dataframe[hc_filt][features].std(axis=0)
        clip_filt = ((dataframe[features] - mean) ** 2) ** 0.5 > n_sigmas * std
        clip_filt = clip_filt.sum(axis=1) > n_extreme
        dataframe = dataframe[~clip_filt]
        delta_n = n_0 - len(dataframe)
    return dataframe


def sigma_clipping_exclude(df, features=vars.COLUMNS_NAME, n_sigmas=3, n_extreme=10):
    """Perform sigma clipping within each scanner using sigma_clipping_exclude_step function.

    Parameters
    ----------
    df : NDFrame of shape [n_subjects, n_features]
        DataFrame with the subject data.
    features : list, default=vars.COLUMNS_NAME
        List of the features to be considered on the effect size test. Default are the 101 FreeSurfer regions.
    n_sigmas : float, default=3
        Threshold for excluding subsets in standard deviations from the mean.
    n_extreme : int, default=10
        Number of features out of the `n_sigmas` range.

    Returns
    -------
    dataframe : NDFrame of shape [n_subjects, n_features]
        DataFrame with the cleaned dataset.

    """
    delta_total = len(df)
    df = pd.concat(
        [sigma_clipping_exclude_step(dset, features, n_sigmas, n_extreme)
         for _, dset in df.groupby("Dataset")]
    )
    delta_total -= len(df)
    click.echo(
        f"================== Total of {delta_total} filtered by the sigma =======================")
    return df.drop(columns=["Diagn", "Dataset"])


def scatter_plot(df_original, df_resampled, x='Age', y='Left-Lateral-Ventricle', hue="Diagn",
                 colors=["#6c6c6c", "#FF7070"]):
    """Scatter plot comparing original and resampled sample across ages for one brain region.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    x : str, default='Age'
        Name of variable on x axis of scatter plot. Must be a column in both DataFrames.
    y : str, default='Left-Lateral-Ventricle'
        Name of variable on y axis of scatter plot. Must be a column in both DataFrames.
    hue : str, default='Diagn'
        Name of variable to split the data by.  Must be a column in both DataFrames.
    colors : list of str, default=["#6c6c6c", "#FF7070"]
        List of colour names for hue.

    Returns
    -------
    fig : plt
        Scatter plot.
    """
    ymin, ymax = df_original[y].quantile([0.0001, 0.995])
    xmin, xmax = df_original[x].quantile([0.0001, 0.995])
    labels = pd.concat([df_original[hue], df_resampled[hue]]).unique()
    possible_markers = ["1", "x", "+", "2", "3"]
    markers = {label: mark for label, mark in zip(labels, cycle(possible_markers))}
    fig, axis = plt.subplots(figsize=(16*0.7, 9*0.7), ncols=2, nrows=1, sharey=True, sharex=True)
    sns.scatterplot(data=df_original, x=x, y=y, hue=hue, markers=markers,
                    palette=colors, style=hue, ax=axis[0], alpha=0.7, s=50)
    sns.scatterplot(data=df_resampled, x=x, y=y, hue=hue, markers=markers,
                    palette=colors, style=hue, ax=axis[1], alpha=0.7, s=50)
    axis[0].set_xlim(xmin, xmax)
    axis[0].set_ylim(ymin, ymax)
    plt.tight_layout()
    return fig


def KDE_plot(df_original, df_resampled, x="Age", hue="Diagn", colors=["#6c6c6c", "#00999e", "#FF7070"]):
    """Kernel density estimate (KDE) plot comparing original and resampled sample across ages.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    x : str, default='Age'
        Name of variable on x axis of KDE plot. Must be a column in both DataFrames.
    hue : str, default='Diagn'
        Name of variable to split the data by.  Must be a column in both DataFrames.
    colors : list of str, default=["#6c6c6c", "#FF7070"]
        List of colour names for hue.

    Returns
    -------
    fig : plt
        KDE plot.
    """
    xmin, xmax = pd.concat([df_original, df_resampled])[x].quantile(q=[0.0001, 0.9999])
    fig, axis = plt.subplots(figsize=(16*0.7, 9*0.7), ncols=1, nrows=3, sharex=True)
    sns.kdeplot(data=df_original, x=x, hue=hue,
                fill=True, common_norm=True, palette=colors, alpha=.4, linewidth=2, ax=axis[0])
    sns.kdeplot(data=df_original, x=x, hue=hue,
                fill=True, common_norm=False, palette=colors, alpha=.4, linewidth=2, ax=axis[1])
    sns.kdeplot(data=df_resampled, x=x, hue=hue, fill=True, common_norm=True, palette=colors,
                alpha=.5, linewidth=2, ax=axis[2])
    axis[0].grid(True)
    axis[1].grid(True)
    axis[2].grid(True)
    axis[0].set_title("Original Data - Common Normalization", y=.98, loc="left")
    axis[1].set_title("Original Data - Individual Normalization", y=.98, loc="left")
    axis[2].set_title("resampled Data  - Individual Normalization", y=.98, loc="left")
    plt.xlim(xmin, xmax)
    plt.minorticks_on()
    plt.tight_layout()
    return fig


def histogram_plot(df_original, df_resampled, x="Age", hue="Gender", stacked=False, colors=["#13c1c7", "#f7ef69"], colors2=["#13c1c7", "#f7ef69"]):
    """Histogram plot comparing original and resampled sample across ages, split by gender.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    x : str, default='Age'
        Name of variable on x axis of histogram. Must be a column in both DataFrames.
    hue : str, default='Gender'
        Name of variable to split the data by. Must be a column in both DataFrames.
    stacked : bool, default=False
        Whether multiple categories should be represented as stacked or overlapping.
    colors : list of str, default=["#6c6c6c", "#FF7070"]
        List of colour names for hue.

    Returns
    -------
    fig : plt
        Histogram plot.
    """
    xmin, xmax = pd.concat([df_original, df_resampled])[x].quantile(q=[0.0001, 0.9999])
    bins = np.arange(xmin, xmax + 1)
    fig, axis = plt.subplots(figsize=(16*0.7, 9*0.7), ncols=1, nrows=2, sharex=True)
    if stacked is True:
        sns.histplot(data=df_original, bins=bins, x=x, hue=hue, multiple='stack', palette=colors, ax=axis[0])
        sns.histplot(data=df_resampled, bins=bins, x=x, hue=hue, multiple='stack', palette=colors2, ax=axis[1])
    else:
        sns.histplot(data=df_original, bins=bins, x=x, hue=hue, palette=colors, ax=axis[0])
        sns.histplot(data=df_resampled, bins=bins, x=x, hue=hue, palette=colors2, ax=axis[1])
    axis[0].grid(True)
    axis[1].grid(True)
    axis[0].set_title("Original Data", y=0.98, loc="left")
    axis[1].set_title("resampled Data", y=0.98, loc="left")
    sns.move_legend(axis[0], loc='upper left', bbox_to_anchor=(1.02, 1))
    sns.move_legend(axis[1], loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.minorticks_on()
    plt.tight_layout()
    return fig


def violin_plot_gender_age(df_original, df_resampled, x="Age", y="Diagn", hue="Gender", colors=["#13c1c7", "#f7ef69"]):
    """Violin plot comparing original and resampled sample across ages by diagnosis, split by age and gender.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    x : str, default='Age'
        Name of variable on x axis of violin plot. Must be a column in both DataFrames.
    y : str, default='Diagn'
        Name of variable on y axis of violin plot. Must be a column in both DataFrames.
    hue : str, default='Gender'
        Name of variable to split the data by. Must be a column in both DataFrames.
    colors : list of str, default=["#6c6c6c", "#FF7070"]
        List of colour names for hue.

    Returns
    -------
    fig : plt
        Histogram plot.
    """
    fig, axis = plt.subplots(figsize=(16*0.7, 9*0.7), nrows=2, sharex=True)
    sns.violinplot(x=x, y=y, hue=hue, data=df_original, ax=axis[0], palette=cycle(colors), width=0.95,
                   inner="box", scale="width", bw=0.45, scale_hue=False, split=True, cut=0.1, orient="h", fontsize=7)
    sns.violinplot(x=x, y=y, hue=hue, data=df_resampled, ax=axis[1],
                   palette=cycle(colors), width=0.95, inner="box", scale="width", bw=0.45, scale_hue=False, split=True,
                   cut=0.1, orient="h", fontsize=7)
    axis[0].grid(True)
    axis[1].grid(True)
    axis[0].minorticks_on()
    axis[1].minorticks_on()
    axis[0].set_title("Original Data", y=0.98, loc="left")
    axis[1].set_title("resampled Data", y=0.98, loc="left")
    plt.tight_layout()
    return fig


def undersample_biobank(df, create_vars, label_vars=['Age'], n_class=100, random_state=243):
    """Undersample UK Biobank-Scanner01 dataset to ensure a more even age distribution.

    This function selects the Biobank-Scanner01 dataset, applies undersampling, and then merges with the
    original dataset again.

    Please note: This function does not take gender balance into account.

    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame with data.
    create_vars : list of str
        List of column names
    label_vars : list of str, default=['Age']
        List of column names that are taken into account when resampling, e.g. Age
    n_class int, default=100
        Subjects per age group in Biobank-Scanner01 after resampling.
    random_state : int, default=243
        Random seed.

    Returns
    -------
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    """
    # Select Biobank scanner, apply undersampling, then merge to original again
    df_biobank = df[df.scanner == 'BIOBANK-SCANNER01']
    df_no_biobank = df[df.scanner != 'BIOBANK-SCANNER01']
    # print(df.shape, df_biobank.shape, df_no_biobank.shape)
    print('The sample size of BIOBANK-SCANNER01 before resampling is ', df_biobank.shape[0])

    # dataset_map contains all the scanners in train_test
    dataset_map = dict(enumerate(df_biobank['Dataset'].cat.categories))

    X, y = df_biobank[create_vars], df_biobank[label_vars]

    # -------------------------------------------------------------------------------------
    # Create dictionary of age groups to undersample
    larger = df_biobank.Age.value_counts() > n_class
    filtered_values_larger = df_biobank.Age.value_counts().loc[larger].index
    filtered_df_biobank_larger = df_biobank[df_biobank.Age.isin(filtered_values_larger)]

    sampling_strategy_dict_larger = {}
    for i in filtered_df_biobank_larger.Age.unique():
        sampling_strategy_dict_larger[i] = n_class

    # Define undersampling function
    under = RandomUnderSampler(sampling_strategy=sampling_strategy_dict_larger, random_state=random_state)

    # Implement undersampling
    X_resampled, y_resampled = under.fit_resample(X, y)

    df_biobank_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    df_biobank_resampled.loc[:, ("Dataset")] = df_biobank_resampled.dataset_code.map(dataset_map).astype("category")
    df_biobank_resampled.loc[:, ("scanner")] = df_biobank_resampled["Dataset"]
    df_biobank_resampled.index.name = "id"
    df_biobank_resampled.reset_index(inplace=True)
    df_biobank_resampled.loc[:, ("participant_id")] = df_biobank_resampled["id"]

    # Renaming IDs to avoid duplication issues in rest of pipeline
    df_biobank_resampled.id = df_biobank_resampled.Dataset.astype(str) + '_' + df_biobank_resampled.id.astype(str)

    print('The sample size of BIOBANK-SCANNER01 after resampling is ', df_biobank_resampled.shape[0])

    # Merge resampled Biobank df into original dataframe
    df_resampled = pd.concat([df_biobank_resampled, df_no_biobank])
    df_resampled.reset_index(inplace=True)

    return df_resampled


def plot_freesurfer_regions(df_original, df_resampled, x=vars.COLUMNS_NAME):
    """Bar plot comparing mean regional volumes in original and resampled sample.
    This could also be used to assess effect of harmonisation.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    x : str, default=COLUMNS_NAME
        Name of variables on x axis of bar plot. Must be columns in both DataFrames.
        Default are the 101 FreeSurfer regional volumes.
    hue : str, default='Dataset'
        Name of variable to split the data by (if any). Must be a column in both DataFrames.

    Returns
    -------
    fig : plt
        Bar plot.
    """
    # Get mean regional volumes
    df_original_means = df_original[x].mean()
    df_resampled_means = df_resampled[x].mean()

    # Create bar plot
    ind = np.arange(101)
    width = 0.40
    fig = plt.figure(figsize=(28, 10))
    plt.bar(ind, df_original_means, width, label='Original')
    plt.bar(ind + width, df_resampled_means, width, label='Resampled')

    plt.ylabel('Mean volume')
    plt.title('Mean FreeSurfer volumes before and after resampling')
    # plt.rcParams['figure.figsize'] = (20, 10)
    plt.xticks(ind + width / 2, vars.COLUMNS_NAME, rotation=45, ha='right')
    plt.legend(loc='best')

    return fig


def make_resample_plot(original, resampled, temp_dir):
    """Plot resampled data.

    This function makes use of most of the other functions in this script.

    Parameters
    ----------
    original : DataFrame
        Pandas DataFrame with original data.
    resampled : DataFrame
        Pandas DataFrame with resampled data.
    temp_dir : Path
        Path to temporary directory.

    Returns
    -------
    plot_paths : List of Path
        List of paths to saved plots comparing original and resampled data, incl.
        - violin plot of age vs gender,
        - histogram of age vs gender,
        - histogram of age vs diagnosis,
        - KDE plot of age vs diagnosis,
        - scatter plots of age vs each region of interest (split by diagnosis),
        - scatter plot of signal-to-noise ratio in whole brain vs grey matter only (split by diagnosis), (?)
    """
    plot_paths = []

    # Violin plot for age (x axis) and gender (y axis)
    gender_age_violin = violin_plot_gender_age(original, resampled)
    gender_age_violin_path = f"{temp_dir}/gender_age_violin.png"
    gender_age_violin.savefig(gender_age_violin_path, dpi=150)
    plt.close()
    plot_paths.append(gender_age_violin_path)

    # Histogram for age distribution, split by gender
    COLORS = [color for color, _ in zip(colors_fullset, pd.concat([original.Gender, resampled.Gender]).unique())]
    histogram_age_gender = histogram_plot(original, resampled, x="Age", hue="Gender", colors=COLORS, colors2=COLORS)
    histogram_age_gender_path = f"{temp_dir}/histogram_age_gender.png"
    histogram_age_gender.savefig(histogram_age_gender_path, dpi=150)
    plt.close()
    plot_paths.append(histogram_age_gender_path)

    # Histogram for age distribution, split by diagnosis
    COLORS_DIAGN = [color for color, _ in zip(colors_fullset, pd.concat([original.Diagn, resampled.Diagn]).unique())]
    histogram_age_diagn = histogram_plot(original, resampled, x="Age", hue="Diagn",
                                         colors=COLORS_DIAGN, colors2=COLORS_DIAGN)
    histogram_age_diagn_path = f"{temp_dir}/histogram_age_diagn.png"
    histogram_age_diagn.savefig(histogram_age_diagn_path, dpi=150)
    plt.close()
    plot_paths.append(histogram_age_diagn_path)

    # Histogram for age distribution, split by scanner
    n_scanner_categories = original.scanner.value_counts().shape[0]
    n_scanner_categories_2 = resampled.scanner.value_counts().shape[0]
    COLORS_SCANNER = sns.color_palette('Paired', n_colors=n_scanner_categories)
    COLORS_SCANNER_2 = sns.color_palette('Paired', n_colors=n_scanner_categories_2)
    histogram_age_scanner = histogram_plot(original, resampled, x="Age",
                                           hue="scanner", stacked=True,
                                           colors=COLORS_SCANNER, colors2=COLORS_SCANNER_2)
    histogram_age_scanner_path = f"{temp_dir}/histogram_age_scanner.png"
    histogram_age_scanner.savefig(histogram_age_scanner_path, dpi=150)
    plt.close()
    plot_paths.append(histogram_age_scanner_path)

    KDE_age_diagn = KDE_plot(original, resampled, x="Age", hue="Diagn", colors=COLORS_DIAGN)
    KDE_age_diagn_path = f"{temp_dir}/KDE_age_diagn.png"
    KDE_age_diagn.savefig(KDE_age_diagn_path, dpi=150)
    plt.close()
    plot_paths.append(KDE_age_diagn_path)

    yvar = vars.COLUMNS_NAME[1]
    scatter_plot_age_diagn = scatter_plot(original, resampled, x="Age", y=yvar,
                                          hue="Diagn", colors=COLORS_DIAGN)
    scatter_plot_age_diagn_path = f"{temp_dir}/scatter_age_{yvar}_diagn.png"
    scatter_plot_age_diagn.savefig(scatter_plot_age_diagn_path, dpi=150)
    plt.close()
    plot_paths.append(scatter_plot_age_diagn_path)

    xvar = "snr_total"
    yvar = "snr_gm"
    scatter_snr_diagn = scatter_plot(original, resampled, x=xvar, y=yvar, hue="Diagn", colors=COLORS_DIAGN)
    scatter_plot_path = f"{temp_dir}/scatter_{xvar}_{yvar}_diagn.png"
    scatter_snr_diagn.savefig(scatter_plot_path, dpi=150)
    plt.close()
    plot_paths.append(scatter_plot_path)

    # Plot mean volums of FreeSurfer regions in original vs resampled datasets
    fs_regions_plot = plot_freesurfer_regions(original, resampled)
    fs_regions_plot_path = f"{temp_dir}/fs_regional_volumes.png"
    fs_regions_plot.savefig(fs_regions_plot_path, dpi=150)
    plt.close()
    plot_paths.append(fs_regions_plot_path)

    return plot_paths


def resample_data_preparation(input_path, disease_label_list, validation_datasets_list, min_size, seed,
                              ignore_test=False):
    """Create dataset for resampling.

    Parameters
    ----------
    input_path : str
        Path for the clean data, the output of the script src.clean_data. (?)
    disease_label_list : list of integers
        List of the disease codes to include.
    validation_datasets_list: list of strings
        List of strings that represents the datasets that must be reserved for validation.
    min_size : int
        Minimun number of subjects in a single scanner.
    seed : int
        Random seed.
    ignore_test : boolean, default=False
        Boolean to see if clinical samples (Diagn != 1) should be ignored. If false, clinical data are included.

    Returns
    -------
    train_test : DataFrame
        Pandas DataFrame including data for train (and test, if ignore_test=False) only.
    validation : DataFrame
        Pandas DataFrame including data for validation only.
    """
    dtype_dict = {"Gender": int, "Age": int, "Diagn": int, "Dataset": "category"}
    dataset = combine_dataset_files(input_path).astype(dtype_dict)

    # Normalise regional volumes by total brain volume
    norm_var = "EstimatedTotalIntraCranialVol"
    dataset.loc[:, vars.COLUMNS_NAME] = dataset[vars.COLUMNS_NAME].divide(
        dataset[norm_var].astype(float).values, axis=0)

    # Create subset of validation data
    is_validation = pd.concat([dataset.Dataset.str.contains(scan)
                              for scan in validation_datasets_list], axis=1).any(axis=1)
    if is_validation.sum() == 0:
        raise ValueError("There are no subjects in the validation sample.")
    validation = dataset[is_validation]

    # Include/exclude clinical data (test data) and create final dataset
    if ignore_test:
        train = dataset[(dataset["Diagn"] == 1) & (~is_validation)]
        train_ids = remove_small_scanners(dataset, train[["id"]], "train", min_size)
        train_test = train[train.id.isin(train_ids.id)]
    else:
        is_participant_target = dataset.Diagn.isin([1] + disease_label_list)
        datasets_with_target_disorder = dataset[(~is_validation) & is_participant_target].Dataset.unique()
        is_test = dataset.Dataset.isin(datasets_with_target_disorder) & dataset.Diagn.isin([1] + disease_label_list)
        test = dataset[is_test]
        test_ids = remove_small_scanners(dataset, test[["id"]], "test", min_size)
        test = test[test.id.isin(test_ids.id)]

        train = dataset[(dataset["Diagn"] == 1) & (~is_validation) & (~is_test)]
        train_ids = remove_small_scanners(dataset, train[["id"]], "train", min_size)
        train = train[train.id.isin(train_ids.id)]

        train_test = pd.concat([train, test]).set_index("id")
    train_test.loc[:, "dataset_code"] = train_test.Dataset.cat.codes

    return train_test, validation


def save_resampled_data(resampled, validation, disease_label_list, temp_dir, seed,
                        rois=vars.COLUMNS_NAME, qc_vars=vars.QC_FEATURES + ['mriqc_prob']):
    """Save resampled data.

    Parameters
    ----------
    resampled : DataFrame
        Pandas DataFrame containing resampled train/test data.
    validation : DataFrame
        Pandas DataFrame containing validation data.
    disease_label_list : list of integers
        List of the disease codes to include.
    temp_dir : Path
        Path to temporary directory.
    seed : int
        Random seed.
    rois : list of str, default=vars.COLUMNS_NAME
        List of included brain regions. The default is 101 FreeSurfer-preprocessed volumes.
    qc_vars : list of str, default=vars.QC_FEATURES + ['mriqc_prob']
        List of variables from quality checking, incl. final scores from MRIQC (raw data)

    Returns
    -------
    metrics_dict : dict
        Sample sizes of test dataset split by diagnosis.
    """
    metrics_dict = {}

    # Save separate files for resampled train, test and validation IDs
    resampled.loc[:, ("participant_id")] = resampled["id"]
    validation.loc[:, ("participant_id")] = validation["id"]
    train, test = train_test_split(resampled, random_state=seed)
    train_ids_path = f"{temp_dir}/train_ids.csv"
    test_ids_path = f"{temp_dir}/test_ids.csv"
    validation_ids_path = f"{temp_dir}/validation_ids.csv"
    train[["id"]].to_csv(train_ids_path, index=False)
    test[["id"]].to_csv(test_ids_path, index=False)
    validation[["id"]].to_csv(validation_ids_path, index=False)

    # Save separate test ID files for each included diagnosis, not used in Neurofind
    for disease_label in disease_label_list:
        disease_ids_path = f"{temp_dir}/test_ids_{disease_label:03d}.csv"
        disease_selection = test.Diagn.isin([1, int(disease_label)])
        if disease_selection.sum() == 0:
            print(test.Diagn.unique(), test.shape, test[disease_selection].shape)
            raise ValueError(f"Empty test_ids_{disease_label:03d}")
        tests = test[disease_selection]
        tests[["id"]].to_csv(disease_ids_path, index=False)

        # Get sample size for each included diagnosis in the test data
        metrics_dict[f"N_test_ids_{disease_label:03d}"] = tests.shape[0]

    # Save separate validation ID files for each included diagnosis
    for disease_label in disease_label_list:
        validation_ids_path = f"{temp_dir}/validation_ids_{disease_label:03d}.csv"
        is_participant_target = validation.Diagn == disease_label
        datasets_with_target_disorder = validation[is_participant_target].Dataset.unique()
        is_in_datasets_with_target_disorder = validation.Dataset.isin(datasets_with_target_disorder)
        tmp_val = validation[is_in_datasets_with_target_disorder & validation.Diagn.isin([1, disease_label])]
        tmp_val[["id"]].to_csv(validation_ids_path, index=False)

    # Create one dataset contain all data (resampled train and test + validation data)
    validation = validation.drop(columns=[var for var in validation.columns if var not in resampled.columns])
    resampled = resampled.drop(columns=[var for var in resampled.columns if var not in validation.columns])
    dataset = pd.concat([resampled, validation])

    norm_var = "EstimatedTotalIntraCranialVol"
    dataset.loc[:, rois] = dataset[rois].mul(
        dataset[norm_var].astype(float).values, axis=0)

    # Create and save three separate files containing demographic data, FreeSurfer data, or QC data only
    participants = dataset[["id", "participant_id", "Age", "Gender", "Diagn", "Dataset"]]
    fs = dataset[["id", "EstimatedTotalIntraCranialVol"] + rois]
    qc = dataset[["id"] + qc_vars]

    participants_path = f"{temp_dir}/participants.tsv"
    fs_path = f"{temp_dir}/freesurferData.csv"
    qc_path = f"{temp_dir}/qc.tsv"

    participants.to_csv(participants_path, sep="\t", index=False)
    fs.to_csv(fs_path, index=False)
    qc.to_csv(qc_path, sep="\t", index=False)

    return metrics_dict


@click.command(help="")
@click.option("--input_path")
@click.option("--validation_datasets_list", cls=PythonLiteralOption, default=str(vars.VALIDATION_DATASETS))
@click.option("--disease_label_list", cls=PythonLiteralOption, default=str(vars.DISEASE_LABEL_LIST))
@click.option("--target_vars", cls=PythonLiteralOption, default=["Gender", "Age", "Diagn"])
@click.option("--min_size", default=10)
@click.option("--minimum_n_diagnosis", default=6)
@LogRun()
def create_ids(
    input_path,
    validation_datasets_list=vars.VALIDATION_DATASETS,
    disease_label_list=vars.DISEASE_LABEL_LIST,
    target_vars=["Gender", "Age", "Diagn"],
    min_size=10,
    minimum_n_diagnosis=6,
):
    """Create lists of ids of resampled data.

    Parameters
    ----------
    input_path : str
        Path for the clean data, the output of the script src.clean_data.
    validation_datasets_list: list of strings, default=vars.VALIDATION_DATASETS
        List of strings that represents the datasets that must be reserved for validation.
    disease_label_list : list of integers, default=vars.DISEASE_LABEL_LIST
        List of the disease codes to be included.
    target_vars: list of strings, default=["Gender", "Age", "Diagn"]
        Variables to be observed while resampling.
    min_size : int, default=10
        Minimum number of subjects in a single scanner.
    minimum_n_diagnosis : int, default=6
        Minimum number of subjects in a single scanner with each diagnosis.

    Returns
    -------
    Outputs:

    train_ids.csv : File
        Comma separated file containing the list of subjects selected for the training set.
    test_ids.csv : File
        Comma separated file containing the list of subjects selected to compose for the test set.

    Metrics:

    N_participants : int
        Number of participants.
    N_scanners : int
        Number of different scanners.
    """
    click.echo("*********************** CREATE IDS ***************************")
    with mlflow.start_run(run_name="clean_ids", nested=True) as run:
        TempDir = tempfile.TemporaryDirectory(dir=run.info.artifact_uri.replace('file://', ''))
        temp_dir = Path(TempDir.name)

        # Define random seed
        seed = 0
        np.random.seed(seed)

        # Prepare data for resampling, such as splitting into train/test and validation and removing small scanners
        train_test, validation = resample_data_preparation(
            input_path,
            disease_label_list,
            validation_datasets_list,
            min_size,
            seed,
            ignore_test=True,
        )

        # Get sample sizes before resampling
        n_traintest_before_resample_total = train_test.shape[0]
        n_validation_total = validation.shape[0]
        n_traintest_before_resample = train_test.Dataset.cat.remove_unused_categories().value_counts().to_frame()
        n_validation = validation.groupby('Dataset', observed=True).Diagn.value_counts()

        create_vars = vars.COLUMNS_NAME + vars.QC_FEATURES
        create_vars += ["EstimatedTotalIntraCranialVol", "dataset_code", 'mriqc_prob', 'Gender', 'Diagn']

        click.echo("*********************** RESAMPLING DATA ***************************")
        # Undersample only; no gender balance
        resampled = undersample_biobank(train_test,
                                        create_vars,
                                        ['Age'],
                                        n_class=55,
                                        random_state=seed)

        # Get sample sizes after resampling
        n_traintest_after_resample_total = resampled.shape[0]
        n_traintest_after_resample = resampled.Dataset.value_counts().to_frame()

        # Perform sigma clipping on resampled data to remove outliers (from training data only)
        resampled_clipped = sigma_clipping_exclude(resampled, features=vars.COLUMNS_NAME, n_sigmas=3, n_extreme=3)

        # Merge with previous dataset to add Dataset and Diagn columns
        resampled_clipped = pd.merge(resampled_clipped, resampled[['id', 'Dataset', 'Diagn']], on='id')

        # Get sample sizes after sigma clipping the resampled sample
        # Note: This should be the final training sample used in the rest of the pipeline
        n_resampled_after_clipping_total = resampled_clipped.shape[0]
        n_resampled_after_clipping = resampled_clipped.Dataset.value_counts().to_frame()

        # Create plots comparing distribution before and after resampling (and sigma clipping)
        make_resample_plot(train_test, resampled_clipped, temp_dir)

        # Plot mean volums of FreeSurfer regions in resampled vs resampled AND clipped datasets
        # This is to visualise the effect of removing outliers by sigma clipping
        fs_regions_clipped_plot = plot_freesurfer_regions(resampled, resampled_clipped)
        fs_regions_clipped_plot_path = f"{temp_dir}/fs_regional_volumes.png"
        fs_regions_clipped_plot.savefig(fs_regions_clipped_plot_path, dpi=150)
        plt.close()

        # Save and log data and metrics
        metrics_dict = save_resampled_data(resampled_clipped, validation, disease_label_list, temp_dir, seed)

        # ------------------------------------------------------------------------------------------------------
        # Log ML artifacts and metrics
        n_traintest_before_resample_path = temp_dir / 'n_traintest_before_resample.csv'
        n_traintest_after_resample_path = temp_dir / 'n_traintest_after_resample.csv'
        n_traintest_after_resample_and_clipping_path = temp_dir / 'n_traintest_after_resample_and_clipping.csv'
        n_validation_path = temp_dir / 'n_validation.csv'

        n_traintest_before_resample.to_csv(n_traintest_before_resample_path)
        n_traintest_after_resample.to_csv(n_traintest_after_resample_path)
        n_resampled_after_clipping.to_csv(n_traintest_after_resample_and_clipping_path)
        n_validation.to_csv(n_validation_path)

        mlflow.log_artifacts(temp_dir)

        metrics_dict["n_traintest_before_resample"] = n_traintest_before_resample_total
        metrics_dict["n_traintest_after_resample"] = n_traintest_after_resample_total
        metrics_dict["n_traintest_after_resample_and_clipping"] = n_resampled_after_clipping_total
        metrics_dict["n_validation"] = n_validation_total
        mlflow.log_metrics(metrics_dict)

        TempDir.cleanup()
        print("**************************************************************")


if __name__ == "__main__":
    create_ids()
