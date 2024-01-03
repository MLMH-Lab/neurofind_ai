"""Script to develop a brain age prediction model.

This script refers to the brainage.py file for the definition of the machine learning model for
brain age prediction (importing BrainAge). The model is trained on healthy control data
and is then applied to an independent test set including both healthy controls and clinical subjects.

This script also creates various scatterplots and KDE plots to illustrate the results.
"""
from itertools import cycle
from pathlib import Path
import tempfile
import warnings

import click
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import tensorflow as tf

from src.brainage import BrainAge
from src import definitions
from src.utils import load_dataset
from src.create_figures import get_box_plot

colors = cycle(["#FF7070", "#6c6c6c", "#f7ef69", "#13c1c7", "#00999e"])

warnings.filterwarnings("ignore")


def plot_age_vs_predicted_age(df, hue="Gender", title=""):
    """Create scatterplot of age vs predicted age separated by gender.

    Parameters
    ----------
    df : NDFrame
        Pandas dataframe containing age, predicted age, and gender per subject.
    hue : str, default="Gender"
        Variable name by which to separate the data in the plot.

    Returns
    -------
    fig : Figure
        Scatterplot of age vs predicted age.
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax = sns.scatterplot(data=df, x="Age", y="Age_pred", hue=hue, ax=ax, palette="colorblind")
    plt.title(title)
    return fig


def plot_age_vs_age_gap(df, title=""):
    """Create scatterplot of age vs age gap separated by gender.

    The plot also includes a regression line.
    This plot illustrates the age bias / regression to the mean present in the model.

    Parameters
    ----------
    df : NDFrame
        Pandas dataframe containing age, age gap, and gender per subject.

    Returns
    -------
    fig : Figure
        Scatterplot.
    """
    xlims = [df.Age.min(), df.Age.max()]
    ln = LinearRegression()
    ln.fit(df[["Age"]].values, df[["age_gap"]].values)
    a = ln.coef_[0][0]
    b = ln.intercept_[0]
    x_reg = np.linspace(xlims[0], xlims[1])
    y_reg = a * x_reg + b
    mean_age = df.Age.mean()
    mean_gap = df.age_gap.mean()
    std_gap = df.age_gap.std()
    df.Gender = df.Gender.map({0: "Female", 1: "Male"})
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax = sns.scatterplot(data=df, x="Age", y="age_gap", hue="Gender", ax=ax, palette=cycle(["#ffc107", "#13c1c7"]))
    plt.plot(x_reg, y_reg, label=f"Age Gap = {a:0.2f}*Age + {b:0.2f}", color="#FF7070", lw=3)
    ax.hlines(mean_gap, xlims[0],  xlims[1], label=f"Mean = {mean_gap:0.3f}", ls=":", color="k")
    ax.hlines(mean_gap - 1.96 * std_gap, xlims[0], xlims[1],
              label=f"-1.96STD = {mean_gap - 1.96 * std_gap:0.3f}", ls="--", color="k")
    ax.hlines(mean_gap + 1.96 * std_gap, xlims[0], xlims[1],
              label=f"+1.96STD = {mean_gap + 1.96 * std_gap:0.3f}", ls="--", color="k")
    ax.vlines(mean_age, mean_gap - 1.96 * std_gap, mean_gap + 1.96 * std_gap,
              label=f"Mean Age = {mean_age:0.1f}", ls="-", color="#6c6c6c")
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=12, ncol=2)
    plt.xlabel("Age (Years)", fontsize=24)
    plt.ylabel("Age Gap (Years)", fontsize=24)
    plt.title(title)
    plt.ylim(mean_gap - 3 * std_gap, mean_gap + 3 * std_gap)
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    return fig


def plot_age_vs_age_gap_by_diagn(df, title=""):
    """Create scatterplot of age vs age gap separated by diagnosis.

    The plot also includes a regression line.
    This plot illustrates the age bias / regression to the mean present in the model.

    Parameters
    ----------
    df : NDFrame
        Pandas dataframe containing age, age gap, and diagnosis per subject.
    y_var_name = str, default="age_gap"
        Variable on y axis; either age gap (default) or predicted age ("Age_pred")
    hue : str, default="Gender"
        Variable name by which to separate the data in the plot
    title : str, default=""
        Title of the plot

    Returns
    -------
    fig : Figure
        Scatterplot.
    """
    xlims = [df.Age.min(), df.Age.max()]
    ln = LinearRegression()
    ln.fit(df[["Age"]].values, df[["age_gap"]].values)
    a = ln.coef_[0][0]
    b = ln.intercept_[0]
    x_reg = np.linspace(xlims[0], xlims[1])
    y_reg = a * x_reg + b
    mean_age = df.Age.mean()
    mean_gap = df.age_gap.mean()
    std_gap = df.age_gap.std()
    disease_dict = definitions.DISEASE_DICT.copy()
    disease_dict[1] = "HC"
    df.loc[:, ("diagn")] = df.Diagn.map(disease_dict)
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax = sns.scatterplot(data=df[df.diagn != "HC"], x="Age", y="age_gap", hue="diagn", ax=ax, palette="colorblind")
    ax = sns.scatterplot(data=df[df.diagn == "HC"], x="Age", y="age_gap", hue="diagn", ax=ax, color="#4caf50",
                         marker="x")
    plt.plot(x_reg, y_reg, label=f"Age Gap = {a:0.2f}*Age + {b:0.2f}", color="#FF7070", lw=3)
    ax.hlines(mean_gap, xlims[0], xlims[1], label=f"Mean = {mean_gap:0.3f}", ls=":", color="k")
    ax.hlines(mean_gap - 1.96 * std_gap, xlims[0], xlims[1],
              label=f"-1.96STD = {mean_gap - 1.96 * std_gap:0.3f}", ls="--", color="k")
    ax.hlines(mean_gap + 1.96 * std_gap, xlims[0], xlims[1],
              label=f"+1.96STD = {mean_gap + 1.96 * std_gap:0.3f}", ls="--", color="k")
    ax.vlines(mean_age, mean_gap - 1.96 * std_gap, mean_gap + 1.96 * std_gap,
              label=f"Mean Age = {mean_age:0.1f}", ls="-", color="#6c6c6c")
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=12, ncol=2)
    plt.xlabel("Age (Years)", fontsize=24)
    plt.ylabel("Age Gap (Years)", fontsize=24)
    plt.title(title)
    try:
        plt.ylim(mean_gap - 3 * std_gap, mean_gap + 3 * std_gap)
    except ValueError:
        pass
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    return fig


def plot_age_gap_KDE_by_diagn(df):
    disease_dict = definitions.DISEASE_DICT.copy()
    disease_dict[1] = "HC"
    df.loc[:, ("Diagnosis")] = df.Diagn.map(disease_dict)
    fig, ax = plt.subplots(1, 1, figsize=(16*0.7, 9*0.7))
    sns.kdeplot(data=df, x="age_gap", hue="Diagnosis", ax=ax, common_norm=False, palette=colors)
    plt.tick_params(labelsize=15)
    plt.ylabel("Normalized Kernel Density Estimate", fontsize=17)
    plt.xlabel("Age Gap (Years)", fontsize=17)
    plt.grid()
    plt.minorticks_on()
    plt.tight_layout()
    return fig


@click.command(help="")
@click.option("--input_path")
@click.option("--ids_path")
def run_model(input_path, ids_path):
    with mlflow.start_run(run_name='run_model') as run:
        print('********************* LOAD DATA ****************************')
        TempDir = tempfile.TemporaryDirectory(dir=run.info.artifact_uri.replace('file://', ''))
        temp_dir = Path(TempDir.name)

        # Load the dataset and separate the data into training and validation
        # The volumes are automatically normalised by total intracranial volume in load_dataset
        dataset = load_dataset(input_path, ids_path,
                               disease_label_list=definitions.DISEASE_LABEL_LIST, relative_volumes=True)
        # Remove potential duplicates
        is_test = dataset.copy().groups.str.contains('test_ids_').values
        dataset = dataset[~is_test]

        is_validation = dataset.Dataset.isin(definitions.VALIDATION_DATASETS)
        dataset_train = dataset[~is_validation].copy()
        dataset_val = dataset[is_validation].copy()

        # Check sample sizes
        # These values are later saved to mlflow
        n_train_total = dataset_train.shape[0]
        n_validation_total = dataset_val.shape[0]
        n_train_scanner_df = dataset_train.scanner.value_counts().to_frame()
        n_val_scanner_df = dataset_val.scanner.value_counts().to_frame()
        n_val_diagnosis_df = dataset_val.Diagn.value_counts().to_frame()

        print('********************* TRAIN BRAIN AGE MODEL ****************************')
        # Set random seed
        np.random.seed(0)
        tf.random.set_seed(0)

        # Define model
        # regressor_type can be Support vector regression, Random forest regression, Elastic net, Linear regression, RVM
        model = BrainAge(
            scaler_type='RobustScaler',
            regressor_type='Support vector regression',
        )

        # Fit the model on training data
        model.fit(dataset_train[definitions.COLUMNS_NAME].values, dataset_train['Age'].values)

        print('******************* INFERENCE MODEL **************************')

        # Get best model parameters
        BEST_PARAMS = model.regressor.get_params()

        # Create subject-level prediction results in training and validation datasets
        # Add the age predictions in a new column
        dataset_train.loc[:, ("Age_pred")] = model.predict(dataset_train[definitions.COLUMNS_NAME])
        dataset_val.loc[:, ("Age_pred")] = model.predict(dataset_val[definitions.COLUMNS_NAME])

        # Calculate brain age gap (predicted age - chronological age) in a new column
        dataset_train.loc[:, ("age_gap")] = dataset_train['Age_pred'] - dataset_train['Age']
        dataset_val.loc[:, ("age_gap")] = dataset_val['Age_pred'] - dataset_val['Age']

        # ---------------------------------------------------------------------------------------------------
        # Saving MLflow artifacts

        # Save scaler and regressor (model)
        model.save(temp_dir)

        # Save brain age prediction results from training and validation datasets in one file
        gap_path = temp_dir / 'brain_age_gap_fullsample.csv'
        dataset = pd.concat([dataset_train, dataset_val])
        dataset[["id", "Dataset", "Diagn", "Gender", "groups", "Age", "age_gap",
                 "Age_pred"]].to_csv(gap_path, index=False)
        mlflow.log_artifact(gap_path)

        # Save sample sizes
        # Note: The sample sizes should be the same as the rest of the pipeline, so this is just a checking point
        n_train_scanner_path = temp_dir / 'n_train_scanner.csv'
        n_val_scanner_path = temp_dir / 'n_val_scanner.csv'
        n_val_diagn_path = temp_dir / 'n_val_diagn.csv'

        n_train_scanner_df.to_csv(n_train_scanner_path)
        n_val_scanner_df.to_csv(n_val_scanner_path)
        n_val_diagnosis_df.to_csv(n_val_diagn_path)

        mlflow.log_artifact(n_train_scanner_path)
        mlflow.log_artifact(n_val_scanner_path)
        mlflow.log_artifact(n_val_diagn_path)

        # Calculate mean absolute error (MAE), standard deviation (std), and mean age gap in training and validation data
        MAE_train = dataset_train["age_gap"].abs().mean().round(3)
        MAE_val_hc = dataset_val[dataset_val.Diagn == 1]["age_gap"].abs().mean().round(3)
        STD_val_hc = dataset_val[dataset_val.Diagn == 1]["age_gap"].abs().std().round(3)
        MEAN_AGE_GAP_train = dataset_train["age_gap"].mean().round(3)
        MEAN_AGE_GAP_val_hc = dataset_val[dataset_val.Diagn == 1]["age_gap"].mean().round(3)
        STD_AGE_GAP_val_hc = dataset_val[dataset_val.Diagn == 1]["age_gap"].std().round(3)
        print(MEAN_AGE_GAP_train, MEAN_AGE_GAP_val_hc)
        print(MAE_train, MAE_val_hc)

        # Get MAE from validation set per diagnosis (within each scanner)
        mae_dict = {}
        for scanner_name in dataset_val.scanner.unique():
            dataset = dataset_val[dataset_val.scanner == scanner_name]
            # print(scanner_name)
            for diagn_name in dataset.Diagn.unique():
                # print(diagn_name)
                var_name = str(scanner_name) + '_' + str(diagn_name) + "_mae_test"
                mae = dataset[dataset.Diagn == diagn_name]["age_gap"].abs().mean().round(3)
                mae_dict[var_name] = mae

        # Get mean BAG from validation set per diagnosis (within each scanner)
        mean_bag_dict = {}
        for scanner_name in dataset_val.scanner.unique():
            dataset = dataset_val[dataset_val.scanner == scanner_name]
            # print(scanner_name)
            for diagn_name in dataset.Diagn.unique():
                # print(diagn_name)
                var_name = str(scanner_name) + '_' + str(diagn_name) + "_mean_bag_test"
                mean_bag = dataset[dataset.Diagn == diagn_name]["age_gap"].mean().round(3)
                mean_bag_dict[var_name] = mean_bag

        # Log MAE test per diagnosis on mlflow
        mean_bag_test = pd.DataFrame(mean_bag_dict.values(), index=mean_bag_dict.keys(), columns=["mean_bag_test"])
        mean_bag_test.to_html(f"{temp_dir}/mean_bag_test_per_diagn.html")
        mlflow.log_artifact(f"{temp_dir}/mean_bag_test_per_diagn.html")

        mae_test = pd.DataFrame(mae_dict.values(), index=mae_dict.keys(), columns=["mae_test"])
        mae_test.to_html(f"{temp_dir}/mae_test_per_diagn.html")
        mlflow.log_artifact(f"{temp_dir}/mae_test_per_diagn.html")

        # Create box plots for age gaps in validation dataset using function in create_figures.py
        # Separate plots are created for split by diagnosis and split by diagnosis within each scanner
        boxplot_val_bydiagn_path = get_box_plot(dataset_val, metric='age_gap',
                                                grouping_var='groups', output_path=temp_dir, filename='boxplot_bydiagn')
        boxplot_val_byscanner_path = get_box_plot(
            dataset_val, metric='age_gap', grouping_var='scanner', output_path=temp_dir, filename='boxplot_byscanner')

        dataset_val = dataset_val.set_index('id')

        # Log figures on mlflow
        mlflow.log_artifact(boxplot_val_bydiagn_path)
        mlflow.log_artifact(boxplot_val_byscanner_path)

        # Plot age vs predicted age in train and validation data
        fig = plot_age_vs_predicted_age(dataset_train, hue="Gender", title="Age vs predicted age in the training data")
        fig.savefig(f"{temp_dir}/age_vs_predicted_age_train.png")
        fig = plot_age_vs_predicted_age(dataset_val, hue="Diagn", title="Age vs predicted age in the validation data")
        fig.savefig(f"{temp_dir}/age_vs_predicted_age_val.png")

        # Plot age vs age gap
        fig = plot_age_vs_age_gap(dataset_train, f"Train - MAE = {MAE_train}")
        plt.close()
        fig = plot_age_vs_age_gap(dataset_val, f"Validation - MAE = {MAE_val_hc}")
        fig.savefig(f"{temp_dir}/age_gap_val.png")
        plt.close()
        fig = plot_age_vs_age_gap_by_diagn(dataset_train, f"Train - MAE = {MAE_train}")
        fig.savefig(f"{temp_dir}/age_gap_train_by_diagn.png")

        for name, group in dataset_val.groupby("Diagn"):
            hc = dataset_val[dataset_val.scanner.isin(group.scanner.unique()) & (dataset_val.Diagn == 1)]
            mae = group["age_gap"].abs().mean().round(3)
            fig = plot_age_vs_age_gap_by_diagn(pd.concat([group.copy(), hc]), f"{name} - MAE = {mae}")
            fig.savefig(f"{temp_dir}/age_gap_test_by_diagn_{name}.png")
            plt.close()
        for name, group in dataset_val.groupby("Dataset"):
            fig = plot_age_gap_KDE_by_diagn(group)
            fig.savefig(f"{temp_dir}/age_gap_KDE_{name}.png")
            plt.close()

        # Log best model parameters
        best_params = pd.DataFrame(BEST_PARAMS.values(), index=BEST_PARAMS.keys(), columns=["param"])
        best_params_path = f"{temp_dir}/best_params.html"
        best_params.to_html(best_params_path)
        mlflow.log_artifact(best_params_path)

        # Log group-level performance metrics
        mlflow.log_metrics(
            {'MAE_TRAIN': MAE_train,
             'MAE_VAL_HC': MAE_val_hc,
             'STD_VAL_HC': STD_val_hc,
             'MEAN_AGE_GAP_train': MEAN_AGE_GAP_train,
             'MEAN_AGE_GAP_val_hc': MEAN_AGE_GAP_val_hc,
             'STD_AGE_GAP_val_hc': STD_AGE_GAP_val_hc,
             'n_train_total': n_train_total,
             'n_validation_total': n_validation_total,
             }
        )

        model_path = temp_dir
        mlflow.log_artifacts(model_path, 'model')

        print('**************************************************************')
        TempDir.cleanup()


if __name__ == '__main__':
    run_model()
