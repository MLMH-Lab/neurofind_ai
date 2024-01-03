"""Pipeline of Neurofind training.

The pipeline executes the following steps:

* Organize the data (fetch_data.py)
* Clean the datasets (clean_data.py)
* Create ids for splitting training and test datasets with matching of different criteria (create_resampled_ids.py)
* Harmonize different scanners (run_harmoniser.py) [1]
(Please note: Please refer to the Neuroharmony publication and github repository for this step.)
* Performs a grid search of the Adversarial autoencoder (run_gridsearch_aae.py)
* Develop brain age prediction model (brain_age.py)

References:
[1] https://neuroharmony.readthedocs.io/en/latest/
"""
import git
import tempfile
from pathlib import Path

import click
import mlflow
import numpy as np
import tensorflow as tf
from numpy import linspace

from src.definitions import DISEASE_LABEL_LIST, VALIDATION_DATASETS
from src.utils import get_or_run, get_or_create_experiment_id, get_or_create_experiment_id_for_branch, link_artefacts


def relog_runs(list_of_runs, temp_path):
    """Relog the information of the subruns in the main run.

    By default only the information logged in this file is added to the main run.
    This function adds all information of the subruns in the main run.

    Parameters
    ----------
    list_of_runs : list
        List of runs to relog.
    temp_path : str or Path object
        Path to save the information temporarily while relogging artefacts.
    """
    main_run_params = {}
    for run in list_of_runs:
        main_run_params.update(run.data.params)
    for run in list_of_runs:
        entry_point_name = run.data.tags["mlflow.project.entryPoint"]
        main_run_params[f"subrun_{entry_point_name}"] = run.info.run_id
        mlflow.log_param('main_run_params', main_run_params)
        relog_subfolder = f"{temp_path}/{entry_point_name}/"
        Path(relog_subfolder).mkdir(exist_ok=True)
        link_artefacts(run.info.artifact_uri.replace('file://', ''), relog_subfolder)
        mlflow.log_param('link_artefacts', link_artefacts)
    mlflow.log_artifacts(temp_path)

    mlflow.log_params(main_run_params)

    main_run_metrics = {}
    for run in list_of_runs:
        main_run_metrics.update(run.data.metrics)
    mlflow.log_metrics(main_run_metrics)


@click.command("pipeline")
@click.option("--verify_commit", default=True)
@click.option("--git_commit", default=None)
@click.option("--use_cache", default=True)
@click.option("--experiment", default="None")
def neurofind_train(verify_commit, git_commit, use_cache, experiment):
    """Run Neurofind training pipeline.

    Parameters
    ----------
    verify_commit: bool, default: True
        If False, ignore commit hash to reuse previous runs.
    git_commit: str, default: None
        Hash for a specific commit for which we want to reuse a certain run.
    use_cache : bool or str, default: True
        If True, search for runs with the same `git_commit` hash to skip the run and reuse previous results.
    experiment : int or str, default: "None"
        The ID or name of the experiments.

    """
    if experiment == "None":
        experiment_id = get_or_create_experiment_id_for_branch()
    else:
        experiment_id = get_or_create_experiment_id(experiment)
    work_branch = git.Repo(search_parent_directories=True).head.object.name_rev.split(" ")[1]
    git_message = git.Repo(search_parent_directories=True).head.object.message.replace("\n", "")
    with mlflow.start_run(nested=True, experiment_id=experiment_id, run_name=work_branch):
        with mlflow.start_run(nested=True, experiment_id=experiment_id, run_name=git_message):
            if git_commit == "None":
                git_commit = git.Repo(search_parent_directories=True).head.object.hexsha
            print("*********************** FETCHING DATA ************************")
            raw_path = "data/raw/"

            # Set random seed
            np.random.seed(42)
            tf.random.set_seed(42)

            fetch_run = get_or_run(
                "fetch_data",
                {"raw_path": raw_path},
                git_commit,
                verify_commit=verify_commit,
                use_cache=use_cache,
                experiment_id=experiment_id,
            )
            fetch_path = fetch_run.info.artifact_uri.replace('file://', '')
            mlflow.log_param('fetch_path', fetch_path)

            # Define quality threshold; data below the threshold are included
            mriqc_th = 0.7

            # Define age range of training data
            min_age = 20
            max_age = 80
            dataset_same_scanner = False

            # Define datasets to use, incl. training, testing, validation
            use_only_datasets = ['BIOBANK', 'HCP', 'HumanConnectomeProject-Aging', 'IXI',
                                 'AIBL', 'MCIC', 'COBRE']

            print("*********************** CLEANING DATA ************************")
            clean_run = get_or_run(
                "clean_data",
                {
                    "fetch_path": fetch_path,
                    "mriqc_th": mriqc_th,
                    "min_age": min_age,
                    "max_age": max_age,
                    "dataset_same_scanner": dataset_same_scanner,
                    "use_only_datasets": use_only_datasets,
                },
                git_commit,
                verify_commit=verify_commit,
                use_cache=use_cache,
                experiment_id=experiment_id,
            )

            clean_path = clean_run.info.artifact_uri.replace('file://', '')
            mlflow.log_param('clean_path', clean_path)

            # This section implements resampling
            create_ids_run = get_or_run(
                "create_resampled_ids",
                {
                    "input_path": clean_path,
                    "validation_datasets_list": VALIDATION_DATASETS,
                    "disease_label_list": DISEASE_LABEL_LIST,
                    "target_vars": ["Gender", "Age", "Diagn"],
                    "min_size": 2,
                    "minimum_n_diagnosis": 6,
                },
                git_commit,
                verify_commit=verify_commit,
                use_cache=use_cache,
                experiment_id=experiment_id,
            )
            use_cache = False

            input_path = create_ids_run.info.artifact_uri.replace('file://', '')
            mlflow.log_param('input_path', input_path)
            ids_path = create_ids_run.info.artifact_uri.replace('file://', '')
            mlflow.log_param('ids_path', ids_path)

            # Implement harmonisation using Neuroharmony
            # Please refer to the Neuroharmony publication and repository for this step [1]
            covars = ["Gender", "scanner", "Age"]
            eliminate_variance = ["Gender", "scanner", "Age"]
            relative_volumes = "True"

            # Harmoniser for scanner, gender, age
            harmoniser_run = get_or_run(
                "run_harmoniser",
                {
                    "input_path": input_path,
                    "ids_path": ids_path,
                    "covars": covars,
                    "eliminate_variance": eliminate_variance,
                    "relative_volumes": relative_volumes,
                    "min_age": min_age,
                    "max_age": max_age,
                    "refit": True,
                    "use_only_datasets": use_only_datasets,
                },
                git_commit,
                verify_commit=verify_commit,
                use_cache=use_cache,
                experiment_id=experiment_id,
            )

            harmoniser_path = harmoniser_run.info.artifact_uri.replace('file://', '')
            mlflow.log_param('harmoniser_path', harmoniser_path)

            # Harmoniser for scanner, gender
            harmoniser_run_without_age = get_or_run(
                "run_harmoniser_without_age",
                {
                    "input_path": input_path,
                    "ids_path": ids_path,
                    "covars": covars,
                    "eliminate_variance": ['Gender', 'scanner'],
                    "relative_volumes": relative_volumes,
                    "min_age": min_age,
                    "max_age": max_age,
                    "refit": True,
                },
                git_commit,
                verify_commit=verify_commit,
                use_cache=use_cache,
                experiment_id=experiment_id,
            )

            harmoniser_run_without_age_path = harmoniser_run_without_age.info.artifact_uri
            mlflow.log_param('harmoniser_run_without_age_path', harmoniser_run_without_age_path)

            # Run AAE
            pre_scaler_list = ["RobustScaler"]
            pos_scaler_list = ["RobustScaler"]

            h_dim_list = [
                2*[25],
                2*[50],
                2*[75],
                2*[100],
            ]
            z_dim_list = [5, 10, 20, 32, 50]

            base_learning_rate_list = [2e-5]
            max_learning_rate_list = [0.005]
            gamma_list = list(linspace(0.03, 0.04, 5).round(3))
            batch_size_list = [64, 32, 16, 8]

            ae_run_harmonized = get_or_run(
                "gridsearch_aae_harmonized",
                {
                    "input_path": harmoniser_path,
                    "ids_path": input_path,
                    "pre_scaler_list": pre_scaler_list,
                    "pos_scaler_list": pos_scaler_list,
                    "base_learning_rate_list": base_learning_rate_list,
                    "max_learning_rate_list": max_learning_rate_list,
                    "gamma_list": gamma_list,
                    "h_dim_list": h_dim_list,
                    "z_dim_list": z_dim_list,
                    "batch_size_list": batch_size_list,
                },
                git_commit,
                verify_commit=verify_commit,
                use_cache=use_cache,
                experiment_id=experiment_id,
            )

            deviations_harm_path = ae_run_harmonized.info.artifact_uri
            mlflow.log_param('deviations_harm_path', deviations_harm_path)

            brain_age_harm = get_or_run(
                "brain_age_harm",
                {
                    "input_path": harmoniser_run_without_age_path,
                    "ids_path": ids_path,
                },
                git_commit,
                verify_commit=verify_commit,
                use_cache=use_cache,
                experiment_id=experiment_id,
            )

            brain_age_harm_path = brain_age_harm.info.artifact_uri
            mlflow.log_param('brain_age_harm_path', brain_age_harm_path)

            temp_path = tempfile.mkdtemp()
            relog_runs(
                [
                    fetch_run,
                    clean_run,
                    create_ids_run,
                    harmoniser_run,
                    ae_run_harmonized,
                    brain_age_harm,
                ],
                temp_path=temp_path,
            )


if __name__ == '__main__':
    neurofind_train()
