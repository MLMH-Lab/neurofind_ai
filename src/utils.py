"""Helper functions."""
from contextlib import redirect_stdout
from io import StringIO
from numbers import Number
from os import symlink
from pathlib import Path
from warnings import warn
import functools
import logging

from mlflow.entities import RunStatus
from mlflow.tracking.client import MlflowClient
from mlflow.utils import env
from mlflow.utils import mlflow_tags
from mlflow.utils.databricks_utils import is_in_databricks_notebook, get_notebook_id
from scipy.stats import hmean
import ast
import click
import git
import mlflow
import numpy as np
import pandas as pd
import six

from src.definitions import COLUMNS_NAME, DISEASE_LABEL_LIST, DISEASE_NAME_LIST, DISEASE_ABBREVIATION_LIST

formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

_EXPERIMENT_ID_ENV_VAR = "MLFLOW_EXPERIMENT_ID"
_EXPERIMENT_NAME_ENV_VAR = "MLFLOW_EXPERIMENT_NAME"
_RUN_ID_ENV_VAR = "MLFLOW_RUN_ID"
_active_run_stack = []
_active_experiment_id = None
SEARCH_MAX_RESULTS_PANDAS = 100000
NUM_RUNS_PER_PAGE_PANDAS = 10000
_active_experiment_id = None


def get_runs(run_id=None, experiment_id=None):
    """Get subruns in a main run.

    If the run do not have subruns it returns None.
    If a run_id is given it return the run with that run_id or returns None if it is not found.

    Parameters
    ----------
    run_id : str, default : None
        Id of a particular run. If not specified returns the first main run in the experiment.
    experiment_id : str, default : None
        Id of a particular experiment. If not specified uses the experiment with the title of the current branch.

    Returns
    -------
    runs_info : NDFrame or None
        Returns a NDFrame with the information of the subruns in a particular run or None if none is found.
    """
    if experiment_id is None:
        work_branch = git.Repo(search_parent_directories=True).head.object.name_rev.split(" ")[1]
        experiment = mlflow.get_experiment_by_name(work_branch)
        if experiment is None:
            experiment_id = mlflow.create_experiment(work_branch)
        else:
            experiment_id = experiment.experiment_id
    runs_info = mlflow.search_runs(experiment_id).set_index("run_id")
    runs_info.columns
    if run_id is None:
        if "params.subrun_fetch_data" in runs_info.columns:
            main_runs = runs_info[~runs_info["params.subrun_fetch_data"].isna()].sort_values(
                "end_time", ascending=False
            )
            run = main_runs.iloc[0]
            return runs_info[runs_info["tags.mlflow.parentRunId"] == run.name]
        else:
            return None
    else:
        run = runs_info.loc[run_id]
        return run


def load_input_data(input_path):
    """Load input_data.

    Parameters
    ----------
    input_path: str
        Path for the freesurferData and participants file, typically it would be the harmoniser output path.

    Returns
    -------
    freesurferdata: NDFrame
        DataFrame with freesurfer volumetric data.
    participants: NDFrame
        DataFrame with demographic data.
    """
    # participants_dtype = {"Gender": int, "Age": int, "Diagn": int, "Dataset": "category", 'id': str}
    # fs_dtype = {'id': str}
    fs = pd.read_csv(f"{input_path}/freesurferData.csv")
    # fs = pd.read_csv(f"{input_path}/freesurferData.csv").astype(fs_dtype)
    participants = pd.read_csv(f"{input_path}/participants.tsv", sep="\t")
    # participants = pd.read_csv(f"{input_path}/participants.tsv", sep="\t").astype(participants_dtype)
    return fs, participants


def load_dataset(input_path, ids_path=None, disease_label_list=None, relative_volumes=False, return_groups=True):
    """Combine freesurfer volumes, demographics and quality controls metrics according to list of ids.

    Parameters
    ----------
    input_path : str or Path
        Path to folder containing the freesurferData.csv, participants.tsv and qc.tsv files. Which contain the
        freesurfer volumes, demographics and quality controls metrics, respectively.
    ids_path : str or Path, default : None
        Path to the lists of ids for the training and test sets. The folder must contain the files train_ids.csv and
        test_ids.csv. If None, all subjects are included in the final dataframe.
    relative_volumes : bool, default : False
        If True divides all regions by the total intracranial volume (EstimatedTotalIntraCranialVol).

    Returns
    -------
    NDFrame (n_subjects, n_features)
        Dataframe containing the freesurfer volumes, demographics and quality controls metrics according to list of ids.
    """
    fs, participants = load_input_data(input_path)
    if relative_volumes:
        fs.loc[:, COLUMNS_NAME] = fs[COLUMNS_NAME].divide(
            fs["EstimatedTotalIntraCranialVol"].astype(float).values, axis=0
        )
    dataset = fs.merge(participants, on="id", how="inner").copy()
    if Path(f"{input_path}/qc.tsv").exists():
        qc = pd.read_csv(f"{input_path}/qc.tsv", sep="\t").astype({"id": str})
        dataset = pd.merge(dataset, qc, on="id").copy()
    n_subjects = []
    if ids_path is not None:
        data_chunks = []
        # Train ids should always come in a single file
        ids_path = str(ids_path).replace("file://", "")
        ids_train = pd.read_csv(f"{ids_path}/train_ids.csv").astype({"id": str})
        if return_groups:
            ids_train.loc[:, ("groups")] = "train_ids"
        chunck = dataset.merge(ids_train, on="id", how="inner").drop_duplicates().copy()
        data_chunks.append(chunck)
        n_subjects.append(chunck.shape[0])
        # The number of test ids sample may vary
        test_ids_files = Path(ids_path).glob("test_*")
        for test_ids_file in test_ids_files:
            ids_test_group = pd.read_csv(test_ids_file).astype({"id": str})
            if return_groups:
                ids_test_group.loc[:, ("groups")] = str(test_ids_file.name).replace(".csv", "")
            chunck = dataset.merge(ids_test_group, on="id", how="inner").drop_duplicates().copy()
            data_chunks.append(chunck)
            n_subjects.append(chunck.shape[0])
        if disease_label_list:
            for disease_label in disease_label_list:
                print("Loading disease label ", disease_label)
                ids_validation_disease = pd.read_csv(
                    f"{ids_path}/validation_ids_{disease_label:03d}.csv"
                ).astype({"id": str})
                if return_groups:
                    ids_validation_disease.loc[:, ("groups")] = f"validation_ids_{disease_label:03d}"
                chunck = dataset.merge(ids_validation_disease, on="id", how="inner").drop_duplicates().copy()
                data_chunks.append(chunck)
            n_subjects.append(chunck.shape[0])
        else:
            validation_ids_files = Path(ids_path).glob("validation_ids_*")
            for validation_ids_file in validation_ids_files:
                ids_validation_disease = pd.read_csv(validation_ids_file).astype({"id": str})
                if return_groups:
                    ids_validation_disease.loc[:, ("groups")] = str(test_ids_file.name).replace(".csv", "")
                chunck = dataset.merge(ids_validation_disease, on="id", how="inner").drop_duplicates().copy()
                data_chunks.append(chunck)
            n_subjects.append(chunck.shape[0])
        dataset_out = pd.concat(data_chunks, axis=0)
    else:
        dataset_out = dataset
    dataset_out = dataset_out[~dataset_out.isna().any(axis=1)]
    dataset_out.loc[:, ("Gender")].replace("male", 1, inplace=True)
    dataset_out.loc[:, ("Gender")].replace("female", 0, inplace=True)
    dataset_out.loc[:, ("Gender")].replace("M", 1, inplace=True)
    dataset_out.loc[:, ("Gender")].replace("F", 0, inplace=True)
    dataset_out.loc[:, ("Gender")].replace("1.0", 1, inplace=True)
    dataset_out.loc[:, ("Gender")].replace("0.0", 0, inplace=True)
    dataset_out.loc[:, ("Diagn")].fillna(-1, inplace=True)
    dataset_out.loc[:, ("Age")].fillna(-1, inplace=True)
    dataset_out.loc[:, ("Gender")] = dataset_out["Gender"]
    dataset_out.loc[:, ("Diagn")] = dataset_out["Diagn"].astype(float).astype(int)
    dataset_out.loc[:, ("Age")] = dataset_out["Age"].astype(float).astype(int)
    dataset_out.loc[:, ("scanner")] = dataset_out["Dataset"]
    return dataset_out


def combine_dataset_files(input_path, ids_path=None, relative_volumes=True, disease_label_list=DISEASE_LABEL_LIST):
    """Combine freesurfer volumes, demographics and quality controls metrics according to list of ids.

    Parameters
    ----------
    input_path : str or Path
        Path to folder containing the freesurferData.csv, participants.tsv and qc.tsv files. Which contain the
        freesurfer volumes, demographics and quality controls metrics, respectively.
    ids_path : str or Path, default : None
        Path to the lists of ids for the training and test sets. The folder must contain the files train_ids.csv and
        test_ids.csv. If None, all subjects are included in the final dataframe.
    relative_volumes : bool, default : None
        If True divides all regions by the total intracranial volume (EstimatedTotalIntraCranialVol).

    Returns
    -------
    NDFrame (n_subjects, n_features)
        Dataframe containing the freesurfer volumes, demographics and quality controls metrics according to list of ids.
    """
    input_path = str(input_path).replace("file://", "")
    if ids_path is not None:
        ids_path = str(ids_path).replace("file://", "")
    fs, participants = load_input_data(input_path)
    dataset = pd.merge(participants, fs, on="id")

    if relative_volumes:
        fs.loc[:, COLUMNS_NAME] = fs[COLUMNS_NAME].divide(
            fs["EstimatedTotalIntraCranialVol"].astype(float).values, axis=0
        )
    if ids_path is not None:
        ids_train, ids_test, ids_validation = load_ids(
            ids_path, return_groups=True, disease_label_list=disease_label_list)
        ids = pd.concat([ids_train, ids_test])
        dataset.copy().merge(ids, on="id")

    if Path(f"{input_path}/qc.tsv").exists():
        qc = pd.read_csv(f"{input_path}/qc.tsv", sep="\t")
        dataset = pd.merge(dataset, qc, on="id")
    dataset = dataset[~dataset.isna().any(axis=1)]
    dataset.loc[:, ("Gender")].replace("male", 1, inplace=True)
    dataset.loc[:, ("Gender")].replace("female", 0, inplace=True)
    dataset.loc[:, ("Diagn")].fillna(-1, inplace=True)
    dataset.loc[:, ("Age")].fillna(-1, inplace=True)
    dataset.loc[:, ("Diagn")] = dataset["Diagn"].astype(int)
    dataset.loc[:, ("Age")] = dataset["Age"].round(0).astype(int)
    dataset.loc[:, ("scanner")] = dataset["Dataset"]
    return dataset


class PythonLiteralOption(click.Option):
    """Class used to parse list of values to click."""

    def type_cast_value(self, ctx, value):
        """Parse string as a list function."""
        try:
            return ast.literal_eval(value)
        except ValueError:
            raise click.BadParameter(value)


def cliff_delta(x, y):
    """Calculate the effect size using the Cliff's delta.

    Parameters
    ----------
    x : array_like
        1-D arrays representing a property of the subjects of a particular group.
    y : array_like
        1-D arrays representing a property of the subjects of a particular group.

    Returns
    -------
    effect_size_score: float
        The Cliff's delta effect size from the comparison of the two groups.
        Cliff's delta ranges from -1 to 1.
    """
    x = x.astype("float32")
    x = x[np.isfinite(x)]
    y = y.astype("float32")
    y = y[np.isfinite(y)]

    lx, ly = x.shape[0], y.shape[0]
    xv, yv = np.meshgrid(x, y)

    effect_size_score = (np.nansum(xv > yv) - np.nansum(xv < yv)) / (lx * ly)
    return effect_size_score if abs(effect_size_score) > 0 else 0.0


def verify_array(array_like):
    """Verify array is a ndarray and expand dimensions if necessary."""
    array_like = np.array(array_like)
    if len(array_like.shape) != 2:
        array_like = np.expand_dims(array_like, axis=1)
    return array_like


def effect_size_score(
    deviations,
    diagn,
    groups=None,
    disease_label_list=DISEASE_LABEL_LIST,
    disease_name_list=DISEASE_NAME_LIST,
    disease_abbreviation_list=DISEASE_ABBREVIATION_LIST,
):
    """Calculate the mean effect size to the different."""
    diagn = verify_array(diagn)
    deviations = verify_array(deviations)

    healthy_label = 1
    effect_size_dict = {}
    if groups is not None:
        groups_ids = np.unique(groups)
    for disease_label, disease_name, disease_abbreviation in zip(
        disease_label_list, disease_name_list, disease_abbreviation_list
    ):
        if groups is not None:
            groups = verify_array(groups)
            for groups_id in groups_ids:
                hc_deviation = deviations[(diagn == healthy_label) & (groups == groups_id)]
                patient_deviation = deviations[(diagn == disease_label) & (groups == groups_id)]

                if len(patient_deviation):
                    effect_size = cliff_delta(hc_deviation, patient_deviation)
                    dict_entry = f"ef_{disease_name}_{groups_id}_score".lower().replace(" ", "_")
                    effect_size_dict[dict_entry] = np.abs(effect_size)
        else:
            hc_deviation = deviations[diagn == healthy_label]
            patient_deviation = deviations[diagn == disease_label]

            if len(patient_deviation):
                effect_size = cliff_delta(hc_deviation, patient_deviation)
                effect_size_dict[f"ef_{disease_name}_score".lower().replace(" ", "_")] = np.abs(effect_size)
    efs = np.fromiter([ef for ef in effect_size_dict.values() if ef > 0], dtype=float)
    effect_size_dict["ef_hmean_score"] = hmean(efs)
    return effect_size_dict


def _get_experiment_id_from_env():
    experiment_name = env.get_env(_EXPERIMENT_NAME_ENV_VAR)
    if experiment_name is not None:
        exp = MlflowClient().get_experiment_by_name(experiment_name)
        return exp.experiment_id if exp else None
    return env.get_env(_EXPERIMENT_ID_ENV_VAR)


def _get_experiment_id():
    deprecated_default_exp_id = "0"

    return (
        _active_experiment_id or _get_experiment_id_from_env() or (is_in_databricks_notebook() and get_notebook_id())
    ) or deprecated_default_exp_id


def get_or_create_experiment_id_for_branch():
    """Get or create the experiment with the branch title and return the experiment_id.

    Return
    ------
    experiment_id: str
        ID for the experiment with the name of the current branch.
    """
    work_branch = git.Repo(search_parent_directories=True).head.object.name_rev.split(" ")[1]
    experiment = mlflow.get_experiment_by_name(work_branch)
    if not experiment:
        experiment_id = mlflow.create_experiment(name=work_branch)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def get_or_create_experiment_id(experiment):
    """Get experiment if from experiment name or create if it doesn't exist.

    Parameters
    ----------
    experiment : str
        The experiment name.

    Returns
    -------
    str
        Experiment ID.
    """
    experiment = mlflow.get_experiment_by_name(experiment)
    if not experiment:
        experiment_id = mlflow.create_experiment(name=experiment)
    else:
        experiment_id = experiment.experiment_id
    return experiment_id


def setup_logger(name, log_file, level=logging.INFO):
    """Setups as many loggers as you want."""
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class LogRun(object):
    """Instead of printing to the terminal it will save a run.log file at the artifacts folder of the active run.

    Simply import the class and add @LogRun() at the top of the function you want to log.
    """

    def __init__(self):
        """Initialize."""
        pass

    def __call__(self, fn):
        """Capture the output of things printed to the terminal."""

        @functools.wraps(fn)
        def decorated(*args, **kwargs):
            f = StringIO()
            with redirect_stdout(f):
                result = fn(*args, **kwargs)
            output_dir = mlflow.search_runs().iloc[0].artifact_uri.replace("file://", "")
            self.logger = setup_logger("test", f"{output_dir}/run.log")
            self.logger.info(f.getvalue())
            return result

        return decorated


def already_ran(entry_point_name, parameters, git_commit=None, experiment_id=None, verify_commit=True):
    """Verifies the existence of a run with the same parameters and on the same git hash.

    Best-effort detection of if a run with the given entrypoint name,
    parameters, and experiment id already ran. The run must have completed
    successfully and have at least the parameters provided.

    Parameters
    ----------
    entrypoint : str
        The taget mlflow entrypoint. Name of the project entry point associated with the current run, if any.
    parameters : dict
        Key-value input parameters of your choice. Both keys and values are strings.
    git_commit : str
        The git hash of a git commit.
    use_cache : bollean, default=True
        If True it tries to reuse existing runs.
    experiment_id : str
        ID of the experiment under which to create the current run.
        If experiment_id argument is unspecified, will look for valid experiment in the following order: activated using
        set_experiment, MLFLOW_EXPERIMENT_NAME environment variable, MLFLOW_EXPERIMENT_ID environment variable, or the
        default experiment as defined by the tracking server.
    verify_commit : bool
        Verify the commit hash to resuse the run. If False uses runs even if they are not in the previous commit.

    Returns
    -------
    run : mlflow.ActiveRun
        A active mlflow run, object that acts as a context manager wrapping the run’s state.
    """
    experiment_id = experiment_id if experiment_id is not None else _get_experiment_id()
    client = mlflow.tracking.MlflowClient()
    all_run_infos = reversed(client.list_run_infos(experiment_id))
    for run_info in all_run_infos:
        full_run = client.get_run(run_info.run_id)
        tags = full_run.data.tags
        run_entry_point_name = tags.get(mlflow_tags.MLFLOW_PROJECT_ENTRY_POINT, None)
        if run_entry_point_name != entry_point_name:
            continue
        match_failed = False
        for param_key, param_value in six.iteritems(parameters):
            run_value = full_run.data.params.get(param_key)
            if run_value is not None:
                if "[" in run_value:
                    run_value = ast.literal_eval(ast.literal_eval(run_value))
            if isinstance(param_value, Number):
                param_value = str(param_value)
            if "path" not in param_key:
                if run_value != param_value:
                    match_failed = True
                    break
        if match_failed:
            continue

        if run_info.to_proto().status != RunStatus.FINISHED:
            continue

        # if verify_commit:
        #     if git_commit is None:
        #         raise ("Verify_commit is True, but git_commit is None.")
        #     previous_version = tags.get(mlflow_tags.MLFLOW_GIT_COMMIT, None)
        #     if git_commit != previous_version:
        #         continue
        git_starus = git.Repo(search_parent_directories=True).index.diff(None)
        if git_starus:
            changed_files = ",\n".join([item.a_path for item in git_starus])
            warn(f"We are reusing {entry_point_name}, but these files have uncommitted changes:\n{changed_files}")
        return client.get_run(run_info.run_id)
    return False


def link_artefacts(artifact_uri, new_artifact_uri):
    """Create Symbolic links to artifacts while reusing runs.

    Parameters
    ----------
    artifact_uri: str
        Path for the original artifacts in the existing run.
    new_artifact_uri: str
        Path for the new artifacts in the copied run.
    """
    artifacts = Path(artifact_uri).rglob("*")
    for artifact in artifacts:
        symlink(artifact.absolute(), f"{new_artifact_uri}/{artifact.name}")


def relog(run, experiment_id):
    """Fill fields in the run log while reusing runs.

    Parameters
    ----------
    run : mlflow.ActiveRun
        A active mlflow run, object that acts as a context manager wrapping the run’s state.

    Returns
    -------
    run : mlflow.ActiveRun
        Modified mlflow run, object that acts as a context manager wrapping the run’s state.
    """
    with mlflow.start_run(
        run_name=run.data.tags["mlflow.project.entryPoint"], nested=True, experiment_id=experiment_id
    ) as new_run:
        mlflow.log_params(run.data.params)
        mlflow.log_metrics(run.data.metrics)
        source = run.data.tags["mlflow.source.name"]
        entryPoint = run.data.tags["mlflow.project.entryPoint"]
        run.data.tags["mlflow.source.name"] = f"{source}/{entryPoint}"
        link_artefacts(
            run.info.artifact_uri.replace('file://', '').replace("file://", ""),
            Path(new_run.info.artifact_uri.replace('file://', '').replace("file://", "")).absolute(),
        )
        return run


def get_or_run(entrypoint, parameters, git_commit, use_cache=True, experiment_id=None, verify_commit=True):
    """Get existent run with the same parameters or run the code if none is found.

    A search is done to determine if the entry point was previously run with the same parameters and the same code.
    To determine if the code is the same we look in the git hash of the current branch.
    If there are uncommitted changes we warn the user and list the files with changes, but the run is reused.
    Parameters
    ----------
    entrypoint : str
        The taget mlflow entrypoint. Name of the project entry point associated with the current run, if any.
    parameters : dict
        Key-value input parameters of your choice. Both keys and values are strings.
    git_commit : str
        The git hash of a git commit.
    use_cache : boolean, default=True
        If True it tries to reuse existing runs.
    experiment_id :
        ID of the experiment under which to create the current run.
        If experiment_id argument is unspecified, will look for valid experiment in the following order: activated using
        set_experiment, MLFLOW_EXPERIMENT_NAME environment variable, MLFLOW_EXPERIMENT_ID environment variable, or the
        default experiment as defined by the tracking server.

    Returns
    -------
    existing_run : mlflow.ActiveRun
        A active mlflow run, object that acts as a context manager wrapping the run’s state.
    """
    if verify_commit in [False, "False", "false", "n"]:
        verify_commit = False
    if use_cache not in [False, "False", "false", "n"]:
        existing_run = already_ran(entrypoint, parameters, git_commit, experiment_id, verify_commit=verify_commit)
        if existing_run:
            existing_run = relog(existing_run, experiment_id)
            existing_run.data.tags["reused"] = "True"
            return existing_run
    submitted_run = mlflow.run(".", entrypoint, parameters=parameters, experiment_id=experiment_id)
    run = mlflow.tracking.MlflowClient().get_run(submitted_run.run_id)
    run.data.tags["reused"] = "False"

    try:
        mlflow.log_param("git_commit", git_commit)
    except mlflow.exceptions.MlflowException:
        pass
    return run
