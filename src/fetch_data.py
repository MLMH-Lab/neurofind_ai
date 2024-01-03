#!/usr/bin/env python
"""Script used to download and organise the study data.
Please note: The data used in the development of Neurofind is not available to download by the public, 
so this script needs to be adapted to your study setup.

This script consists of the following steps:
-   Download files for all scanners from network-attached storage (NAS) server,
    incl. demographic information (participants.tsv), imaging data (freesurferData.csv),
    quality assessments for the raw data (group_T1w.tsv and mriqc_prob.csv).
-   Organise data (merge into one dataset; rename IDs to include dataset name and participant_id)
-   Save data ("participants.tsv", "freesurferData.csv", "qc.tsv")
"""
from pathlib import Path
from shutil import copyfile
import tempfile

import click
import mlflow
import pandas as pd

from src.definitions import COLUMNS_NAME, QC_FEATURES
from src.utils import LogRun


def get_scanners_list(bids_data):
    """Generate a list of scanners based on the folders inside a BIDS-like folder.

    Parameters
    ----------
    bids_data : str
        Path to the BIDS directory.

    Returns
    -------
    scanner_list : list
        List of the scanners found inside the BIDS directory.
    """
    dataset_paths = [dataset_path for dataset_path in Path(bids_data).glob("*") if dataset_path.is_dir()]
    scanner_paths = []
    for dataset_path in dataset_paths:
        scanner_paths.extend([scanner_path for scanner_path in dataset_path.glob("*") if scanner_path.is_dir()])
    scanner_list = [scanner_path.name for scanner_path in scanner_paths]
    return scanner_list


def download_files(local_dir, selected_path, dataset_prefix, path_nas):
    """Download the files necessary for the study from network-attached storage (NAS).

    These files include:
        - participants.tsv: Demographic data
        - freesurferData.csv: Neuroimaging data preprocessed using FreeSurfer
        - group_T1w.tsv and mriqc_prob.csv: Raw data quality control metrics obtained using MRIQC tool

    Parameters
    ----------
    local_dir: PosixPath
        Path indicating local path to store the data.
    selected_path: PosixPath
        Path indicating external path with the data.
    dataset_prefix: str
        Datasets prefix.
    path_nas: PosixPath
        Path indicating NAS system.
    """
    dataset_path = local_dir / dataset_prefix
    dataset_path.mkdir(exist_ok=True, parents=True)

    try:
        copyfile(str(selected_path / "participants.tsv"), str(dataset_path / "participants.tsv"))
        # print(selected_path)
        # print(pd.read_csv(Path(dataset_path / "participants.tsv"), delimiter="\t").shape)
    except (FileNotFoundError, StopIteration):
        print(f"{dataset_prefix} does not have participants.tsv")

    try:
        copyfile(
            str(path_nas / "FreeSurfer_preprocessed" / dataset_prefix / "freesurferData.csv"),
            str(dataset_path / "freesurferData.csv"),
        )

    except (FileNotFoundError, StopIteration):
        print(f"{dataset_prefix} does not have freesurferData.csv")

    try:
        copyfile(str(path_nas / "MRIQC" / dataset_prefix / "group_T1w.tsv"), str(dataset_path / "group_T1w.tsv"))
    except (FileNotFoundError, StopIteration):
        print(f"{dataset_prefix} does not have group_T1w.tsv")

    try:
        mriqc_prob_path = next((path_nas / "MRIQC" / dataset_prefix).glob("*unseen_pred.csv"))
        copyfile(str(mriqc_prob_path), str(dataset_path / "mriqc_prob.csv"))
    except (FileNotFoundError, StopIteration):
        print(f"{dataset_prefix} does not have *unseen_pred.csv")


def merge_files(selected_path):
    """Merge all type of data into one DataFrame.

    Perform the merge of all type of data (i.e. demographics, neuroimaging,
    quality metrics) into a single pandas DataFrame from a specific dataset.

    Parameters
    ----------
    selected_path: PosixPath
        Path to the selected dataset.

    Returns
    -------
    dataset_df: pandas DataFrame
        DataFrame with all type of data.
    """
    participants = pd.read_csv(f"{selected_path}/participants.tsv", sep="\t")
    fs = pd.read_csv(f"{selected_path}/freesurferData.csv")
    mriqc = pd.read_csv(f"{selected_path}/group_T1w.tsv", sep="\t")
    prob_mriqc = pd.read_csv(f"{selected_path}/mriqc_prob.csv")

    if len(mriqc) != len(prob_mriqc):
        print(f"{selected_path} mriqc files with different sizes")
        raise

    mriqc_data = pd.merge(mriqc, prob_mriqc, on="image_id")
    if mriqc_data.shape[0] == 0:
        print("MRIQC merge returns zero subjects")
        return None
    mriqc_data["mriqc_prob"] = mriqc_data["prob_y"]

    dataset = pd.merge(fs, mriqc_data, on="image_id")
    if dataset.shape[0] == 0:
        print("MRIQCxFS merge returns zero subjects")
        return None

    dataset = pd.merge(dataset, participants, on="image_id")
    if dataset.shape[0] == 0:
        print("DATASETxPARTICIPANTS merge returns zero subjects")
        return None

    dataset["id"] = dataset["Dataset"] + "_" + dataset["image_id"]

    if sum(dataset.duplicated(subset="id")) > 0:
        print(f"{selected_path} contains duplicate subjects")
        dataset = dataset.drop_duplicates(subset="id", keep=False)

    return dataset


@click.command(help="Path to the Network Attached Storage system.")
@click.option("--raw_path", default="data/raw/")
@LogRun()
def fetch_data(raw_path="data/raw/"):
    """Perform organization of the datasets.

    Parameters
    ----------
    raw_path: str
        Path to the raw data. The target folder must contain the following subfolders:
        BIDS_data, FreeSurfer_preprocessed, MRIQC;
        Each of this folders must contain a series of dataset folders. Each folder inside of the dataset folders are
        considered to be a different scanner. For example: BIDS_data/dataset/scanner01/

    Outputs
    -------
    participants.tsv: File
        Tab-separated file containing the subjects' demographic information.
    freesurferData.csv: File
        Comma-separated file containing the raw regional volumes of the 101 FreeSurfer regions for each subject.
    qc.tsv: File
        Tab-separated file containing the quality measures for the subjects' images from MRIQC.

    Metrics
    -------
    N_participants: int
        Number of participants.
    N_scanners: int
        Number of different scanners.
    """
    with mlflow.start_run(run_name="fetch_data", nested=True) as run:
        TempDir = tempfile.TemporaryDirectory(dir=run.info.artifact_uri.replace('file://', ''))
        temp_dir = Path(TempDir.name)

        print("*********************** DOWNLOAD DATA ************************")
        # Create list of scanners from the included datasets
        raw_path = raw_path.replace("file://", "")
        DATASET_LIST = get_scanners_list(f"{raw_path}/BIDS_data/")
        raw_path = Path(raw_path)
        for selected_path in (raw_path / "BIDS_data").iterdir():
            if not selected_path.is_dir():
                continue

            for selected_subdir in selected_path.iterdir():
                if not selected_subdir.is_dir():
                    continue

                scanner_name = selected_subdir.stem
                if scanner_name not in DATASET_LIST:
                    continue
                # print(f"Fetching {scanner_name}")

                if (selected_subdir / "participants.tsv").is_file():
                    download_files(temp_dir, selected_subdir, f"{selected_path.stem}/{scanner_name}", raw_path)

        print("********************* COMBINE DATASETS ***********************")
        dataset_list = []
        # Loop over dataset names
        for selected_path in temp_dir.iterdir():
            # Loop over scanner within each dataset
            for selected_path in selected_path.iterdir():
                # print(selected_path)
                # Merge all data into one file
                try:
                    merged = merge_files(selected_path)
                    if merged is None:
                        print(f"Zero subjects after match in {str(selected_path)}")
                    else:
                        dataset_list.append(merged)
                except (KeyError, ValueError, FileNotFoundError) as err:
                    print(f"{err.__class__.__name__}: {err} with {str(selected_path)}")

        dataset = pd.concat(dataset_list, axis=0)
        dataset.Dataset.sort_values().unique()

        participants = dataset[["id", "participant_id", "Age", "Gender", "Diagn", "Dataset"]]

        participants.Diagn = pd.to_numeric(participants["Diagn"], errors="coerce")

        fs = dataset[["id", "EstimatedTotalIntraCranialVol"] + COLUMNS_NAME]

        qc = dataset[["id", "mriqc_prob"] + QC_FEATURES]

        # Saving MLflow artifacts
        participants_path = temp_dir / "participants.tsv"
        fs_path = temp_dir / "freesurferData.csv"
        qc_path = temp_dir / "qc.tsv"

        participants.to_csv(participants_path, sep="\t", index=False)
        fs.to_csv(fs_path, index=False)
        qc.to_csv(qc_path, sep="\t", index=False)

        print(f"Uploading participants_path.tsv: {participants_path}")
        print(f"Uploading freesurferData.csv: {fs_path}")
        print(f"Uploading qc.tsv: {qc_path}")
        mlflow.log_artifact(participants_path)
        mlflow.log_artifact(fs_path)
        mlflow.log_artifact(qc_path)

        mlflow.log_metrics({"N_participants": participants.shape[0], "N_scanners": len(participants.Dataset.unique())})

        print("**************************************************************")
        TempDir.cleanup()


if __name__ == "__main__":
    fetch_data()
