"""Grid search of parameters for the Adversarial Autoencoder (AAE).

This script trains and optimises the AAE model, which is one of the main components of Neurofind.
"""
from pathlib import Path
import tempfile

from numpy import array, median, mean
from scipy.stats import iqr
import click
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from src import utils
from src.autoencoders import GridSearchAAE
from src import definitions
from src.create_figures import get_box_plot


def create_reconstruction_error_hist(estimator, X, output_path):
    """Plot the loss value per epoch for the autoencoder, the generator and the discriminator.

    Parameters
    ----------
    estimator : Autoencoder object
        Trained autoencoder model.
    X : array-like of shape(n_samples, n_features) [n_sample, [freesurferData, age, gender, diagn]]
        Training vector, where n_samples is the number of subjects and
        n_features is the number of features in the freesurferData plus age, gender, tiv and diagnosis.
        In order to work the model needs to receive a array with the all the features in the freesurferData followed
        by the age, gender and diagn.
    output_path : type
        Path to folder where we want to save the output image.

    Returns
    -------
    str
        Path for the image saved.
    """
    X = X.values
    norm_fs = X[:, :-4] / X[:, -2:-1]
    median_std = median(norm_fs.std(axis=0))
    iqr_std = iqr(norm_fs.std(axis=0))
    plt.figure(figsize=(16 / 3, 9 / 3))
    try:
        plt.hist(estimator.recons_error_, bins=30)
    except ValueError:
        pass
    ylim = plt.gca().get_ylim()
    plt.errorbar(
        [median_std], [mean(ylim)], xerr=[iqr_std], label=f"$median \\pm iqr = {median_std:.2g} \\pm {iqr_std:.2g}$"
    )
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    plt.xlabel("|Reconstruction Error|")
    plt.ylabel("N")
    plt.legend()
    plt.tight_layout()
    img_path = f"{output_path}/reconstruction_error_hist.png"
    plt.savefig(img_path, dpi=200)
    plt.close()
    return img_path


def create_loss_epoch_plot(estimator, output_path):
    """Plot the loss value per epoch for the autoencoder, the genarator and the discriminator.

    Parameters
    ----------
    estimator: Autoencoder object
        Trained autoencoder model.
    output_path : type
        Path to folder where we want to save the output image.

    Returns
    -------
    str
        Path for the image saved.
    """
    autoencoder_loss = estimator.ae_loss
    generator_loss = estimator.gen_loss
    descriminator_loss = estimator.dc_acc

    plt.figure(figsize=(16 / 3, 9 / 3))
    plt.plot(range(1, len(autoencoder_loss) + 1), autoencoder_loss, color="#f44336", label="Autoencoder")
    plt.plot(range(1, len(generator_loss) + 1), generator_loss, color="#4caf50", label="Generator")
    plt.plot(range(1, len(descriminator_loss) + 1), descriminator_loss, color="#ffc107", label="Descriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.ylim(0, 5)
    plt.tight_layout()
    img_path = f"{output_path}/epoch_loss.png"
    plt.savefig(img_path, dpi=200)
    plt.close()
    return img_path


def aae_data_preparation(input_path, ids_path, disease_label_list, validation_datasets, only_hc=False):
    input_path = input_path.replace("file://", "")
    ids_path = ids_path.replace("file://", "")

    dataset = utils.load_dataset(input_path, ids_path, disease_label_list=disease_label_list, relative_volumes=True)

    # Removing the test_id subjects specific for diagnoses, as these are duplicates from earlier versions
    is_test = dataset.copy().groups.str.contains('test_ids_').values
    dataset = dataset[~is_test]

    is_validation = dataset.copy().groups.str.contains('validation').values
    y = dataset.copy()[~is_validation][["id", "Diagn"]].set_index("id")
    X = dataset.copy()[~is_validation][["id"] + definitions.COLUMNS_NAME + ["Age", "Gender"]].set_index("id")
    groups = dataset.copy()[~is_validation][["id", "groups"]].set_index("id")
    if only_hc:
        X = X.copy()[(y == 1).values]
        groups = groups.copy()[(y == 1).values]
        y = y.copy()[(y == 1).values]
    return X, y, groups, dataset, is_validation


def aae_organize_results(dataset, gridsearchaee, is_validation):
    dataset_apply = dataset[["id"] + definitions.COLUMNS_NAME + ["Age", "Gender"]].set_index("id").copy()
    predicted_deviations = gridsearchaee.best_estimator_.predict(dataset_apply.values)
    predicted_deviations = pd.DataFrame(predicted_deviations,
                                        index=dataset.id,
                                        columns=["deviations"],
                                        ).reset_index()
    predicted_deviations_regions = gridsearchaee.best_estimator_.predict_regional_deviations(dataset_apply)
    predicted_deviations_regions = pd.DataFrame(predicted_deviations_regions,
                                                index=dataset.id,
                                                columns=definitions.COLUMNS_NAME,
                                                ).reset_index()
    predicted_deviations = (
        predicted_deviations.merge(dataset[["id", "Age", "Gender", "Diagn", "scanner", "groups"]], on="id")
        .drop_duplicates()
        .set_index("id")
    )
    predicted_deviations_regions = (
        predicted_deviations_regions.merge(dataset[["id", "Age", "Gender", "Diagn", "scanner", "groups"]], on="id")
        .drop_duplicates()
        .set_index("id")
    )
    return predicted_deviations, predicted_deviations_regions


def summarize_deviations(dataset, predicted_deviations, is_validation, validation_datasets):
    """Plot the loss value per epoch for the autoencoder, the genarator and the discriminator.

    Parameters
    ----------
    dataset: pandas DataFrame
        DataFrame containing subject-level regional volumes and demographic information.
    predicted_deviations : pandas DataFrame
        DataFrame containing subject-level deviations and demograhic information with index ID.
    is_validation : ndarray of bool
        Subject-level data to show whether each subject is part of the validation datasets.
    validation_dataset : list of str
        List containing scanner names of all included validation datasets.

    Returns
    -------
    test_deviations : pandas Series
        Pandas Series containing subject-level deviation scores in the validation datasets with index ID.
    dataset_test: pandas DataFrame
        DataFrame contain all subjects.
    """
    # Get the deviations for the validation scanners only
    dataset_test = predicted_deviations[predicted_deviations.scanner.isin(validation_datasets)].copy()

    # Extract subject-level deviations from dataset_test as a pandas Series
    test_deviations = dataset_test['deviations']

    return test_deviations, dataset_test


def aae_save_results(temp_dir,
                     predicted_deviations,
                     predicted_deviations_regions,
                     predicted_deviations_train,
                     pretrained_model,
                     gridsearchaee,
                     reconstruction_error_hist_path,
                     loss_epoch_plot_path,
                     box_plot_bydiagn_path, box_plot_byscanner_path,
                     box_plot_bydiagn_path_train, box_plot_byscanner_path_train):
    # Save deviations
    deviations_path_train = f"{temp_dir}/aae_deviations_train.csv"
    predicted_deviations_train.to_csv(deviations_path_train)
    deviations_path = f"{temp_dir}/aae_deviations_validation.csv"
    predicted_deviations.to_csv(deviations_path)

    deviations_regions_path = f"{temp_dir}/aae_deviations_regions.csv"
    predicted_deviations_regions.to_csv(deviations_regions_path)

    # Save model
    # if pretrained_model is None:
    #     gridsearchaee.save(temp_dir, "GridSearchAAE.joblib")
    #     gridsearchaee_path = f"{temp_dir}/GridSearchAAE.joblib"
    #     mlflow.log_artifact(gridsearchaee_path)

    mlflow.log_artifacts(temp_dir)

    keras_path = Path(f"{temp_dir}/keras/")
    keras_path.mkdir(exist_ok=True)

    mlflow.log_artifact(deviations_path_train)
    mlflow.log_artifact(deviations_path)
    mlflow.log_artifact(deviations_regions_path)
    mlflow.log_artifacts(keras_path)
    mlflow.log_artifact(reconstruction_error_hist_path)
    mlflow.log_artifact(loss_epoch_plot_path)

    pd.DataFrame(gridsearchaee.best_params_).to_html(f"{temp_dir}/best_params.html")
    pd.DataFrame(gridsearchaee.best_score_.sort_values(ascending=False)).to_html(f"{temp_dir}/score.html")
    mlflow.log_artifact(f"{temp_dir}/score.html")
    mlflow.log_artifact(f"{temp_dir}/best_params.html")
    for key, value in gridsearchaee.best_score_.iteritems():
        try:
            mlflow.log_metric(key.replace("'", ""), value)
        except mlflow.exceptions.MlflowException:
            mlflow.log_metric("bad_name", value)
            print(f"bad key name {key}!!!!!!!!")
    try:
        mlflow.log_metric("model_performance", array(gridsearchaee.best_score_).mean())
    except ZeroDivisionError:
        mlflow.log_metric("model_performance", 0)
    mlflow.log_artifact(box_plot_bydiagn_path)
    mlflow.log_artifact(box_plot_byscanner_path)
    mlflow.log_artifact(box_plot_bydiagn_path_train)
    mlflow.log_artifact(box_plot_byscanner_path_train)


@ click.command(help="")
@ click.option("--input_path")
@ click.option("--ids_path")
@ click.option("--disease_label_list", cls=utils.PythonLiteralOption, default=[])
@ click.option("--validation_datasets", cls=utils.PythonLiteralOption, default=[])
@ click.option("--pre_scaler_list", cls=utils.PythonLiteralOption, default="['RobustScaler']")
@ click.option("--pos_scaler_list", cls=utils.PythonLiteralOption, default="['RobustScaler']")
@ click.option("--base_learning_rate_list", cls=utils.PythonLiteralOption, default="[0.0001]")
@ click.option("--max_learning_rate_list", cls=utils.PythonLiteralOption, default="[0.005]")
@ click.option("--gamma_list", cls=utils.PythonLiteralOption, default="[0.98]")
@ click.option("--h_dim_list", cls=utils.PythonLiteralOption, default="[[10, 10, 10], [5, 5, 5], ]")
@ click.option("--z_dim_list", cls=utils.PythonLiteralOption, default="[10, 20, 30, 40, 50]")
@ click.option("--batch_size_list", cls=utils.PythonLiteralOption, default="[256]")
@ click.option("--pretrained_model", default=None)
def perform_search(
    input_path,
    ids_path,
    disease_label_list=[],
    validation_datasets=[],
    pre_scaler_list=["RobustScaler"],
    pos_scaler_list=["RobustScaler"],
    base_learning_rate_list=[0.0001],
    max_learning_rate_list=[0.005],
    gamma_list=[0.98],
    h_dim_list=[[10, 10, 10], [5, 5, 5]],
    z_dim_list=[10, 20, 30, 40, 50],
    batch_size_list=[256],
    pretrained_model=None,
):
    """Perform grid search adversarial autoencoders.

    Parameters
    ----------
    input_path : str
        Path for the freesurferData and participants file, typically it would be the harmoniser output path.
    ids_path : str
        Path for the list of group ids, typically it would be the path for the output of the create_ids.
    pre_scaler_list : list, | 'RobustScaler' | 'StandardScaler' |, default : ['RobustScaler']
        List of scaler to apply before the input of data to the autoencoder.
    pos_scaler_list : list, | 'RobustScaler' | 'StandardScaler' |, default : ['RobustScaler']
        List of scaler to apply after the input of data to the autoencoder.
    h_dim_list : list of lists of integers, default : [[10, 10, 10], [5, 5, 5], ]
        List of the possible values numbers of neurons in each Dense network layers. One can use any number of layers as
        long as they are passed as a list. For instance h_dim = [10] will create one Dense layer with 10 neurons
        followed by a LeakyReLU layer, h_dim = [10, 100] will create 4 layers, one Dense layer with 10 neurons followed
        by a LeakyReLU layer and another Dense layer with a 100 neurons followed of a second LeakyReLU.
    z_dim_list : list of integers, default : [10, 20, 30, 40, 50]
        The list of numbers of neurons in the bottleneck layer of the Autoencoder.
    batch_size : int, default : [256]
        The list of the batch size limits the number of samples to be shown to the network before a weight update can be
        performed. This same limitation is then imposed when making predictions with the fit model.
    """
    disease_label_list = disease_label_list if len(disease_label_list) else definitions.DISEASE_LABEL_LIST
    validation_datasets = validation_datasets if len(validation_datasets) else definitions.VALIDATION_DATASETS
    with mlflow.start_run(run_name="gridsearch_aae", nested=True) as run:
        TempDir = tempfile.TemporaryDirectory(dir=run.info.artifact_uri.replace('file://', ''))
        temp_dir = Path(TempDir.name)

        X, y, groups, dataset, is_validation = aae_data_preparation(
            input_path,
            ids_path,
            disease_label_list,
            validation_datasets,
            only_hc=True,
        )
        n_train_total = X.shape[0]
        n_validation_total = is_validation.sum()

        # Set random seed
        np.random.seed(0)
        tf.random.set_seed(0)

        parameters = {
            "base_learning_rate": base_learning_rate_list,
            "max_learning_rate": max_learning_rate_list,
            "gamma": gamma_list,
            "h_dim": h_dim_list,
            "z_dim": z_dim_list,
            "batch_size": batch_size_list,
        }
        if pretrained_model in ["None", None]:
            gridsearchaee = GridSearchAAE(param_grid=parameters, train_epochs=3,
                                          max_iter=20, refit=True, criterion='mse',
                                          random_state=0)
            gridsearchaee.fit(X, y)
        else:
            print('Loading pretrained model')
            gridsearchaee = joblib.load(f"{pretrained_model}/GridSearchAAE.joblib")
            gridsearchaee.save_path_ = Path(pretrained_model)
            gridsearchaee.load()

        # Apply the best AAE estimator to dataset
        predicted_deviations, predicted_deviations_regions = aae_organize_results(dataset, gridsearchaee, is_validation)

        test_deviations, dataset_test = summarize_deviations(dataset,
                                                             predicted_deviations,
                                                             is_validation,
                                                             validation_datasets,
                                                             )

        loss_epoch_plot_path = create_loss_epoch_plot(gridsearchaee.best_estimator_, temp_dir)
        reconstruction_error_hist_path = create_reconstruction_error_hist(gridsearchaee.best_estimator_, X, temp_dir)
        predicted_deviations.reset_index(inplace=True)

        # Get predicted deviations from training set only
        predicted_deviations_train = predicted_deviations[~is_validation]
        duplicates_to_drop = predicted_deviations_train.copy().groups.str.contains('test_ids_').values
        predicted_deviations_train = predicted_deviations_train[~duplicates_to_drop]

        print(gridsearchaee.best_params_)
        # Get predicted deviations for validation set only
        predicted_deviations = predicted_deviations[is_validation]
        print(predicted_deviations.deviations.mean())

        # Variable to check if the results are consistent across runs
        aae_train_mean = predicted_deviations_train[predicted_deviations_train.Diagn == 1]['deviations'].mean()
        aae_hc_mean = predicted_deviations[predicted_deviations.Diagn == 1]['deviations'].mean()
        print(aae_train_mean, aae_hc_mean)

        # Create AAE deviation result figures by scanner and by diagnosis
        box_plot_bydiagn_path = get_box_plot(predicted_deviations, metric='deviations',
                                             grouping_var='groups', output_path=temp_dir, filename='boxplot_bydiagn')
        box_plot_byscanner_path = get_box_plot(predicted_deviations, metric='deviations',
                                               grouping_var='scanner', output_path=temp_dir, filename='boxplot_byscanner')

        # Plot deviation scores from training datasets
        box_plot_byscanner_path_train = get_box_plot(predicted_deviations_train, metric='deviations',
                                                     grouping_var='scanner', output_path=temp_dir,
                                                     filename='boxplot_byscanner_train')
        box_plot_bydiagn_path_train = get_box_plot(predicted_deviations_train, metric='deviations',
                                                   grouping_var='Diagn', output_path=temp_dir,
                                                   filename='boxplot_bydiagn_train')

        # -------------------------------------------------------------------------------------------
        # Save model
        gridsearchaee.save(temp_dir, "GridSearchAAE.joblib")
        gridsearchaee_path = f"{temp_dir}/GridSearchAAE.joblib"
        mlflow.log_artifact(gridsearchaee_path)
        mlflow.log_artifacts(temp_dir / 'keras')

        # Saves the network and the relevant metrics
        aae_save_results(temp_dir,
                         predicted_deviations,
                         predicted_deviations_regions,
                         predicted_deviations_train,
                         pretrained_model,
                         gridsearchaee,
                         reconstruction_error_hist_path,
                         loss_epoch_plot_path,
                         box_plot_bydiagn_path,
                         box_plot_byscanner_path,
                         box_plot_bydiagn_path_train,
                         box_plot_byscanner_path_train,
                         )

        # Log sample sizes
        mlflow.log_metrics(
            {'n_train_total': n_train_total,
             'n_validation_total': n_validation_total,
             }
        )

        TempDir.cleanup()


if __name__ == "__main__":
    perform_search()
