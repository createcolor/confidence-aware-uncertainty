# Confidence-aware-Uncertainty

## Pixel-level uncertainty estimation in segmentation tasks using confidence-aware ensembles
### Quick start
To reproduce the results of the uncertainty estimation using CAE on the RIGA dataset presented in the supplementary material of the paper, locate the relevant data in the `datasets` folder and run the following commands:

```bash
python segmentation/train_segmentator.py -c segmentation/configs/train_ensembles.json
```
```bash
python segmentation/train_segmentator.py -c segmentation/configs/train_cae.json
```
```bash
python segmentation/test_segmentator.py -c segmentation/configs/test_ensemble.json
```
```bash
python segmentation/test_segmentator.py -c segmentation/configs/test_cae.json
```
```bash
python segmentation/rejection_curves_multiclass.py -c segmentation/configs/rejection_curve_tv_cae.json -f true -s results
```
This will produce two ensembles stored in the `nets` folder, their predictions stored in the `predictions` folder, and rejection curves calculated via the $\mathrm{\textit{TV}(Ens, CAE)}$ method stored in the `results` folder.
### Training models
For training basic ensembles, CAEs and MCMC models, use `train.segmentator.py`. For training ABNN models, use `train_abnn.py`.

To train a model, run
```bash
python train_segmentator.py -c [config_path]
```
or
```bash
python train_abnn.py -c [config_path]
```

Training process is curated via a `.json` config file with following arguments:
* `net_dir_name (str)`: name with which the model will be saved;
* `architecture (str)`: name of the architecture. Currently, only `unet` is supported;
* `device_type (str)`: device type that will be used by torch during training;
* `n_channels (int)`: number of input channels. Use 1 for LIDC and 3 for RIGA;
* `n_classes (int)`: number of classes to predict. Use 1 for LIDC and 3 for RIGA;
* `epochs_total (int)`: number of epochs to train the model for;
* `checkpoint_step (int)`: how often to checkpoint the model. Note that currently only the model parameters are saved, so continuing learning with correct optimizator parameters is impossible;
* `validation step (int)`: how often to perform validation steps;
* `nets_number (int)`: number of nets in an ensemble to train;
* `learn_from_scratch (bool)`: whether to learn a model from scratch or fine-tune an existing model. If `false`, it is expected that the following parameters are provided alongside:
	* `learn_from_model (str)`: path to the model to fine-tune;
	* `pretrained_epochs (int)`: number of epochs the initial model for trained for;
	* `use_previous_lr (bool)`: whether or not to initialize the learning rate with the last value obtained during training of the initial model. Currently, `true` value does not function properly;
* `net_dir_path (str)`: path to save the model to;
* `loss_func (str)`: loss function to use for training. Note that no final activation is performed during training, so proper choices would be `CrossEntropyLoss` and `BCEWithLogitsLoss` and not `BCELoss`. When training ABNN models, use `ABNNLoss` for multiclass segmentation and `BinaryABNNLoss` for single class segmentation;
* `batch_size (int)`: batch size used for training;
* `train_part (float)`: fraction of the dataset to be used for training; `1 - train_part` is used for validation;
* `dataset (dict)`: dictionary specifying the following dataset parameters:
	* `type (str)`: type of dataset. `LIDC` and `RIGA` are supported;
	* `data_path (str)`: path to data directory;
	* `gt_mode (str)`: how to supply labels. Use `mOH` or `mSVLS` for training CAEs and `expert{id}` for other models, where `id` is an integer;
	* `params (dict)`: dictionary of parameters to supply to the dataset; see `dataset.py` for class definitions to learn which parameters you can supply;
* `optimizer (dict)`: dictionary specifying the following optimizer parameters:
	* `type (str)`: type of optimizator. Use `SGLD` for training MCMC models;
	* `params (dict)`: dictionary of parameters to supply to the optimizer, such as the learning rate;
* `lr_scheduler (dict)`: dictionary specifying the following scheduler parameters:
	* `type (str)`: type of scheduler. Note that only `StepLr` has been tested;
	* `parameters (dict)`: dictionary of parameters to supply to the scheduler.

Training an ensemble creates model checkpoints at a specified interval, a folder containing seeds for reproducible data spliting, a folder containing tensorboard data and a copy of the config for logging purposes.
### Testing models
For testing basic ensembles and CAEs, use `test_segmentator.py`. For testing MCMC models, use `test_mcmc.py`. For testing ABNN models, use `test_abnn.py`.

To test a model, run
```bash
python test_segmentator.py -c [config_path]
```
or
```bash
python test_mcmc.py -c [config_path]
```
or
```bash
python test_abnn.py -c [config_path]
```

Testing process is curated via a `.json` config file with following arguments:
* `model (str)`: name of the model to test;
* `architecture (str)`: name of the architecture. Currently, only `unet` is supported;
* `device (str)`: device type that will be used by torch during training;
* `n_channels (int)`: number of input channels. Use 1 for LIDC and 3 for RIGA;
* `n_classes (int)`: number of classes to predict. Use 1 for LIDC and 3 for RIGA;
* `epochs (int)`: number of epochs the model was trained for;
* `nets (int)`: number of nets in an ensemble to train;
* `models_path (str)`: path to the directory where the model is saved;
* `save_path (str)`: path to save predictions;
* `dataset (dict)`: dictionary specifying the following dataset parameters:
	* `type (str)`: type of dataset. `LIDC` and `RIGA` are supported;
	* `data_path (str)`: path to data directory;
	* `gt_mode (str)`: how to supply labels. Use `mOH` or `mSVLS` for training CAEs and `expert{id}` for other models, where `id` is an integer;
	* `params (dict)`: dictionary of parameters to supply to the dataset; see `dataset.py` for class definitions to learn which parameters you can supply.

When testing MCMC models, also provide
* `checkpoint (int)`: step size with which the checkpoints were taken.

When testing ABNN models, also provide
* `samples (int)`: how many predictions should be sampled for each model.

Testing an ensemble creates a `predictions.pt` file containing a tensor of predictions of the model and a `Dice.json` file containing dice scores on individual samples.

### Rejection curves
For plotting rejection curves in single class segmentation tasks (e.g. LIDC), use `rejection_curves.py`. For multiclass segmentation tasks (e.g. RIGA), use `rejection_curves_multiclass.py`.

To calculate and plot rejections curves, run
```bash
python rejection_curves.py -c [config_path] -s [save_path] [-f True]
```
or
```bash
python rejection_curves_multiclass.py -c [config_path] -s [save_path] [-f True]
```

`-f True` is specified when the final activation (sigmoid/softmax) should be applied to the output provided in `expert_predictions`.

The process of calculating rejection curves is curated via a `.json` config file with following arguments:
* `models_predictions (str)`: path to predictions of the main ensemble;
* `expert_predictions (str)`: path to the predictions of an auxiliary ensemble;
* `models_method (str)`: method to calculate uncertainty from main ensemble predictions; has to be one of `"var"`, `"bern_var"` or `"none"`;
* `expert_method (str)`: method to calculate uncertainty from auxiliary ensemble predictions; has to be one of `"var"`, `"bern_var"`, `"mcmc"`, `"abnn"` or `"none"`;
* `models_ids (int or list(int))`: specifies which models in the main ensemble to use:
	* if `int`, uses first `models_ids` models of the ensemble;
	* if `list(int)`, uses indices from `models_ids` to select models from the ensemble;
* `expert_ids (int or list(int))`: similar to `models_ids` but for an auxiliary ensemble;
* `dataset (str)`: type of dataset. `LIDC` and `RIGA` are supported;
* `data_path (str)`: path to data directory;
* `expert_id (int)`: which expert the main ensemble was trained to predict.

Running the script generates a `.json` dict for each class containing pairs of (throwaway rate, Dice score) as well as a `.png` plot of curves for all classes.

#### Applying different methods
For all methods analyzed in the paper, `models_predictions` shoud be a path to the `predictions.pt` file of a basic ensemble. Use the following instructions to evaluate rejection curves for various methods:

* for the $\mathrm{\textit{TV}(Ens, -)}$ method, set `models_method` to `"var"` and `expert_method` to `"none"`. `expert_predictions` is ignored;
* for the $\mathrm{\textit{TV}(Ens, Ens)}$ method, set `expert_predictions` to be a path to the `predictions.pt` file of a basic ensemble, `models_method` to `"var"` and `expert_method` to `"bern_var"`;
* for the $\mathrm{\textit{TV}(Ens, CAE)}$ method, set `expert_predictions` to be a path to the `predictions.pt` file of a CAE ensemble, `models_method` to `"var"` and `expert_method` to `"bern_var"`;
* for the $\mathrm{\textit{TV}(Ens, Exps)}$ method, set `expert_predictions` to be a path to the `.pt` file containing averaged expert assessments, `models_method` to `"var"` and `expert_method` to `"bern_var"`;
* for the MCMC method, set `expert_predictions` to be a path to the **folder** where predictions of all MCMC checkpoints are stored, `models_method` to `"none"` and `expert_method` to `"mcmc"`;
* for the ABNN method, set `expert_predictions` to be a path to the `predictions.pt` file of an ABNN model/ensemble, `models_method` to `"none"` and `expert_method` to `"abnn"`.