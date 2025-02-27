## To plot rejection curves
1. Calculate uncertainties by running **calculate_uncertainties.py**, **DDU.py** or **HUQ.py**.

   Examples:

   `python calculate_uncertainties.py --path2save_markup_dir=[path to the directory where output uncertainties will be saved] --experts_pred=[optional: path to the experts values predicted by neural net] --path2ensemble_outputs=[optional: path to the neural nets ensemble (CE) outputs] --experts_markup=[optional: path to the ground truth experts votes]`

   `python DDU.py --path2test_config=[path to the config for testing nets] --pca_components_num=[ints from 2 to 10 are good] --test_on_val_data=[bool, testing on test or validation] --path2save_dir=[path to the directory for outputs] --path2outputs_train=[path to precomputed net's outputs on train dataset] --path2outputs_test=[path to precomputed net's outputs on validation dataset]`

   `python HUQ.py --path2test_config=[path to the config for testing nets] --pca_components_num=[ints from 2 to 10 are good] --path2save_dir=[path to the directory for outputs] --path2ddu_outputs=[path to the directory with precomputed DDU outputs] --path2outputs_train=[path to precomputed net's outputs on train dataset]`

2. Generate data (points coordinates) for the rejection curves. To do it, you need:

   2.1 to create **a config file** (the example is rejection_curves_args.json);

   2.2 to run **rejection_curves.py**.

   `python rejection_curves.py --test_markup=[path to the test markup] --dir2save=[path to the directory for outputs] --config=[path to the config, the example is in nn/rejection_curves_args.json] --net_num=[the number of nets in the ensemble]`

3. Plot rejection curves:
   
   3.1 In **dash_visualizer.py** in line 17 set *path2rejection_curves_data* as path to the config from 2.1 and in line 18 *rej_curves_data_dir* (path to the directory where outputs of rejection_curves.py are located).

   3.2 Run **dash_visualizer.py** to get an interactive rejection curves plot.

   `python dash_visualizer.py --dataset_dir=[path to the directory with data(images)] --test_markup=[path to the directory with the test markup]`

## To calculate uncertainties
### HUQ
1. Calculate neural network's outputs on train+validation and test by running **test_classifier.py** two times with the corresponding configs.

 `python test_classifier.py --test_config=[path to the config for testing nets] --output_dir=[path to the directory for outputs]`

2. Run **DDU.py** on test set.

`python DDU.py --path2test_config=[path to the config for testing nets] --pca_components_num=[ints from 2 to 10 are good] --test_on_val_data=[bool, testing on test or validation] --path2save_dir=[path to the directory for outputs] --path2outputs_train=[path to precomputed net's outputs on train dataset] --path2outputs_test=[path to precomputed net's outputs on validation dataset]`

3. Run **HUQ.py**.

 `python HUQ.py --path2test_config=[path to the config for testing nets] --pca_components_num=[ints from 2 to 10 are good] --path2save_dir=[path to the directory for outputs] --path2ddu_outputs=[path to the directory with precomputed DDU outputs] --path2outputs_train=[path to precomputed net's outputs on train dataset]`