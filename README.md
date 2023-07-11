## *Predicting appropriateness of antibiotic treatment among ICU patients with hospital acquired infection*
Public repository containing code for model predicting antibiotic treatment appropriateness, as described in the manuscript "Predicting appropriateness of antibiotic treatment among ICU patients with hospital acquired infection"

#### Authors:
Ella 1*, Ella 2*, Dan Coster, Asaf Wasserman, Daniel Bernstein, Ron Shamir

#### Modules
The repository contains the following modules:
* **`data_processing`** Code for preprocessing the data, including time-series feature creation, imputation, outlier removal, etc. 
* **`feature_selection`** Code for feature selection methods and filteration of correlated and redundent features.
* **`feature_stats`** Code for conducting statistical tests on the features.
* **`models`** Code for the BalancedDataEnsemble model and a generic class for ABX appropriateness prediction model.
