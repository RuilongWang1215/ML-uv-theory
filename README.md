# ML-uv-theory

This project focuses on using symbolic regression algorithms to develop interpretable predictive models for correction term of uv-theory, specifically using symbolic regression methods like [PySR](https://github.com/MilesCranmer/PySR) and [DSO](https://github.com/dso-org/deep-symbolic-optimization). Note that to install the package of DSO, the python version cannot be higher than 3.6.

## Table of Contents
- [Data](#data)
  - [Raw Data](#raw-data)
  - [Processed data](#data-processing)
- [Training](#training)
- [Results](#results)
- [Result Analysis](#result-analysis)

## Data

### Raw Data
The raw data includes thermodynamic property measurements for various substances:
- `real_ufrac_argon.csv`: Data for argon
- `real_ufrac_methane.csv`: Data for methane
- `real_ufrac_water.csv`: Data for water

These files contain the initial data before any transformations are applied.

### Data Processing
Prior to training, several data preprocessing steps are performed:
1. **Data Combination**: All raw data files are merged into a unified dataset.
2. **Feature Engineering**: New substances-specific features are created using RDkit to enhance the predictive power of the models. Only features that are non-zero for at least one substances are kept.
3. **Normalization**: Features are scaled to a consistent range of 1-100. This ensures stable model convergence while maintaining the integrity of restriction fulfillment evaluations.
4. **Outlier Filtering**: Outliers are identified and removed to improve model robustness.

The script `data_preprocessing.py` handles these transformations and outputs the processed data, which is saved in `data/processed_data`.

## Training

The main training script is `main.py`, where you can select different algorithms and configure training parameters, such as iteration, train_test_split ratio, data used, training framework (standard or step-wise), and the number of trainings. After each training, `restriction.py` is used to check if the result fulfills the requirements, if no, the algorithm will repeat the training until all the requirements are fulfilled. You can adjust the restriction depending on your case.

- **Algorithm Details**:
  - **PySR**: The settings for the PySR symbolic regression algorithm can be found in `SR_PYSR.py`.
  - **DSO**: The settings for the Deep Symbolic Optimization (DSO) algorithm are in `SR_DSO.py`.

Feel free to experiment with these settings to tune the models for your specific use case.

## Results

After training, results are stored in separate directories based on the algorithm used:
- **PySR Results**: Training outputs and model checkpoints for PySR are stored in the `result_pysr` folder.
- **DSO Results**: Training outputs and model checkpoints for DSO are stored in the `result_dso` folder.

These directories contain all relevant information for analyzing the performance and structure of the models produced by each algorithm.

## Result Analysis

To analyze and interpret the results of each algorithm:
- **DSO Analysis**: Use `dso_analysis.ipynb` to explore and evaluate the results from the DSO model.
- **PySR Analysis**: Use `pysr_analysis.ipynb` to explore and evaluate the results from the PySR model.

