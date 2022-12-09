# Japan Household Carbon Emission Footprint

This is the public repository maintained by the author team for the publication of the article "Demographic transition hinders climate change mitigation in the aging and shrinking Japanese residential sector". 

Restricted by the data policy of the Japan government, the original data or survey cannot be shared publicly on the github, therefore only the preprocessed data are available in this repository. Please contact us for further information if you are interested in the original data. 

## Get started
You should get a properly installed `Python3` environment ready before running the repository. It is highly recommended to start with `miniconda3` or `anaconda3` for the environment setup. Once the environment is ready, please run the following code to install the dependencies:

> pip install -r requirements.txt

All the dependencies required in the modelling of our paper will be installed automatically. Then, ensuring the original data (if requested) is available under `data-local/` before running the tests, you can follow up the steps: 

- Step 1: get preprocessing done
> python preprocess.py

It outputs, 1) `data/feature-desc-<timestamp>.xlsx`, 2) `cache-preprocess-data-<timstamp>.xlsx`

- Step 2: run clustering model
> python cluster.py

It outputs, 1) `data/variable-importance-figure-<timestamp>.png`, 2) `data/cache-clustered-<timestamp>.csv`, 3) `data/cache-scaled-data-<timestamp>.csv`

- Step 3: output statistical results
> python stats.py

It outputs, 1) `data/img/dist-emission-by-<type>-<timestamp>.png` images for the distribution of per capita emissions by types, 2) `data/household-by-prefecture-cluster-<timestamp>.xlsx`, 3) `data/emission-by-fuel-<timestamp>.xlsx`, 4) `data/emission-by-usage-<timestamp>.xlsx`, 5) `data/figure2-emission-by-potential-<timestamp>.xlsx`

- Step 4: projection data
> python predict.py

It outputs, 1) `data/est-pv-coef-<timestamp>.xlsx`, 2) `data/est-pv-coef-<timestamp>.xlsx`, 3) `data/figure3-projection-emission-<timestamp>.xlsx`, 4) `data/japan-pop-projection-cluster-<timestamp>.xlsx`

Only the main data relevant to clustering and projection have been included in this repository. For the other relevant data and scripts, please contact us for details. 

## Contact us
- Yi Wu, y_wu.21@ucl.ac.uk
- Yin Long, longyinutokyo@gmail.com>