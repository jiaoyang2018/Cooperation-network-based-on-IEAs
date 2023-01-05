# Cooperation-network-based-on-IEAs

These are the codes for the implementation of the construction and analysis of the international environmental cooperation network based on International Environmental Agreements (IEAs), as described in our paper:


Carattini, Stefano, Sam Fankhauser, Jianjian Gao, Caterina Gennaioli, and Pietro Panzarasa. [What does network analysis teach us about international environmental cooperation?](https://www.sciencedirect.com/science/article/pii/S0921800922003317). Ecological Economics 205 (2023): 107670.


## Data 

The data on international environmental agreements(IEAs) is from [ECOLEX](https://www.ecolex.org/). Please contact us for the data.

## Codes

Network construction and analysis are conducted by **Python**, while the regression analysis is performed by **Stata**.

```
- weighted_network.py: functions defined to calcualte weighted gloabl and local measures of networks;

- dataset_construction.py: functions defined to produce statistically significant one-mode networks and calculate local and global measures by calling functions in weighted_network.py;

- appendix.py: functions defined to perform the calculations in the appendix;

- dataset_construction.ipynb: examples of how to use functions in dataset_construction.py;

- results_analysis_main_paper.ipynb: codes for tables and figures in the main paper;

- results_analysis_appendix.ipynb: codes for tables and figures in the appendix;

- appendix_regression.do: Stata codes for the regression tables in the appendix.
```

## How to use the functions defined in weighted_network.py, dataset_construction.py and appendix.py

```
import weighted_network as wn

import dataset_construction as dc

import appendix as ap
```

## Cite 

Please cite our paper if you use these codes in your own work:

```
@article{carattini2021does,
  title={What does network analysis teach us about international environmental cooperation?},
  author={Carattini, Stefano and Fankhauser, Sam and Gao, Jianjian and Gennaioli, Caterina and Panzarasa, Pietro},
  journal={Ecological economics},
  year={2022}
}
```
