# PdM
============================

> Folder Structure for this Github

    .
    ├── input
    | 	├── *.csv                   
    ├── modules/
    |	├── DE_mod/
    |	|      ├── DE_train.py
    |	|      └── DE_deploy.py
    |	├── FC_mod/
    |	|      ├── CM_train.py
    |	|      ├── DL_train.py
    |	|      └── best_deploy.py
    |	└── helper_class/       
    |	       ├── time_shift.py
    |	       └── classical_model.py            
    ├── output/
    |	├── DE_output/
    |	|      ├── DE_output.txt
    |	|      └── *.csv
    |	└── FC_output/
    |	       ├── DL_output/
    |	       |       └── *.h5
    |	       ├── CL_output/
    |              |       └── *.pickle
    |	       └── FC_pred/
    |                      └── results.csv                  
    ├── results/
    |       ├── DE_results/
    |       |       ├── DE_techniques.csv
    |       |       └── DE_time.csv
    |       └── FC_results/        
    |               ├── DL_results/
    |               |      ├── roc.png
    |               |      ├── conf.png
    |               |      └── *_time.csv
    |               └── CL_results/    
    |                      ├── roc.png
    |                      ├── conf.png
    |                      └── *_time.csv                           
    ├── README.md
    └── requirements.txt



========================

#### Below we will go through the description of what each folder does

## Input

> The Input folder contains the input file, in our case it is a .csv

## Modules

> The modules folder contains the modules which we will be using for fault detection
> The modules for now are Data Enrichment and Fault Classification

#### Data Enrichment

> This module takes in a dataframe and performs an imputation technique that fills in
> missing data.

#### Fault Classification

> This module predicts whether or not a given dataset of sensor readings is a fault
> or not a fault.

## Output

> Outputs of each module are placed here, the outputs here are intended to be read as inputs for 
> the next module in the PdM pipeline

## Results 

> Results of each module are placed here, the results are resulting tables, graphs, scores of 
> the module calculates. These results are not read by the pipeline, they are meant to be read
> by a human interpreter.
