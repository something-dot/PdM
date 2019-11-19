# PdM
============================

Folder Structure for this Github

    .
    ├── input
    | 	├── *.csv                   
    ├── modules/
    |	├── DE_mod/
    |	|  ├── DE_train.py
    |	|  └── DE_deploy.py
    |	├── FC_mod/
    |	|  ├── FC_train.py
    |	|  └── FC_deploy.py
    |	└── helper_class/       
    |	   ├── time_shift.py
    |	   └── classical_model.py            
    ├── output/
    |	├── DE_output/
    |       |  └── DE_output.txt
    |	└── FC_output/
    |	   ├── DL_output/
    |	   |   └── *.h5
    |	   ├── CL_output/
    |          |   └── *.pickle
    |	   └── FC_pred/
    |              └── results.csv                  
    ├── results/
    |   ├── DE_results/
    |   |   ├── DE_techniques.csv
    |   |   └── DE_time.csv
    |   └── FC_results/        
    |       ├── DL_results/
    |       |   ├── roc.png
    |       |   ├── conf.png
    |       |   └── *_time.csv
    |       └── CL_results/    
    |           ├── roc.png
    |           ├── conf.png
    |           └── *_time.csv                           
    ├── README.md
    └── requirements.txt