# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, we use machine learning to predict customer churn using bank data.  
After developing the experimental code in a Jupyter notebook, the code is converted to production ready code via python scripts. The following clean code principles were applied in order to ensure that the code was production ready:  
- Writing unit tests
- Catching errors via try / catch statements
- Logging the errors
- Generating EDA and evaluation images 
- PEP8 standards


## Running Files
The files can be directly from the terminal using the following commands:  

pip install -r requirements.txt  
python churn_library.py

By running the commands above, the following directories should be populated:  

images/eda  
images/results  
models  

To run the tests, use the following command in the terminal:  

python churn_script_logging_and_tests.py  

By running the command above, the following directory should be populated:  

logs  

## Files in the Repo
- data
  - bank_data.csv 
- images
  - eda
    - churn_histogram.png
    - correlation_heatmap.png
    - customer_age_histogram.png
    - marital_status_histogram.png
    - total_trans_ct_histogram.png
  - results
    - feature_importances.png
    - lr.png
    - rf.png
    - roc_curve.png
- logs
  - churn_library.log
- models
  - logistic_model.pkl
  - rfc_model.pkl
- churn_library.py
- churn_notebook.ipynb
- churn_script_logging_and_tests.py
- README.md
- requirements.txt




