# Detecting the Higgs :b:oson (ML project 1)
Here's a quick explanation on how to run the provided code.

The results obtained can vary depending on the machine used to run the code.

## Prerequisites
- The files ```run.py``` and ```implementations.py```should be located in the "scripts" folder.
- The "data" folder should be located on the same level as the "scripts" folder.
- All the code was written with Python ```3.7.6```.
- We use matplotlib and seaborn for some visualizations in notebook. Check
```requirements.txt``` file for exact version of each library.
## Execution
To replicate our submission on AICrowd, simply run the following command : ``` python run.py```.
(N.B : To actually retrain the model and recompute the optimal hyper-parameters, run ``` python run.py --recompute_params``` ).

## Submission
The submission ``` submission-ridge.csv``` can be found in the "data" folder.
Then, it simply is a matter of uploading the file to AICrowd, but you already knew that I suppose. :)

## All scripts files description
- ```exploratory_data_analysis.ipynb```: This notebook contains various plots
we did for exploring the data.
- ```implementations.py```: Machine learning algorithms, loss functions,
cross-validation, data preprocessing, finding best params
- ```proj1_helpers.py```: Helper methods for reading the data and making predictions.
- ```project1.ipynb```: Notebook used in exploring various ML models.
- ```run.py```: Final pipeline which outputs the best predictions.
