4/20 To-do:
- Joshua: Website
- Wyatt: Models
- Haman: Report

- Goal is to try and finish as much as we can before Thursday so that we have things to discuss with prof. Pijyan

Website specifications:
- Data selection:
  - Allow the user to upload their own dataset
  - Allow the user to select our dataset (the csv that we compiled using the APIs)
  - Allow the user to visualize the dataset (classification task vs. regression task)
- Data pre-processing:
  - Allow the user to select which data pre-processing steps they want to include from:
    - string2factor
    - log
    - zv & nzv
    - impute_knn
    - center / scale / pca
    - unknown / other / dummy
  - Allow the user to determine training-split ratio
- Model Tuning & Selection:
  - Allow the user to select hyperparameters (make sure that the hyperparameter values make sense)
  - Allow the user to select tuning metric / loss function
  - Allow the user to visualize hyperparameter results
- Model results:
  - Allow the user to run the model on the dataset
  - Visualize some of the results

4/2 To-do:
- Wyatt: API integration for data-fetching
- Joshua: Data exploration & cleanup (PCA & MLE)
- Haman: Data exploration & cleanup

Step 1: Data Acquisition & Pre-processing
- Data selection: We chose the NY public tax records, with a task of using regression to estimate property values based on 139 features and 11.5 million rows. 
  - Since 11.5 million rows seems to be too many data points, we can narrow our dataset down to focus on specific neighborhoods which will hopefully reduce the number of rows we can work with, without reducing the information we can gain from the remaining features.
- Our data is from data.cityofnewyork.us
- Data exploration / cleanup:
  - Finding NA fields to estimate how many columns we should run MLE / other data imputation algorithms on
  - Inspecting data types to figure out where we can apply dummy variables / remove categories with a small number of rows
  - Remove columns with low or near-zero variance
  - Inspect the distribution of prices to figure out if we can make it into a normal distribution through log / power transformations
  - Apply standardization / normalization to numeric features so that models can efficiently learn parameters
  - Perform PCA to find the most significant features

Dataset Ideas:
- NY open data: Public tax records. We can make a regression model to predict property prices. https://data.cityofnewyork.us/City-Government/Property-Valuation-and-Assessment-Data-Tax-Classes/8y4t-faws/about_data
- Baseball data: Classification problem for whether a hit is a homerun / not
- Heart attack prediction: Classification problem for low / high risk of a heart attack
- Company bankruptcy classification: Predict whether a company is bankrupt or not based on financial records
