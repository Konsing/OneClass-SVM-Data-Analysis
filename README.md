# Comprehensive Data Analysis and Anomaly Detection

## Project Overview
This Jupyter notebook demonstrates various data analysis techniques, including anomaly detection using the OneClassSVM algorithm. The notebook contains multiple sections, each addressing different datasets and analytical methods.

## Files in the Repository
- **Comprehensive_Data_Analysis_and_Anomaly_Detection.ipynb**: This Jupyter notebook contains code for multiple data analysis tasks, including preprocessing, training models, and visualizing results.

## How to Use
1. **Prerequisites**:
   - Python 3.x
   - Jupyter Notebook or JupyterLab
   - Required Python packages: `pandas`, `numpy`, `matplotlib`, `scikit-learn`

2. **Installation**:
   Ensure you have the required packages installed. You can install them using pip:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```

3. **Running the Notebook**:
   - Open the Jupyter Notebook:
     ```bash
     jupyter notebook Comprehensive_Data_Analysis_and_Anomaly_Detection.ipynb
     ```
   - Execute the cells in the notebook sequentially to perform the various data analysis tasks.

## Sections in the Notebook

### 1. Data Preprocessing
This section includes data cleaning, transformation, and selection of relevant columns for analysis. Specific tasks may include handling missing values, encoding categorical variables, and normalizing numerical features.

### 2. Exploratory Data Analysis (EDA)
Visual and statistical analysis of datasets to understand the underlying patterns and distributions. This section may include:
- Summary statistics
- Distribution plots (histograms, box plots)
- Correlation analysis
- Visualization of relationships between variables

### 3. Anomaly Detection in BMI and Insulin Levels
#### Description:
Uses the OneClassSVM algorithm to detect anomalies in BMI and Insulin levels.
#### Key Steps:
   - Data selection and conversion to NumPy array
   - OneClassSVM initialization and training
   - Prediction of anomalies
   - Visualization of original data and detected anomalies
#### Code Example:
```python
Data = df[["BMI", "Insulin"]]
input = Data.to_numpy()
svm = OneClassSVM(kernel='rbf', gamma=0.001, nu=0.2)
svm.fit(input)
pred = svm.predict(input)
anom_index = where(pred == -1)
values = input[anom_index]
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(20, 6))
ax0.set_title('Original')
ax0.scatter(Data["BMI"], Data["Insulin"])
ax0.set_xlabel("BMI")
ax0.set_ylabel("Insulin")
ax1.set_title('Anomalies')
ax1.scatter(input[:,0], input[:,1])
ax1.scatter(values[:,0], values[:,1])
ax1.set_xlabel("BMI")
ax1.set_ylabel("Insulin")
```

### 4. Additional Analysis (Provide specifics based on the content of your notebook)
- **Clustering Analysis**: Implement clustering algorithms such as K-Means or DBSCAN and visualize the clusters.
- **Regression Models**: Apply linear or polynomial regression models to predict continuous outcomes.
- **Classification Tasks**: Use classification algorithms like Logistic Regression, Decision Trees, or Random Forests to predict categorical outcomes.
- **Time Series Analysis**: Analyze and forecast time series data using methods such as ARIMA or Prophet.

## Visualization
The notebook includes various visualizations to support the analysis, such as scatter plots, histograms, and line charts. Each section's visualizations help in understanding the data and the results of the applied techniques.

## Output
- **Original Data Plot**: Displays the scatter plot of BMI versus Insulin levels without any anomaly distinction.
- **Anomaly Plot**: Shows the same scatter plot but highlights the detected anomalies using the OneClassSVM model.

## Conclusion
This notebook provides a comprehensive approach to data analysis using various techniques. By following the steps in the notebook, users can replicate the analyses on similar datasets or extend them to other data.

If you have any questions or encounter any issues, please feel free to reach out for further assistance.