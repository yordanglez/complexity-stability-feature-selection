```python
!pip install problexity arff

```


```python
import problexity as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV, Ridge, RidgeCV, LogisticRegression, LogisticRegressionCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.utils import resample
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score
import pandas as pd
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning,DataConversionWarning
import pandas as pd


warnings.filterwarnings("ignore")

```

# 1. Defining our datasets for experiments

**Ant Dataset**

The Ant dataset comes from the Apache Ant project, which is a widely used software build tool developed in Java. The Apache Ant project is part of the Apache Software Foundation and has been a reference in many studies on software quality and defects. It is frequently used in defect prediction studies due to its extensive use in the software community and because it contains detailed information on code metrics and defects across historical versions of the project.

**Camel Dataset**

Apache Camel is a software integration framework for Java, developed as part of the Apache Software Foundation. It is used in integration systems to manage and process data between different applications. It is another set of software metrics frequently used in defect prediction studies. It contains information on defects and software quality metrics from the early versions of the project.


**Velocity Dataset**

Apache Velocity is a template engine for Java that allows developers to use templates to generate source code or content dynamically. It is commonly used to separate content from presentation in web applications. It provides software quality metrics and defects found in that version of the template engine.

**JEdit Dataset**

JEdit is a text editor and development environment for programmers, written in Java. It is a popular tool among developers as it offers many advanced features for text editing and code manipulation. This dataset is based on the structure of classes and modules in JEdit and contains code metrics that allow analyzing the quality of the code and predicting defects.





```python
def extract_features_and_target(df, target_column, offset, limit):
    df['defects'] = df['bug'] > 0
    df = df.drop(columns=['name','version','name.1', 'bug'], axis=1)

    df.replace('?', np.nan, inplace=True)
    df.replace('-', np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df[offset:limit]
    # Verify that the target column exists in the DataFrame
    if target_column not in df.columns:
        raise ValueError(f"The column '{target_column}' does not exist in the DataFrame.")

    # Separate X and y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y

ant = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TFM/ant-1.7.csv')
camel = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TFM/camel-1.0.csv')
velocity = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TFM/velocity-1.6.csv')
jedit = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/TFM/jedit-3.2.csv')

datasets = {
    'Ant': extract_features_and_target(ant, 'defects', 0, 1000),
    'Camel': extract_features_and_target(camel, 'defects',0, 1000),
    'Velocity': extract_features_and_target(velocity, 'defects', 0, 1000),
    'Jedit': extract_features_and_target(jedit, 'defects', 0, 1000),
}
```

# What features will we use for experiments?

The features we are using for the experiment come from software metrics that are commonly used in the field of software engineering to assess the quality of the code and predict potential defects. These metrics provide insights into different aspects of the code's structure, such as its complexity, cohesion, and coupling. These factors are important because higher complexity, lower cohesion, and higher coupling are often associated with more defects and maintainability issues in software.

## Overview of the Features:

**wmc**: The weighted methods per class. This is calculated by summing the cyclomatic complexity of all methods in a class. Indicates how complex a class is in terms of the number of methods and their individual complexity. Classes with high WMC can be harder to understand and more prone to errors.

**dit**: Depth of inheritance tree. Measures the number of inheritance levels from a class to the root of the inheritance tree. The deeper the inheritance tree, the harder it may be to understand the behavior of a class, as it is influenced by multiple levels of inheritance. This can increase the likelihood of errors.

**noc**: Number of children. Measures how many classes inherit from a given class. A high number of children indicates that the class is highly reused, which can be good for code reuse but can also increase the likelihood of defects if the base class contains errors or is difficult to maintain.

**cbo**: Coupling between objects. Measures the number of external classes a given class is coupled to (or dependent on). High coupling means that the class depends on many other classes, which can increase complexity and the likelihood of defects.

**rfc**: Response for a class. Measures the number of methods that can be invoked in response to a message received by a class. A high RFC value indicates that the class has many ways to react to different inputs, which can make it more error-prone due to its complexity.

**lcom**: Lack of cohesion of methods. Measures how related the methods in a class are based on the instance variables they share. Low cohesion indicates that the class has too many unrelated responsibilities, making it harder to understand and more prone to defects.

**ca**: Afferent couplings. Measures how many classes depend on a given class. If many classes depend on one class, a defect in the central class can have severe consequences throughout the system.

**ce**: Efferent couplings. Measures how many classes a given class depends on. A high CE value suggests that the class is highly coupled to others, making it more complex and difficult to maintain.

**npm**: Number of public methods. A high number of public methods may indicate a complex API or a class that exposes too much functionality, increasing its complexity.

**lcom3**: Lack of cohesion of methods, version 3. Another measure of the lack of cohesion among a class’s methods. Similar to LCOM, this metric measures how related the responsibilities of a class are.

**loc**: Lines of code. The total number of lines of code in a class. Classes with many lines of code tend to be more complex, harder to maintain, and more prone to errors.

**dam**: Data access metric. Measures the percentage of instance variables that are private or protected (encapsulated). A high DAM value indicates good encapsulation, which is generally desirable for reducing the risk of defects.

**moa**: Measure of aggregation. Measures the number of objects used as attributes within a class. A high MOA value indicates strong aggregation, meaning the class has many dependencies, which increases the risk of defects.

**mfa**: Measure of functional abstraction. Measures the percentage of methods that are inherited. A high MFA value indicates that the class is reusing methods from its parent classes, which can be beneficial for code reuse but may make its behavior harder to predict.

**cam**: Cohesion among methods. Measures cohesion among methods. High cohesion is desirable because it means that the methods in the class focus on a specific task.

**ic**: Inheritance coupling. Measures how many classes are coupled to others through inheritance. A high value indicates strong dependence through inheritance, which can make the system harder to maintain.

**cbm**: Coupling between methods. Measures the number of dependencies between methods within a class. A high value indicates that the methods are tightly coupled, which can make the system harder to modify or extend.

**amc**: Average method complexity. Measures the average cyclomatic complexity of the methods within a class. High complexity in individual methods suggests that they are harder to understand and maintain.

**max_cc**: Maximum cyclomatic complexity. Measures the highest cyclomatic complexity value among all methods in a class. Indicates how complex the most complicated methods in a class are.

**avg_cc**: Average cyclomatic complexity. Measures the average cyclomatic complexity of all the methods in a class. A high average value suggests that the class, as a whole, is harder to understand, increasing the likelihood of errors.

**bug**: A binary indicator that shows whether a class has a bug or defect. This is the target variable in defect prediction models. 1 indicates the class has at least one defect, while 0 means the class is defect-free.



# The following shows the distribution of defects and non-defects across our datasets.


```python
defect_data = []
for dataset_name, (X, y) in datasets.items():
    defect_count = y.value_counts()
    has_defects = defect_count.get(True, defect_count.get(1, 0))
    no_defects = defect_count.get(False, defect_count.get(0, 0))

    defect_data.append({
        'Dataset': dataset_name,
        'Has Defects': has_defects,
        'No Defects': no_defects
    })

df_defects = pd.DataFrame(defect_data)


color_has_defects = '#FF9999'
color_no_defects = '#99CC99'

n_datasets = len(df_defects)
n_cols = 4
n_rows = math.ceil(n_datasets / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))

axes = axes.flatten()

for i, (dataset_name, row) in enumerate(df_defects.iterrows()):
    row_data = pd.DataFrame({
        'Defect Type': ['Has Defects', 'No Defects'],
        'Count': [row['Has Defects'], row['No Defects']]
    })

    bars = row_data.plot(kind='bar', x='Defect Type', y='Count', stacked=True, ax=axes[i], color=[color_has_defects, color_no_defects], legend=False)
    axes[i].set_title(row['Dataset'])
    axes[i].set_ylabel('Number of Instances')
    axes[i].set_xticklabels(['Has Defects', 'No Defects'], rotation=0)


if n_datasets % n_cols != 0:
    for j in range(n_datasets, n_rows * n_cols):
        fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```


    
![png](complexity-stability-selection_files/complexity-stability-selection_7_0.png)
    


#About Complexity

# 2. Calculate the complexity of the entire dataset

We calculate and visualize the complexity of various datasets. The process start by normalizing the features of each dataset and then using a complexity calculator to generate a report that includes various complexity metrics and an overall complexity score for each dataset.


```python
dataset_complexities = {}
for dataset_name, (X, y) in datasets.items():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cc = px.ComplexityCalculator(multiclass_strategy='ova')
    cc.fit(X_scaled, y)
    report = cc.report()
    complexity = report['score']
    dataset_complexities[dataset_name] = report
    print(f'Complexity of dataset {dataset_name}: {complexity:.4f}')


n_datasets = len(dataset_complexities)
fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 6))

for i, (dataset_name, report) in enumerate(dataset_complexities.items()):
    complexity_types = list(report['complexities'].keys())
    complexity_values = list(report['complexities'].values())
    overall_score = report['score']

    df = pd.DataFrame({
        'Complexity Type': complexity_types,
        'Value': complexity_values
    })

    sns.barplot(x='Complexity Type', y='Value', data=df, ax=axes[i])
    axes[i].axhline(y=overall_score, color='red', linestyle='--', label=f'Score: {overall_score:.4f}')
    axes[i].set_xlabel('Complexity Type')
    axes[i].set_ylabel('Value')
    axes[i].set_title(f'Complexity Types for {dataset_name}')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].legend()

plt.tight_layout()
plt.show()

dataset_names = list(dataset_complexities.keys())
complexities = [report['score'] for report in dataset_complexities.values()]

plt.figure(figsize=(7, 5))
plt.plot(dataset_names, complexities, marker='o', linestyle='-', color='green')
plt.xlabel('Dataset')
plt.ylabel('Complexity Score')
plt.title('Line Plot of Dataset Complexities')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()



```

    Complexity of dataset Ant: 0.4060
    Complexity of dataset Camel: 0.3700
    Complexity of dataset Velocity: 0.4020
    Complexity of dataset Jedit: 0.3970



    
![png](complexity-stability-selection_files/complexity-stability-selection_10_1.png)
    



    
![png](complexity-stability-selection_files/complexity-stability-selection_10_2.png)
    


# 3. Define the feature selection models

In this section, we define the feature selection algorithms that will be used in our experiments


```python

models = {
    'LASSO': Lasso(alpha=0.01),
    'LassoCV': LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5),
    # 'Ridge': Ridge(alpha=0.01),
    # 'RidgeCV': RidgeCV(alphas=[0.01, 0.1, 1, 10], cv=5),
    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5),
    'ElasticNetCV': ElasticNetCV(alphas=[0.01, 0.1, 1, 10], l1_ratio=[0.1, 0.5, 0.9], cv=5),
    "SelectKBest": SelectKBest(score_func=f_classif, k=10),
    # "RandomForest": RandomForestClassifier(n_estimators=100),
    'RFE (LogisticRegression)': RFE(estimator=LogisticRegression(),
    n_features_to_select=10),
    # 'RFE (LogisticRegressionCV)': RFE(estimator=LogisticRegressionCV(),
    # n_features_to_select=10)
}
```

# 4. Calculate stability and feature importance

We calculate the stability of feature selection using different algorithms and assess the importance of the selected features. The stability is measured across multiple iterations, ensuring consistency in feature selection, while feature importance highlights the most relevant attributes for each model. This process helps evaluate how reliable the feature selection is for different algorithms.


```python
def calculate_stability_kuncheva(Z):
    M, d = Z.shape
    k_bar = np.mean(np.sum(Z, axis=1))
    pairwise_kuncheva = []

    for i in range(M):
        for j in range(i + 1, M):
            intersection = np.sum(np.logical_and(Z[i], Z[j]))
            kuncheva_index = (intersection - (k_bar**2 / d)) / (k_bar - (k_bar**2 / d))
            pairwise_kuncheva.append(kuncheva_index)

    return np.mean(pairwise_kuncheva)

def calculate_stability_and_importance(X, y, model, M=50):
    Z = np.zeros((M, X.shape[1]))
    feature_importances = np.zeros(X.shape[1])

    for i in range(M):
        X_bootstrap, y_bootstrap = resample(X, y, replace=True)
        model.fit(X_bootstrap, y_bootstrap)
        if hasattr(model, 'coef_'):
            selected_features = np.where(model.coef_ != 0)[0]
            feature_importances[selected_features] += np.abs(model.coef_[selected_features])
        elif hasattr(model, 'feature_importances_'):
            selected_features = np.argsort(model.feature_importances_)[-int(0.5 * X.shape[1]):]
            feature_importances += model.feature_importances_
        elif isinstance(model, RFE):
            selected_features = np.where(model.support_)[0]
            feature_importances[selected_features] += 1
        elif isinstance(model, SelectKBest):
            selected_features = np.where(model.get_support())[0]
            feature_importances[selected_features] += 1

        Z[i, selected_features] = 1

    stability_jaccard = np.mean([jaccard_score(Z[i], Z[j], average='binary')
                                 for i in range(M) for j in range(i + 1, M)])
    stability_kuncheva = calculate_stability_kuncheva(Z)

    feature_importances /= M
    return stability_jaccard, stability_kuncheva, feature_importances

results = {}
table_data = []

for dataset_name, (X, y) in datasets.items():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dataset_results = {}
    for model_name, model in models.items():
        stability_jaccard, stability_kuncheva, feature_importances = calculate_stability_and_importance(X_scaled, y, model)
        dataset_results[model_name] = {
            'stability_jaccard': stability_jaccard,
            'stability_kuncheva': stability_kuncheva,
            'complexity': dataset_complexities[dataset_name]['score'],
            'importances': feature_importances,
            'feature_names': X.columns
        }
        table_data.append([dataset_name, model_name, stability_jaccard, stability_kuncheva, dataset_complexities[dataset_name]['score']])
        # print(f'{dataset_name} - {model_name} -> Stability Jaccard: {stability_jaccard:.4f}, Stability Kuncheva: {stability_kuncheva:.4f}, Complexity: {dataset_complexities[dataset_name]["score"]:.4f}')
    results[dataset_name] = dataset_results

df_table = pd.DataFrame(table_data, columns=['Dataset', 'Model', 'Stability (Jaccard)', 'Stability (Kuncheva)', 'Complexity'])

n_datasets = len(results)
n_models = len(next(iter(results.values())))

fig, axes = plt.subplots(n_datasets, n_models, figsize=(5 * n_models, 5 * n_datasets))

for i, (dataset_name, dataset_results) in enumerate(results.items()):
    for j, (model_name, res) in enumerate(dataset_results.items()):
        feature_importance_df = pd.DataFrame({
            'Feature': res['feature_names'],
            'Importance': res['importances']
        }).sort_values(by='Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=axes[i, j])
        axes[i, j].set_title(f'{dataset_name} ({model_name})')
        axes[i, j].set_xlabel('Importance')
        axes[i, j].set_ylabel('Feature')

plt.tight_layout()
plt.show()

n_datasets = len(results)
n_models = len(next(iter(results.values())))

# Create a grid of subplots with n_datasets rows and n_models columns
fig, axes = plt.subplots(n_datasets, n_models, figsize=(5 * n_models, 5 * n_datasets))

# Loop through each dataset and model combination to create bar plots
for i, (dataset_name, dataset_results) in enumerate(results.items()):
    for j, (model_name, res) in enumerate(dataset_results.items()):
        feature_importance_df = pd.DataFrame({
            'Feature': res['feature_names'],
            'Importance': res['importances']
        }).sort_values(by='Importance', ascending=False)

        # Plot the feature importance as bar plots in the grid
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=axes[i, j])
        axes[i, j].set_title(f'{dataset_name} ({model_name})')
        axes[i, j].set_xlabel('Importance')
        axes[i, j].set_ylabel('Feature')

plt.tight_layout()
plt.show()
```


    
![png](complexity-stability-selection_files/complexity-stability-selection_14_0.png)
    



    
![png](complexity-stability-selection_files/complexity-stability-selection_14_1.png)
    


# Analysis of Complexity vs. Feature Selection Stability

These plots analyze the relationship between dataset complexity and feature selection stability. The stability is measured using two different indices: Jaccard and Kuncheva. Each line represents how the stability of the feature selection algorithms changes as the complexity of the datasets increases, providing insights into the robustness of the selection process across different levels of dataset complexity.


```python
print(df_table.to_string(index=False))

plt.figure(figsize=(12, 8))
for dataset_name, dataset_results in results.items():
    complexities = [res['complexity'] for res in dataset_results.values()]
    stabilities = [res['stability_jaccard'] for res in dataset_results.values()]
    plt.plot(complexities, stabilities, marker='o', label=dataset_name)

plt.xlabel('Dataset Complexity')
plt.ylabel('Feature Selection Stability (Jaccard)')
plt.title('Complexity vs. Stability Curves (Jaccard) - Entire Datasets')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 8))
for dataset_name, dataset_results in results.items():
    complexities = [res['complexity'] for res in dataset_results.values()]
    stabilities = [res['stability_kuncheva'] for res in dataset_results.values()]  # Use Kuncheva stability
    plt.plot(complexities, stabilities, marker='o', label=dataset_name)

plt.xlabel('Dataset Complexity')
plt.ylabel('Feature Selection Stability (Kuncheva)')
plt.title('Complexity vs. Stability Curves (Kuncheva) - Entire Datasets')
plt.legend()
plt.grid(True)
plt.show()



```

     Dataset                    Model  Stability (Jaccard)  Stability (Kuncheva)  Complexity
         Ant                    LASSO             0.472075              0.348398       0.406
         Ant                  LassoCV             0.458161              0.340994       0.406
         Ant               ElasticNet             0.564849              0.231045       0.406
         Ant             ElasticNetCV             0.548161              0.234230       0.406
         Ant              SelectKBest             0.934051              0.927184       0.406
         Ant RFE (LogisticRegression)             0.451350              0.224000       0.406
       Camel                    LASSO             0.292463              0.206010       0.370
       Camel                  LassoCV             0.208631              0.159837       0.370
       Camel               ElasticNet             0.387465              0.122499       0.370
       Camel             ElasticNetCV             0.254013              0.041062       0.370
       Camel              SelectKBest             0.516513              0.333388       0.370
       Camel RFE (LogisticRegression)             0.406742              0.135837       0.370
    Velocity                    LASSO             0.547619              0.286397       0.402
    Velocity                  LassoCV             0.520115              0.273713       0.402
    Velocity               ElasticNet             0.656135              0.231355       0.402
    Velocity             ElasticNetCV             0.598691              0.179630       0.402
    Velocity              SelectKBest             0.659565              0.576490       0.402
    Velocity RFE (LogisticRegression)             0.448467              0.220408       0.402
       Jedit                    LASSO             0.648511              0.413314       0.397
       Jedit                  LassoCV             0.663039              0.447728       0.397
       Jedit               ElasticNet             0.725663              0.240573       0.397
       Jedit             ElasticNetCV             0.705462              0.231003       0.397
       Jedit              SelectKBest             0.743533              0.695184       0.397
       Jedit RFE (LogisticRegression)             0.491405              0.298449       0.397



    
![png](complexity-stability-selection_files/complexity-stability-selection_16_1.png)
    



    
![png](complexity-stability-selection_files/complexity-stability-selection_16_2.png)
    



```python
!jupyter nbconvert --to pdf /content/drive/MyDrive/Colab\ Notebooks/TFM/complexity-stability-selection.ipynb
```

    [NbConvertApp] Converting notebook /content/drive/MyDrive/Colab Notebooks/TFM/complexity-stability-selection.ipynb to pdf
    [NbConvertApp] Support files will be in complexity-stability-selection_files/
    [NbConvertApp] Making directory ./complexity-stability-selection_files
    [NbConvertApp] Making directory ./complexity-stability-selection_files
    [NbConvertApp] Making directory ./complexity-stability-selection_files
    [NbConvertApp] Making directory ./complexity-stability-selection_files
    [NbConvertApp] Making directory ./complexity-stability-selection_files
    [NbConvertApp] Making directory ./complexity-stability-selection_files
    [NbConvertApp] Making directory ./complexity-stability-selection_files
    [NbConvertApp] Writing 73431 bytes to notebook.tex
    [NbConvertApp] Building PDF
    [NbConvertApp] Running xelatex 3 times: ['xelatex', 'notebook.tex', '-quiet']
    [NbConvertApp] Running bibtex 1 time: ['bibtex', 'notebook']
    [NbConvertApp] WARNING | bibtex had problems, most likely because there were no citations
    [NbConvertApp] PDF successfully created
    [NbConvertApp] Writing 793713 bytes to /content/drive/MyDrive/Colab Notebooks/TFM/complexity-stability-selection.pdf

