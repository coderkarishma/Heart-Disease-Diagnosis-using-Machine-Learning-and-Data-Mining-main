#!/usr/bin/env python
# coding: utf-8

# # PROJECT DISSERTATION CODE

# ### Importing the Libraries

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve


# ### Importing the dataset

# In[ ]:


dataset = pd.read_csv('heart_statlog_cleveland_hungary_final.csv')
print('The shape of the data is ', dataset.shape)


# ### A preview of the dataset

# In[ ]:


dataset.head(20).style.set_properties(**{'background-color': '#d8e0d5','color': 'black','border-color': 'darkblack'})


# ### Dividing the features into Numerical and Categorical

# In[ ]:


col = list(dataset.columns)
categorical_features = []
numerical_features = []
for i in col:
    if len(dataset[i].unique()) > 6:
        numerical_features.append(i)
    else:
        categorical_features.append(i)

print('Categorical Features :',*categorical_features)
print('Numerical Features :',*numerical_features)


# ### Mapping the categorical features to their corresponding labels

# In[ ]:


df1 = dataset.copy()
df1['sex'] = df1['sex'].replace({1: 'male', 0: 'female'})
df1['chest pain type'] = df1['chest pain type'].replace({1: 'typical ang', 2: 'atypical ang', 3: 'non-ang', 4: 'asymptomatic'})
df1['fasting blood sugar'] = df1['fasting blood sugar'].replace({1: 'true', 0: 'false'})
df1['resting ecg'] = df1['resting ecg'].replace({0: 'normal', 1: 'ST-T abnormality', 2: 'LVH'})
df1['exercise angina'] = df1['exercise angina'].replace({1: 'yes', 0: 'no'})
df1['ST slope'] = df1['ST slope'].replace({1: 'upsloping', 2: 'flat', 3: 'downsloping'})
df1['target'] = df1['target'].replace({1: 'heart disease', 0: 'no heart disease'})


# ### Visualization of the target feature in the dataset

# In[ ]:


d = list(df1['target'].value_counts())
circle = [d[1] / sum(d) * 100, d[0] / sum(d) * 100]
colors = ['#2ecc71', '#f7342a']
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))

plt.subplot(1, 2, 1)
plt.pie(circle, labels=['Normal', 'Heart Disease'], autopct='%1.1f%%', startangle=90, explode=(0.1, 0), colors=colors,
        wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'antialiased': True}, textprops={'fontsize': 14})
plt.title('Heart Disease (%)', fontsize=18)

# Count plot for heart disease cases
plt.subplot(1, 2, 2)
ax = sns.countplot(x='target', data=df1, palette=colors, edgecolor='black')
for rect in ax.patches:
    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), horizontalalignment='center', fontsize=14)
ax.set_xticklabels(['Normal', 'Heart Disease'], fontsize=14)
ax.set_xlabel('Target', fontsize=16)
ax.set_ylabel('Count', fontsize=16)
plt.title('Cases of Heart Disease', fontsize=18)

plt.show()


# ### Visualization of the categorical features in the dataset

# In[ ]:


fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
for i in range(len(categorical_features) - 1):
    plt.subplot(3, 2, i + 1)
    ax = sns.countplot(x=categorical_features[i], data=df1, hue="target", palette=colors, edgecolor='black')
    for rect in ax.patches:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), horizontalalignment='center', fontsize=11)
    title = categorical_features[i] + ' vs Heart Disease'
    plt.legend(['Normal', 'Heart Disease'])
    plt.title(title)

plt.tight_layout()
plt.show()


# ### Data Preprocessing

# In[ ]:


# Data Preprocessing function
def data_preprocessing(dataset):
    # Handle missing values
    missing_values = dataset.isnull().sum()
    
    # Replace the outlier with the mode of the column
    mode_st_slope = dataset['ST slope'].mode()[0]
    dataset['ST slope'] = dataset['ST slope'].replace(0, mode_st_slope)
    print("Outlier replaced with mode:", mode_st_slope)
    
    # Split the data into features and target
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    
    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test, missing_values

# Perform data preprocessing
X_train, X_test, y_train, y_test, missing_values = data_preprocessing(dataset)

# Print dataset information and missing values
print("The dataset information:")
print(dataset.info())
print("\nMissing values in each column:")
print(missing_values)


# ### Feature Selection, Training and Evaluation of the Models

# In[ ]:


# Initialize models to evaluate
models = {
    'Decision Tree': DecisionTreeClassifier(criterion='entropy', random_state=0),
    'Random Forest': RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(kernel='linear', random_state=0, probability=True)
}

# Dictionary to store feature importances for each model
feature_importance_info = {model_name: None for model_name in models.keys()}

# Evaluate models and get feature importances
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):  # For models like SVM with coefficients
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        importances = None
    
    feature_importance_info[model_name] = importances

# Aggregate feature importances across models
feature_names = dataset.columns[:-1]
aggregate_importances = np.zeros(len(feature_names))
for model_name, importances in feature_importance_info.items():
    if importances is not None:
        aggregate_importances += importances

# Normalize aggregated importances
aggregate_importances /= len(models)

# Sort indices based on aggregated importances
indices = np.argsort(aggregate_importances)[::-1]

# Create DataFrame to display aggregated feature importances
aggregate_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': aggregate_importances})
aggregate_importance_df = aggregate_importance_df.sort_values(by='Importance', ascending=False)

# Display aggregated feature importances
print("\nAggregate Feature Importances Across Models:")
print(aggregate_importance_df)


# In[ ]:


# Function to evaluate model performance with selected features
def evaluate_model_with_selected_features(model, X_train, X_test, y_train, y_test, indices, num_features):
    top_features = indices[:num_features]
    X_train_selected = X_train[:, top_features]
    X_test_selected = X_test[:, top_features]
    
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Evaluate models with increasing number of features
results = {model_name: [] for model_name in models.keys()}
for num_features in range(1, len(indices) + 1):
    for model_name, model in models.items():
        accuracy = evaluate_model_with_selected_features(model, X_train, X_test, y_train, y_test, indices, num_features)
        results[model_name].append(accuracy)

# Plotting the feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=aggregate_importance_df, palette='viridis')
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Plotting model performance with increasing number of features
plt.figure(figsize=(12, 8))
for model_name, accuracies in results.items():
    plt.plot(range(1, len(indices) + 1), accuracies, label=model_name, marker='o')
plt.title('Model Performance with Increasing Number of Features')
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[ ]:


# Dictionary to store highest accuracy and corresponding number of features for each model
best_accuracy_info = {model_name: {'accuracy': 0, 'num_features': 0} for model_name in models.keys()}

# Evaluate models with increasing number of features
for num_features in range(1, len(indices) + 1):
    for model_name, model in models.items():
        accuracy = evaluate_model_with_selected_features(model, X_train, X_test, y_train, y_test, indices, num_features)
        if accuracy > best_accuracy_info[model_name]['accuracy']:
            best_accuracy_info[model_name]['accuracy'] = accuracy
            best_accuracy_info[model_name]['num_features'] = num_features

# Print the results
for model_name, info in best_accuracy_info.items():
    print(f'Model: {model_name}, Highest Accuracy: {info["accuracy"]}, Number of Features: {info["num_features"]}')


# In[ ]:


# Select top 11 features 
top_n = 11
top_features = indices[:top_n]
X_train_selected = X_train[:, top_features]
X_test_selected = X_test[:, top_features]

# Store model results
model_names = []
accuracies = []
conf_matrices = []
class_reports = []
precision_recall_curves = []
roc_auc_scores = []

# Train and evaluate each model with selected features
for name, model in models.items():
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    
    print(f'{name} Model with Top {top_n} Features:')
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    print('Accuracy:', accuracy)
    
    # Classification Report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_reports.append(class_report)
    print('Classification Report:\n', classification_report(y_test, y_pred))
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrices.append(conf_matrix)
    print('Confusion Matrix:\n', conf_matrix)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    precision_recall_curves.append((precision, recall))
    
    # ROC AUC Score
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test_selected)[:, 1]
    else:
        y_scores = model.decision_function(X_test_selected)
    
    roc_auc = roc_auc_score(y_test, y_scores)
    roc_auc_scores.append(roc_auc)
    print('ROC_AUC Score:\n', roc_auc_score(y_test, y_scores))
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
    print(f'{name} Cross-Validation Scores:', cv_scores)
    print(f'{name} Mean CV Accuracy: {np.mean(cv_scores)}')
    print('\n' + '-'*30 + '\n')
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
    
    model_names.append(name)  # Append the model name to the list

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# ### Number of Data point (support) for each class

# In[ ]:


for name, model in models.items():
    if hasattr(model, 'support_'):
        num_support_vectors_per_class = model.n_support_
        print(f'Number of support vectors for each class: {num_support_vectors_per_class}')
    else:
        print(f'{name} does not use support vectors.')


# In[ ]:


# Apply PCA to reduce to 2 dimensions
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot decision boundary function
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='viridis')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    legend1 = plt.legend(*scatter.legend_elements(), title="Targets")
    plt.gca().add_artist(legend1)
    plt.show()

# Train and plot decision boundaries for each model
for name, model in models.items():
    model.fit(X_train_pca, y_train)
    plot_decision_boundary(model, X_train_pca, y_train, f'{name} Decision Boundary (Train)')
    plot_decision_boundary(model, X_test_pca, y_test, f'{name} Decision Boundary (Test)')


# ### Precision - Recall Curve

# In[ ]:


# Plot the Precision-Recall curves
plt.figure(figsize=(10, 6))
for i, (precision, recall) in enumerate(precision_recall_curves):
    plt.plot(recall, precision, label=model_names[i])

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()


# ### Accuracy 

# In[ ]:


# Set up Seaborn color palette for better color harmony
colors = sns.color_palette("colorblind", len(model_names))

# Increase figure size
plt.figure(figsize=(12, 7))

# Plot the accuracy comparison
bars = plt.bar(model_names, accuracies, width=0.6, color=colors)

# Add values on top of the bars for better readability
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', 
             ha='center', va='bottom', fontsize=12, color='black')

# Add labels, title, and customize appearance
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
plt.title('Accuracy Comparison of the Models', fontsize=18, fontweight='bold', color='#4f4f4f')

# Set y-axis limit and add gridlines
plt.ylim([0, 1.1])
plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)

# Remove top and right spines for a cleaner look
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Adjust layout to prevent clipping
plt.tight_layout()

# Show the plot
plt.show()


# ### Confusion Matrix

# In[ ]:


# Plot the confusion matrix for each model
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()
colors = ['#7971e3', '#eb8e46']
font_size = 14

for i, (name, conf_matrix) in enumerate(zip(model_names, conf_matrices)):
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=axes[i], cmap= colors, cbar=False, annot_kws={"size": font_size})
    axes[i].set_title(f'{name} Confusion Matrix', fontsize=font_size)
    axes[i].set_xlabel('Predicted', fontsize=font_size)
    axes[i].set_ylabel('Actual', fontsize=font_size)
    axes[i].tick_params(axis='both', which='major', labelsize=font_size)

plt.tight_layout()
plt.show()


# ### Classification Report

# In[ ]:


# Extract precision for each class from the class reports
precision_class_0 = [
    report['0']['precision'] for report in class_reports
]
precision_class_1 = [
    report['1']['precision'] for report in class_reports
]

# Plotting function for precision by class
def plot_precision_by_class(model_names, precision_class_0, precision_class_1):
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(index, precision_class_0, bar_width, label='Class 0')
    bars2 = plt.bar(index + bar_width, precision_class_1, bar_width, label='Class 1')
    
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title('Precision by Class for Each Model', fontsize=18, fontweight='bold', color='#4f4f4f')
    plt.xticks(index + bar_width / 2, model_names)
    plt.ylim([0, 1.1])
    plt.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', 
                     ha='center', va='bottom', fontsize=12, color='black')
    
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# Model names for labeling
model_names = ["Decision Tree", "Random Forest", "Naive Bayes", "Support Vector Machine"]

# Plot precision by class
plot_precision_by_class(model_names, precision_class_0, precision_class_1)


# In[ ]:


# Extract recall for each class from the class reports
recall_class_0 = [
    report['0']['recall'] for report in class_reports
]
recall_class_1 = [
    report['1']['recall'] for report in class_reports
]

# Plotting function for recall by class
def plot_recall_by_class(model_names, recall_class_0, recall_class_1):
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(index, recall_class_0, bar_width, label='Class 0')
    bars2 = plt.bar(index + bar_width, recall_class_1, bar_width, label='Class 1')
    
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('Recall', fontsize=14, fontweight='bold')
    plt.title('Recall by Class for Each Model', fontsize=18, fontweight='bold', color='#4f4f4f')
    plt.xticks(index + bar_width / 2, model_names)
    plt.ylim([0, 1.1])
    plt.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', 
                     ha='center', va='bottom', fontsize=12, color='black')
    
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# Model names for labeling
model_names = ["Decision Tree", "Random Forest", "Naive Bayes", "Support Vector Machine"]

# Plot recall by class
plot_recall_by_class(model_names, recall_class_0, recall_class_1)


# In[ ]:


# Extract F1 scores for each class from the class reports
f1_class_0 = [
    report['0']['f1-score'] for report in class_reports
]
f1_class_1 = [
    report['1']['f1-score'] for report in class_reports
]

# Plotting function for F1 score by class
def plot_f1_by_class(model_names, f1_class_0, f1_class_1):
    bar_width = 0.35
    index = np.arange(len(model_names))
    
    plt.figure(figsize=(12, 7))
    bars1 = plt.bar(index, f1_class_0, bar_width, label='Class 0')
    bars2 = plt.bar(index + bar_width, f1_class_1, bar_width, label='Class 1')
    
    plt.xlabel('Model', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.title('F1 Score by Class for Each Model', fontsize=18, fontweight='bold', color='#4f4f4f')
    plt.xticks(index + bar_width / 2, model_names)
    plt.ylim([0, 1.1])
    plt.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', 
                     ha='center', va='bottom', fontsize=12, color='black')
    
    plt.gca().yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.show()

# Model names for labeling
model_names = ["Decision Tree", "Random Forest", "Naive Bayes", "Support Vector Machine"]

# Plot F1 score by class
plot_f1_by_class(model_names, f1_class_0, f1_class_1)


# ### Graphical User Interface

# In[ ]:


rf_model = RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
rf_model.fit(X_train, y_train)

# Save the trained Random Forest model
with open('random_forest_model.pkl', 'wb') as model_file:
    joblib.dump(rf_model, model_file)
    
# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as model_file:
    random_model = joblib.load(model_file)

def predict_heart_disease():
    # Get input values from the user
    try:
        age = int(entry_age.get())
        sex = int(entry_sex.get())
        cp = int(entry_cp.get())
        trestbps = int(entry_trestbps.get())
        chol = int(entry_chol.get())
        fbs = int(entry_fbs.get())
        restecg = int(entry_restecg.get())
        thalach = int(entry_thalach.get())
        exang = int(entry_exang.get())
        oldpeak = float(entry_oldpeak.get())
        slope = int(entry_slope.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")
        return

    # Create the feature array
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope]])
    
    # Predict using the loaded model
    prediction = random_model.predict(features)
    
    # Display the result
    if prediction[0] == 1:
        result_text.set("The patient is likely to have heart disease.")
    else:
        result_text.set("The patient is normal.")
    messagebox.showinfo("Success", result_text.get())

# Create the main window
window = tk.Tk()
window.title("Heart Disease Prediction")
window.geometry("500x600")
window.configure(bg='#d3d3d3')

# Title
title_label = tk.Label(window, text="Heart Disease Prediction System", font=("Helvetica", 20), bg='#d3d3d3', pady=20)
title_label.pack()

# Frame for the input fields
frame = tk.Frame(window, bg='#d3d3d3')
frame.pack(pady=10)

fields = [
    ("Age", "entry_age"),
    ("Sex (1=Male, 0=Female)", "entry_sex"),
    ("Chest Pain Type (1-4)", "entry_cp"),
    ("Resting Blood Pressure", "entry_trestbps"),
    ("Cholesterol", "entry_chol"),
    ("Fasting Blood Sugar (1 if >120mg/dl, 0 otherwise)", "entry_fbs"),
    ("Resting ECG (0-2)", "entry_restecg"),
    ("Max Heart Rate", "entry_thalach"),
    ("Exercise Induced Angina (1=Yes, 0=No)", "entry_exang"),
    ("Oldpeak", "entry_oldpeak"),
    ("ST Slope (1-3)", "entry_slope")
]

entries = {}

for i, (label_text, var_name) in enumerate(fields):
    tk.Label(frame, text=label_text, bg='#d3d3d3').grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entries[var_name] = tk.Entry(frame)
    entries[var_name].grid(row=i, column=1, padx=10, pady=5, sticky="w")
globals().update(entries)

# Button to predict
predict_button = tk.Button(frame, text="Predict", command=predict_heart_disease, bg='#ff69b4', fg='white', font=("Helvetica", 12))
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=20)

# Label to display the result
result_text = tk.StringVar()
result_label = tk.Label(window, textvariable=result_text, fg="blue", font=("Helvetica", 12), bg='#d3d3d3')
result_label.pack(pady=10)

# Start the GUI loop
window.mainloop()


# ### References

# Harris, C.R., Millman, K.J., van der Walt, S.J., et al. (2020). Array programming with NumPy. Nature, 585(7825), 357-362.
# 
# Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95.
# 
# Manu Siddhartha (2020). Heart Disease Dataset (Comprehensive). [online] IEEE DataPort. Available at: https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive [Accessed 1 Jan. 2024].
# 
# Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.
# 
# Python Software Foundation.(n.d.). Tkinter - Python interface to Tcl/Tk [Software]. Available at: https://docs.python.org/3/library/tkinter.html [Accessed 1 June. 2024}
# 
# Python Software Foundation. (n.d.). joblib - Lightweight pipelining in Python [Software]. Available at: https://joblib.readthedocs.io/ [Accessed 5 June. 2024].
# 
# Waskom, M. L. (2021). seaborn: statistical data visualization. Journal of Open Source Software, 6(60), 3021.
