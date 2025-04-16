# 1. Prepare Problem
# a) Load libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

# b) Load dataset
def load_data(filename):
    data = pd.read_csv(filename, keep_default_na=False, na_values=[''])
    return data

filename = '/Users/aizirekibraimova/Desktop/Sleep_health_and_lifestyle_dataset.csv'
data = load_data(filename)

# 2. Summarize Data
# a) Descriptive statistics
def summarize_data(data):
    print("Dataset shape:", data.shape)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nData types and missing values:")
    print(data.info())
    print("\nDescriptive statistics:")
    print(data.describe().round(2))
    print("\nSleep Disorder distribution:")
    print(data['Sleep Disorder'].value_counts(dropna=False))

summarize_data(data)

# b) Data visualizations
def visualize_data(data):
    # Sleep Disorder distribution
    plt.figure(figsize=(8,5))
    data['Sleep Disorder'].value_counts().plot(kind='bar')
    plt.title('Sleep Disorder Distribution')
    plt.ylabel('Count')
    plt.savefig('sleep_disorder_distribution.png')
    plt.show()
    
    # Correlation heatmap
    numeric_data = data.select_dtypes(include=['float64', 'int64'])
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    plt.show()

visualize_data(data)

# 3. Prepare Data
# a) Data Cleaning
def clean_data(data):
    # Handle duplicates
    data = data.drop_duplicates()
    
    # Split Blood Pressure
    data[['Systolic', 'Diastolic']] = data['Blood Pressure'].str.split('/', expand=True).astype(float)
    data['Pulse Pressure'] = data['Systolic'] - data['Diastolic']
    data.drop(['Person ID', 'Blood Pressure'], axis=1, inplace=True)
    
    return data

data = clean_data(data)

# b) Feature Selection
def select_features(data):
    # All features selected initially
    return data

data = select_features(data)

# c) Data Transforms
def transform_data(data):
    # Encode categorical features
    categorical_cols = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    # Separate features and target
    X = data.drop('Sleep Disorder', axis=1)
    y = data['Sleep Disorder']
    
    return X, y, label_encoders

X, y, label_encoders = transform_data(data)

# 4. Evaluate Algorithms
# a) Split-out validation dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features after split to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Handle class imbalance
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

# b) Test options and evaluation metric
scoring = 'accuracy'

# c) Spot Check Algorithms
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "SVM": SVC(class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier()
}

# d) Compare Algorithms
def evaluate_models(models, X_train, y_train, X_test, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"\n{name} Performance:")
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=label_encoders['Sleep Disorder'].classes_)
        disp.plot()
        plt.title(f'{name} Confusion Matrix')
        plt.savefig(f'{name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.show()
    
    return results

results = evaluate_models(models, X_train_res, y_train_res, X_test_scaled, y_test)

# 5. Improve Accuracy
# a) Algorithm Tuning
def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    print("\nBest parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)
    
    return grid_search.best_estimator_

best_model = tune_model(X_train_res, y_train_res)

# b) Ensembles
# Already included Random Forest and Gradient Boosting in initial evaluation

# 6. Finalize Model
# a) Predictions on validation dataset
y_pred = best_model.predict(X_test_scaled)
print("\nFinal Model Performance:")
print(classification_report(y_test, y_pred))

# b) Create standalone model on entire training dataset
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
final_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42
)
final_model.fit(X_scaled, y)

# c) Save model for later use
joblib.dump(final_model, 'final_sleep_model.pkl')
joblib.dump(final_scaler, 'final_scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12,8))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

print("\nFeature Importance:")
print(feature_importance)

