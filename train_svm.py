import numpy as np
from joblib import dump, load
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from utils.recongnition import extract_features
from utils.datasets import load_food101

# Load Food-101 dataset (take only 10 images for each category)
X, y, categories = load_food101("data/food-101/images", image_size=(32, 32), num_samples_per_category=10)

# Train, test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract features for all images
X_train_feats = np.array([
    extract_features(img) 
    for img in tqdm(X_train, desc='Extracting features from training images', dynamic_ncols=True)
])

X_test_feats = np.array([
    extract_features(img) 
    for img in tqdm(X_test, desc='Extracting features from test images', dynamic_ncols=True)
])

# Define model
pipeline = make_pipeline(
    StandardScaler(),
    SVC(kernel='rbf')
)

# Define parameter grid
param_grid = {
    'svc__C': np.logspace(-3, 3, 7),  # ~ [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    'svc__gamma': np.logspace(-3, 3, 7),  # ~ [0.001, 0.01, 0.1, 1, 10, 100, 1000]
}

# Setup grid search
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=10,                  # Number of folds in cross-validation
    scoring='accuracy',     # Evaluation metric
    n_jobs=-1,              # Use all available processors
    verbose=3               # Display all informations
)

# Train classifier
grid_search.fit(X_train_feats, y_train)

# Get best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Predictions using the model with best parameters
y_pred = grid_search.predict(X_test_feats)

# Evaluate model
print(classification_report(y_test, y_pred))

# Save the best SVM model
best_svm_model = grid_search.best_estimator_
dump(best_svm_model, 'classification_models/svm.joblib')

# # Load the model
# loaded_svm_model = load('best_svm_model.joblib')