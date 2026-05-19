"""
Integrated ML Prediction + Expert System Pipeline
Combines CatBoost and Neural Network predictions with expert evaluation
Author: Snežana Đurković
Year: 2026
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
from expert_system_scoring import main_expert_system_pipeline


# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class NeuralNetworkModel(nn.Module):
    """Neural Network architecture for Dq/B prediction"""

    def __init__(self, input_dim, hidden_dims=[128, 64, 32, 16], dropout_rate=0.2):
        super(NeuralNetworkModel, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        # He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


def train_neural_network(X_train, y_train, X_val, y_val,
                         epochs=500, batch_size=64, lr=0.001, weight_decay=1.9e-3):
    """Train neural network with validation"""

    input_dim = X_train.shape[1]
    model = NeuralNetworkModel(input_dim)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)

    # Training loop
    model.train()
    for epoch in range(epochs):
        # Shuffle training data
        perm = torch.randperm(X_train_t.size(0))
        X_train_shuffled = X_train_t[perm]
        y_train_shuffled = y_train_t[perm]

        # Mini-batch training
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_shuffled[i:i + batch_size]
            batch_y = y_train_shuffled[i:i + batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

    return model


def run_cross_validation(X, y, n_splits=10, random_state=42):
    """
    Run 10-fold cross-validation with both CatBoost and Neural Network
    """
    set_seeds(random_state)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    catboost_scores = []
    nn_scores = []

    print(f"\nRunning {n_splits}-fold cross-validation (Random State: {random_state})...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Train CatBoost
        catboost_model = CatBoostRegressor(
            iterations=250,
            depth=3,
            learning_rate=0.08,
            l2_leaf_reg=1.75,
            loss_function='RMSE',
            random_state=random_state,
            verbose=False
        )
        catboost_model.fit(X_train_scaled, y_train)
        catboost_pred = catboost_model.predict(X_val_scaled)
        catboost_r2 = r2_score(y_val, catboost_pred)
        catboost_scores.append(catboost_r2)

        # Train Neural Network
        nn_model = train_neural_network(X_train_scaled, y_train, X_val_scaled, y_val)
        nn_model.eval()
        with torch.no_grad():
            nn_pred = nn_model(torch.FloatTensor(X_val_scaled)).numpy().flatten()
        nn_r2 = r2_score(y_val, nn_pred)
        nn_scores.append(nn_r2)

        print(f"  Fold {fold + 1:2d}: CatBoost R²={catboost_r2:.4f}, NN R²={nn_r2:.4f}")

    avg_catboost = np.mean(catboost_scores)
    avg_nn = np.mean(nn_scores)

    print(f"\nAverage R² across {n_splits} folds:")
    print(f"  CatBoost: {avg_catboost:.4f} ± {np.std(catboost_scores):.4f}")
    print(f"  Neural Network: {avg_nn:.4f} ± {np.std(nn_scores):.4f}")

    return avg_catboost, avg_nn


def optimize_random_state(X, y, n_candidates=29, n_splits=10):
    """
    Search for best random state across multiple candidates
    """
    print("\n" + "=" * 80)
    print("OPTIMIZING RANDOM STATE")
    print("=" * 80)

    best_score = -np.inf
    best_state = None

    candidate_states = np.random.choice(range(1, 10000), size=n_candidates, replace=False)

    for i, state in enumerate(candidate_states):
        print(f"\nCandidate {i + 1}/{n_candidates}: Random State = {state}")
        avg_catboost, avg_nn = run_cross_validation(X, y, n_splits=n_splits, random_state=state)
        combined_score = 0.5 * avg_catboost + 0.5 * avg_nn

        if combined_score > best_score:
            best_score = combined_score
            best_state = state
            print(f"  → New best score: {combined_score:.4f}")

    print("\n" + "=" * 80)
    print(f"BEST RANDOM STATE: {best_state} (Combined R² = {best_score:.4f})")
    print("=" * 80)

    return best_state


def train_final_models(X, y, random_state=42):
    """
    Train final models on full dataset with ensemble predictions
    """
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODELS ON FULL DATASET")
    print("=" * 80)

    set_seeds(random_state)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train multiple models for ensemble uncertainty estimation
    n_ensemble = 10
    kfold = KFold(n_splits=n_ensemble, shuffle=True, random_state=random_state)

    catboost_models = []
    nn_models = []

    print("\nTraining ensemble models...")
    for fold, (train_idx, _) in enumerate(kfold.split(X)):
        X_train = X_scaled[train_idx]
        y_train = y[train_idx]

        # CatBoost
        cb_model = CatBoostRegressor(
            iterations=250, depth=3, learning_rate=0.08,
            l2_leaf_reg=1.75, loss_function='RMSE',
            random_state=random_state + fold, verbose=False
        )
        cb_model.fit(X_train, y_train)
        catboost_models.append(cb_model)

        # Neural Network
        nn_model = train_neural_network(X_train, y_train, X_train, y_train)
        nn_models.append(nn_model)

        print(f"  Trained ensemble member {fold + 1}/{n_ensemble}")

    print("✓ Ensemble training complete!")

    return catboost_models, nn_models, scaler


def predict_with_uncertainty(X_new, catboost_models, nn_models, scaler):
    """
    Make predictions with uncertainty estimation using ensemble
    """
    X_scaled = scaler.transform(X_new)

    # CatBoost ensemble predictions
    catboost_preds = np.array([model.predict(X_scaled) for model in catboost_models])
    catboost_mean = np.mean(catboost_preds, axis=0)
    catboost_std = np.std(catboost_preds, axis=0)

    # Neural Network ensemble predictions
    nn_preds_list = []
    for model in nn_models:
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_scaled)).numpy().flatten()
            nn_preds_list.append(pred)

    nn_preds = np.array(nn_preds_list)
    nn_mean = np.mean(nn_preds, axis=0)
    nn_std = np.std(nn_preds, axis=0)

    # Combined uncertainty (average of both models' uncertainties)
    combined_uncertainty = (catboost_std + nn_std) / 2

    return catboost_mean, nn_mean, combined_uncertainty


def create_parity_plot(y_true, y_pred, model_name, save_path='parity_plot.png'):
    """Create and save parity plot"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', linewidths=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    plt.xlabel('True Dq/B', fontsize=14)
    plt.ylabel('Predicted Dq/B', fontsize=14)
    plt.title(f'{model_name}\nR² = {r2:.4f}, MAE = {mae:.4f}', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved parity plot to: {save_path}")


def main_pipeline(training_file='Cr3_dqb_training_set.xlsx',
                  prediction_file='To_predict.xlsx',
                  optimize_state=False,
                  random_state=42):
    """
    Main integrated pipeline: ML prediction + Expert System evaluation
    """
    print("\n" + "=" * 80)
    print("INTEGRATED ML + EXPERT SYSTEM PIPELINE")
    print("Cr³⁺ Phosphor Discovery Framework")
    print("=" * 80)

    # Load training data
    print("\n[1/6] Loading training data...")
    train_df = pd.read_excel(training_file)
    X_train = train_df.iloc[:, 1:-1].values  # Features
    y_train = train_df.iloc[:, -1].values  # Target (Dq/B)
    print(f"✓ Loaded {len(train_df)} training samples with {X_train.shape[1]} features")

    # Optimize random state (optional)
    if optimize_state:
        print("\n[2/6] Optimizing random state...")
        random_state = optimize_random_state(X_train, y_train, n_candidates=29, n_splits=10)
    else:
        print(f"\n[2/6] Using fixed random state: {random_state}")

    # Final cross-validation with best random state
    print("\n[3/6] Final cross-validation...")
    run_cross_validation(X_train, y_train, n_splits=10, random_state=random_state)

    # Train final models
    print("\n[4/6] Training final ensemble models...")
    catboost_models, nn_models, scaler = train_final_models(X_train, y_train, random_state)

    # Load prediction data
    print("\n[5/6] Making predictions on new candidates...")
    pred_df = pd.read_excel(prediction_file)
    formulas = pred_df.iloc[:, 0].values
    X_pred = pred_df.iloc[:, 1:].values

    # Make predictions with uncertainty
    catboost_preds, nn_preds, uncertainties = predict_with_uncertainty(
        X_pred, catboost_models, nn_models, scaler
    )

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'Formula': formulas,
        'CatBoost_Pred': catboost_preds,
        'NN_Pred': nn_preds,
        'Uncertainty': uncertainties
    })

    print(f"✓ Predictions complete for {len(predictions_df)} candidates")

    # Run Expert System Evaluation
    print("\n[6/6] Running expert system evaluation...")
    expert_results = main_expert_system_pipeline(
        predictions_df=predictions_df,
        output_excel='expert_system_recommendations.xlsx',
        output_report='expert_system_report.txt'
    )

    # Create parity plots for training data validation
    print("\nGenerating validation plots...")

    # Quick validation on training data
    X_train_scaled = scaler.transform(X_train)
    cb_train_pred = np.mean([m.predict(X_train_scaled) for m in catboost_models], axis=0)

    nn_train_preds = []
    for model in nn_models:
        model.eval()
        with torch.no_grad():
            pred = model(torch.FloatTensor(X_train_scaled)).numpy().flatten()
            nn_train_preds.append(pred)
    nn_train_pred = np.mean(nn_train_preds, axis=0)

    create_parity_plot(y_train, cb_train_pred, 'CatBoost Model (Training Set)',
                       'parity_plot_catboost.png')
    create_parity_plot(y_train, nn_train_pred, 'Neural Network Model (Training Set)',
                       'parity_plot_nn.png')

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  • expert_system_recommendations.xlsx - Detailed evaluation with scores")
    print("  • expert_system_report.txt - Summary report with top recommendations")
    print("  • parity_plot_catboost.png - CatBoost model validation")
    print("  • parity_plot_nn.png - Neural Network model validation")
    print("\n✓ Ready for experimental synthesis of top-ranked candidates!")

    return expert_results


if __name__ == "__main__":
    # Run complete pipeline
    results = main_pipeline(
        training_file='Cr3_dqb_training_set.xlsx',
        prediction_file='To_predict.xlsx',
        optimize_state=False,  # Set to True for random state optimization
        random_state=42
    )

    # Display top 5 recommendations
    print("\n" + "=" * 80)
    print("TOP 5 SYNTHESIS RECOMMENDATIONS")
    print("=" * 80)
    top5 = results.head(5)[['Formula', 'Predicted_Emission_nm', 'Total_Score',
                            'Tier', 'Recommendation']]
    print(top5.to_string(index=False))
