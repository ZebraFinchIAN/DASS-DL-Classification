"""
DASS Multi-Label Deep Learning Classifier - EXAMPLE USAGE
Demonstrates full pipeline: Data Prep → Training → Evaluation

Bu script synthetic örnek veri ile çalışır.
Kendi verin ile kullanmak için satır ~150'de değişiklik yap.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from dass_multilabel_classifier import (
    DASSMultiTaskNet, FocalLoss, DASSTrainer, 
    prepare_data, evaluate_model, 
    plot_training_history, plot_confusion_matrices
)

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 150
GAMMA = 2.0  # Focal loss gamma
# GAMMA = 3.5 için extreme imbalance (Religion 378:1, Race 1197:1)

# ============================================================================
# SYNTHETIC DATA CREATION (Test için)
# ============================================================================

def create_synthetic_dass_data(n_samples=1000, random_state=42):
    """
    Test amaçlı synthetic DASS veri oluştur
    
    Includes:
    - 42 DASS questions (Q1A-Q42A, scale 1-4)
    - Response times (Q1E-Q42E, milliseconds)
    - Demographic features (one-hot encoded)
    - DASS scores (calculated from answers)
    - Multi-label targets (Depression/Anxiety/Stress severity)
    """
    
    np.random.seed(random_state)
    
    data = {}
    
    # 1. DASS Answers (Q1A-Q42A, scale 1-4)
    for i in range(1, 43):
        data[f'Q{i}A'] = np.random.randint(1, 5, n_samples)
    
    # 2. Response times (Q1E-Q42E, milliseconds)
    for i in range(1, 43):
        data[f'Q{i}E'] = np.random.exponential(7, n_samples)
    
    # 3. Demographic features (one-hot encoded)
    # Gender (3 categories)
    data['gender_1'] = np.random.randint(0, 2, n_samples)  # Male
    data['gender_2'] = np.random.randint(0, 2, n_samples)  # Female
    data['gender_3'] = np.random.randint(0, 2, n_samples)  # Other
    
    # Education (4 categories)
    for i in range(1, 5):
        data[f'education_{i}'] = np.random.randint(0, 2, n_samples)
    
    # Urban (3 categories)
    for i in range(1, 4):
        data[f'urban_{i}'] = np.random.randint(0, 2, n_samples)
    
    # Age & family size
    data['age'] = np.random.randint(18, 80, n_samples)
    data['familysize'] = np.random.randint(1, 8, n_samples)
    
    # 4. DASS Scores (calculated from answers)
    # Depression: Q3, Q5, Q10, Q13, Q16, Q17, Q21, Q24, Q26, Q31, Q34, Q37, Q38, Q42
    depression_items = [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42]
    data['Depression_Score'] = sum([data[f'Q{i}A'] for i in depression_items]) / 2  # Normalize
    
    # Anxiety: Q2, Q4, Q7, Q9, Q15, Q19, Q20, Q23, Q25, Q28, Q30, Q36, Q40, Q41
    anxiety_items = [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41]
    data['Anxiety_Score'] = sum([data[f'Q{i}A'] for i in anxiety_items]) / 2
    
    # Stress: Q1, Q6, Q8, Q11, Q12, Q14, Q18, Q22, Q27, Q29, Q32, Q33, Q35, Q39
    stress_items = [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]
    data['Stress_Score'] = sum([data[f'Q{i}A'] for i in stress_items]) / 2
    
    # 5. Multi-label targets (one-hot encoded)
    # DASS Severity thresholds: Normal < Mild < Moderate < Severe < Extremely Severe
    
    # Depression severity
    dep_severity = np.zeros((n_samples, 5))
    for i in range(n_samples):
        score = data['Depression_Score'][i]
        if score < 10:
            dep_severity[i, 0] = 1  # Normal
        elif score < 14:
            dep_severity[i, 1] = 1  # Mild
        elif score < 21:
            dep_severity[i, 2] = 1  # Moderate
        elif score < 28:
            dep_severity[i, 3] = 1  # Severe
        else:
            dep_severity[i, 4] = 1  # Extremely Severe
    
    # Anxiety severity (similar approach)
    anx_severity = np.zeros((n_samples, 5))
    for i in range(n_samples):
        score = data['Anxiety_Score'][i]
        if score < 8:
            anx_severity[i, 0] = 1
        elif score < 10:
            anx_severity[i, 1] = 1
        elif score < 15:
            anx_severity[i, 2] = 1
        elif score < 20:
            anx_severity[i, 3] = 1
        else:
            anx_severity[i, 4] = 1
    
    # Stress severity
    str_severity = np.zeros((n_samples, 5))
    for i in range(n_samples):
        score = data['Stress_Score'][i]
        if score < 15:
            str_severity[i, 0] = 1
        elif score < 19:
            str_severity[i, 1] = 1
        elif score < 26:
            str_severity[i, 2] = 1
        elif score < 34:
            str_severity[i, 3] = 1
        else:
            str_severity[i, 4] = 1
    
    # Add one-hot encoded columns
    for j in range(5):
        data[f'Depression_Status_{["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"][j]}'] = dep_severity[:, j]
        data[f'Anxiety_Status_{["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"][j]}'] = anx_severity[:, j]
        data[f'Stress_Status_{["Normal", "Mild", "Moderate", "Severe", "Extremely Severe"][j]}'] = str_severity[:, j]
    
    df = pd.DataFrame(data)
    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("\n" + "="*70)
    print("DASS Multi-Label Deep Learning Classifier")
    print("Strategy: Class Weights + Focal Loss + Task-Specific Models")
    print("="*70)
    
    print(f"\n✓ Device: {DEVICE}")
    print(f"✓ Random seed: {SEED}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    print(f"✓ Learning rate: {LEARNING_RATE}")
    print(f"✓ Focal Loss gamma: {GAMMA}")
    
    # Set seeds for reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # ─────────────────────────────────────────────────────────────────────
    # 1. CREATE DATA
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*70}")
    print("1. CREATE DATA")
    print(f"{'─'*70}")
    
    # Option A: Synthetic data (test)
    df = create_synthetic_dass_data(n_samples=1000, random_state=SEED)
    print(f"\n✓ Synthetic data created: {df.shape}")
    
    # Option B: Real data (uncomment to use)
    # df = pd.read_csv('your_processed_dass_data.csv')
    # print(f"\n✓ Data loaded: {df.shape}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 2. PREPARE DATA (Stratified Split + Class Weights)
    # ─────────────────────────────────────────────────────────────────────
    
    data = prepare_data(df, test_size=0.2, val_size=0.1, device=DEVICE)
    
    # ─────────────────────────────────────────────────────────────────────
    # 3. CREATE DATALOADERS
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*70}")
    print("2. CREATE DATALOADERS")
    print(f"{'─'*70}")
    
    train_dataset = TensorDataset(
        data['X_train'],
        data['y_train']['depression'],
        data['y_train']['anxiety'],
        data['y_train']['stress']
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(
        data['X_val'],
        data['y_val']['depression'],
        data['y_val']['anxiety'],
        data['y_val']['stress']
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n✓ DataLoaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # ─────────────────────────────────────────────────────────────────────
    # 4. INITIALIZE MODEL & LOSS FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*70}")
    print("3. INITIALIZE MODEL & LOSS FUNCTIONS")
    print(f"{'─'*70}")
    
    input_dim = data['X_train'].shape[1]
    model = DASSMultiTaskNet(input_dim=input_dim, num_classes=5)
    
    # Focal Loss with Class Weights (STEP 1 + STEP 2: Hocanın Tavsiyesi)
    loss_fns_dict = {
        'depression': FocalLoss(
            alpha=data['class_weights']['depression'],
            gamma=GAMMA,
            device=DEVICE
        ),
        'anxiety': FocalLoss(
            alpha=data['class_weights']['anxiety'],
            gamma=GAMMA,
            device=DEVICE
        ),
        'stress': FocalLoss(
            alpha=data['class_weights']['stress'],
            gamma=GAMMA,
            device=DEVICE
        )
    }
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n✓ Model initialized:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture: Shared (256→128→64) + 3 Task Heads")
    
    # ─────────────────────────────────────────────────────────────────────
    # 5. TRAIN
    # ─────────────────────────────────────────────────────────────────────
    
    trainer = DASSTrainer(model, device=DEVICE, lr=LEARNING_RATE)
    history = trainer.fit(train_loader, val_loader, loss_fns_dict, epochs=EPOCHS, gamma=GAMMA)
    
    # ─────────────────────────────────────────────────────────────────────
    # 6. EVALUATE
    # ─────────────────────────────────────────────────────────────────────
    
    results = evaluate_model(model, data['X_test'], data['y_test'], device=DEVICE)
    
    # ─────────────────────────────────────────────────────────────────────
    # 7. VISUALIZE & SAVE
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n{'─'*70}")
    print("4. SAVE RESULTS")
    print(f"{'─'*70}\n")
    
    plot_training_history(history, save_path='training_history.png')
    plot_confusion_matrices(results, save_path='confusion_matrices.png')
    
    # Save model
    torch.save(model.state_dict(), 'dass_model_focal_loss.pt')
    print("✓ Model saved to dass_model_focal_loss.pt")
    
    # ──────────────────────────────────────���──────────────────────────────
    # 8. SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    print("✓ Training Complete!")
    print(f"\n📊 Final Results:")
    print(f"  Depression Macro F1:  {results['depression']['macro_f1']:.4f}")
    print(f"  Anxiety Macro F1:     {results['anxiety']['macro_f1']:.4f}")
    print(f"  Stress Macro F1:      {results['stress']['macro_f1']:.4f}")
    
    print(f"\n📈 Files saved:")
    print(f"  - training_history.png")
    print(f"  - confusion_matrices.png")
    print(f"  - dass_model_focal_loss.pt")
    
    print(f"\n💡 Hocanın Tavsiyesi uygulandı:")
    print(f"  ✓ STEP 1: Class Weights (inverse frequency)")
    print(f"  ✓ STEP 2: Focal Loss (γ={GAMMA})")
    print(f"  ✓ STEP 3: (Optional) Gender split edebilirsin")
    print(f"  ✓ STEP 4: Macro F1 metriği kullandık (Accuracy değil)")
    print(f"  ✓ STEP 5: SMOTE kullanmadık (Stratified split yeterli)")
    
    print(f"\n🎯 Next steps:")
    print(f"  1. Kendi verin ile test et")
    print(f"  2. GAMMA değerini tune et (3.5 for extreme imbalance)")
    print(f"  3. Dropout/LR/Epochs adjust et")
    print(f"  4. Production'a deploy et")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
