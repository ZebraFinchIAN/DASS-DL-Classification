"""
DASS Multi-Label Deep Learning Classifier
Extreme Imbalance Handling (Religion: 378:1, Race: 1197:1)
Strategy: Class Weights → Focal Loss → Task-Specific Models → SMOTE (last resort)

Author: ZebraFinch AI
Date: 2026-05-05
Based on: Hocanın Tavsiyesi (Class Weights + Focal Loss + No SMOTE)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. FOCAL LOSS - Extreme Imbalance için En Önemli Bileşen
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: (1 - pt)^gamma * CE(p, y)
    
    Neden Focal Loss?
    - Standard CE: zor örnekleri easy negatifler "eziyor"
    - Focal Loss: easy samples downweight → hard samples focus
    - Extreme imbalance'da ~15-20% performans iyileştirmesi
    
    Parametreler:
        alpha: class weights (inverse frequency)
        gamma: focusing parameter (2 = balanced, 3-4 = extreme imbalance)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean', device='cpu'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) logits
            targets: (N,) class indices
        """
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        
        # pt = prob of correct class
        p_t = torch.exp(-ce_loss)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# 2. MULTI-TASK NEURAL NETWORK - Shared + Task-Specific Heads
# ============================================================================

class DASSMultiTaskNet(nn.Module):
    """
    Multi-task learning architecture:
    - Shared feature extraction layers (Depression/Anxiety/Stress ortak patterns)
    - 3 task-specific heads (Depression, Anxiety, Stress)
    
    Avantajlar:
    - Shared layers = transfer learning
    - Task-specific heads = her görev kendi pattern'i öğrenebilir
    - Daha az parameter = daha hızlı eğitim
    
    Architecture:
        Input (130) → Shared (256) → BN → ReLU → Dropout(0.4)
                    → Shared (128) → BN → ReLU → Dropout(0.3)
                    → Shared (64)  → BN → ReLU → Dropout(0.2)
                    ├─ Depression Head → 32 → ReLU → Dropout(0.2) → 5 (softmax)
                    ├─ Anxiety Head    → 32 → ReLU → Dropout(0.2) → 5 (softmax)
                    └─ Stress Head     → 32 → ReLU → Dropout(0.2) → 5 (softmax)
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 dropout_rates=[0.4, 0.3, 0.2], num_classes=5):
        super(DASSMultiTaskNet, self).__init__()
        self.num_classes = num_classes
        
        # Shared layers
        layers_list = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers_list.append(nn.Linear(prev_dim, hidden_dim))
            layers_list.append(nn.BatchNorm1d(hidden_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers_list)
        
        # Task-specific heads
        self.depression_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        self.anxiety_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        self.stress_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        shared = self.shared_layers(x)
        depression = self.depression_head(shared)
        anxiety = self.anxiety_head(shared)
        stress = self.stress_head(shared)
        return depression, anxiety, stress


# ============================================================================
# 3. CLASS WEIGHT HESAPLAMA - STEP 1: Temel Zorunlu Adım
# ============================================================================

def calculate_class_weights_per_task(y_train_dict, num_classes=5, device='cpu'):
    """
    Hocanın Tavsiyesi - STEP 1: Class Weights Hesapla
    
    Balanced class weights = inverse frequency
    Ağır imbalance için weights otomatik büyüyor
    
    Örnek (Religion 378:1):
        Muslim (freq=0.997) → weight=0.003
        Sikh (freq=0.0001) → weight=10.0
    
    Args:
        y_train_dict: {'depression': array, 'anxiety': array, 'stress': array}
        num_classes: 5 (Normal, Mild, Moderate, Severe, Extremely Severe)
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict of torch tensors with weights for each task
    """
    weights_dict = {}
    
    for task_name, y_true in y_train_dict.items():
        # One-hot → single label
        if len(y_true.shape) > 1 and y_true.shape[1] > 1:
            y_single = np.argmax(y_true, axis=1)
        else:
            y_single = y_true.flatten()
        
        # Sklearn's balanced weights (1 / frequency)
        class_weights = compute_class_weight(
            'balanced',
            classes=np.arange(num_classes),
            y=y_single.astype(int)
        )
        
        weights_tensor = torch.from_numpy(class_weights).float().to(device)
        weights_dict[task_name] = weights_tensor
        
        print(f"\n{task_name.upper()} - Class Weights (Inverse Frequency):")
        print(f"  Normal (0):           {class_weights[0]:.4f}")
        print(f"  Mild (1):             {class_weights[1]:.4f}")
        print(f"  Moderate (2):         {class_weights[2]:.4f}")
        print(f"  Severe (3):           {class_weights[3]:.4f}")
        print(f"  Extremely Severe (4): {class_weights[4]:.4f}")
        
        # Imbalance ratio
        class_counts = np.bincount(y_single.astype(int), minlength=num_classes)
        max_class = np.max(class_counts)
        min_class = np.min(class_counts[class_counts > 0])
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        print(f"  Imbalance Ratio: {imbalance_ratio:.1f}:1")
    
    return weights_dict


# ============================================================================
# 4. VERİ HAZIRLAMA - Stratified Split + Normalization (SMOTE YOK!)
# ============================================================================

def prepare_data(df, test_size=0.2, val_size=0.1, random_state=42, device='cpu'):
    """
    Hocanın Tavsiyesi - STEP 5: SMOTE Kaçın
    
    Bunun yerine:
    - Stratified Split (imbalance'ı korur)
    - Standard Scaling
    - Class Weights (Focal Loss'ta kullanılır)
    
    Args:
        df: Preprocessed dataframe (one-hot encoded)
        test_size: % test set (default 20%)
        val_size: % validation set (default 10% of train)
        random_state: Reproducibility
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict with X_train/val/test, y_train/val/test, weights, scaler
    """
    
    print("\n" + "="*70)
    print("DATA PREPARATION (Strategy: Stratified Split + Class Weights)")
    print("="*70)
    
    # Feature seç (DASS + demographics, targets hariç)
    exclude_cols = [col for col in df.columns if 
                   col.startswith(('Depression_Status', 'Anxiety_Status', 'Stress_Status'))]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    
    print(f"\n✓ Features selected: {X.shape[1]}")
    print(f"  Samples: {X.shape[0]}")
    
    # Targets extract
    depression_cols = [col for col in df.columns if col.startswith('Depression_Status')]
    anxiety_cols = [col for col in df.columns if col.startswith('Anxiety_Status')]
    stress_cols = [col for col in df.columns if col.startswith('Stress_Status')]
    
    y_depression = df[depression_cols].values
    y_anxiety = df[anxiety_cols].values
    y_stress = df[stress_cols].values
    
    # Single label for stratification
    y_depression_single = y_depression.argmax(axis=1)
    
    print(f"\n✓ Target Distribution (Depression - for stratification):")
    unique, counts = np.unique(y_depression_single, return_counts=True)
    for u, c in zip(unique, counts):
        pct = 100 * c / len(y_depression_single)
        print(f"  Class {u}: {c:6d} ({pct:5.2f}%)")
    
    # Stratified split
    print(f"\n✓ Stratified Split (by Depression severity):")
    
    X_temp, X_test, dep_temp, dep_test, anx_temp, anx_test, str_temp, str_test = train_test_split(
        X, y_depression, y_anxiety, y_stress,
        test_size=test_size,
        stratify=y_depression_single,
        random_state=random_state
    )
    
    X_train, X_val, dep_train, dep_val, anx_train, anx_val, str_train, str_val = train_test_split(
        X_temp, dep_temp, anx_temp, str_temp,
        test_size=val_size / (1 - test_size),
        stratify=dep_temp.argmax(axis=1),
        random_state=random_state
    )
    
    print(f"  Train: {X_train.shape[0]} ({100*X_train.shape[0]/X.shape[0]:.1f}%)")
    print(f"  Val:   {X_val.shape[0]} ({100*X_val.shape[0]/X.shape[0]:.1f}%)")
    print(f"  Test:  {X_test.shape[0]} ({100*X_test.shape[0]/X.shape[0]:.1f}%)")
    
    # StandardScaler (fit on train only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n✓ StandardScaler fitted on training data")
    
    # Torch tensors'a convert
    X_train_tensor = torch.from_numpy(X_train_scaled).float().to(device)
    X_val_tensor = torch.from_numpy(X_val_scaled).float().to(device)
    X_test_tensor = torch.from_numpy(X_test_scaled).float().to(device)
    
    y_train_dict = {'depression': dep_train, 'anxiety': anx_train, 'stress': str_train}
    y_val_dict = {'depression': dep_val, 'anxiety': anx_val, 'stress': str_val}
    y_test_dict = {'depression': dep_test, 'anxiety': anx_test, 'stress': str_test}
    
    # Class weights hesapla (STEP 1)
    print(f"\n✓ Computing class weights (Focal Loss):")
    weights_dict = calculate_class_weights_per_task(y_train_dict, num_classes=5, device=device)
    
    # Convert labels to tensors (one-hot → indices)
    y_train_tensors = {
        'depression': torch.from_numpy(np.argmax(dep_train, axis=1)).long().to(device),
        'anxiety': torch.from_numpy(np.argmax(anx_train, axis=1)).long().to(device),
        'stress': torch.from_numpy(np.argmax(str_train, axis=1)).long().to(device)
    }
    
    y_val_tensors = {
        'depression': torch.from_numpy(np.argmax(dep_val, axis=1)).long().to(device),
        'anxiety': torch.from_numpy(np.argmax(anx_val, axis=1)).long().to(device),
        'stress': torch.from_numpy(np.argmax(str_val, axis=1)).long().to(device)
    }
    
    y_test_tensors = {
        'depression': torch.from_numpy(np.argmax(dep_test, axis=1)).long().to(device),
        'anxiety': torch.from_numpy(np.argmax(anx_test, axis=1)).long().to(device),
        'stress': torch.from_numpy(np.argmax(str_test, axis=1)).long().to(device)
    }
    
    return {
        'X_train': X_train_tensor, 'X_val': X_val_tensor, 'X_test': X_test_tensor,
        'y_train': y_train_tensors, 'y_val': y_val_tensors, 'y_test': y_test_tensors,
        'class_weights': weights_dict, 'scaler': scaler, 'feature_cols': feature_cols
    }


# ============================================================================
# 5. TRAINING LOOP - Focal Loss + Class Weights + Early Stopping
# ============================================================================

class DASSTrainer:
    """
    Multi-task trainer with:
    - Focal Loss (STEP 2: Hocanın Tavsiyesi)
    - Class Weights (STEP 1: Hocanın Tavsiyesi)
    - Early Stopping
    - Learning Rate Scheduling
    - Gradient Clipping
    """
    
    def __init__(self, model, device='cpu', lr=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_macro_f1': [], 'val_macro_f1': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 20
    
    def train_epoch(self, train_loader, loss_fns_dict, gamma=2.0):
        """One training epoch with all three tasks"""
        self.model.train()
        total_loss = 0
        all_preds = {'depression': [], 'anxiety': [], 'stress': []}
        all_targets = {'depression': [], 'anxiety': [], 'stress': []}
        
        for X_batch, y_dep_batch, y_anx_batch, y_str_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_dep_batch = y_dep_batch.to(self.device)
            y_anx_batch = y_anx_batch.to(self.device)
            y_str_batch = y_str_batch.to(self.device)
            
            # Forward pass
            pred_dep, pred_anx, pred_str = self.model(X_batch)
            
            # Focal Loss (STEP 2) + Class Weights (STEP 1)
            loss_dep = loss_fns_dict['depression'](pred_dep, y_dep_batch)
            loss_anx = loss_fns_dict['anxiety'](pred_anx, y_anx_batch)
            loss_str = loss_fns_dict['stress'](pred_str, y_str_batch)
            
            # Multi-task loss (equal weighted)
            loss = (loss_dep + loss_anx + loss_str) / 3
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            all_preds['depression'].append(pred_dep.detach().cpu().numpy())
            all_preds['anxiety'].append(pred_anx.detach().cpu().numpy())
            all_preds['stress'].append(pred_str.detach().cpu().numpy())
            all_targets['depression'].append(y_dep_batch.detach().cpu().numpy())
            all_targets['anxiety'].append(y_anx_batch.detach().cpu().numpy())
            all_targets['stress'].append(y_str_batch.detach().cpu().numpy())
        
        epoch_loss = total_loss / len(train_loader)
        
        # Metrics
        acc_scores, macro_f1_scores = [], []
        for task in ['depression', 'anxiety', 'stress']:
            preds = np.concatenate(all_preds[task]).argmax(axis=1)
            targets = np.concatenate(all_targets[task])
            acc = np.mean(preds == targets)
            macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
            acc_scores.append(acc)
            macro_f1_scores.append(macro_f1)
        
        return epoch_loss, np.mean(acc_scores), np.mean(macro_f1_scores)
    
    @torch.no_grad()
    def validate(self, val_loader, loss_fns_dict):
        """Validation epoch"""
        self.model.eval()
        total_loss = 0
        all_preds = {'depression': [], 'anxiety': [], 'stress': []}
        all_targets = {'depression': [], 'anxiety': [], 'stress': []}
        
        for X_batch, y_dep_batch, y_anx_batch, y_str_batch in val_loader:
            X_batch = X_batch.to(self.device)
            y_dep_batch = y_dep_batch.to(self.device)
            y_anx_batch = y_anx_batch.to(self.device)
            y_str_batch = y_str_batch.to(self.device)
            
            pred_dep, pred_anx, pred_str = self.model(X_batch)
            
            loss_dep = loss_fns_dict['depression'](pred_dep, y_dep_batch)
            loss_anx = loss_fns_dict['anxiety'](pred_anx, y_anx_batch)
            loss_str = loss_fns_dict['stress'](pred_str, y_str_batch)
            
            loss = (loss_dep + loss_anx + loss_str) / 3
            total_loss += loss.item()
            
            all_preds['depression'].append(pred_dep.cpu().numpy())
            all_preds['anxiety'].append(pred_anx.cpu().numpy())
            all_preds['stress'].append(pred_str.cpu().numpy())
            all_targets['depression'].append(y_dep_batch.cpu().numpy())
            all_targets['anxiety'].append(y_anx_batch.cpu().numpy())
            all_targets['stress'].append(y_str_batch.cpu().numpy())
        
        epoch_loss = total_loss / len(val_loader)
        
        acc_scores, macro_f1_scores = [], []
        for task in ['depression', 'anxiety', 'stress']:
            preds = np.concatenate(all_preds[task]).argmax(axis=1)
            targets = np.concatenate(all_targets[task])
            acc = np.mean(preds == targets)
            macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
            acc_scores.append(acc)
            macro_f1_scores.append(macro_f1)
        
        return epoch_loss, np.mean(acc_scores), np.mean(macro_f1_scores)
    
    def fit(self, train_loader, val_loader, loss_fns_dict, epochs=150, gamma=2.0):
        """Full training loop with early stopping"""
        
        print("\n" + "="*70)
        print(f"TRAINING WITH FOCAL LOSS (gamma={gamma}) + CLASS WEIGHTS")
        print("="*70)
        
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, loss_fns_dict, gamma)
            val_loss, val_acc, val_f1 = self.validate(val_loader, loss_fns_dict)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_macro_f1'].append(train_f1)
            self.history['val_macro_f1'].append(val_f1)
            
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1:3d}/{epochs}")
                print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                print(f"  Train Acc:  {train_acc:.4f} | Val Acc:  {val_acc:.4f}")
                print(f"  Train F1:   {train_f1:.4f} | Val F1:   {val_f1:.4f}")
                print(f"  Patience:   {self.patience_counter}/{self.max_patience}")
            
            if self.patience_counter >= self.max_patience:
                print(f"\n✓ Early stopping at epoch {epoch+1}")
                self.model.load_state_dict(self.best_model_state)
                break
        
        return self.history


# ============================================================================
# 6. EVALUATION - Per-Class Metrics (Hocanın Tavsiyesi STEP 4: Macro F1!)
# ============================================================================

def evaluate_model(model, X_test_tensor, y_test_dict, device='cpu', task_names=['depression', 'anxiety', 'stress']):
    """
    Hocanın Tavsiyesi - STEP 4: Macro F1 ile Ölçü (Accuracy değil!)
    
    İmbalanced data'da Accuracy yanıltıcı:
    - Model tüm samples'ı "Muslim" tahmin ediyor → %99.7 accuracy
    - Ama tüm azınlık classları kaçırıyor → işe yaramaz
    
    Macro F1 = per-class F1'in ortalaması → her class eşit ağırlık
    """
    
    model.eval()
    
    with torch.no_grad():
        X_test = X_test_tensor.to(device)
        pred_dep, pred_anx, pred_str = model(X_test)
    
    predictions = {
        'depression': pred_dep.cpu().numpy(),
        'anxiety': pred_anx.cpu().numpy(),
        'stress': pred_str.cpu().numpy()
    }
    
    results = {}
    class_names = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
    
    print("\n" + "="*70)
    print("EVALUATION RESULTS (Macro F1 > Accuracy for Imbalanced Data)")
    print("="*70)
    
    for task in task_names:
        y_true = y_test_dict[task].cpu().numpy()
        y_pred = predictions[task].argmax(axis=1)
        
        accuracy = np.mean(y_pred == y_true)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0, labels=range(5))
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0, labels=range(5))
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0, labels=range(5))
        
        conf_matrix = confusion_matrix(y_true, y_pred, labels=range(5))
        
        results[task] = {
            'y_true': y_true, 'y_pred': y_pred, 'accuracy': accuracy,
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class, 'f1_per_class': f1_per_class,
            'confusion_matrix': conf_matrix
        }
        
        print(f"\n{'─'*70}")
        print(f"{task.upper()}")
        print(f"{'─'*70}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy (NOT recommended):  {accuracy:.4f}")
        print(f"  Macro F1 (RECOMMENDED):      {macro_f1:.4f}  ← BU'NA BAK!")
        print(f"  Weighted F1:                 {weighted_f1:.4f}")
        
        print(f"\nPer-Class Metrics:")
        print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Support':>12}")
        print(f"{'-'*68}")
        
        class_counts = np.bincount(y_true, minlength=5)
        for i, class_name in enumerate(class_names):
            print(f"{class_name:<20} {precision_per_class[i]:>12.4f} {recall_per_class[i]:>12.4f} "
                  f"{f1_per_class[i]:>12.4f} {class_counts[i]:>12d}")
    
    return results


# ============================================================================
# 7. VİZÜALİZASYON
# ============================================================================

def plot_training_history(history, save_path='training_history.png'):
    """Training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Focal Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Per-Sample Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['train_macro_f1'], label='Train', linewidth=2)
    axes[2].plot(history['val_macro_f1'], label='Val', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Macro F1 (Recommended)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history saved to {save_path}")
    plt.close()


def plot_confusion_matrices(results, task_names=['depression', 'anxiety', 'stress'], 
                            save_path='confusion_matrices.png'):
    """Confusion matrices for all tasks"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    class_names_short = ['N', 'M', 'Mo', 'S', 'ES']
    
    for idx, task in enumerate(task_names):
        cm = results[task]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=class_names_short, yticklabels=class_names_short,
                   cbar=False)
        axes[idx].set_title(f'{task.capitalize()}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved to {save_path}")
    plt.close()
