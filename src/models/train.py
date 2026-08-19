import os
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb

logger = logging.getLogger(__name__)

class ModelPipeline:
    def __init__(self, output_dir: str = 'outputs'):
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, 'models')
        self.figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.best_model = None

    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42):
        """Scales data, trains baseline and ensemble models, and outputs metrics."""
        
        # Data Splitting
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
        
        # Scaling
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=X.columns, index=X_test.index)
        
        # Define base estimators
        estimators = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=random_state, class_weight='balanced'),
            'LightGBM': lgb.LGBMClassifier(random_state=random_state, verbose=-1)
        }

        results = {}
        for name, clf in estimators.items():
            logger.info(f"Training {name}...")
            clf.fit(X_train_scaled, y_train)
            preds = clf.predict(X_test_scaled)
            
            macro_f1 = f1_score(y_test, preds, average='macro')
            acc = accuracy_score(y_test, preds)
            results[name] = {'Macro-F1': macro_f1, 'Accuracy': acc}
            
            logger.info(f"{name} - Accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")

        # Train Meta-Ensemble
        logger.info("Training Voting Classifier ensemble...")
        voting_clf = VotingClassifier(
            estimators=[(name, clf) for name, clf in estimators.items()],
            voting='soft',
            n_jobs=-1
        )
        voting_clf.fit(X_train_scaled, y_train)
        preds = voting_clf.predict(X_test_scaled)
        results['Voting Ensemble'] = {
            'Macro-F1': f1_score(y_test, preds, average='macro'), 
            'Accuracy': accuracy_score(y_test, preds)
        }
        
        self.best_model = voting_clf
        self._plot_confusion_matrix(y_test, preds, voting_clf.classes_, "Voting Ensemble")
        self._save_artifacts()
        
        logger.info("\n" + classification_report(y_test, preds, target_names=voting_clf.classes_))
        return results

    def _plot_confusion_matrix(self, y_true, y_pred, classes, model_name: str):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'cm_{model_name.replace(" ", "_").lower()}.png'), dpi=300)
        plt.close()

    def _save_artifacts(self):
        joblib.dump(self.best_model, os.path.join(self.models_dir, 'classifier.pkl'))
        joblib.dump(self.scaler, os.path.join(self.models_dir, 'scaler.pkl'))
        logger.info(f"Model artifacts saved to {self.models_dir}/")
