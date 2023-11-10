import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dill
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import cross_val_predict
from src.logger import logger
from src.exception import CustomException
from utils import load_object
import os

class ModelVisualization:
    def __init__(self, model_path):
        self.model_path = model_path

    def visualize(self):
        try:
            model = load_object(self.model_path)
            X = model['X']
            y = model['y']
            clf = model['clf']
            y_pred = cross_val_predict(clf, X, y, cv=5)
            conf_mx = confusion_matrix(y, y_pred)
            precision, recall, thresholds = precision_recall_curve(y, y_pred)
            feature_importances = clf.feature_importances_
            # code to visualize the model
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].matshow(conf_mx, cmap=plt.cm.gray)
            ax[0].set_xlabel('Predicted')
            ax[0].set_ylabel('Actual')
            ax[1].plot(recall, precision, 'b-', linewidth=2)
            ax[1].set_xlabel('Recall')
            ax[1].set_ylabel('Precision')
            ax[1].set_xlim([0, 1])
            ax[1].set_ylim([0, 1])
            plt.show()
            logger.info("Model visualization successful")
            return feature_importances
        except FileNotFoundError:
            logger.error("Model file not found")
            raise CustomException("Model file not found")
        except Exception as e:
            logger.error(f"Error visualizing model: {e}")
            raise CustomException("Error visualizing model")

    def save_feature_importance(self, feature_importances, model_name):
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(feature_importances)), feature_importances)
            ax.set_xticks(range(len(feature_importances)))
            ax.set_xticklabels(X.columns, rotation=90)
            ax.set_title(f"Feature Importance for {model_name}")
            plt.savefig(f"Figuras/{model_name}_feature_importance.png")
            logger.info("Feature importance saved successfully")
        except Exception as e:
            logger.error(f"Error saving feature importance: {e}")
            raise CustomException("Error saving feature importance")

    def save_confusion_matrix(self, conf_mx, model_name):
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.matshow(conf_mx, cmap=plt.cm.gray)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f"Confusion Matrix for {model_name}")
            plt.savefig(f"Figuras/{model_name}_confusion_matrix.png")
            logger.info("Confusion matrix saved successfully")
        except Exception as e:
            logger.error(f"Error saving confusion matrix: {e}")
            raise CustomException("Error saving confusion matrix")

    def save_precision_recall_curve(self, precision, recall, model_name):
        try:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(recall, precision, 'b-', linewidth=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title(f"Precision-Recall Curve for {model_name}")
            plt.savefig(f"Figuras/{model_name}_precision_recall_curve.png")
            logger.info("Precision-Recall curve saved successfully")
        except Exception as e:
            logger.error(f"Error saving precision-recall curve: {e}")
            raise CustomException("Error saving precision-recall curve")


