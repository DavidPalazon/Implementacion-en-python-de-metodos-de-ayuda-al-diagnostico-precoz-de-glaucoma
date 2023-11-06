from dataclasses import dataclass
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier

from src.components.data_ingestion_OCT import OCTDataProcessor
from src.components.data_ingestion_RET import RETDataProcessor
from src.components.data_transformation import DELTA_METRICS
from src.exception import CustomException
from src.logger import logger, results_logger
from src.utils import get_params_grid, get_random_params, save_object

@dataclass
class ModelTrainerConfig:
    """Clase que representa la configuración del entrenador de modelos."""
    trained_model_path: str = os.path.join('models', 'Model.pkl')



class ModelTrainer:
    """Clase que representa un entrenador de modelos."""

    def __init__(self, config=ModelTrainerConfig()):
        self.model_trainer_config = config

    
    @staticmethod
    def save_best_results(filtered_cv_results, search):
        content = (
            f"Resultados de busqueda {search}:\n"
            f"Sensitivity: {filtered_cv_results['mean_test_sensitivity']:0.3f} (±{2*filtered_cv_results['std_test_sensitivity']:0.03f})\n"
            f"Precision: {filtered_cv_results['mean_test_precision']:0.3f} (±{2*filtered_cv_results['std_test_precision']:0.03f})\n"
            f"Accuracy: {filtered_cv_results['mean_test_accuracy']:0.3f} (±{2*filtered_cv_results['std_test_accuracy']:0.03f})\n"
            f"Specificity: {filtered_cv_results['mean_test_specificity']:0.3f} (±{2*filtered_cv_results['std_test_specificity']:0.03f})\n"
            f"Parámetros: {filtered_cv_results['params']} \n"
        )

        results_logger.info(content)

    
    def refit_strategy(self, cv_results, search):

        # List of columns to be used
        COLUMNS = [
            "mean_score_time",
            "mean_test_sensitivity",
            "std_test_sensitivity",
            "mean_test_precision",
            "std_test_precision",
            "mean_test_specificity",
            "std_test_specificity",
            "mean_test_accuracy",
            "std_test_accuracy",
            "rank_test_sensitivity",
            "rank_test_precision",
            "params",
        ]

        precision_threshold = 0.5

        cv_results_ = pd.DataFrame(cv_results)
        high_precision_cv_results = cv_results_[cv_results_["mean_test_precision"] > precision_threshold]
        high_precision_cv_results = high_precision_cv_results[COLUMNS]

        best_recall = high_precision_cv_results["mean_test_sensitivity"].max()
        best_recall_std = high_precision_cv_results["mean_test_sensitivity"].std()

        high_recall_cv_results = high_precision_cv_results[
            high_precision_cv_results["mean_test_sensitivity"] > best_recall - best_recall_std
        ]

        top_recall_high_precision_index = high_recall_cv_results["mean_test_sensitivity"].idxmax()

        self.save_best_results(high_recall_cv_results.loc[top_recall_high_precision_index], search)
        return top_recall_high_precision_index


    def train_model(self, pipe,model_name, X, y) -> object:
        """Entrena un modelo de clasificación.

        Parámetros:
        - pipe: Pipeline de sklearn.
        - model_name: Nombre del modelo.
        - X: Datos de entrada.
        - y: Etiquetas.

        Retorna:
        - Modelo entrenado.

        """
        # Definir métricas de evaluación
        scoring = {
            'precision': make_scorer(precision_score, average='binary'),
            'sensitivity': make_scorer(recall_score, average='binary'),
            'accuracy': make_scorer(accuracy_score),
            'specificity': make_scorer(recall_score, average='binary', pos_label=0)
        }

        # Definir validación cruzada
        cv_hiper = StratifiedShuffleSplit(n_splits=50, test_size=0.25, random_state=27)

        # Primero: RandomizedSearchCV
        param_space_random = get_random_params(model_name)
        randomized_search = RandomizedSearchCV(pipe[model_name], param_distributions=param_space_random, n_iter=100, scoring=scoring, cv=cv_hiper, n_jobs=-1, refit=lambda x: self.refit_strategy(x, 'Aleatoria'), random_state=27)
        randomized_result = randomized_search.fit(X, y)

        # Segundo: GridSearchCV con los mejores hiperparámetros
        param_space_grid = get_params_grid(randomized_result, model_name)
        grid_search = GridSearchCV(pipe[model_name], param_grid=param_space_grid, scoring=scoring, cv=cv_hiper, n_jobs=-1, refit=lambda x: self.refit_strategy(x, 'Cuadricula'))
        grid_result = grid_search.fit(X, y)

        # Guardar el modelo
        save_object(self.model_trainer_config.trained_model_path, grid_result)

        return grid_result

    
    def train(self) -> None:
        """Entrena el modelo."""
        try:
            # Definir directorios
            DATA_DIRECTORY_OCT = os.path.join('data', 'BBDD_Original', 'DatosOCT', 'Datos Pacientes v13.6_anonimizado_OCT.xlsx')
            DATA_DIRECTORY_RET = os.path.join('data', 'BBDD_Original', 'DatosRET', 'DatosPacientes_01_04_RET_dpp.mat')
            DATA_OUTPUT = os.path.join('data', 'BBDD_Nueva')

            # Crear objetos DataIngestion y procesar los datos
            processor_oct = OCTDataProcessor(DATA_DIRECTORY_OCT, DATA_OUTPUT)
            processor_oct.process_data()

            processor_ret = RETDataProcessor(DATA_DIRECTORY_RET, DATA_OUTPUT)
            processor_ret.process_data()

            # Cargar los datos
            df_oct = pd.read_csv(os.path.join(DATA_OUTPUT, 'Datos_SECT_OCT.csv'), index_col=0)
            df_ret = pd.read_csv(os.path.join(DATA_OUTPUT, 'Datos_SECT_RET.csv'), index_col=0)

            # Calcular las métricas de asimetría
            oct_metrics = DELTA_METRICS.calculate_all(df_oct)
            ret_metrics = DELTA_METRICS.calculate_all(df_ret)

            # Unir los datos de OCT y RET
            df = pd.concat([oct_metrics, ret_metrics], axis=1)

            
            # Definir el pipeline
            pipe = {
                'DecisionTree': DecisionTreeClassifier(random_state=27),
                'RandomForest': RandomForestClassifier(n_jobs=-1, random_state=27, oob_score=True)
            }

            # Entrenar el modelo
            for model_name in pipe:
                self.train_model(pipe, model_name, df, df['Glaucoma'])

            logger.info("Modelo entrenado exitosamente.")

        except Exception as e:
            logger.error("Error al entrenar el modelo.", exc_info=True)
            raise CustomException("Error al entrenar el modelo.", e, sys.exc_info()[2])
        


            








        

    

