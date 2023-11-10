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
    trained_model_path: str = os.path.join('models') # Carpeta donde se guardarán los modelos entrenados.



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

    
    def best_model(self, cv_results, search) -> int:
        # Filtrar los modelos con la precision mayor a 0.5
        high_precision_cv_results = pd.DataFrame(cv_results)[pd.DataFrame(cv_results)["mean_test_precision"] > 0.5]

        # Filtrar los modelos con la sensibilidad media mayor a la máxima media menos la máxima desviación estándar
        best_recall_models = high_precision_cv_results[high_precision_cv_results["mean_test_sensitivity"] > high_precision_cv_results["mean_test_sensitivity"].max() - high_precision_cv_results["std_test_sensitivity"].max()]

        # Identificar el mejor modelo por su sensibilidad (máxima media y mínima desviación estándar)
        best_model_index = best_recall_models["std_test_sensitivity"].idxmin()

        self.save_best_results(best_recall_models.loc[best_model_index], search)
        return best_model_index

    def train_model(self, pipe ,model_name: str,metrica: str, X, y) -> object:
        """Entrena un modelo de clasificación.

        Parámetros:
        - pipe: Pipeline de sklearn.
        - model_name: Nombre del modelo.
        - metrica: Nombre de la métrica de asimetría utilizada.
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
        cv_hiper = StratifiedShuffleSplit(n_splits=100, test_size=0.25, random_state=27)

        # Primero: RandomizedSearchCV
        param_space_random = get_random_params(model_name)
        randomized_search = RandomizedSearchCV(pipe[model_name], param_distributions=param_space_random, n_iter=200, scoring=scoring, cv=cv_hiper, n_jobs=-1, refit=lambda x: self.best_model(x, 'Aleatoria'), random_state=27)
        randomized_result = randomized_search.fit(X, y)

        # Segundo: GridSearchCV con los mejores hiperparámetros
        param_space_grid = get_params_grid(randomized_result, model_name)
        grid_search = GridSearchCV(pipe[model_name], param_grid=param_space_grid, scoring=scoring, cv=cv_hiper, n_jobs=-1, refit=lambda x: self.best_model(x, 'Cuadricula'))
        grid_result = grid_search.fit(X, y)

        # Guardar el modelo entrenado
        save_object(f'{self.model_trainer_config.trained_model_path}/{model_name}_{metrica}_model.pkl', grid_result) 

        return grid_result

    
    def train(self) -> dict:
        """Entrena el modelo.
        
        Retorna:
        - Diccionario con los modelos entrenados.
        """
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
            oct_metrics: dict = DELTA_METRICS.calculate_all(df_oct)
            ret_metrics: dict = DELTA_METRICS.calculate_all(df_ret)

            # Definir el pipeline
            pipe = {
                'DecisionTree': DecisionTreeClassifier(random_state=27),
                'RandomForest': RandomForestClassifier(n_jobs=2, random_state=27, oob_score=True)
            }

            # Lista para almacenar los modelos entrenados
            trained_models = dict()

            # Unir los datos de OCT y RET
            for key_oct, key_ret in zip(oct_metrics.keys(), ret_metrics.keys()):
                df_oct = oct_metrics[key_oct]
                df_ret = ret_metrics[key_ret]
                df_octret = pd.merge(df_oct, df_ret, on=['Paciente', 'Glaucoma'], how='inner', suffixes=('_OCT', '_RET'))

                # Entrenar el modelo
                for model_name in pipe:
                    trained_model = self.train_model(pipe, model_name, key_oct, df_oct, df_oct['Glaucoma'])
                    #Almacena los resultados en un diccionario con la clave del modelo, la métrica de asimetría, el path del modelo y el modelo entrenado sin utilizar append
                    trained_models[(f'{self.model_trainer_config.trained_model_path}/{model_name}_{key_oct}_OCT.pkl', trained_model)]
                    logger.info("Modelo entrenado exitosamente para OCT.")
                    
                    trained_model = self.train_model(pipe, model_name, key_ret, df_ret, df_ret['Glaucoma'])
                    trained_models[(f'{self.model_trainer_config.trained_model_path}/{model_name}_{key_ret}_RET.pkl', trained_model)]
                    logger.info("Modelo entrenado exitosamente para RET.")
                    
                    if key_oct == key_ret:
                        key_octret = key_oct
                        trained_model = self.train_model(pipe, model_name, f'{key_oct}_{key_ret}', df_octret, df_octret['Glaucoma'])
                        trained_models[(f'{self.model_trainer_config.trained_model_path}/{model_name}_{key_octret}_OCTRET.pkl', trained_model)]
                        logger.info("Modelo entrenado exitosamente para OCT-RET.")
                    
            logger.info("Modelos entrenados exitosamente.")
            return trained_models

        except Exception as e:
            logger.error("Error al entrenar el modelo.", exc_info=True)
            raise CustomException("Error al entrenar el modelo.", e, sys.exc_info()[2])


if __name__ == "__main__":
    model_trainer = ModelTrainer()
    model_trainer.train()









        

    

