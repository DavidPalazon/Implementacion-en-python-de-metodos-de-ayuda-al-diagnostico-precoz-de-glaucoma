import os
import sys

import pandas as pd
import numpy as np
import dill


from pathlib import Path

from src.exception import CustomException
from src.logger import logger

def save_object(file_path: Path, obj: object) -> None:
    """Guarda un objeto en un archivo .pkl.

    Parámetros:
    - file_path: Ruta del archivo.
    - obj: Objeto a guardar.

    """

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
            logger.info(f'Objeto guardado exitosamente en {file_path}')
    
    except Exception as e:
        logger.error(f'Ocurrió un error al guardar el objeto: {e}', exc_info=True)
        raise CustomException(f'Error al guardar el objeto en {file_path}', e, sys.exc_info()[2])
    
def load_object(file_path: Path) -> object:
    """Carga un objeto desde un archivo .pkl.

    Parámetros:
    - file_path: Ruta del archivo.

    Retorna:
    - Objeto cargado desde el archivo.
    """
    try:
        with open(file_path, 'rb') as f:
            obj = dill.load(f)
            logger.info(f'Objeto cargado exitosamente desde {file_path}')
            return obj
    except Exception as e:
        logger.error(f'Ocurrió un error al cargar el objeto: {e}', exc_info=True)
        raise CustomException(f'Error al cargar el objeto desde {file_path}', e, sys.exc_info()[2])


def get_random_params(model_name: str) -> list:
    """Obtiene una lista de parámetros aleatorios para el pipeline dado.

    Parámetros:
    - model_name: Nombre del modelo.
    
    Retorna:
    - Lista de parámetros aleatorios.
    """
    if model_name == 'DecisionTree':
        return [
            {
                "smote__sampling_strategy": [float(x) for x in np.arange(0.4, 1.0, 0.1)],
                "smote__k_neighbors": [2, 3, 4, 5, 6],
                "selectfrommodel__max_features": [2, 3, 4, 5, 6],
                "decisiontreeclassifier__criterion": ['gini', 'entropy', 'log_loss'],
                "decisiontreeclassifier__max_depth": [int(x) for x in np.arange(1, 7)],
                "decisiontreeclassifier__min_samples_split": [float(x) for x in np.arange(0.05, 0.3, 0.05)],
                "decisiontreeclassifier__splitter": ['best', 'random'],
                "decisiontreeclassifier__min_samples_leaf": [1, 5, 10, 15, 20],
                "decisiontreeclassifier__class_weight": [None, 'balanced']

            }
        ]
    elif model_name == 'RandomForest':
        return [
            {
                "smote__sampling_strategy": [float(x) for x in np.arange(0.5, 1.1, 0.1)],
                "smote__k_neighbors": [3, 4, 5, 6],
                "randomforestclassifier__n_estimators": [int(x) for x in np.linspace(100, 500, 10)],
                "randomforestclassifier__criterion": ['gini', 'entropy', 'log_loss'],
                "randomforestclassifier__min_samples_split": [float(x) for x in np.arange(0.05, 0.3, 0.05)],
                "randomforestclassifier__min_samples_leaf": [1, 5, 10, 15, 20],
                "randomforestclassifier__max_features": ['sqrt', 'log2'],
                "randomforestclassifier__class_weight": [None, 'balanced', 'balanced_subsample'],
                "randomforestclassifier__bootstrap": [True],
                "randomforestclassifier__oob_score": [True],
                "randomforestclassifier__warm_start": [True, False],
                "randomforestclassifier__max_samples": [0.5, 0.8, 1.0],
                "randomforestclassifier__max_depth": [int(x) for x in np.arange(1, 7)]
            }
        ]
    else:
        raise ValueError(f"{model_name} : Modelo no sujeto a optimización.")


def get_params_grid(puntuaciones, model_name: str) -> list:
    """Obtiene una lista de parámetros para el pipeline dado.

    Parámetros:
    - model_name: Nombre del modelo.

    Retorna:
    - Lista de parámetros.
    """
    

    if  model_name == 'DecisionTree':
        return [
            {
                "smote__sampling_strategy": [puntuaciones.best_params_["smote__sampling_strategy"] - 0.1, 
                                            puntuaciones.best_params_["smote__sampling_strategy"],
                                            puntuaciones.best_params_["smote__sampling_strategy"] + 0.1],
                "smote__k_neighbors": [puntuaciones.best_params_["smote__k_neighbors"]],
                "selectfrommodel__max_features": [puntuaciones.best_params_['selectfrommodel__max_features']],
                "decisiontreeclassifier__criterion": ['entropy', 'gini', 'log_loss'],
                "decisiontreeclassifier__max_depth": [puntuaciones.best_params_["decisiontreeclassifier__max_depth"] - 1,
                                                    puntuaciones.best_params_["decisiontreeclassifier__max_depth"],
                                                    puntuaciones.best_params_["decisiontreeclassifier__max_depth"] + 1],
                "decisiontreeclassifier__min_samples_split": [puntuaciones.best_params_["decisiontreeclassifier__min_samples_split"] - 0.05,
                                                            puntuaciones.best_params_["decisiontreeclassifier__min_samples_split"],
                                                            puntuaciones.best_params_["decisiontreeclassifier__min_samples_split"] + 0.05],
                "decisiontreeclassifier__splitter": [puntuaciones.best_params_["decisiontreeclassifier__splitter"]],
                "decisiontreeclassifier__min_samples_leaf": [puntuaciones.best_params_["decisiontreeclassifier__min_samples_leaf"]],
                "decisiontreeclassifier__class_weight": [puntuaciones.best_params_["decisiontreeclassifier__class_weight"]]
            }
        ]
    elif model_name == 'RandomForest':
        return [
            {
            "smote__sampling_strategy" : [puntuaciones.best_params_["smote__sampling_strategy"] - 0.1 , puntuaciones.best_params_["smote__sampling_strategy"] , puntuaciones.best_params_["smote__sampling_strategy"] +0.1],
            "smote__k_neighbors" : [puntuaciones.best_params_["smote__k_neighbors"]],
            "randomforestclassifier__n_estimators": [puntuaciones.best_params_["randomforestclassifier__n_estimators"]],
            "randomforestclassifier__criterion": ['gini', 'entropy', 'log_loss'],
            "randomforestclassifier__min_samples_split" : [puntuaciones.best_params_['randomforestclassifier__min_samples_split']],
            "randomforestclassifier__min_samples_leaf" : [puntuaciones.best_params_['randomforestclassifier__min_samples_leaf']-5,puntuaciones.best_params_['randomforestclassifier__min_samples_leaf'],puntuaciones.best_params_['randomforestclassifier__min_samples_leaf']+5 ],
            "randomforestclassifier__max_features" : [puntuaciones.best_params_['randomforestclassifier__max_features']],
            "randomforestclassifier__class_weight": [puntuaciones.best_params_['randomforestclassifier__class_weight']],
            "randomforestclassifier__bootstrap": [True],
            "randomforestclassifier__oob_score": [True],
            "randomforestclassifier__warm_start": [puntuaciones.best_params_['randomforestclassifier__warm_start']],
            "randomforestclassifier__max_samples": [puntuaciones.best_params_['randomforestclassifier__max_samples']-0.1,puntuaciones.best_params_['randomforestclassifier__max_samples'],puntuaciones.best_params_['randomforestclassifier__max_samples']+0.1],
            "randomforestclassifier__max_depth": [puntuaciones.best_params_['randomforestclassifier__max_depth']-1,puntuaciones.best_params_['randomforestclassifier__max_depth'],puntuaciones.best_params_['randomforestclassifier__max_depth']+1]
            }
        ]
    else:
        raise ValueError(f"{model_name} : Modelo no sujeto a optimización.")