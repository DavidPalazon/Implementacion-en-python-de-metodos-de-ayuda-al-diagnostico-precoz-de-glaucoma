import os
from dataclasses import dataclass
import sys

import pandas as pd
import numpy as np
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

from src.logger import logger
from src.exception import CustomException  # Asegúrate de que esta excepción esté definida en tu código.


@dataclass
class PreprocessorConfig:
    """Clase que representa la configuración del preprocesador."""
    preprocessor_path: str = os.path.join('models', 'preprocessor.pkl')


class Preprocessor:
    """Clase que representa un preprocesador de datos."""

    def __init__(self, config: PreprocessorConfig = PreprocessorConfig()) -> None:
        """Constructor de la clase.

        Parámetros:
        - config: Configuración del preprocesador.

        """
        self.preprocessor_config = config
        logger.info("Preprocessor initialized with config: %s", config)

    def preprocessing_pipelines(self) -> object:
        """Crea el pipeline de preprocesamiento.

        Retorna:
        - Pipeline de preprocesamiento.

        """
        try:
            smt = SMOTE(random_state=27)
            clf_dt = DecisionTreeClassifier(random_state=27)
            clf_rf = RandomForestClassifier(n_jobs=-1, random_state=27, oob_score=True)
            SFM = SelectFromModel(estimator=clf_rf)

            pipeline_dt = make_pipeline(smt, SFM, clf_dt)
            pipeline_rf = make_pipeline(smt, clf_rf)
            pipeline = {'DecisionTree': pipeline_dt, 'RandomForest': pipeline_rf}

            logger.info("Preprocesador creado correctamente.")
            return pipeline
        
        except Exception as e:

            logger.error("Error al crear el preprocesador %s", str(e))
            raise CustomException("Error al crear el preprocesador.", e, sys.exc_info()[2])
        




