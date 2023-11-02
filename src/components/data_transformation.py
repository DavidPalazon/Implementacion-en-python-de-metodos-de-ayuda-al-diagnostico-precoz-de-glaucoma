import pandas as pd
import numpy as np
from src.logger import logger
from src.exception import CustomException

class DeltaMetric:
    """
    Clase que representa una métrica de asimetría entre sectores OD y OS.
    """

    def __init__(self, name, operation):
        """
        Constructor de la clase.

        Parámetros:
        - name: Nombre de la métrica.
        - operation: Función lambda que define la operación a realizar.
        """
        self.name = name
        self.operation = operation

    def calculate(self, df):
        """
        Calcula la métrica de asimetría para el DataFrame dado.

        Parámetros:
        - df: DataFrame con los datos.

        Retorna:
        - DataFrame con las métricas de asimetría.
        """
        try:
            new_df = pd.DataFrame()  # Resultados
            sectores = ['TS', 'T', 'TI', 'G', 'NS', 'NI', 'N']  # Sectores específicos

            # Filtrar columnas por sectores y ojo
            columnas_OD = [col for col in df.columns if '_OD' in col and col.split('_')[0] in sectores]
            columnas_OS = [col for col in df.columns if '_OS' in col and col.split('_')[0] in sectores]

            # Verificar que hay la misma cantidad de columnas OD y OS
            if len(columnas_OD) != len(columnas_OS):
                raise ValueError("La cantidad de columnas OD y OS no coincide")

            # Media de los sectores para cada ojo
            df_OD_media = df[columnas_OD].mean()
            df_OS_media = df[columnas_OS].mean()

            # Calcular asimetría para cada sector
            for c1, c2 in zip(columnas_OD, columnas_OS):
                result = self.operation(df, c1, c2, df_OD_media, df_OS_media)
                new_df[c1.split("_")[0]] = result

            logger.info(f'Métrica {self.name} calculada exitosamente')
            return new_df

        except Exception as e:
            logger.error(f'Error calculando la métrica {self.name}: {e}', exc_info=True)
            raise CustomException(f"Error al calcular la métrica {self.name}")

class DeltaMetricCollection:
    """
    Clase que representa una colección de métricas de asimetría.
    """

    def __init__(self, metrics):
        """
        Constructor de la clase.

        Parámetros:
        - metrics: Lista de objetos DeltaMetric.
        """
        self.metrics = metrics

    def calculate_all(self, df):
        """
        Calcula todas las métricas de asimetría para el DataFrame dado.

        Parámetros:
        - df: DataFrame con los datos.

        Retorna:
        - Diccionario con los nombres de las métricas como claves y los DataFrames con las métricas de asimetría como valores.
        """
        result = {}
        for metric in self.metrics:
            result[metric.name] = metric.calculate(df)
        return result

# Definiciones de operaciones para cada métrica
operations = {
    'Delta_A': lambda df, c1, c2, _, __: df[c1] - df[c2],
    'Delta_B': lambda df, c1, c2, _, __: abs(df[c1] - df[c2]),
    'Delta_C': lambda df, c1, c2, _, __: (df[c1] - df[c2]) / (df[c1] + df[c2]),
    'Delta_D': lambda df, c1, c2, _, __: abs(df[c1] - df[c2]) / (df[c1] + df[c2]),
    'Delta_E': lambda df, c1, c2, df_OD_media, df_OS_media: (df[c1] - df[c2]) / (df_OD_media[c1] + df_OS_media[c2]),
    'Delta_F': lambda df, c1, c2, df_OD_media, df_OS_media: abs(df[c1] - df[c2]) / (df_OD_media[c1] + df_OS_media[c2]),
    'Delta_G': lambda df, c1, c2, _, __: (df[c1] - df[c2]) / (df['G_OD'] + df['G_OS']),
    'Delta_H': lambda df, c1, c2, _, __: abs(df[c1] - df[c2]) / (df['G_OD'] + df['G_OS']),
    'Delta_I': lambda df, c1, c2, _, __: np.sqrt(abs(df[c1] - df[c2]) / (df[c1] + df[c2])),
    'Delta_J': lambda df, c1, c2, _, __: ((df[c1] + df['G_OD']) / df['G_OS']) - ((df[c2] + df['G_OS']) / df['G_OD']),
    'Delta_K': lambda df, c1, c2, _, __: abs(((df[c1] + df['G_OD']) / df['G_OS']) - ((df[c2] + df['G_OS']) / df['G_OD'])),
    'Delta_L': lambda df, c1, c2, _, __: np.sqrt(abs(((df[c1] + df['G_OD']) / df['G_OS']) - ((df[c2] + df['G_OS']) / df['G_OD'])))
}

# Crear objetos DeltaMetric a partir de las definiciones de operaciones
metrics = [DeltaMetric(name, op) for name, op in sorted(operations.items())]

# Crear objeto DeltaMetricCollection a partir de la lista de objetos DeltaMetric
DELTA_METRICS = DeltaMetricCollection(metrics)

