import pandas as pd
import pytest
from src.components.data_transformation import DELTA_METRICS, calculate_delta
from src.exception import CustomException
from src.logger import logger

# Cargar el archivo de base como una fixture
@pytest.fixture
def base_data():
    df = pd.read_csv('data/BBDD_Nueva/Datos_SECT_OCTRET.csv')
    columns = [col for col in df.columns if 'OCT' in col]
    df = df[columns]
    df = df.rename(columns=lambda x: x.replace('_OCT', ''))
    return df

# Función de test para cada métrica en DELTA_METRICS
@pytest.mark.parametrize('metric_name', DELTA_METRICS.keys())
def test_delta_metrics(metric_name, base_data):
    # Carga el dataframe esperado
    df_expected = pd.read_csv(f'data/BBDD_Nueva/Datos_SECT_OCTRET_{metric_name.lower()}.csv')
    
    # Filtra y renombra las columnas del archivo esperado
    # Renombra las columnas del archivo esperado
    columns = [col for col in df_expected.columns if 'OCT' in col]
    df_expected = df_expected[columns]
    rename_columns = {col: col.replace('_OCT', '').replace(metric_name.lower() + '_', '') for col in df_expected.columns}
    df_expected.rename(columns=rename_columns, inplace=True)


    # Registra un mensaje indicando que el archivo de datos esperado se cargó con éxito
    logger.info(f'Archivo de datos esperados cargado exitosamente para la métrica {metric_name.lower()}')

    # Call the metric function with the base data
    result = DELTA_METRICS[metric_name](base_data)
    columns = [col.replace(metric_name.lower() + '_', '') for col in result.columns]
    result.columns = columns

    
    print(result)
    # Check that the result is equal to the expected DataFrame
    pd.testing.assert_frame_equal(result, df_expected, check_dtype=False)
