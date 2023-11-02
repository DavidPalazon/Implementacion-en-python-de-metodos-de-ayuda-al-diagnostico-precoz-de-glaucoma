import pandas as pd
import sys
from pathlib import Path
import scipy.io as sc
from src.exception import CustomException
from src.logger import logger

class RETDataProcessor:
    def __init__(self, data_directory, data_output):
        self.data_directory = data_directory
        self.data_output = data_output

    def load_mat_file(self, file_path):
        try:
            mat = sc.loadmat(file_path)
            mat = {k: v for k, v in mat.items() if k[0] != '_'}
            df = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
            return df

        except Exception as e:
            logger.error(f'Ocurrió un error al convertir .mat a DataFrame: {e}')
            raise CustomException("Error al convertir .mat a DataFrame", e, sys.exc_info()[2])

    def process_data(self):
        try:
            # Cargar y procesar el conjunto de datos Datos_SECT_RET
            datos_sect_ret = self.load_mat_file(self.data_directory)
            datos_sect_ret.set_index('pacientes', inplace=True)
            datos_sect_ret.index.name = 'paciente'
            order_cols = [f'BBDDret_{i}' for i in range(1, 15)]
            datos_sect_ret = datos_sect_ret[order_cols]
            new_columns = ['TS_OD', 'T_OD', 'TI_OD', 'NS_OD', 'N_OD', 'NI_OD', 'G_OD', 
                           'TS_OS', 'T_OS', 'TI_OS', 'NS_OS', 'N_OS', 'NI_OS', 'G_OS']
            datos_sect_ret.columns = new_columns
            datos_sect_ret = datos_sect_ret.loc[~(datos_sect_ret == 0).all(axis=1)]

            # Guardar a CSV
            datos_sect_ret.to_csv(self.data_output / 'Datos_SECT_RET.csv')
            logger.info('Datos RET procesados y guardados exitosamente en CSV')

        except FileNotFoundError:
            logger.error(f'Archivo no encontrado: {self.data_directory}')
            raise CustomException("No se encontró el archivo especificado", FileNotFoundError, sys.exc_info()[2])

        except Exception as e:
            logger.error(f'Ocurrió un error: {e}', exc_info=True)
            raise CustomException("Error al procesar los datos", e, sys.exc_info()[2])

# Definir directorios
DATA_DIRECTORY_RET = Path('data', 'BBDD_Original', 'DatosRET','DatosPacientes_01_04_RET_dpp.mat')
DATA_OUTPUT = Path('data', 'BBDD_Nueva')

# Crear objeto DataIngestion y procesar los datos
processor = RETDataProcessor(DATA_DIRECTORY_RET, DATA_OUTPUT)
processor.process_data()

    