import pandas as pd
from pathlib import Path
import sys
from src.exception import CustomException
from src.logger import logger

class OCTDataProcessor:
    def __init__(self, data_directory_oct, data_output):
        self.data_directory_oct = data_directory_oct
        self.data_output = data_output
        self.data = None
    
    def load_data(self):
        try:
            # Load the Excel file into a DataFrame
            self.data = pd.read_excel(self.data_directory_oct, index_col=[0, 1, 2])
            logger.info(f'Datos cargados exitosamente desde {self.data_directory_oct}')
            
        except FileNotFoundError as e:
            error_msg = f'Archivo no encontrado: {self.data_directory_oct}'
            logger.error(error_msg)
            raise CustomException(error_msg, e, sys.exc_info()[2])
            
        except Exception as e:
            error_msg = f'Ocurrió un error al cargar los datos: {e}'
            logger.error(error_msg, exc_info=True)
            raise CustomException(error_msg, e, sys.exc_info()[2])

    
    def transpose_data(self):
        # Transponer el DataFrame para tener a los pacientes como índice
        self.data = self.data.transpose()
        self.data.index.name = 'Paciente'
    
    def convert_multilevel_columns(self):
        # Convertir columnas multi-nivel a columnas de un solo nivel
        columnas_sin_multinivel = ['_'.join(filter(pd.notna, col)).strip() for col in self.data.columns]
        columnas_sin_multinivel[-1], columnas_sin_multinivel[-2] = 'CV (DM)_OI', 'CV (DM)_OD'
        self.data.columns = columnas_sin_multinivel
    
    def convert_date_column(self):
        # Convertir la columna de fecha a tipo datetime
        self.data['Fecha nacimiento (dd/mm/aaaa)'] = pd.to_datetime(self.data['Fecha nacimiento (dd/mm/aaaa)'], format='%Y/%m/d', errors='coerce')
    
    def create_pio_columns(self):
        # Crear nuevas columnas combinando PIO Neumático y PIO Perkins
        self.data['PIO_OD'] = self.data['PIO Neumático_OD'].combine_first(self.data['PIO Perkins_OD'])
        self.data['PIO_OI'] = self.data['PIO Neumático_OI'].combine_first(self.data['PIO Perkins_OI'])
    
    def create_indicator_columns(self):
        # Crear columnas indicadoras basadas en la disponibilidad de datos
        self.data['PIO_Neumatico'] = (self.data[['PIO Neumático_OD', 'PIO Neumático_OI']].notna().any(axis=1)).astype(int)
        self.data['PIO_Perkins'] = (self.data[['PIO Perkins_OD', 'PIO Perkins_OI']].notna().any(axis=1)).astype(int)
    
    def clean_data(self):
        # Eliminar columnas innecesarias y manejar la columna 'Glaucoma'
        cols_to_drop = ['PIO Neumático_OD', 'PIO Neumático_OI', 'PIO Perkins_OD', 'PIO Perkins_OI'] 
        cols_to_drop += [col for col in self.data.columns if "Excavación Papilar" in col or "CV" in col]
        self.data.drop(columns=cols_to_drop, inplace=True)
        self.data['Glaucoma'] = self.data['Glaucoma'].str.contains('Sí', na=False).astype(int)
        # Limpiar el índice 'Paciente'
        self.data.index = self.data.index.str.replace('#', '').str.slice(stop=3).astype(int)
    
    def convert_selected_columns(self):
        # Convertir columnas seleccionadas y reemplazar NaNs
        columns_to_convert = ['Paquimetría_OD', 'Paquimetría_OI', 'Longitud axial_OD', 
                              'Longitud axial_OI', 'PIO_OD', 'PIO_OI',
                              'Oct N Óptico_TS_OD', 'Oct N Óptico_T _OD', 'Oct N Óptico_TI_OD',
                              'Oct N Óptico_NS_OD', 'Oct N Óptico_N _OD', 'Oct N Óptico_NI_OD',
                              'Oct N Óptico_G_OD', 'Oct N Óptico_TS_OI', 'Oct N Óptico_T _OI',
                              'Oct N Óptico_TI_OI', 'Oct N Óptico_NS_OI', 'Oct N Óptico_N _OI',
                              'Oct N Óptico_NI_OI', 'Oct N Óptico_G_OI']
        for col in columns_to_convert:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col].fillna(self.data.groupby('Glaucoma')[col].transform('mean'), inplace=True)
    
    def clean_column_names(self):
        # Eliminar espacios en blanco en los nombres de las columnas
        self.data.columns = self.data.columns.str.replace(' ', '')
        # Cambiar las columnas 'OctNÓptico' para mostrar sólo el indicador del sector (T, TS, TI, etc.) y el ojo (OD y OS)
        self.data.rename(columns=lambda x: x.replace('OctNÓptico_', ''), inplace=True)
        # Cambiar las columnas de 'Fáquico' y 'Pseudofáquico' para mostrar sus abreviaturas y el ojo correspondiente
        self.data.rename(columns={'Fáquico(P)/Pseudofáquico(PSQ)_OD': 'Faq_OD', 'Fáquico(P)/Pseudofáquico(PSQ)_OI': 'Faq_OI'}, inplace=True)
    
    def save_data(self):
        # Guardar el DataFrame procesado en un archivo CSV
        self.data.to_csv(self.data_output / 'Datos_SECT_OCT.csv')
        logger.info(f'Datos procesados y guardados exitosamente en {self.data_output / "Datos_SECT_OCT.csv"}')
    
    def process_data(self):
        self.load_data()
        self.transpose_data()
        self.convert_multilevel_columns()
        self.convert_date_column()
        self.create_pio_columns()
        self.create_indicator_columns()
        self.clean_data()
        self.convert_selected_columns()
        self.clean_column_names()
        self.save_data()

# Definir la ruta del archivo Excel
DATA_DIRECTORY_OCT = Path('data', 'BBDD_Original', 'DatosOCT', 'Datos Pacientes v13.6_anonimizado_OCT.xlsx')
DATA_OUTPUT = Path('data', 'BBDD_Nueva')

processor = OCTDataProcessor(DATA_DIRECTORY_OCT, DATA_OUTPUT)
processor.process_data()

