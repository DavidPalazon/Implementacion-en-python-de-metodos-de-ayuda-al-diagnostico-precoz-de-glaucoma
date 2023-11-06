import pandas as pd
from pathlib import Path
import sys
from src.exception import CustomException
from src.logger import logger

class OCTDataProcessor:
    """Clase que representa un procesador de datos de OCT."""
    def __init__(self, data_directory_oct: Path, data_output: Path) -> None:
        """Constructor de la clase.
        
        Parámetros:
        - data_directory_oct: Ruta del archivo Excel con los datos.
        - data_output: Ruta de salida para guardar los datos procesados.
        
        """
        self.data_directory_oct = data_directory_oct
        self.data_output = data_output
        self.data = None
    
    def load_data(self) -> None:
        """Carga los datos desde el archivo Excel.

        Retorna:
        - DataFrame con los datos cargados.

        """
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

    
    def transpose_data(self) -> None:
        """Transpone el DataFrame para tener a los pacientes como índice.

        Retorna:
        - DataFrame transpuesto.
        
        """
        
        self.data = self.data.transpose()
        self.data.index.name = 'Paciente'
    
    def convert_multilevel_columns(self) -> None:
        """Convierte las columnas multinivel en una sola columna.

        Retorna:
        - DataFrame con columnas multinivel convertidas.

        """
        columnas_sin_multinivel = ['_'.join(filter(pd.notna, col)).strip() for col in self.data.columns]
        columnas_sin_multinivel[-1], columnas_sin_multinivel[-2] = 'CV (DM)_OI', 'CV (DM)_OD'
        self.data.columns = columnas_sin_multinivel
    
    def convert_date_column(self) -> None:
        """Convierte la columna 'Fecha nacimiento (dd/mm/aaaa)' a formato de fecha.
        
        Retorna:
        - DataFrame con la columna 'Fecha nacimiento (dd/mm/aaaa)' convertida.
        """
        self.data['Fecha nacimiento (dd/mm/aaaa)'] = pd.to_datetime(self.data['Fecha nacimiento (dd/mm/aaaa)'], format='%Y/%m/d', errors='coerce')
    
    def create_pio_columns(self) -> None:
        """Crea columnas para PIO Neumático y PIO Perkins combinando los datos del mismo paciente para cada ojo.
            El objetivo es tener una sola columna para PIO OD y otra para PIO OI, para posteriormente crear dos columnas que definan
            si el paciente tiene PIO Neumático y/o PIO Perkins.

        
        Retorna:
        - DataFrame con las columnas creadas.
        """
        self.data['PIO_OD'] = self.data['PIO Neumático_OD'].combine_first(self.data['PIO Perkins_OD'])
        self.data['PIO_OI'] = self.data['PIO Neumático_OI'].combine_first(self.data['PIO Perkins_OI'])
    
    def create_indicator_columns(self) -> None:
        """Crea columnas indicadoras para PIO Neumático y PIO Perkins.
            El objetivo es tener una columna que indique si el paciente tiene PIO Neumático y otra que indique si tiene PIO Perkins.

        Retorna:
        - DataFrame con las columnas creadas.
        """
        self.data['PIO_Neumatico'] = (self.data[['PIO Neumático_OD', 'PIO Neumático_OI']].notna().any(axis=1)).astype(int)
        self.data['PIO_Perkins'] = (self.data[['PIO Perkins_OD', 'PIO Perkins_OI']].notna().any(axis=1)).astype(int)
    
    def clean_data(self) -> None:
        """Limpia el DataFrame eliminando columnas innecesarias y manejando la columna 'Glaucoma'.

        Retorna:
        - DataFrame limpio.
        """
        cols_to_drop = ['PIO Neumático_OD', 'PIO Neumático_OI', 'PIO Perkins_OD', 'PIO Perkins_OI'] 
        cols_to_drop += [col for col in self.data.columns if "Excavación Papilar" in col or "CV" in col]
        self.data.drop(columns=cols_to_drop, inplace=True)
        self.data['Glaucoma'] = self.data['Glaucoma'].str.contains('Sí', na=False).astype(int)
        # Limpiar el índice 'Paciente'
        self.data.index = self.data.index.str.replace('#', '').str.slice(stop=3).astype(int)
    
    def convert_selected_columns(self) -> None:
        """Convierte las columnas seleccionadas a numéricas y reemplaza los NaNs por la media de los pacientes con y sin glaucoma.

        Retorna:
        - DataFrame con las columnas convertidas y NaNs reemplazados.
        """
        columns_to_convert = ['Paquimetría_OD', 'Paquimetría_OI', 'Longitud axial_OD', 
                              'Longitud axial_OI', 'PIO_OD', 'PIO_OI',
                              'Oct N Óptico_TS_OD', 'Oct N Óptico_T _OD', 'Oct N Óptico_TI_OD',
                              'Oct N Óptico_NS_OD', 'Oct N Óptico_N _OD', 'Oct N Óptico_NI_OD',
                              'Oct N Óptico_G_OD', 'Oct N Óptico_TS_OI', 'Oct N Óptico_T _OI',
                              'Oct N Óptico_TI_OI', 'Oct N Óptico_NS_OI', 'Oct N Óptico_N _OI',
                              'Oct N Óptico_NI_OI', 'Oct N Óptico_G_OI']
        for col in columns_to_convert:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            self.data[col].fillna(self.data.groupby('Glaucoma')[col].transform('most_frequent'), inplace=True)
    
    def clean_column_names(self) -> None:
        """Limpia los nombres de las columnas.

        Retorna:
        - DataFrame con los nombres de las columnas limpios.
        """
        self.data.columns = self.data.columns.str.replace(' ', '')
        self.data.rename(columns=lambda x: x.replace('OctNÓptico_', ''), inplace=True)
        self.data.rename(columns={'Fáquico(P)/Pseudofáquico(PSQ)_OD': 'Faq_OD', 'Fáquico(P)/Pseudofáquico(PSQ)_OI': 'Faq_OI'}, inplace=True)
    
    def save_data(self) -> None:
        """Guarda el DataFrame procesado en un archivo CSV."""

        self.data.to_csv(self.data_output / 'Datos_SECT_OCT.csv')
        logger.info(f'Datos procesados y guardados exitosamente en {self.data_output / "Datos_SECT_OCT.csv"}')
    
    def process_data(self) -> None:
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


