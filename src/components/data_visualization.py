import sys
import os
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from datetime import datetime
from src.logger import logger, results_logger
from src.exception import CustomException
from typing import List, Dict, Union, Tuple, Any

# Estableciendo configuraciones globales para visualizaciones
plt.rcParams['figure.figsize'] = (20, 20)
sb.set_style('whitegrid')
sb.set_context('paper', font_scale=1.5)

class Visualization:
    """Clase encargada de visualizar datos a partir de un DataFrame"""

    def __init__(self, df: pd.DataFrame) -> None:
        """Inicializador de la clase Visualization

        Args:
            df (pd.DataFrame): DataFrame con los datos a visualizar.
        """
        self.df = df

    def _preprocess_pairplot_data(self) -> pd.Series:
        """Preprocesa los datos para el pairplot"""

        self.df = self.df.rename(columns={'glaucoma': 'Diagnostico'})
        replacements = {0: 'Sano', 1: 'Glaucoma'}
        return self.df['Diagnostico'].replace(replacements)

    def pinta_pairplot(self, sectors: List[str], df_name: str) -> None:
        """Genera un pairplot para el DataFrame

        Args:
            sectors (list): Lista de sectores a incluir en el pairplot.
            df_name (str): Nombre del DataFrame.
        """
        df_copy = self.df.copy()
        df_copy._preprocess_pairplot_data()

        df_g0 = df_copy[df_copy['Diagnostico'] == 'Sano']
        df_g1 = df_copy[df_copy['Diagnostico'] == 'Glaucoma']

        g0_means, g0_std = df_g0[sectors].mean(), df_g0[sectors].std()
        g1_means, g1_std = df_g1[sectors].mean(), df_g1[sectors].std()

        plot = sb.pairplot(self.df, hue='Diagnostico', height=4, vars=sectors)
        plot.map_offdiag(sb.kdeplot, levels=10, color=".2")
        plot.map_diag(sb.kdeplot, color=".2")

        for i in range(len(sectors)):
            for j in range(len(sectors)):
                if i != j:
                    p0 = plot.axes[i, j].scatter(g0_means[sectors[j]], g0_means[sectors[i]], color='yellow', marker='o', label='Sano - Mean')
                    p1 = plot.axes[i, j].scatter(g1_means[sectors[j]], g1_means[sectors[i]], color='green', marker='D', label='Glaucoma - Mean')
                    plot.axes[i, j].legend(handles=[p0, p1], loc='upper right', fontsize=10)

        for i, sector in enumerate(sectors):
            plot.axes[i, i].axvline(g0_means[sector], color='red', label=r'Sano - Mean: {:.4f} $\pm$ {:.4f}'.format(g0_means[sector], 2 * g0_std[sector]))
            plot.axes[i, i].axvline(g1_means[sector], color='deepskyblue', label=r'Glaucoma - Mean: {:.4f} $\pm$ {:.4f}'.format(g1_means[sector], 2 * g1_std[sector]))
            plot.axes[i, i].legend(loc='upper right', fontsize='small')

        plot.fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.96)
        plot.fig.savefig(f'Figuras/Pair_Plot_{df_name}', pad_inches=0)
        plt.show()


    def pintar_boxplot(self, nombre:str) -> None:
        """Genera boxplots para cada columna del DataFrame"""
        columnas_df = self.df.iloc[:, :-1].columns.to_list()
        graficas_por_fila = 5
        filas = len(columnas_df) // graficas_por_fila + (len(columnas_df) % graficas_por_fila > 0)
        columnas = min(len(columnas_df), graficas_por_fila)
        fig, axes = plt.subplots(filas, columnas, figsize=(20, filas * 5))
        for i, col in enumerate(columnas_df):
            fila = i // graficas_por_fila
            columna = i % graficas_por_fila
            sb.boxplot(data=self.df[col], ax=axes[fila, columna]).set_title(col)
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.96)
        fig.savefig(f'Figuras/Boxplot_{nombre}', pad_inches=0)
        plt.show()

    def pinta_distribuciones(self, nombre:str) -> None:
        """Genera histogramas para cada columna del DataFrame"""
        df_sqrt = np.sqrt(self.df)
        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(15, 10))
        axes = axes.flat
        columnas = df_sqrt.columns
        for i, colum in enumerate(columnas):
            sb.histplot(data=df_sqrt, x=colum, stat="density", kde=True, color=(list(plt.rcParams['axes.prop_cycle'])*5)[i]["color"], line_kws={'linewidth': 2}, alpha=0.3, ax=axes[i], hue='Diagnostico').set_title(colum, fontsize=7, fontweight="bold")
            axes[i].tick_params(labelsize=6)
            axes[i].set_xlabel("")
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle('Distribución variables numéricas', fontsize=10, fontweight="bold")
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.96)
        fig.savefig(f'Figuras/Distribuciones_{nombre}', pad_inches=0)
        plt.show()

    def pinta_distribuciones_clases(self, nombre: str) -> None:
        """Genera histogramas por clase para cada columna del DataFrame"""
        df_g0 = self.df[self.df['Diagnostico'] == 'Sano']
        df_g1 = self.df[self.df['Diagnostico'] == 'Glaucoma']
        fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(15, 10))
        axes = axes.flat
        columnas = df_g0.columns
        for i, colum in enumerate(columnas):
            sb.histplot(data=df_g0, x=colum, stat="density", kde=True, color='blue', line_kws={'linewidth': 2}, alpha=0.3, ax=axes[i], label='Sano').set_title(colum, fontsize=7, fontweight="bold")
            sb.histplot(data=df_g1, x=colum, stat="density", kde=True, color='red', line_kws={'linewidth': 2}, alpha=0.3, ax=axes[i], label='Glaucoma')
            axes[i].tick_params(labelsize=6)
            axes[i].set_xlabel("")
            axes[i].legend()
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        fig.suptitle('Distribución variables numéricas por clase', fontsize=10, fontweight="bold")
        fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.96)
        fig.savefig(f'Figuras/Distribuciones_Clases_{nombre}', pad_inches=0)
        plt.show()

    def pintar_mapa_calor(self, nombre:str) -> None:
        """Genera un heatmap para el DataFrame"""
        sb.heatmap(self.df.corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm').set_title(f'Mapa de calor - {nombre}')
        sb.fig.savefig(f'Figuras/Mapa_Calor_{nombre}', pad_inches=0)
        plt.show()


class DataAnalyzer:
    """Clase encargada de analizar datos a partir de un conjunto de DataFrames"""

    def __init__(self, dataframes:dict):
        """Inicializador de la clase DataAnalyzer

        Args:
            dataframes (dict): Diccionario con DataFrames a analizar.
        """
        self.dataframes = dataframes

    def _get_dataframe(self, df_name: str) -> pd.DataFrame:
        """Obtiene un DataFrame específico

        Args:
            df_name (str): Nombre del DataFrame a obtener.

        Returns:
            pd.DataFrame: DataFrame con los datos.
        """
        try:
            return self.dataframes[df_name]
        except KeyError:
            raise CustomException(f"DataFrame con nombre {df_name} no encontrado.", sys.exc_info()[2])

    def visualize(self, df_name:str) -> None:
        """Analiza un DataFrame específico generando diferentes visualizaciones

        Args:
            df_name (str): Nombre del DataFrame a analizar.

        Raises:
            CustomException: Error al analizar el DataFrame.

        Returns:
            None
        """
        
        try:
            df_copy = self._get_dataframe(df_name)
            visualizer = Visualization(df_copy)
            visualizer.pinta_pairplot(0, 7, df_name)
            logger.info('Visualización exitosa del PairPlot')
            visualizer.pintar_boxplot(df_name)
            logger.info('Visualización exitosa del BoxPlot')
            visualizer.pinta_distribuciones(df_name)
            logger.info('Visualización exitosa de las distribuciones')
            visualizer.pinta_distribuciones_clases(df_name)
            logger.info('Visualización exitosa de las distribuciones por clase')
            visualizer.pintar_mapa_calor(df_name)
            logger.info('Visualización exitosa del Mapa de Calor')

        except CustomException as ce:
            logger.error(f"Se encontró un error en la visualización: {ce}",ce, sys.exc_info()[2])

    def visualize_all(self) -> None:
        """Analiza todos los DataFrames generando diferentes visualizaciones"""
        for df_name in self.dataframes:
            self.visualize(df_name)

   

    def save_to_results(self, method: str, df_name: str, data: Dict[str, Tuple[str, str]], results_dir_root: str = 'Resultados') -> None:
        """Guarda datos en el directorio 'resultados'."""
        # Crear estructura de directorio con la fecha actual
        date_str = datetime.now().strftime('%Y-%m-%d')
        results_dir = f'{results_dir_root}/{date_str}/{method}'
        os.makedirs(results_dir, exist_ok=True)
        
        with open(f'{results_dir}/{df_name}.txt', 'w') as f:
            for feature, values in data:
                line = f"{feature}: {values[0]}, {values[1]}\n"
                f.write(line)
                results_logger.info(line)  # Registro de resultados en el logger de resultados

    def perform_ttest(self, df_name: str) -> None:
        """Realiza un t-test entre las columnas de un DataFrame y la columna 'Diagnostico'."""
        df = self.dataframes[df_name]

        try:
            logger.info(f'Calculando t-test para el DataFrame {df_name}')
            results = {}
            for column in df.columns:
                if column != 'Glaucoma':
                    grupo_sano = df[df['Glaucoma'] == 0][column]
                    grupo_glaucoma = df[df['Glaucoma'] == 1][column]
                    t_stat, p_val = ttest_ind(grupo_glaucoma, grupo_sano)
                    results[column] = [f'T_stat = {t_stat:.4f}', f'P-valor = {p_val:.4f}']
            self.save_to_results('t-test', df_name, results.items())
            logger.info(f't-test calculado exitosamente para el DataFrame {df_name}')
        except Exception as e:
            logger.error(f'Error calculando t-test para el DataFrame {df_name}: {e}', exc_info=True)
            raise CustomException(f'Error calculando t-test para el DataFrame {df_name}',e, sys.exc_info()[2])


    def perform_chi2(self, df_name: str) -> None:
        """Realiza un chi2 entre las columnas de un DataFrame y la columna 'Glaucoma'

        Args:
            df_name (str): Nombre del DataFrame a analizar.
        """
        df = self.dataframes[df_name]
        try:
            logger.info(f'Calculando chi2 para el DataFrame {df_name}')

            y = df['Glaucoma']
            X = df.drop('Glaucoma', axis=1)

            significant_features = {}
            for col in X.columns:
                contingency = pd.crosstab(X[col], y)
                res = chi2_contingency(contingency)
                if res[1] < 0.05:
                    significant_features[col] = [f'Chi2 = {res[0]},P-valor ={res[1]}']
            
            self.save_to_results('chi2', df_name, significant_features)
            logger.info(f'Chi2 calculado exitosamente para el DataFrame {df_name}')
        except Exception as e:
            logger.error(f'Error calculando chi2 para el DataFrame {df_name}: {e}', exc_info=True)
            raise CustomException(f'Error calculando chi2 para el DataFrame {df_name}',e, sys.exc_info()[2])
