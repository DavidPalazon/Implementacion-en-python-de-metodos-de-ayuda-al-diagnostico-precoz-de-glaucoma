import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# Estableciendo configuraciones globales para visualizaciones
plt.rcParams['figure.figsize'] = (20, 20)
plt.style.use('ggplot')

class Visualization:
    """Clase encargada de visualizar datos a partir de un DataFrame"""

    def __init__(self, df):
        """Inicializador de la clase Visualization

        Args:
            df (pd.DataFrame): DataFrame con los datos a visualizar.
        """
        self.df = df
        sb.set_style('whitegrid')
        sb.set_context('paper', font_scale=1.5)

    def pinta_pairplot(self, sectors, nombre):
        """Genera un Pairplot para visualizar relaciones bivariadas en un DataFrame.

        Args:
            sectors (list): Lista de nombres de columnas a considerar.
            nombre (str): Nombre para etiquetas y títulos.
        """
        self.df = self.df.rename(columns={'glaucoma': 'Diagnostico'})
        replacements = {0: 'Sano', 1: 'Glaucoma'}
        self.df['Diagnostico'] = self.df['Diagnostico'].replace(replacements)
        
        df_g0 = self.df[self.df['Diagnostico'] == 'Sano']
        df_g1 = self.df[self.df['Diagnostico'] == 'Glaucoma']
        
        g0_means = df_g0[sectors].mean()
        g0_std = df_g0[sectors].std()
        g1_means = df_g1[sectors].mean()
        g1_std = df_g1[sectors].std()

        plot = sb.pairplot(self.df, hue='Diagnostico', height=4, vars=sectors, kind='scatter', diag_kws={'color': 'green'})
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
        plot.fig.savefig(f'Figuras/Pair_Plot_{nombre}', pad_inches=0)
        plt.show()


    def pintar_boxplot(self, nombre):
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

    def pinta_distribuciones(self, nombre):
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

    def pinta_distribuciones_clases(self, nombre):
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

    def pintar_mapa_calor(self, nombre):
        """Genera un heatmap para el DataFrame"""
        sb.heatmap(self.df.corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm').set_title(f'Mapa de calor - {nombre}')
        sb.fig.savefig(f'Figuras/Mapa_Calor_{nombre}', pad_inches=0)
        plt.show()

class DataAnalyzer:
    """Clase encargada de analizar datos a partir de un conjunto de DataFrames"""

    def __init__(self, dataframes):
        """Inicializador de la clase DataAnalyzer

        Args:
            dataframes (dict): Diccionario con DataFrames a analizar.
        """
        self.dataframes = dataframes

    def analyze(self, df_name):
        """Analiza un DataFrame específico generando diferentes visualizaciones

        Args:
            df_name (str): Nombre del DataFrame a analizar.
        """
        df = self.dataframes[df_name]
        visualizer = Visualization(df)
        visualizer.pinta_pairplot(0, 7, df_name)
        visualizer.pintar_boxplot(df_name)
        visualizer.pinta_distribuciones(df_name)
        visualizer.pinta_distribuciones_clases(df_name)
        visualizer.pintar_mapa_calor(df_name)

    def analyze_all(self):
        """Analiza todos los DataFrames generando diferentes visualizaciones"""
        for df_name in self.dataframes:
            self.analyze(df_name)

        
