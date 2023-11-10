# IMPLEMENTACIÓN EN PYTHON DE MÉTODOS DE AYUDA AL DIAGNÓSTICO DE GLAUCOMA BASADO EN CARACTERÍSTICAS EXTRAÍDAS DE AMBOS OJOS EN RETINOGRAFÍAS Y TOMOGRAFÍAS DE COHERENCIA ÓPTICA

# IMPORTANTE: AHORA MISMO EL CÓDIGO ESTA TERMINANDO DE SER DESARROLLADO PARA UN POSTERIOR USO MAS SIMPLE, MEJORADO Y ESCALABLE. APESAR DE ELLO, LA METODOLODÍA ES LA MISMA QUE LA EXPUESTA EN LOS ARTICULOS Y SE OBTENDRAN LOS MISMOS RESULTADOS. DISCULPEN LAS MOLESTIAS.

## Descripción
Este repositorio contiene el código y la documentación del proyecto realizado para el diagnóstico precoz de glaucoma utilizando Python. Se enfoca en el análisis de retinografías y tomografías de coherencia óptica (OCT) para detectar características indicativas de glaucoma.

## Contenido del Repositorio
src/: Contiene los scripts de Python desarrollados para el análisis de imágenes y la implementación de algoritmos de aprendizaje automático.
venv/: Describe el entorno de anaconda utilizado para el desarrollo del código. Se aporta para el archivo '''venv.yml''' y '''requirements.txt''' para instalarlo mediante conda y mantener la compatibilidad de paquetes en las distintas maquinas que se ejecute.
articulos/: Incluye el artículo presentado en el CASEIB2023 y el Trabajo de Fin de Grado (TFG) que respalda este proyecto.
presentacion/: Materiales utilizados para presentaciones, incluyendo diapositivas y pósteres.
.gitignore: Archivo para excluir archivos y carpetas no deseados del repositorio.
### Restricciones de Acceso
Por razones de privacidad y seguridad, la base de datos original que contiene información sensible de pacientes no se incluye en este repositorio.

## Uso
Este proyecto está diseñado para facilitar la investigación y el análisis en el diagnóstico precoz de glaucoma mediante el uso de técnicas de procesamiento de imágenes y aprendizaje automático. A continuación, se detallan los pasos para configurar y utilizar el proyecto.

### Requisitos Previos
Antes de comenzar, asegúrate de tener instalado Python en tu sistema. Este proyecto se ha desarrollado utilizando Python 3.x. También se recomienda tener instalado Anaconda o Miniconda para gestionar los entornos virtuales y las dependencias.

### Configuración del Entorno
Clonar el Repositorio:

Primero, clona este repositorio en tu máquina local usando:
``` bash
git clone https://github.com/DavidPalazon/Implementacion-en-python-de-metodos-de-ayuda-al-diagnostico-precoz-de-glaucoma.git
```
Navega al directorio clonado:
```` bash
cd Implementacion-en-python-de-metodos-de-ayuda-al-diagnostico-precoz-de-glaucoma
````
Configurar el Entorno de Anaconda:

Crea un nuevo entorno de Anaconda utilizando el archivo venv.yml (asegurate que especificas correctamente su ruta):
``` bash
conda env create -f venv.yml
```
Activa el entorno:
```` bash
conda activate nombre_del_entorno
````
O, si prefieres usar pip, instala las dependencias usando requirements.txt:

```` bash
pip install -r venv/requirements.txt
````
Ejecución de Scripts
Dentro del entorno activado, puedes ejecutar los scripts individuales ubicados en la carpeta src/. Por ejemplo: 
```` bash
python src/nombre_del_script.py
````
Sigue las instrucciones específicas en cada script para realizar análisis de imágenes o entrenar modelos de aprendizaje automático.

Uso de Jupyter Notebooks
Si el proyecto incluye Jupyter Notebooks, puedes iniciar Jupyter Notebook o JupyterLab en tu navegador para explorarlos y ejecutarlos interactivamente:

```` bash
jupyter notebook
````

```` bash
jupyter lab
````
Navega a la ubicación de los notebooks dentro de la interfaz de usuario de Jupyter y ábrelos para interactuar con ellos.

## Contacto
Para preguntas, colaboraciones o solicitudes de acceso a datos restringidos, por favor contacte a [david.palazon@edu.upct.es].
