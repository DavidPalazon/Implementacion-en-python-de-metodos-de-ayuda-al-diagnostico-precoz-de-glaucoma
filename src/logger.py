import logging
import os
from datetime import datetime

# Crear y configurar el directorio de logs
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)

# --- Logger principal ---
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}_app.log"
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)
logger = logging.getLogger('app_logger')
logger.setLevel(logging.DEBUG)  # Cambia esto a logging.INFO para reducir la salida

# Handlers para el logger principal
app_c_handler = logging.StreamHandler()  # Salida a la consola
app_f_handler = logging.FileHandler(LOG_FILE_PATH)  # Salida a un archivo basado en la fecha
app_c_handler.setLevel(logging.WARNING)  # Solo mostrará warnings y errores en la consola
app_f_handler.setLevel(logging.DEBUG)  # Guardará todos los logs en el archivo

# Formatos para el logger principal
app_c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
app_f_format = logging.Formatter('[%(asctime)s] %(lineno)d - %(name)s - %(levelname)s - %(message)s')
app_c_handler.setFormatter(app_c_format)
app_f_handler.setFormatter(app_f_format)

logger.addHandler(app_c_handler)
logger.addHandler(app_f_handler)

# --- Logger de resultados ---
RESULTS_LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}_results.log"
RESULTS_LOG_FILE_PATH = os.path.join(logs_dir, RESULTS_LOG_FILE)
results_logger = logging.getLogger('results_logger')
results_logger.setLevel(logging.INFO)  # Guardaremos solo resultados

# Handler para el logger de resultados
results_f_handler = logging.FileHandler(RESULTS_LOG_FILE_PATH)  
results_f_handler.setLevel(logging.INFO)  # Nivel para resultados

# Formato para el logger de resultados
results_format = logging.Formatter('[%(asctime)s] %(message)s')
results_f_handler.setFormatter(results_format)

results_logger.addHandler(results_f_handler)

# Ejemplo de uso
if __name__ == '__main__':
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical message')
    
    results_logger.info('Resultado: El paciente XYZ fue diagnosticado con glaucoma.')
