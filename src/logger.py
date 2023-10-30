import logging
import os
from datetime import datetime


# Configurar el nombre y la ruta del archivo de log basado en la fecha actual
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d')}.log"
logs_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Crear y configurar el logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Cambia esto a logging.INFO para reducir la salida

# Crear manejadores (handlers) para determinar a dónde irá la salida del log
c_handler = logging.StreamHandler()  # Salida a la consola
f_handler = logging.FileHandler(LOG_FILE_PATH)  # Salida a un archivo basado en la fecha

# Establecer niveles para cada manejador
c_handler.setLevel(logging.WARNING)  # Solo mostrará warnings y errores en la consola
f_handler.setLevel(logging.DEBUG)  # Guardará todos los logs en el archivo

# Crear formatos y añadirlos a los manejadores
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('[%(asctime)s] %(lineno)d - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Añadir manejadores al logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# Ejemplo de uso
if __name__ == '__main__':
    logger.debug('Debug message')
    logger.info('Info message')
    logger.warning('Warning message')
    logger.error('Error message')
    logger.critical('Critical message')

