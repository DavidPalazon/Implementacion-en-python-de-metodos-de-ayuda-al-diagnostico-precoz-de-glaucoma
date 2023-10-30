import sys
import traceback

def error_message_detail(error, tb):
    """Construye un mensaje de error detallado."""
    file_name = tb.tb_frame.f_code.co_filename
    func_name = tb.tb_frame.f_code.co_name
    error_message = f"Error en el script [{file_name}] en la función [{func_name}] línea [{tb.tb_lineno}] con mensaje: {str(error)}"
    return error_message

class CustomException(Exception):
    """Excepción personalizada que proporciona un mensaje detallado del error."""

    def __init__(self, error, tb):
        self.error = error
        self.tb = tb

    def __str__(self):
        return error_message_detail(self.error, self.tb)

# Ejemplo de uso




