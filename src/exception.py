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

def risky_math_operation(x, y):
    """Función de ejemplo que realiza una operación matemática riesgosa (división)."""
    try:
        result = x / y
        return result
    except ZeroDivisionError as e:
        raise CustomException(e, sys.exc_info()[2]) from e

# Ejemplo de uso
if __name__ == "__main__":
    try:
        print(risky_math_operation(5, 0))
    except CustomException as ce:
        print(f"Se capturó una excepción personalizada: {ce}")



