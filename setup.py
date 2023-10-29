from setuptools import setup, find_packages

HYPEN_E_DOT = '-e .'

def get_requirements(filename='requirements.txt'):
    with open(filename, 'r') as file:
        requirements = file.readlines()
        requirements = [req.strip() for req in requirements if req.strip()]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='implementacion-en-python-de-metodos-de-ayuda-al-diagnostico-de-glaucoma',
    version='0.1.0',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            # Si tienes algún script que quieras que sea ejecutable desde la línea de comandos, puedes agregarlo aquí.
            # 'nombre_del_script=nombre_del_paquete.modulo:funcion_principal',
        ],
    },
    author='David Palazón Palau',
    author_email='david.palazon@edu.upct.es',
    description='Proyecto completo de ML para el diagnostico precoz de glaucoma',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DavidPalazon/Implementacion-en-python-de-metodos-de-ayuda-al-diagnostico-precoz-de-glaucoma.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
