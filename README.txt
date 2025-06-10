IMPORTANTE:
Antes de todo hay que crear el entorno virtual. Debido a que no se puede compartir directamente,
hay que instalar las dependencias. Desde el programa Visual Studio Code, en una terminal de 
python, ejecuta uno a uno estos comandos:

python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt

---------------------------------------------------------------------------------------->

Esta carpeta contiene dos partes del detector de señas:
1. DETECTOR DE SEÑAS (ABECEDARIO)

	ARCHIVOS: Entrenamiento.py, Code/Prediccion.py
	FUNCIONAMIENTO: Al ejecutar el Entrenamiento.py se empezará a crear un modelo a partir
	de la base de datos y con Prediccion.py se puede poner en practica este modelo.
	COMO FUNCIONA: Debido a que crear el modelo cuesta 3 horas, aquí tienes el enlace para
	descargar: https://drive.google.com/file/d/1riVtC595q52prfPe9v6m9n_zi9xnRj0B/view?usp=sharing
	Pega los dos modelos en la carpeta Code y ejecuta Prediccion.py

2. DETECTOR DE SEÑAS (PALABRAS)

	ARCHIVOS: MP_hands_Action (codigo más nuevo), MP_holistic_Action (codigo anterior)
	FUNCIONAMIENTO: El entrenamiento y la ejecución se encuentran en el mismo archivo,
	el programa debería interpretar los tres gestos entrenados.
	COMO FUNCIONA: Al ejecutar cualquiera de los dos codigos, se debería poder interpretar
	los gestos tal y como se ve en este video de presentación: https://www.youtube.com/watch?v=eA_JtX0rEds
	(El entrenamiento se encuentra comentado)

---------------------------------------------------------------------------------------->

COMANDOS UTILES:

- Ejecutar entorno virtual: manos/Scripts/activate
- Ejecutar archivo .py: pyhton [nombre_archivo].py

En teoria con estos dos ya sirve para el contenido de esta carpeta.