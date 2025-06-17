
# Detector de lenguaje de señas en tiempo real

_El objetivo de este proyecto es el de desarrollar un sistema de reconocimiento de lenguaje de datos. Para ello se usa una camara que detecta los gestos de las manos y el sistema es capaz de identificar y traducir gestos de la Lengua de Señales Americana `ASL` a texto._

_Para la realización de este programa se ha trabajado a partir del [WLASL dataset](https://www.kaggle.com/datasets/utsavk02/wlasl-complete) descrito en "Word-level Deep Sign Language Recognition from Video: En Nueva Large-Scale Dataset and Methods Comparison". Puedes encontrar el repositorio de github [aquí](https://github.com/dxli94/WLASL)._

>La solución utiliza una arquitectura Inflated 3D ConvNet (I3D) preentrenada en ImageNet y optimizada para clasificar las señas. 


El modelo procesa entradas de vídeo RGB y combina predicciones a nivel de vídeo y de fotograma para mejorar la precisión del reconocimiento.


### Descarga el dataset

El conjunto de datos utilizado en este proyecto es "WLASL", disponible [aquí](https://www.kaggle.com/datasets/utsavk02/wlasl-complete) en Kaggle.

Una vez descargado es recomendable guardarlo en la misma ruta que el directorio WLASL. Cree una carpeta "data" y dentro pon el conjunto de datos.

## Pasos a ejecutar

Estas instrucciones te permitirán obtener una copia del proyecto en tu máquina local de la forma más clara y sencilla posible.

1. Descarga el repositorio, ya sea clonando mediante el siguiente comando o descargandolo como zip:

 ```
 git clone https://github.com/1631153/Sign-Recognition-TFG.git
 
 ```

2. Instalar los requisitos:

```
pip install -r requirements.txt
```
3. Una vez todo esté instalado podrás ser capaz de ejecutar los archivos de **entrenamiento** y **test**, además del de la camara para probar el funcionamiento.

No obstante, hay que tener en cuenta que el proyecto utiliza Cuda y PyTorch, por lo que se requiere un sistema con gráficos NVIDIA. Además, para ejecutar el sistema se necesita un mínimo de 4-5 GB de memoria GPU dedicada.

## Diferencias clave entre versiones

Como se puede apreciar, hay diferentes versiónes modificadas del train_i3d.py. A continuación se dicen los cambios más notables de cada versión:

- Javi_train_v1 : Sustituido el bucle de entrenamiento por **funciones separadas de entrenamiento y validación con early stopping y cálculo de precisión**.

- Javi_train_v2 : Sustituido el entrenamiento basado en acumulación de gradientes y validación por pasos por un esquema más tradicional con **funciones separadas para entrenamiento y validación por época, métricas detalladas (accuracy y F1)**, y estructura modular y clara.

- Javi_train_v3 : Sustituido el entrenamiento de todo el modelo por un **entrenamiento con capas congeladas (fine-tuning)**, enfocándose solo en `Mixed_5c` como capa entrenable para una adaptación más eficiente del modelo preentrenado.

- Javi_train_v4 : Sustituido el entrenamiento estándar por uno con **balanceo de clases mediante class_weights, cálculo de métricas avanzadas (accuracy y F1), y uso de early stopping para evitar sobreentrenamiento**.

- Javi_train_v5 : Sustituido el entrenamiento con class weights, congelación parcial de capas y early stopping por un **esquema más simple basado en CrossEntropyLoss sin pesos**, entrenamiento completo del modelo y guardado del mejor modelo según la precisión.

- Javi_train_v6 : Sustituido el entrenamiento estándar con scheduler fijo por uno **adaptativo usando ReduceLROnPlateau basado en la precisión de validación**, con mejoras en el seguimiento de métricas por batch y aplicación de early stopping más prolongado.

- Javi_train_v7 : Sustituido el entrenamiento con CrossEntropyLoss y etiquetas como índices de clase por **BCEWithLogitsLoss con etiquetas one-hot, adaptando la interpolación temporal y simplificando el control de overfitting al eliminar el early stopping**.
