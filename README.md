# Anexos digitales del proyecto de tesis: detección automática de fallas en la calidad de audio

## 1. Descripción del problema

En los procesos de postproducción televisiva, el control de calidad de audio suele realizarse mediante revisión manual por parte de operadores técnicos. Este enfoque depende en gran medida de la experiencia del evaluador, introduce subjetividad, consume tiempo operativo y dificulta la estandarización del proceso. Como consecuencia, pueden pasar desapercibidas fallas técnicas relevantes como saturación, clipping, silencios anómalos, problemas de fase, variaciones de sonoridad o defectos que afectan la inteligibilidad del contenido.

## 2. Descripción de la solución

La solución desarrollada en este proyecto consiste en un sistema basado en aprendizaje automático supervisado para apoyar la detección automática de fallas en la calidad de audio de contenidos televisivos. La propuesta parte de archivos audiovisuales reales, construye un dataset etiquetado, extrae características acústicas clásicas, entrena modelos de clasificación y los integra en un prototipo funcional con interfaz gráfica.

El sistema permite:

* analizar archivos en formato MXF o WAV;
* extraer o reconstruir audio cuando corresponde;
* medir sonoridad conforme a EBU R128;
* segmentar el audio en ventanas de análisis;
* calcular variables acústicas relevantes;
* aplicar modelos de clasificación;
* generar un reporte HTML con hallazgos técnicos y resultado global.

## 3. Estructura del repositorio

El repositorio se organiza en las siguientes carpetas:

* `01_paso0_dataset`: generación del dataset etiquetado a partir del archivo fuente.
* `02_paso1_extraccion`: extracción de características acústicas.
* `03_paso2_modelos`: entrenamiento y evaluación de modelos de clasificación.
* `04_prototipo_codigo`: código fuente del prototipo en Python.
* `05_reportes_html`: reportes HTML generados por el sistema.
* `06_evidencias`: capturas y material visual complementario.

## 4. Requisitos técnicos y dependencias

### Requisitos generales

* Python 3.10 o superior
* FFmpeg
* FFprobe
* Sistema operativo Windows (recomendado para la versión usada en el proyecto)

### Dependencias principales de Python

El desarrollo utiliza librerías como:

* `numpy`
* `pandas`
* `scikit-learn`
* `librosa`
* `soundfile`
* `matplotlib`
* `joblib`
* `tkinter`

### Dependencias externas

* `ffmpeg`: para extracción y conversión de audio
* `ffprobe`: para inspección de streams de audio en archivos MXF

## 5. Instrucciones de ejecución paso a paso

### 5.1 Preparación del entorno

1. Instalar Python.
2. Instalar FFmpeg y FFprobe.
3. Verificar que ambos estén disponibles desde la línea de comandos.
4. Instalar las librerías necesarias de Python.

Ejemplo:

```
pip install numpy pandas scikit-learn librosa soundfile matplotlib joblib
```

### 5.2 Ejecución metodológica por etapas

#### PASO 0: Generación del dataset etiquetado

Ubicación: `01_paso0_dataset`

Objetivo:

* generar el archivo consolidado de etiquetas a partir de la matriz original de trabajo.

Archivo principal:

* `GENERACION DATASET DESDE XLSX.ipynb`

Resultado esperado:

* `dataset_v1_etiquetado_v2.csv`

#### PASO 1: Extracción de características acústicas

Ubicación: `02_paso1_extraccion`

Objetivo:

* procesar los segmentos de audio asociados al dataset etiquetado y calcular las variables acústicas utilizadas por los modelos.

Archivo principal:

* `1.EXTRACCION.ipynb`

Resultado esperado:

* `features_dataset_v1.csv`

#### PASO 2: Entrenamiento y evaluación de modelos

Ubicación: `03_paso2_modelos`

Objetivo:

* entrenar y comparar los modelos de clasificación utilizados en el proyecto.

Archivos principales:

* `2.BASELINE REGRESION LOGISTICA.ipynb`
* `3.MODELO COMPARATIVO RANDOM FOREST.ipynb`

Resultado esperado:

* métricas de validación y prueba;
* matrices de confusión;
* validación cruzada;
* importancia de variables.

### 5.3 Ejecución del prototipo

Ubicación: `04_prototipo_codigo`

1. Verificar rutas y disponibilidad de modelos serializados.

2. Ejecutar la interfaz principal:

   python app_gui.py

3. Seleccionar un archivo de entrada en formato MXF o WAV.

4. Elegir el modelo de inferencia:

   * Regresión Logística
   * Random Forest

5. Ejecutar el análisis.

6. Revisar el reporte HTML generado.

## 6. Explicación general del pipeline

El desarrollo completo del proyecto se organizó en varios pipelines metodológicos y funcionales.

### 6.1 Pipeline del PASO 0: generación del dataset etiquetado

Este pipeline parte del archivo fuente de etiquetado y realiza:

1. lectura de la matriz original;
2. validación de columnas obligatorias;
3. conversión de tiempos a formato numérico;
4. revisión de consistencia entre etiquetas;
5. generación del dataset consolidado `dataset_v1_etiquetado_v2.csv`.

### 6.2 Pipeline del PASO 1: extracción de características acústicas

Este pipeline toma como entrada los segmentos de audio asociados a los registros etiquetados y realiza:

1. lectura de los segmentos;
2. extracción de variables acústicas clásicas;
3. validación de valores;
4. construcción del archivo `features_dataset_v1.csv`.

Las variables utilizadas fueron:

* `rms_mean`
* `crest_factor_db`
* `true_peak_dbfs`
* `short_term_lufs_mean`
* `loudness_var_proxy_db`
* `silence_ratio`
* `stereo_correlation`
* `spectral_centroid_mean`
* `spectral_flatness_mean`
* `clipping_ratio`
* `near_ceiling_ratio`

### 6.3 Pipeline del PASO 2: entrenamiento y evaluación de modelos

Este pipeline recibe el dataset de features y realiza:

1. selección de variables predictoras;
2. construcción de la salida binaria;
3. división estratificada en entrenamiento, validación y prueba;
4. entrenamiento de Regresión Logística;
5. entrenamiento de Random Forest;
6. comparación de métricas;
7. análisis de matrices de confusión;
8. validación cruzada e importancia de variables.

### 6.4 Pipeline del prototipo funcional

El prototipo implementa un flujo operativo orientado al usuario técnico:

1. **Entrada del archivo**

   * selección de archivo MXF o WAV desde la interfaz gráfica.

2. **Validación e inspección**

   * verificación de la estructura del archivo;
   * inspección de audio en caso de archivos MXF.

3. **Extracción o reconstrucción de audio**

   * generación de un WAV de trabajo cuando corresponde.

4. **Medición global EBU R128**

   * cálculo de sonoridad del archivo completo.

5. **Segmentación**

   * división del audio en ventanas consecutivas de 5 segundos.

6. **Extracción de características**

   * cálculo de variables acústicas por segmento.

7. **Inferencia**

   * aplicación del modelo seleccionado;
   * obtención de predicción y probabilidad por segmento.

8. **Lógica de decisión**

   * combinación de evidencia del modelo y reglas operativas.

9. **Generación del reporte**

   * producción de un archivo HTML con resultado global, estado EBU, porcentaje de segmentos problemáticos y hallazgos por intervalo temporal.

## 7. Modelos implementados

### Regresión Logística

* utilizada como modelo baseline;
* alta interpretabilidad;
* menor costo computacional;
* comportamiento práctico consistente en varios casos reales.

### Random Forest

* utilizado como modelo comparativo;
* mejores métricas internas de validación y prueba;
* mayor capacidad para capturar relaciones no lineales;
* útil como referencia académica y experimental.

## 8. Evidencias incluidas

El repositorio contiene además:

* notebooks del desarrollo metodológico;
* código fuente del prototipo;
* reportes HTML generados automáticamente;
* capturas de interfaz y salidas del sistema.

## 9. Observación final

Este repositorio constituye el anexo digital del proyecto de tesis y complementa la memoria escrita principal. Su propósito es documentar de manera reproducible el desarrollo metodológico, experimental y técnico de la solución propuesta.
