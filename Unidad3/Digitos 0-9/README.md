# Tarea 7. Clasificación de dígitos 0-9 (MNIST)

Alumna: Miriam Alejandra Bazaldua Machado

## ¿Qué hace este proyecto?

Se entrena una red neuronal con **MNIST**, un conjunto de dígitos escritos a mano, y el modelo resultante se guarda en
un archivo. Una aplicación en Python abre ese archivo y permite dibujar un número con el mouse; mientras se dibuja, el
programa muestra qué dígito reconoce.

La tarea pedía:

1. Trabajar con el dataset MNIST.
2. Entrenar el modelo y guardarlo.
3. Cargar el modelo guardado en una aplicación de Python y probarlo en tiempo real.

## Archivos

| Archivo | Descripción |
|---|---|
| `entrenar_modelo_mnist.ipynb` | Cuaderno de entrenamiento y guardado del modelo |
| `reconocer_digito.py` | Aplicación con lienzo que clasifica el dígito dibujado |
| `modelo_mnist.keras` | Modelo entrenado, generado por el cuaderno |
| `requirements.txt` | Dependencias de la aplicación |

## Cómo ejecutarlo

### Paso 1: entrenar

1. Subir `entrenar_modelo_mnist.ipynb` a Google Colab (Archivo > Subir cuaderno).
2. Elegir Entorno de ejecución > Ejecutar todo (dura menos de un minuto).
3. El archivo `modelo_mnist.keras` se descarga solo al terminar; hay que guardarlo en esta carpeta, junto con
   `reconocer_digito.py`.

### Paso 2: probar en tiempo real

```bash
pip install -r requirements.txt
python reconocer_digito.py
```

Se abre una ventana con tres zonas: a la izquierda la **pizarra** donde se dibuja con el mouse, a la derecha una
**tarjeta** que muestra en grande el dígito detectado y una barra con la certeza del modelo, y debajo una **gráfica de
barras** con la probabilidad de cada dígito del 0 al 9.

| Control | Acción |
|---|---|
| Clic izquierdo y arrastrar | Dibujar |
| C | Borrar el lienzo |
| Q o ESC | Salir |

Como la aplicación usa una ventana gráfica, debe correrse en una computadora local (no en Colab ni Codespaces).
TensorFlow necesita una versión de Python compatible, normalmente de la 3.10 a la 3.13; se verifica con
`python --version`.

## Estructura del modelo

Es una red densa (perceptrón multicapa) con estas capas:

| Capa | Para qué sirve |
|---|---|
| Rescaling | Lleva los píxeles de 0-255 a 0-1 |
| Flatten | Convierte la imagen de 28 x 28 en un vector de 784 valores |
| Dense 256 (ReLU) | Primera capa oculta |
| Dropout 0.2 | Ignora al azar el 20 % de las neuronas al entrenar para evitar el sobreajuste |
| Dense 128 (ReLU) | Segunda capa oculta |
| Dense 10 (softmax) | Probabilidad de cada dígito |

Se entrenó durante 10 épocas con Adam y lotes de 128 imágenes. En las 10,000 imágenes de prueba alcanza cerca del
98 % de precisión. El cuaderno también incluye las curvas de entrenamiento, la matriz de confusión y ejemplos de
imágenes mal clasificadas.

## Procesamiento del dibujo

Un dibujo hecho con el mouse no se parece a una imagen de MNIST: es mayor, está en otro lugar y el trazo es más
grueso. Por eso, antes de clasificar, `reconocer_digito.py` lo prepara como están preparados los datos de MNIST: recorta
el área del trazo, lo escala a 20 píxeles en su lado más largo, lo centra en una imagen de 28 x 28 y lo desplaza para
que su centro de masa quede en el medio. Sin este paso, el modelo falla bastante más.
