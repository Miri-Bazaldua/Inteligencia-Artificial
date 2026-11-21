# Alumna: Miriam Alejandra Bazaldua Machado

# Controles:
#   Clic izquierdo (mantener)  -> dibujar
#   C                          -> borrar todo
#   Q o ESC                    -> salir

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia mensajes innecesarios de TensorFlow

import cv2
import numpy as np
import keras

RUTA_ARCHIVO_MODELO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_mnist.keras")
TAM_LIENZO = 280 
GROSOR_PINCEL = 20

MARGEN = 14
ALTO_TITULO = 46
ANCHO_TARJETA = 262
ALTO_GRAFICA = 130
ALTO_PIE = 40

X_LIENZO = MARGEN
Y_LIENZO = ALTO_TITULO + MARGEN
ANCHO_VENTANA = MARGEN * 3 + TAM_LIENZO + ANCHO_TARJETA
ALTO_VENTANA = Y_LIENZO + TAM_LIENZO + MARGEN + ALTO_GRAFICA + ALTO_PIE

# Paleta de colores (formato BGR de OpenCV)
COLOR_FONDO = (38, 30, 27)
COLOR_PANEL = (58, 47, 42)
COLOR_PIZARRA = (28, 22, 20)
COLOR_TRAZO = (255, 235, 210)
COLOR_ACENTO = (60, 170, 255)
COLOR_TEXTO = (235, 228, 222)
COLOR_TEXTO_SUAVE = (150, 140, 134)
COLOR_BARRA = (110, 98, 92)

TITULO_VENTANA = "Pizarra"


def normalizar_dibujo(lienzo):
    pixeles_trazo = cv2.findNonZero(lienzo)
    if pixeles_trazo is None:
        return None  # no hay trazo todavía

    # Se recorta el rectángulo más pequeño que contiene el trazo
    x0, y0, ancho, alto = cv2.boundingRect(pixeles_trazo)
    region = lienzo[y0:y0 + alto, x0:x0 + ancho]

    # Se redimensiona para que el lado más largo quede en 20 px
    factor = 20.0 / max(ancho, alto)
    ancho_nuevo = max(1, int(round(ancho * factor)))
    alto_nuevo = max(1, int(round(alto * factor)))
    region = cv2.resize(region, (ancho_nuevo, alto_nuevo), interpolation=cv2.INTER_AREA)

    # Se pega en medio de un fondo negro de 28x28
    resultado = np.zeros((28, 28), dtype=np.uint8)
    margen_x = (28 - ancho_nuevo) // 2
    margen_y = (28 - alto_nuevo) // 2
    resultado[margen_y:margen_y + alto_nuevo, margen_x:margen_x + ancho_nuevo] = region

    # Se desplaza la imagen hasta que el centro de masa quede en (14, 14)
    mom = cv2.moments(resultado)
    if mom["m00"] > 0:
        cx = mom["m10"] / mom["m00"]
        cy = mom["m01"] / mom["m00"]
        traslacion = np.float32([[1, 0, 14 - cx], [0, 1, 14 - cy]])
        resultado = cv2.warpAffine(resultado, traslacion, (28, 28))
    return resultado


def clasificar(red, lienzo):
    # Devuelve la probabilidad de cada dígito (10 valores), o None si el lienzo está vacío
    imagen = normalizar_dibujo(lienzo)
    if imagen is None:
        return None
    entrada = imagen.reshape(1, 28, 28).astype("float32")
    return red(entrada, training=False).numpy()[0]


def texto_centrado(pantalla, texto, centro_x, y, escala, color, grosor=1):
    (ancho, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_DUPLEX, escala, grosor)
    cv2.putText(pantalla, texto, (int(centro_x - ancho / 2), y), cv2.FONT_HERSHEY_DUPLEX, escala, color, grosor,
                cv2.LINE_AA)


def armar_ventana(lienzo, probs):
    pantalla = np.full((ALTO_VENTANA, ANCHO_VENTANA, 3), COLOR_FONDO, dtype=np.uint8)

    pantalla[:ALTO_TITULO] = COLOR_PANEL
    cv2.putText(pantalla, "Pizarra", (MARGEN, 31), cv2.FONT_HERSHEY_DUPLEX, 0.75, COLOR_TEXTO, 1,
                cv2.LINE_AA)
    cv2.line(pantalla, (0, ALTO_TITULO), (ANCHO_VENTANA, ALTO_TITULO), COLOR_ACENTO, 2)

    opacidad = (lienzo.astype(np.float32) / 255.0)[:, :, None]
    fondo = np.full((TAM_LIENZO, TAM_LIENZO, 3), COLOR_PIZARRA, dtype=np.float32)
    trazo = np.full((TAM_LIENZO, TAM_LIENZO, 3), COLOR_TRAZO, dtype=np.float32)
    pizarra = (fondo * (1 - opacidad) + trazo * opacidad).astype(np.uint8)
    pantalla[Y_LIENZO:Y_LIENZO + TAM_LIENZO, X_LIENZO:X_LIENZO + TAM_LIENZO] = pizarra
    cv2.rectangle(pantalla, (X_LIENZO - 1, Y_LIENZO - 1), (X_LIENZO + TAM_LIENZO, Y_LIENZO + TAM_LIENZO),
                  COLOR_ACENTO, 1)

    # Tarjeta de resultado
    tx = X_LIENZO + TAM_LIENZO + MARGEN
    ty = Y_LIENZO
    pantalla[ty:ty + TAM_LIENZO, tx:tx + ANCHO_TARJETA] = COLOR_PANEL
    centro = tx + ANCHO_TARJETA // 2
    texto_centrado(pantalla, "DIGITO DETECTADO", centro, ty + 34, 0.55, COLOR_TEXTO_SUAVE)

    if probs is None:
        texto_centrado(pantalla, "?", centro, ty + 165, 4.5, COLOR_TEXTO_SUAVE, 3)
        texto_centrado(pantalla, "Dibuja un numero", centro, ty + 225, 0.6, COLOR_TEXTO_SUAVE)
    else:
        ganador = int(np.argmax(probs))
        certeza = float(probs[ganador])
        texto_centrado(pantalla, str(ganador), centro, ty + 165, 5.0, COLOR_ACENTO, 4)

        # Barra de certeza
        bx0, bx1, by = tx + 24, tx + ANCHO_TARJETA - 24, ty + 205
        cv2.rectangle(pantalla, (bx0, by), (bx1, by + 12), COLOR_FONDO, -1)
        cv2.rectangle(pantalla, (bx0, by), (bx0 + int((bx1 - bx0) * certeza), by + 12), COLOR_ACENTO, -1)
        texto_centrado(pantalla, f"Certeza: {certeza:.1%}", centro, ty + 248, 0.6, COLOR_TEXTO)

    # Gráfica de barras verticales con la probabilidad de cada dígito
    gy0 = Y_LIENZO + TAM_LIENZO + MARGEN
    cv2.putText(pantalla, "Probabilidad por digito", (MARGEN, gy0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLOR_TEXTO_SUAVE, 1, cv2.LINE_AA)
    base = gy0 + ALTO_GRAFICA - 20
    alto_max = ALTO_GRAFICA - 52
    ancho_col = (ANCHO_VENTANA - 2 * MARGEN) / 10
    for d in range(10):
        p = 0.0 if probs is None else float(probs[d])
        es_ganador = probs is not None and d == int(np.argmax(probs))
        color = COLOR_ACENTO if es_ganador else COLOR_BARRA
        x0 = int(MARGEN + d * ancho_col + ancho_col * 0.2)
        x1 = int(MARGEN + d * ancho_col + ancho_col * 0.8)
        cv2.rectangle(pantalla, (x0, base - max(int(p * alto_max), 2)), (x1, base), color, -1)
        texto_centrado(pantalla, str(d), (x0 + x1) / 2, base + 16, 0.5, COLOR_TEXTO if es_ganador else COLOR_TEXTO_SUAVE)
        if probs is not None and p >= 0.01:
            texto_centrado(pantalla, f"{p:.0%}", (x0 + x1) / 2, base - int(p * alto_max) - 6, 0.38, color)

    cv2.putText(pantalla, "Mouse: dibujar   |   C: borrar   |   Q o ESC: salir", (MARGEN, ALTO_VENTANA - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXTO_SUAVE, 1, cv2.LINE_AA)
    return pantalla


def iniciar():
    if not os.path.exists(RUTA_ARCHIVO_MODELO):
        print("No se encontró el modelo:", RUTA_ARCHIVO_MODELO)
        print("Primero entrena con el cuaderno y copia modelo_mnist.keras a esta carpeta.")
        return

    red = keras.models.load_model(RUTA_ARCHIVO_MODELO)
    print("Modelo cargado. Ya puedes dibujar un dígito en la ventana.")

    lienzo = np.zeros((TAM_LIENZO, TAM_LIENZO), dtype=np.uint8)
    trazo = {"activo": False, "previo": None, "modificado": False}

    def al_mover_mouse(evento, x, y, banderas, extra):
        x -= X_LIENZO
        y -= Y_LIENZO
        dentro = 0 <= x < TAM_LIENZO and 0 <= y < TAM_LIENZO
        if evento == cv2.EVENT_LBUTTONDOWN and dentro:
            trazo["activo"] = True
            trazo["previo"] = (x, y)
            cv2.circle(lienzo, (x, y), GROSOR_PINCEL // 2, 255, -1)
            trazo["modificado"] = True
        elif evento == cv2.EVENT_MOUSEMOVE and trazo["activo"]:
            cv2.line(lienzo, trazo["previo"], (x, y), 255, GROSOR_PINCEL)
            trazo["previo"] = (x, y)
            trazo["modificado"] = True
        elif evento == cv2.EVENT_LBUTTONUP:
            trazo["activo"] = False

    cv2.namedWindow(TITULO_VENTANA)
    cv2.setMouseCallback(TITULO_VENTANA, al_mover_mouse)

    probs = None
    while True:
        # Solo se vuelve a clasificar si el dibujo cambió
        if trazo["modificado"]:
            probs = clasificar(red, lienzo)
            trazo["modificado"] = False

        cv2.imshow(TITULO_VENTANA, armar_ventana(lienzo, probs))
        tecla = cv2.waitKey(15) & 0xFF

        if tecla in (ord("c"), ord("C")):
            lienzo[:] = 0
            probs = None
        elif tecla in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(TITULO_VENTANA, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar()
