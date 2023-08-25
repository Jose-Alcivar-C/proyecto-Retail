import gradio as gr
import cv2
from recursos import textoSalida
from constants import MODELO_ENTRENADO_PROPIO, MODELO_ENTRENADO_RETAIL

from ultralytics import YOLO

#Cargar modelo entrenado
modelo_retail = YOLO(MODELO_ENTRENADO_RETAIL)
modelo_propio = YOLO(MODELO_ENTRENADO_PROPIO)

# La interfaz solo recibe una entrada (La imagen ingresada en el cargador de path de imagenes), por lo
# que solo se define un parametro de entrada en la funcion.
def show_results(loaded_image):

    outputs_propios = modelo_propio.predict(source=loaded_image)[0]

    outputs_retail = modelo_retail.predict(source=loaded_image)[0]
    
    ploteo_propios = outputs_propios.plot()
    ploteo_retail = outputs_retail.plot()
   
    results1 = outputs_propios.cpu().numpy()
    results2 = outputs_retail.cpu().numpy()
    
    salida = dict()

    # Se recorre cada boundingBox detectado y se reporta a que producto corresponde.

    for i, det in enumerate(results2.boxes.xyxy):
            
            id_detectado = results2.names.get(results2.boxes.cls[i])

            if(str(id_detectado)) in salida:
                salida[str(id_detectado)] = salida[str(id_detectado)]+1
            else:
                salida[str(id_detectado)] = 1


    for i, det in enumerate(results1.boxes.xyxy):
        
        id_detectado = results1.names.get(results1.boxes.cls[i])

        if(str(id_detectado)) in salida:
            salida[str(id_detectado)] = salida[str(id_detectado)]+1
        else:
            salida[str(id_detectado)] = 1


    # Se retornan las 2 salidas definidas(imagen y texto): la imagen resultante (image) y un texto indicando
    # los productos y cantidades encontrados
    return cv2.cvtColor(ploteo_propios, cv2.COLOR_BGR2RGB), cv2.cvtColor(ploteo_retail, cv2.COLOR_BGR2RGB), textoSalida(salida)


inputs = [gr.components.Image(type="filepath", label="Imagen original"),
         ]
outputs= [gr.components.Image(type="numpy", label="Resultados propios"),
          gr.components.Image(type="numpy", label="Resultados retail"), 
          gr.Textbox(label="Productos:")
         ]
#examples  = "./productos_vitrina"

interface = gr.Interface(fn=show_results, 
                         inputs=inputs,
                         outputs=outputs,
                         title="Deteccion y conteo de productos",
                         #examples=examples,
                        )

interface.launch()

