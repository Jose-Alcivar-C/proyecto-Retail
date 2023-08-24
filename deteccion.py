import gradio as gr
import cv2
from recursos import textoSalida, colorCaja

from ultralytics import YOLO

#Cargar modelo entrenado
modelo_entrenado_propio = "./modelos_finales/yolov8x_propios/weights/best.pt"
modelo_entrenado_retail = "./modelos_finales/yolov8x_retail/weights/last.pt"

model = YOLO(modelo_entrenado_retail)

#Definir funcion que ejecuta la interfaz definida (en este caso es solo una interfaz, pero pueden ser algunas)
#La interfaz solo recibe una entrada (La imagen ingresada en el cargador de path de imagenes), por lo
# q ue solo se define un parametro de entrada en la funcion.
def show_results(loaded_image):
    
    #Se generan las salidas (detecciones) pidiendo al modelo que prediga a partir de la imagen de entrada
    outputs = model.predict(source=loaded_image)[0]
    
    ploteo = outputs.plot()
    
    results = outputs.cpu().numpy()
    
    salida = dict()

    #Se recorre cada boundingBox detectado y para cada uno se pinta un rectangulo y se escribe un id.

    for i, det in enumerate(results.boxes.xyxy):
        
        id_detectado = results.names.get(results.boxes.cls[i])

        if(str(id_detectado)) in salida:
            salida[str(id_detectado)] = salida[str(id_detectado)]+1
        else:
            salida[str(id_detectado)] = 1

    #Se retornan las 2 salidas definidas(imagen y texto): la imagen resultante (image) y un texto indicando cuantos boundingBox se encontraron
    #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), len(results.boxes)
    return cv2.cvtColor(ploteo, cv2.COLOR_BGR2RGB), textoSalida(salida)


inputs = [gr.components.Image(type="filepath", label="Input Image"),
         ]
outputs= [gr.components.Image(type="numpy", label="Output Image"),
          gr.Textbox(label="Total:")
         ]
#examples  = ""

interface = gr.Interface(fn=show_results, 
                         inputs=inputs,
                         outputs=outputs,
                         title="Object Detection",
                         
                         #examples=examples,
                        )
interface.launch()
