import gradio as gr
import cv2
from recursos import textoSalida, colorCaja

from ultralytics import YOLO

#Cargar modelo entrenado
modelo_entrenado_propio = "./modelos_finales/yolov8x_test_01/weights/best.pt"
modelo_entrenado_retail = "./modelos_finales/yolov8x_retail/weights/best.pt"

modelo_retail = YOLO(modelo_entrenado_retail)
modelo_propio = YOLO(modelo_entrenado_propio)

# La interfaz solo recibe una entrada (La imagen ingresada en el cargador de path de imagenes), por lo
# que solo se define un parametro de entrada en la funcion.
def show_results(loaded_image):

    outputs_inicial = modelo_retail.predict(source=loaded_image)[0]

    outputs_final = modelo_propio.predict(outputs_inicial.plot())[0]
    
    ploteo = outputs_final.plot()
   
    results1 = outputs_inicial.cpu().numpy()
    results2 = outputs_final.cpu().numpy()
    
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
    return cv2.cvtColor(ploteo, cv2.COLOR_BGR2RGB), textoSalida(salida)


inputs = [gr.components.Image(type="filepath", label="Input Image"),
         ]
outputs= [gr.components.Image(type="numpy", label="Output Image"),
          gr.Textbox(label="Total:")
         ]
examples  = "./productos_vitrina"

interface = gr.Interface(fn=show_results, 
                         inputs=inputs,
                         outputs=outputs,
                         title="Object Detection",
                         examples=examples,
                        )
interface.launch()
