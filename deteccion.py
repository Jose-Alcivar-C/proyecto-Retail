import gradio as gr
import cv2
from recursos import textoSalida, colorCaja

from ultralytics import YOLO

#Cargar modelo entrenado
modelo_entrenado = "./runs/yolov8m_test_08/weights/best.pt"

model = YOLO(modelo_entrenado)

#Definir funcion que ejecuta la interfaz definida (en este caso es solo una interfaz, pero pueden ser algunas)
#La interfaz solo recibe una entrada (La imagen ingresada en el cargador de path de imagenes), por lo
# q ue solo se define un parametro de entrada en la funcion.
def show_results(loaded_image):
    #Se generan las salidas (detecciones) pidiendo al modelo que prediga a partir de la imagen de entrada
    outputs = model.predict(source=loaded_image)
    results = outputs[0].cpu().numpy()
    #Se carga la imagen usando openCV para poder editarla
    image = cv2.imread(loaded_image)
    #Se recorre cada boundingBox detectado y para cada uno se pinta un rectangulo y se escribe un id.
    
    salida = dict()

    for i, det in enumerate(results.boxes.xyxy):
        
        id_detectado = results.names.get(results.boxes.cls[i])

        cv2.rectangle(image,
                      (int(det[0]), int(det[1])),
                      (int(det[2]), int(det[3])),
                      color=colorCaja(id_detectado),
                      thickness=4,
                      lineType=cv2.LINE_AA
                     )

        if(str(id_detectado)) in salida:
            salida[str(id_detectado)] = salida[str(id_detectado)]+1
        else:
            salida[str(id_detectado)] = 1

        cv2.putText(image,
                    text =f"id:{str(id_detectado)}",
                    org=(int(det[0]), int(det[1])),
                    fontFace =cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=2,
                    color=colorCaja(id_detectado),
                    thickness=4,
                    lineType=cv2.LINE_AA
                   )
    #Se retornan las 2 salidas definidas(imagen y texto): la imagen resultante (image) y un texto indicando cuantos boundingBox se encontraron
    #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), len(results.boxes)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), textoSalida(salida)


inputs = [gr.components.Image(type="filepath", label="Input Image"),
         ]
outputs= [gr.components.Image(type="numpy", label="Output Image"),
          gr.Textbox(label="Total:")
         ]
examples  = "./output/test/images/"

interface = gr.Interface(fn=show_results, 
                         inputs=inputs,
                         outputs=outputs,
                         title="Object Detection",
                         #En la interfaz se pueden incluir ejemplos de lo que se espera como entrada o entradas. En este caso,
                         # la entrada es una imagen por lo que se pueden poner imagenes de ejemplo (deben estar subidas en el repositorio
                         # y con el path correctamente referenciado)
                         examples=examples,
                        )
interface.launch()
