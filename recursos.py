def textoSalida(diccionario):
    
    impresion = ""

    if(len(diccionario)>0):
        
        for dato in diccionario:
            
            impresion = impresion + f"Productos con id {dato}, hay {str(diccionario[dato])}\n"

    else:
        impresion = "La imagen no contiene productos reconocidos."
        
    return impresion


def colorCaja(idDetectado):

    colores = {
        "75":(0, 0, 255), 
        "78":(0, 189, 255), 
        "79":(0, 255, 158), 
        "125":(135, 255, 0), 
        "127":(255, 162, 0), 
        "128":(255, 0, 135), 
        "140":(151, 0, 255)
        }

    return colores[str(idDetectado)]
