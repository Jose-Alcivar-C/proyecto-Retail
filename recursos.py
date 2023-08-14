from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d
import logging
import platform


def emojis(str=''):
    # Return platform-dependent emoji-safe version of string
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


LOGGER = logging.getLogger("yolov8")  # define globally (used in train.py, val.py, detect.py, etc.)
if platform.system() == 'Windows':
    for fn in LOGGER.info, LOGGER.warning:
        setattr(LOGGER, fn.__name__, lambda x: fn(emojis(x)))


def plot_results(file='path/to/results.csv', dir=''):
    # Plot training results.csv. Usage: from utils.plots import *; plot_results('path/to/results.csv')
    save_dir = Path(file).parent if file else Path(dir)
    fig, ax = plt.subplots(2, 5, figsize=(12, 6), tight_layout=True)
    ax = ax.ravel()
    files = list(save_dir.glob('results*.csv'))
    assert len(files), f'No results.csv files found in {save_dir.resolve()}, nothing to plot.'
    for f in files:
        try:
            data = pd.read_csv(f)
            s = [x.strip() for x in data.columns]
            x = data.values[:, 0]
            for i, j in enumerate([1, 2, 3, 4, 5, 8, 9, 10, 6, 7]):
                y = data.values[:, j].astype('float')
                # y[y == 0] = np.nan  # don't show zero values
                ax[i].plot(x, y, marker='.', label=f.stem, linewidth=2, markersize=8)  # actual results
                ax[i].plot(x, gaussian_filter1d(y, sigma=3), ':', label='smooth', linewidth=2)  # smoothing line
                ax[i].set_title(s[j], fontsize=12)
                # if j in [8, 9, 10]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except Exception as e:
            LOGGER.info(f'Warning: Plotting error for {f}: {e}')
    ax[1].legend()
    fig.savefig("./ploteos/results.png", dpi=200)
    plt.close()


def textoSalida(diccionario):
    
    impresion = ""

    for dato in diccionario:
        
        impresion = impresion + f"Productos con id {dato}, hay {str(diccionario[dato])}\n"

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
