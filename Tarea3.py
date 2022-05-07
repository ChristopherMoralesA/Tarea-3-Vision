#Importar las librerías por utilizar
import os
import numpy as np
from skimage import io
from skimage.transform import resize, probabilistic_hough_line,hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from skimage.color import rgb2gray

#-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

#PARAMETROS
#Tamaño de trabajo
SIZE=(1544,1158)

#numero de colores incluyendo el fondo
N_COLORS = 2

MAX_RHO = 18
MAX_THETA = 5*np.pi/180

#-*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

#FUNCIONES

def edit_image(image):
    y,x = image.shape[:2]
    y_div = y//3
    x_div = x//3
    x_off = 250
    cropped = image[y_div:2*y_div,x_div+x_off:2*x_div+x_off]
    resized = resize(cropped,(SIZE[0],SIZE[1]),preserve_range=True).astype(int)
    return resized

def recreate_image(codebook, labels, w, h):
    return codebook[labels].reshape(w, h, -1)

def img_to_2_colors(img):
    # Copia de la imagen por cuantificar
    img_5_colors = img
    # Convert to floats instead of the default 8 bits integer coding. Dividing by
    # 255 is important so that plt.imshow behaves works well on float data (need to
    # be in the range [0-1])
    img_5_colors = np.array(img_5_colors, dtype=np.float64) / 255
    # Load Image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img_5_colors.shape)
    assert d == 3
    image_array = np.reshape(img_5_colors, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=N_COLORS, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    colores = np.rint(np.array(kmeans.cluster_centers_, dtype=np.float64) *255)
    colores = np.array(colores,dtype=np.int32)
    img_quant = recreate_image(kmeans.cluster_centers_, labels, w, h)
    return colores,img_quant

def dist_puntos(p0,p1):
    return np.sqrt((p1[1]-p0[1])**2+(p1[0]-p0[0])**2)

def delete_long(linesdl):
    newlinesdl = []
    for linedl in linesdl:
        p0dl, p1dl = linedl
        len = dist_puntos(p0dl,p1dl)
        if len < SIZE[1]/2:
            newlinesdl.append(linedl)
    return newlinesdl

def lineas2puntos(lineas):
    puntos = []
    for i in lineas:
        puntos.append(i[0])
        puntos.append(i[1])
    return puntos

def centroides_dados(puntos):
    kmeans = KMeans(n_clusters=3, random_state=0).fit(puntos)
    centroides = kmeans.cluster_centers_
    return centroides

def image_circles(edges):
    circles = []
    hough_radii = np.arange(5, 10)
    hough_res = hough_circle(edges, hough_radii)
    # Select the most prominent 18 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=18)
    for i in range(len(cx)):
        temp = (cx[i],cy[i])
        circles.append(temp)
    return circles

def uniq_circles(circulos):
    min_diff = 2
    uniq = []
    uniq.append(circulos[0])
    for i in circulos:
        uniq_circ = True
        for k in uniq:
            dist = dist_puntos(i,k)
            #Si no es unico
            if dist <= min_diff:
                uniq_circ = False
        if uniq_circ:
            uniq.append(i)
    return uniq

def segmentacion_por_dado(dados,puntos_unicos):
    dist_max = 40
    dado_correspondiente = []
    for i in puntos_unicos:
        for k in range(len(dados)):
            dist = dist_puntos(i,dados[k])
            if dist <= dist_max:
                dado_correspondiente.append(k)
    puntos_d1 = dado_correspondiente.count(0)
    puntos_d2 = dado_correspondiente.count(1)
    puntos_d3 = dado_correspondiente.count(2)
    total = puntos_d1 + puntos_d2 + puntos_d3
    return [puntos_d1,puntos_d2,puntos_d3],total

def cuenta_puntos(image):
    img_edit = edit_image(image)
    COLORES, img_quant=img_to_2_colors(img_edit)
    image_gray = rgb2gray(img_quant)*255**3/2
    edgesHT = canny(image_gray, 2, 1, 25)
    linesHT1 = probabilistic_hough_line(edgesHT, threshold=10, line_length=30, line_gap=3)
    linesHT2 = delete_long(linesHT1)
    puntosHT = lineas2puntos(linesHT2)
    centroides = centroides_dados(puntosHT)
    c_puntos_dados = image_circles(edgesHT)
    puntos_unicos = uniq_circles(c_puntos_dados)
    puntos_por_dado,total = segmentacion_por_dado(centroides,puntos_unicos)
    return puntos_por_dado,total

def main():
    if os.path.exists("report_file.txt"):
        os.remove("report_file.txt")
    report_file = open("report_file.txt", "x")
    path = os.getcwd()
    folder = 'Imagenes_por_analizar'
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)
    for file in files:
        res_report = ''
        file_path = os.path.join(folder_path, file)
        foto = io.imread(file_path)
        puntos_por_dado, puntos_totales = cuenta_puntos(foto)
        resultado = f": Valor de cada dado: {puntos_por_dado[0]}, {puntos_por_dado[1]} y {puntos_por_dado[2]}, y los puntos totales son {puntos_totales}."
        res_report =file + resultado + '\n'
        report_file.write(res_report)
    report_file.close()

if __name__ == "__main__": main()