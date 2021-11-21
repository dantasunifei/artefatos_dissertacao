import csv
import cv2
import cv2 as cv
import numpy as np
import math  

#variáveis globais
largura = 2

''' Distância entre dois pontos 2D '''
def DistanciaDoisPontos(x1,y1,x2,y2):
	dif = 0
	y1 = y2 = 0
	d1 = math.sqrt(pow((x2-0),2) - pow((y2-0),2))
	d2 = math.sqrt(pow((x1-0),2) - pow((y1-0),2))
	dif = d1 - d2	
	return dif 

''' Transformada de Hough Probabilístico '''
def HLP(dst, cdstP, minLineLength, maxLineGap):
    linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, minLineLength, maxLineGap)
    resultsP = 1
    soma_tga = 0
    if linesP is not None:
        for i in range(0, len(linesP)):
                l = linesP[i][0]
                distancia = DistanciaDoisPontos(l[0], l[1], l[2], l[3])
                tga = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180. / np.pi
                if distancia > 25:
                    resultsP = resultsP + 1
                    soma_tga += tga
    return cdstP, resultsP, soma_tga/resultsP

''' Encontra o ângulo das retas'''
def findAngulo(img):
    cimg = img.copy() # numpy function
    dst = cv.Canny(cimg, 50, 100, None, 5)   
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)
    cdstP, _, _ = HLP(dst, cdstP, 100, 5) 
    dstHLP = cv.Canny(cdstP, 50, 100, None, 5)   
    cdstP, resultsP, tgaP = HLP(dstHLP, cdstP, 100, 5) 
    return tgaP

''' Define a ação a ser executada '''
def defineAcao(tga):
    if tga == 0:
        return 0
    elif tga < 0:
        tga = -1. * tga
    elif tga < 30: #esquerda/desce
        return 2
    elif tga > 60: #direita/sobe
        return 1
    else:
        return 0  #mantem
