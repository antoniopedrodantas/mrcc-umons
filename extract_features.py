# Defintion de toute les fonctions à appeller dans l'interface
import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.io import imread
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern
from skimage import exposure
from skimage import img_as_ubyte

def generateHistogramme_HSV(filenames):
    if not os.path.isdir("./descriptors/HSV"):
        os.mkdir("./descriptors/HSV")
    i = 0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames + "/" + path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([img], [0], None, [180], [0, 180])
            histS = cv2.calcHist([img], [1], None, [256], [0, 256])
            histV = cv2.calcHist([img], [2], None, [256], [0, 256])
            feature = np.concatenate(
                (histH, np.concatenate((histS, histV), axis=None)), axis=None)

            num_image, _ = path.split(".")
            # with open("./descriptors/HSV/"+str(num_image)+".txt", 'w+') as f:
            #     np.savetxt(f ,feature)
            np.savetxt("./descriptors/HSV/"+str(num_image)+".txt" ,feature)
            i += 1
        else:
            print("Problem loading image: ", path)
    print("Finished HSV indexation (2/5)")


def generateHistogramme_Color(filenames):
    if not os.path.isdir("./descriptors/BGR"):
        os.mkdir("./descriptors/BGR")
    i = 0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        if img is not None:
            histB = cv2.calcHist([img], [0], None, [256], [0, 256])
            histG = cv2.calcHist([img], [1], None, [256], [0, 256])
            histR = cv2.calcHist([img], [2], None, [256], [0, 256])
            feature = np.concatenate(
                (histB, np.concatenate((histG, histR), axis=None)), axis=None)

            num_image, _ = path.split(".")
            with open("./descriptors/BGR/"+str(num_image)+".txt", 'w+') as f:
                np.savetxt(f ,feature)
            i += 1
        else:
            print("Problem loading image: ", path)
    print("Finished BGR indexation (1/5)")


# def generateSIFT(filenames):
#     if not os.path.isdir("./descriptors/SIFT"):
#         os.mkdir("./descriptors/SIFT")
#     i = 0
#     for path in os.listdir(filenames):
#         img = cv2.imread(filenames+"/"+path)
#         featureSum = 0
#         sift = cv2.SIFT_create()
#         kps, des = sift.detectAndCompute(img, None)

#         num_image, _ = path.split(".")
#         with open("./descriptors/SIFT/"+str(num_image)+".txt", 'w+') as f:
#             np.savetxt(f ,des)

#         featureSum += len(kps)
#         i += 1
#     print("Indexation SIFT terminée !!!!")


def generateORB(filenames):
    if not os.path.isdir("./descriptors/ORB"):
        os.mkdir("./descriptors/ORB")
    i = 0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        orb = cv2.ORB_create()
        key_point1, descrip1 = orb.detectAndCompute(img, None)

        num_image, _ = path.split(".")
        with open("./descriptors/ORB/"+str(num_image)+".txt", 'w+') as f:
            np.savetxt(f ,descrip1)
        i += 1
    print("indexation ORB terminée !!!!")


## TP3
def generateGLCM(filenames):
    if not os.path.isdir("./descriptors/GLCM"):
        os.mkdir("./descriptors/GLCM")
    distances = [1, -1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    i = 0
    for path in os.listdir(filenames):
        image = cv2.imread(filenames+"/"+path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = img_as_ubyte(gray)
            glcmMatrix = greycomatrix(gray, distances=distances, angles=angles,
                                    normed=True)
            glcmProperties1 = greycoprops(glcmMatrix, 'contrast').ravel()
            glcmProperties2 = greycoprops(glcmMatrix, 'dissimilarity').ravel()
            glcmProperties3 = greycoprops(glcmMatrix, 'homogeneity').ravel()
            glcmProperties4 = greycoprops(glcmMatrix, 'energy').ravel()
            glcmProperties5 = greycoprops(glcmMatrix, 'correlation').ravel()
            glcmProperties6 = greycoprops(glcmMatrix, 'ASM').ravel()
            feature = np.array([glcmProperties1, glcmProperties2, glcmProperties3, glcmProperties4, glcmProperties5, glcmProperties6]).ravel()
            num_image, _ = path.split(".")
            with open("./descriptors/GLCM/"+str(num_image)+".txt", 'w+') as f:
                np.savetxt(f ,feature)
            i += 1
        else:
            print("Problem loading image: ", path)
        
    print("Finished GLCM indexation (3/5)")
    
def generateLBP(filenames):
    if not os.path.isdir("./descriptors/LBP"):
        os.mkdir("./descriptors/LBP")
    i=0
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        if img is not None:
            points=8
            radius=1
            method='default'
            subSize=(70,70)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img,(350,350))
            fullLBPmatrix = local_binary_pattern(img,points,radius,method)
            histograms = []
            for k in range(int(fullLBPmatrix.shape[0]/subSize[0])):
                for j in range(int(fullLBPmatrix.shape[1]/subSize[1])):
                    subVector = fullLBPmatrix[k*subSize[0]:(k+1)*subSize[0],j*subSize[1]:(j+1)*subSize[1]].ravel()
                    subHist,edges = np.histogram(subVector,bins=int(2**points),range=(0,2**points))
                    histograms = np.concatenate((histograms,subHist),axis=None)
            num_image, _ = path.split(".")
            with open("./descriptors/LBP/"+str(num_image)+".txt", 'w+') as f:
                np.savetxt(f ,histograms)
            i+=1
        else:
            print("Problem loading image: ", path)
        
    print("Finished LBP indexation (5/5)")


def generateHOG(filenames):
    if not os.path.isdir("./descriptors/HOG"):
        os.mkdir("./descriptors/HOG")
    i=0
    cellSize = (25,25)
    blockSize = (50,50)
    blockStride = (25,25)
    nBins = 9
    winSize = (350,350)
    for path in os.listdir(filenames):
        img = cv2.imread(filenames+"/"+path)
        if img is not None:
            image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,winSize)
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins)
            feature = hog.compute(image)
            num_image, _ = path.split(".")
            with open("./descriptors/HOG/"+str(num_image)+".txt", 'w+') as f:
                np.savetxt(f ,feature)
            i+=1
        else:
            print("Problem loading image: ", path)
    print("Finished HOG indexation (4/5)")


generateHistogramme_Color("./static/dataset")
generateHistogramme_HSV("./static/dataset")
# Be careful with these lines!
# generateSIFT("./static/dataset")
# generateORB("./static/dataset")
generateGLCM("./static/dataset")
generateHOG("./static/dataset")
generateLBP("./static/dataset")
