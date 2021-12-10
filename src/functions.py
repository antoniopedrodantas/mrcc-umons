#Defintion de toute les fonctions à appeller dans l'interface
import os
import cv2
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
from skimage import io, color, img_as_ubyte
from matplotlib import pyplot as plt
from skimage.feature import hog, greycomatrix, greycoprops, local_binary_pattern



def lbpDescriptor(image):                 # function : return LBP Image
# settings for LBP
  METHOD = 'uniform'
  radius = 3
  n_points = 8 * radius
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  lbp = local_binary_pattern(gray, n_points, radius, METHOD)
  return lbp
	
def extractReqFeatures(fileName,algo_choice):  
    if fileName : 
        img = cv2.imread(fileName)
        resized_img = resize(img, (128*4, 64*4))
            
        if algo_choice==1: #Couleurs
            histB = cv2.calcHist([img],[0],None,[256],[0,256])
            histG = cv2.calcHist([img],[1],None,[256],[0,256])
            histR = cv2.calcHist([img],[2],None,[256],[0,256])
            vect_features = np.concatenate((histB, np.concatenate((histG,histR),axis=None)),axis=None)
        
        elif algo_choice==2: # Histo HSV
            hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
            histH = cv2.calcHist([hsv],[0],None,[180],[0,180])
            histS = cv2.calcHist([hsv],[1],None,[256],[0,256])
            histV = cv2.calcHist([hsv],[2],None,[256],[0,256])
            vect_features = np.concatenate((histH, np.concatenate((histS,histV),axis=None)),axis=None)

        elif algo_choice==3: #SIFT
            sift = cv2.SIFT_create() #cv2.xfeatures2d.SIFT_create() pour py < 3.4 
            # Find the key point
            kps , vect_features = sift.detectAndCompute(img,None)
    
        elif algo_choice==4: #ORB
            orb = cv2.ORB_create()
            # finding key points and descriptors of both images using detectAndCompute() function
            key_point1,vect_features = orb.detectAndCompute(img,None)

        elif algo_choice==5: #GLCM
            distances = [1, -1]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = img_as_ubyte(gray)
            glcmMatrix = greycomatrix(gray, distances=distances, angles=angles,
                                normed=True)
            glcmProperties1 = greycoprops(glcmMatrix, 'contrast').ravel()
            glcmProperties2 = greycoprops(glcmMatrix, 'dissimilarity').ravel()
            glcmProperties3 = greycoprops(glcmMatrix, 'homogeneity').ravel()
            glcmProperties4 = greycoprops(glcmMatrix, 'energy').ravel()
            glcmProperties5 = greycoprops(glcmMatrix, 'correlation').ravel()
            glcmProperties6 = greycoprops(glcmMatrix, 'ASM').ravel()
            vect_features = np.array([glcmProperties1, glcmProperties2, glcmProperties3, glcmProperties4, glcmProperties5, glcmProperties6]).ravel()

        elif algo_choice==6: #HOG
            cellSize = (25,25)
            blockSize = (50,50)
            blockStride = (25,25)
            nBins = 9
            winSize = (350,350)
            image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image,winSize)
            hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nBins)
            vect_features = hog.compute(image)

        elif algo_choice==7: #LBP
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
            vect_features = histograms
			
        np.savetxt("Methode_"+str(algo_choice)+"_requete.txt" ,vect_features)
        return vect_features