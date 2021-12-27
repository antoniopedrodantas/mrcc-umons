# flask imports
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

# other library imports
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# imports from src files
from src.functions import extractReqFeatures
from src.distances import *

app = Flask(__name__)

# tells where the image folder is
filenames = './static'

# function combines all image descriptors chosen
def concat_db_features(features):

    # instanciates new concatenated features
    new_features = []

    # adds instance of every image
    for elem in features[0]:
        new_features.append([elem[0], []])

    # adds concatenation of every chosen descriptor to the desired image
    for i in range(len(features)):
        # holds every image feature for the chosen descriptor
        descriptor = features[i]
        for j in range(len(descriptor)):
            # concatenates what was already there with the new descriptor
            new_features[j][1] = np.concatenate((new_features[j][1], descriptor[j][1]), axis=None)
    return new_features

# loads feautures of every image
def load_features(descriptors):

    # initializes array that will hold every image features for the chosen descriptors
    features_descriptors = []

    for descriptor in descriptors:
        # checks which descriptor will run
        folder_model = ''
        if (descriptor == 1):
            folder_model = './descriptors/BGR'
            print("Loading BGR features...")
        elif (descriptor == 2):
            folder_model = './descriptors/HSV'
            print("Loading HSV features...")
        elif (descriptor == 3):
            folder_model = './descriptors/SIFT'
            print("Loading SIFT features...")
        elif (descriptor == 4):
            folder_model = './descriptors/ORB'
            print("Loading ORB features...")
        elif (descriptor == 5):
            folder_model = './descriptors/GLCM'
            print("Loading GLCM features...")
        elif (descriptor == 6):
            folder_model = './descriptors/HOG'
            print("Loading HOG features...")
        elif (descriptor == 7):
            folder_model = './descriptors/LBP'
            print("Loading LBP features...")

        features = []
        # extracts features from the chosen descriptors
        for j in os.listdir(folder_model):
            data = os.path.join(folder_model, j)
            if not data.endswith(".txt"):
                continue
            feature = np.loadtxt(data)
            features.append(
                (os.path.join(filenames, os.path.basename(data).split('.')[0]+'.jpg'), feature))
        
        # appends features from one descriptor
        features_descriptors.append(features)
    
    return features_descriptors

# searches the most similar images
def search(file_name, features, descriptor, distanceName, results):

    # gets requested image features for every descriptor chosen
    req_features = []
    for elem in descriptor:
        req = extractReqFeatures(file_name, elem)
        req_features.append(req)
    
    # concatenates those features
    concat_req_features = []
    for elem in req_features:
        concat_req_features = np.concatenate((concat_req_features, elem), axis=None)

    # gets the k most similar neighbors
    neighbors = getkVoisins(features, concat_req_features, results, distanceName)

    # gets results names to display on the next webpage
    neighbor_names = []
    for k in range(results):
        neighbor_names.append(neighbors[k][0])

    return neighbor_names

# gets array of rappel and precision to draw the chart on the next webpage
def rappel_precision(file_name, neighbors):
    
    # gets number of neighbors
    number_of_neighbors = len(neighbors)
    # instanciates rappel precision variables
    rappel_precision = []
    rappels = []
    precisions = []

    # gets array of how many images correspond to the same thing and how many don't
    filename_req = os.path.basename(file_name)
    num_image, _ = filename_req.split(".")
    # classe_image_requete = int(num_image)/100
    classe_image_requete = int(num_image[0])
    val = 0
    for j in range(number_of_neighbors):
        if os.name != "nt":
            image_number = neighbors[j].replace("./static/", "")
        else :
            image_number = neighbors[j].replace("./static\\", "")
        # classe_image_proche = (int(image_number.split('.')[0]))/100
        classe_image_proche = int(image_number.split('.')[0][0])
        classe_image_requete = int(classe_image_requete)
        classe_image_proche = int(classe_image_proche)
        if classe_image_requete == classe_image_proche:
            rappel_precision.append(True)  # Bonne classe (pertinant)
            val += 1
        else:
            rappel_precision.append(False)  # Mauvaise classe (non pertinant)
    for i in range(number_of_neighbors):
        j = i
        val = 0
        while(j >= 0):
            if rappel_precision[j]:
                val += 1
            j -= 1
        # gets precision
        precision = (val/(i+1))*100
        #gets rappel
        rappel = (val/number_of_neighbors)*100
        rappels.append(rappel)
        precisions.append(precision)

    return [rappels, precisions]


@app.route('/', methods=["POST", "GET"])
def research():
    if request.method == "POST":

        # checks post validity
        if (request.form.get("file-name") == ""):
            return render_template("index.html")

        file_name_tmp = request.form["file-name"]
        file_name = "./static/" + file_name_tmp

        # TODO: check file_name validity

        # gets number of desired results
        results = int(request.form.get("results"))

        # gets descriptors
        descriptor = []
        if (request.form.get("descriptor_brg") == "BRG"):
            descriptor.append(1)
        if (request.form.get("descriptor_hsv") == "HSV"):
            descriptor.append(2)
        if (request.form.get("descriptor_glcm") == "GLCM"):
            descriptor.append(5)
        if (request.form.get("descriptor_hog") == "HOG"):
            descriptor.append(6)
        if (request.form.get("descriptor_lbp") == "LBP"):
            descriptor.append(7)

        # gets distance
        distance = request.form["distance"]

        # loads features
        features = load_features(descriptor)

        # concatenates features for all descriptors
        new_features = concat_db_features(features)

        # gets top n results
        neighbors = search(file_name, new_features, descriptor, distance, results)

        # calculates rappel and precision
        rappels_precisions = rappel_precision(file_name, neighbors)

        return render_template("results.html", image_path=file_name, neighbors_path=neighbors, rappel=rappels_precisions[0], precision=rappels_precisions[1], number_of_neighbors=len(neighbors))
    else:
        return render_template("index.html")

@app.route('/about', methods=["GET"])
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
