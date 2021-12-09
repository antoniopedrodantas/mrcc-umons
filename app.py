from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

from src.functions import extractReqFeatures
from src.distances import *

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

app = Flask(__name__)

# tells where the image folder is
filenames = './static'


def load_features(descriptor):

    # checks which descriptor will run
    folder_model = ''
    if (descriptor == 1):
        folder_model = './descriptors/BGR'
    elif (descriptor == 2):
        folder_model = './descriptors/HSV'
    elif (descriptor == 3):
        folder_model = './descriptors/SIFT'
    elif (descriptor == 4):
        folder_model = './descriptors/ORB'

    features = []
    print("Loading features...")
    # extracts features from the chosen descriptors
    for j in os.listdir(folder_model):
        data = os.path.join(folder_model, j)
        if not data.endswith(".txt"):
            continue
        feature = np.loadtxt(data)
        features.append(
            (os.path.join(filenames, os.path.basename(data).split('.')[0]+'.jpg'), feature))
    return features


def search(file_name, features, descriptor, distanceName):
    neighbors = ""
    # the second argument is the chosen algorithm
    # in this case it's 1 because it corresponds to the BGR algorithm
    req = extractReqFeatures(file_name, descriptor)
    # defines number of neighbors
    number_of_neighbors = 50
    # generates neighbors
    # distance name can be changed too as requested by the user
    # if descriptor == 3 or descriptor == 4:
    #     distanceName = "Brute force"
    # else:
    #     distanceName = "Bhattacharyya"
    neighbors = getkVoisins(features, req, number_of_neighbors, distanceName)
    neighbor_names = []
    for k in range(number_of_neighbors):
        neighbor_names.append(neighbors[k][0])
    return neighbor_names


def rappel_precision(file_name, neighbors):
    number_of_neighbors = len(neighbors)
    rappel_precision = []
    rappels = []
    precisions = []
    filename_req = os.path.basename(file_name)
    num_image, _ = filename_req.split(".")
    classe_image_requete = int(num_image)/100
    val = 0
    for j in range(number_of_neighbors):
        # will probably need to change the 9 for the other database
        image_number = neighbors[j].replace("./static/", "")
        classe_image_proche = (int(image_number.split('.')[0]))/100
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
        precision = (val/(i+1))*100
        rappel = (val/number_of_neighbors)*100
        rappels.append(rappel)
        precisions.append(precision)


    return [rappels, precisions]


@app.route('/', methods=["POST", "GET"])
def research():
    if request.method == "POST":

        # TODO: check post validity

        file_name_tmp = request.form["file-name"]
        file_name = "./static/" + file_name_tmp

        # TODO: check file_name validity

        # default descriptor is BRG
        descriptor = 1
        # gets descriptor id
        if (request.form["descriptor"] == "BRG"):
            descriptor = 1
        elif (request.form["descriptor"] == "HSV"):
            descriptor = 2
        elif (request.form["descriptor"] == "SIFT"):
            descriptor = 3
        elif (request.form["descriptor"] == "ORB"):
            descriptor = 4

        # gets distance
        distance = ""
        if (descriptor == 1):
            distance = request.form["brg-distance"]
        elif (descriptor == 2):
            distance = request.form["hsv-distance"]
        elif (descriptor == 3):
            distance = request.form["sift-distance"]
        elif (descriptor == 4):
            distance = request.form["orb-distance"]

        # loads features
        features = load_features(descriptor)

        # gets top 50 images
        neighbors = search(file_name, features, descriptor, distance)

        # calculates rappel and precision
        rappels_precisions = rappel_precision(file_name, neighbors)

        return render_template("results.html", image_path=file_name, neighbors_path=neighbors, rappel=rappels_precisions[0], precision=rappels_precisions[1], number_of_neighbors=len(neighbors))
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
