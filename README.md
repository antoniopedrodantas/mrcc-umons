# Multimedia Retrieval & Cloud Computing

## Context

The goal of this project is to develop and host a multimedia indexing and search application on Cloud resources. It is divided into two main parts, one from the Multimedia Retrieval course and other from the Cloud and Edge Computing course.

On the first part we should develop a search engine that exploits the descriptors lectured in the Multimedia Retrieval course. We must create a script that indexes a vehicles database with the descriptors of our choosing giving the possibility to combine them. It should then be able to use those descriptors to retrieve similar images to the ones we provide, using different similarity calculation functions. Afterwards, we should analyse the search results regarding their recall and precision.

The second part aims to host our multimedia search application on a Cloud or Edge resource on the form of Software as a Service, SaaS.

![Search Page](https://i.imgur.com/2JFJu6y.png)

![Results Page](https://i.imgur.com/Yfa3kXu.png)

![About Page](https://i.imgur.com/uQmiiHn.png)

## Instructions

Move a ```/dataset``` folder into the ```/static``` directory to run our project.
Then, run the ```extract_features.py``` script to create our image descriptors.
When this is done, run the ```app.py``` script to initialize the application.
Afterward, the app can be accessed on "0.0.0.0:88" inside of any internet browser.

Further information inside our report, ```mrcc_report.pdf```.