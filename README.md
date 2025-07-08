HUMAN GESTURE RECOGNITION PROJECT

PARTICIPANTS:
ΔΗΜΗΤΡΙΟΣ ΜΠΑΡΟΥΤΗΣ 1084549, ΓΕΩΡΓΙΟΣ ΜHΤΣΑΙΝΑΣ 1084527

Introduction: 

Firstly we used a windows PC and VScode environment, downloaded the necessary python libraries in order to collect the data from the MetaWear device and used a Bluetooth adapter to connect with it. The API used for the data collection script was provided in MetaBase's documentation, although we altered it because it was outdated and many functions werent working properly. Then, we used a mongoDB atlas in order to upload our sensor data and organize it in a specific folder structure. And finally we process the data collected and use Machine Learning in order to classify the different gestures.


Here is how to run the project successfully: 

1) Run the sensor_data_combined.py script for the data collection. The CSVs collected should be put in specific file order provided in the project description.

2) Run the mongo.py to upload .csv files in your mongoDB Atlas database

3) Use the AI_model.py file provided which will automatically take our modified parameters from utils.py and utils_visual.py in order to filter and train the data.

*Notebooks are just for illustrative and informative purposes.


Gestures used: 

1) Rubbing of the nose, class_name: nose rubbing.

2) Knee itching, class_name: scratch.

3) Waving, class_name: wave.


Collection procedure: 

The gestures we had to perform were simple, but we needed to be careful. First of all, there was the problem with the BLE, where it was missing or duplicating data because of partial signal loss, which was solved easily though it took a while to notice. The movements used for the data training and testing had to be very periodic-like movements, in order to minimize the noise. The subjects ( Dimitris and Giorgos in this case) both had to be very careful in the collection process, one was pressing the start button and the other was already performing the said gesture, and data was always graphed afterwards in order to check for any anomalies. Then we had the .csv files all organized by gesture class in folders in order to upload them to our mongo database. 


Possible problems:

1) As mentioned in the introduction, the documentation was outdated, but we managed to make the functions inside the script work, provided you are running on python 3.13.1.

2) The script itself needed some optimization, because running both accelerometer and gyroscope was probably stressful for the BLE itself and had some streaming/logging problems (duplicated and/or missing data)

3) A problem that occurred was that the MetaWear device used BLE (Bluetooth low energy), which we noticed was underperforming if the subject completing the gestures was a little further away, which we easily solved by placing the sensor and the Bluetooth adapter closer together.