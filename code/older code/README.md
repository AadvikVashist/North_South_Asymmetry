
TITAN.PY is everything. It is the only file that is necessary. The others are just supplements.

#tif class
the tiff class does everything necessary to create the dataset images. You select the root folder for the dataset, the cylindrical projection, and the geographic csv and it creates all 96 images in a new folder system, whilst moving the two csv's.

#Titan
controls all the other classes. The default parameters are meant for my computer, but all you have to do is set the directory paramter to the root folder that hass all the datasets. The Titan Data folder has the wavelegnth key and dates for all the datasets, and is meant to serve as the global file location. The indiviudal datasets are sorted with a folder that shares the name of the dataset (T62), and contains the vis.cyl file, and two other folders. the vis.cyl subfolder has all the images and the csv sub folder contains all the important csv information.

To call the class:
if default paramters are set, then the only thing you have to put is the purpose, or if the info is implemented, the info as well. 
Purpose:
Purpose[0] is the main purpose. It will either be data, tilt, or figure.
If purpose [0] is figure, then purpose[1] will ahve the type of figuret that is being created
If purpose [0] is tilt, data, or if_Sh, the data class will be called with the directory information, and the other purpose parameters will be used
If purpose [0] is if_sh, then purpose[1] will either be None to show that all the images will be analyzed or a list with al lthe images that will be analyzed. purpose[2] can also be show or None, show if you want to see the way IF is found. 
If purpose[0] is data, then you can either have it write (purpose[1] == write) or have it run and print the values.
If purpose[0] is tilt, then the tilt of specfic bands will be found (purpose[2])

