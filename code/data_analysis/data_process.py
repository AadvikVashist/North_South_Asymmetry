import os
import os.path as path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from PIL import Image, ImageStat
from scipy.optimize import curve_fit
import time
import math
from sklearn.metrics import r2_score
import PIL
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import brentq
from scipy.stats import linregress

import json
import pyvims
# flyby directory location
# name of csv containing all data (inside flyby folder)
# name of flyby image data (inside flyby folder)
# save loation (default to flyby data)
import numpy as np
import cv2
# surface_windows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True]
surface_windows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True]
ir_surface_windows = [99,100,106,107,108,109,119,120,121,122,135,136,137,138,139,140,141,142,163,164,165,166,167,206,207,210,211,212,213,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352]
class analyze_image:
    def __init__(self):
        self = self
        self.figures = {
            "original_image" : False,
            "shifted_unrefined_image" : True,
            "gaussian_brightness_bands" : True,
            "unrefined_north_south_boundary" : True,
            "cropped_shifted_image" : False,
            "nsb_column" : True,
            "visualized_plotting" : False,
            "boundary_vs_longitude" : False,
            "persist_figures" : 1
        }
        self.figure_keys = {key: index for index, (key, value) in enumerate(self.figures.items())}
    def figure_options(self):
        print("\nCurrent Figure Setttings:\n",*[str(val[0]) + " = " + str(val[1]) for val in self.figures.items()], "\n\n", sep = "\n")
        return self.figures
    def show(self, force = False):
        if force:
            plt.pause(10)
            return
        options = self.figures["persist_figures"]
        if type(options) == int:
            plt.pause(options)
            plt.clf()
            plt.close()
        elif options == "wait":
            plt.waitforbuttonpress()
            plt.close()
        elif options == "wait_till_end":
            return
        elif options == True:
            plt.show()
        else:
            plt.pause(2)
            plt.clf()
            plt.close()
    def get_vmin_vmax(self, image):
        return np.quantile(image, [0.1, 0.9])
    
    def gauss2(self, x, b, a, x0, sigma):
        return b + (a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)))
    def poly12(self, x, a, b, c, d, e, f, g, h, i, j, k, l, m):
        return a*x**12+b*x**11+c*x**10+d*x**9+e*x**8+f*x**7+g*x**6+h*x**5+i*x**4+j*x**3+k*x**2+l*x+m
    def polyXDerivative(self, *args): # 5)
        return [(len(args) - index-1)*value for index, value in enumerate(args)]
    def polyXPlugin(self, x,  *args): # 5)
        y = 0
        for index, value in enumerate(args):
            y+=value*x**(len(args) - index-1)
        return y
    def poly6(self, x, g, h, i, j, a, b, c):  # sextic function
        return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c
    def poly6Prime(self, x, g, h, i, j, a, b, c):  # derivative of sextic function
        return 6*g*x**5+5*h*x**4+4*i*x**3+3*j*x**2+2*a*x+b
    # derivative of sextic function coefficents
    def poly6Derivative(self, g, h, i, j, a, b, c):
        return [6*g, 5*h, 4*i, 3*j, 2*a, b]
    
    def remove_outliers(self, arr, threshold=1.5):
        arr = np.array(arr)
        mask = np.isnan(arr)  # Create a mask for NaN values
        
        mean = np.nanmean(arr)  # Compute mean ignoring NaNs
        std = np.nanstd(arr)  # Compute standard deviation ignoring NaNs
        cutoff = std * threshold
        lower_bound = mean - cutoff
        upper_bound = mean + cutoff
        
        arr[(arr < lower_bound) | (arr > upper_bound)] = np.nan
        arr[mask] = np.nan  # Restore NaN values
        return arr
    def compare_acceleration_to_flux(self, x, minimum, brightness_array, shift: int = 0):
        # return True
        x= int(x) + shift
        brightness_arr = brightness_array - np.min(brightness_array)
        flux = np.mean(brightness_arr[x:int(1.1*x)])/np.mean(brightness_arr[int(0.9*x):x]) - 1
        if minimum and flux < 0:
            return True
        elif flux > 0 and not minimum:
            return True
        else:
            return False
    def min_or_max(self, function, x, amount, recurse): # true is min
        if recurse > 5:
            return None

        before = function(x-amount)
        after = function(x+amount)
        if after > 0 and before > 0 or after < 0 and before < 0:
            return None
        elif before == 0 or after == 0:
            return self.min_or_max(function,x,amount/10, recurse+1)
        elif after > 0 and before < 0:
            return True
        elif before > 0 and after < 0:
            return False
        else:
            raise ValueError("bad")
        # if after > 0 and before < 0:
        #     return True
        # # if after > before and after_two > before_two:
        # #     return True
        # # elif after < before and after_two < before_two:
        # #     return False
        # else:
        #     return None

    def convert_array_to_image(self, array, savename):
        if array is not np.array:
            array = np.array(array, dtype=np.float32)
        else:
            array.astype(np.uint8)
        quintiles = np.quantile(array.compressed(),np.arange(0,1.01,0.01))
        # quintile_range = [quintiles[i]-quintiles[i-1] for i in range(4,len(quintiles-3))]
        # min = 0
        # max = 0
    
        min = quintiles[15]
        max = quintiles[-16]
        
        

        if min != 0:
            array -= min
        max = max-min
        array = np.clip(array, 0, max)
        ranges = np.ptp(array)
        if ranges != 255:
            array *= 255 / ranges
        array.astype(np.int8)
        im = PIL.Image.fromarray(array)
        im = im.convert("L")
        im.save(savename+ ".png")
    def north_facing(self, image, rotation):
        return "not implemented yet"
    def equirectangular_projection(self, cube, index):
        # take image and apply cylindrical projection
        # after cylindrical projection, remove extraneous longitude data
        proj = pyvims.projections.equirectangular.equi_cube(cube,index,3)
        return proj
    def shift_image(self, image, shift, crop : bool = True):
        subtraction = (np.insert(image, 0, np.array(shift*2*[[0]*image.shape[1]]), axis=0) - np.concatenate((image, shift*2*[[0]*image.shape[1]])))
        hc_band = subtraction[2*shift:-2*shift]
        nan_rows = [False for i in range(shift)] + [True for i in range(hc_band.shape[0])] + [False for i in range(shift)]
        if crop:
            imager = image.filled(fill_value=np.nan)
            nan_columns = np.isnan(imager).sum(axis=0) > 0.15 * image.shape[0]
            hc_band = hc_band[:, ~nan_columns]
            return hc_band, nan_rows, nan_columns
        
        return hc_band, nan_rows
    
    def pixel_to_geo(self, pixel, geo_list): #convert pixel to lat or pixel to lon
        return geo_list[int(np.around(pixel))]
    def pixel_to_geo_interpolated(self, pixel, geo_list):
        return geo_list[int(np.around(pixel))]
    def geo_to_pixel(self, geo, geo_list): #convert lat to pixel or lon to pixel
        return np.min(range(len(geo_list)), key=lambda x: abs(geo_list[x]-geo))
    
    def find_zeros(self, f, interval):
        zeros = []
        x = np.linspace(interval[0], interval[1], 1000)
        y = f(x)
        for i in range(len(x) - 1):
            if np.sign(y[i]) != np.sign(y[i + 1]):
                zero = brentq(f, x[i], x[i + 1])
                zeros.append(zero)
        return zeros
    def fit_line_and_get_roots(self, y):
        x = list(range(len(y)))
        fitted_line, _ = curve_fit(self.poly6, x, y)    
        fitted_line_derivative = self.poly6Derivative(*fitted_line)
        try:
            derivativeRoot = np.roots(fitted_line_derivative)
        except:
            return False
        # remove extraneous soulutions (imaginary)
        realDerivativeRoots = derivativeRoot[np.isreal(derivativeRoot)]
        if len(realDerivativeRoots) == 0:
            return False
        drIndex = min(range(len(realDerivativeRoots)), key=lambda x: abs(
            realDerivativeRoots[x]-len(y)/2))  # find value closest to NSA
        derivativeRoots = realDerivativeRoots[drIndex]
        return derivativeRoots.real
    def cubic_spline_derivative_zero_maximum(self, x, y):
        cs = CubicSpline(x,y)
        derivative = cs.derivative()
        zeros = self.find_zeros(derivative, [0, len(x)])

        zeros = [zero for zero in zeros if zero > np.max(x)/6 and zero < np.max(x)*5/6]
        if len(zeros) == 0:
            return np.nan
        values_at_zeros = [cs(zero) for zero in zeros]

        max_min = np.polyfit(x, y, deg= 1)[0]
        if max_min > 0:
            north_south_boundary_index = np.argmax(values_at_zeros)
        else:
            north_south_boundary_index = np.argmin(values_at_zeros)
        return zeros[north_south_boundary_index]
    
    def locate_north_south_boundary_unrefined(self, image):
        self.shifted_image, cropped_rows, cropped_columns = self.shift_image(image, 20)
        top_shift = cropped_rows.index(True)
        if self.figures["gaussian_brightness_bands"]:
            plt.figure(self.figure_keys["gaussian_brightness_bands"])
            for i in range(0,self.shifted_image.shape[1], 10):
                average_brightness =  gaussian_filter(self.shifted_image[:,i], sigma=3)
                plt.plot(range(self.shifted_image.shape[0]), average_brightness)
            plt.title("Raw Brightness Data with Gaussian Filter")
            plt.xlabel("Latitude")
            plt.ylabel("Brightness")
            self.show()
        
        latitudes = list(range(self.shifted_image.shape[0]))
        average_brightness = np.mean(self.shifted_image, axis=1)

        gaussian_y = gaussian_filter(average_brightness, sigma=15)

        cs = CubicSpline(latitudes,gaussian_y)
        plottedy = [cs(xa)for xa in latitudes]
        derivative = cs.derivative()

        # Find the zeros of the derivative function
        # sign_changes = np.where(np.diff(np.sign(derivative(x))) != 0)[0]
        zeros = self.find_zeros(derivative, [0, image.shape[0]])
        zeros = [zero for zero in zeros if zero > image.shape[0]/3 and zero < 2/3*image.shape[0]]
        if len(zeros) == 1:
            return zeros
        elif len(zeros) == 0:
            raise ValueError("didn't work")
        # remove extraneous soulutions (imaginary)

        north_south_mins = [approx for approx in zeros if self.min_or_max(derivative, approx, 5, 0) == True and self.compare_acceleration_to_flux(approx, True, average_brightness,0)]
        north_south_maxes = [approx for approx in zeros if self.min_or_max(derivative, approx, 5, 0) == False and self.compare_acceleration_to_flux(approx, False, average_brightness,0)]

        if self.figures["unrefined_north_south_boundary"] == True:
            plt.figure(self.figure_keys["unrefined_north_south_boundary"])
            plt.plot(latitudes,average_brightness,label="shifted data") #values
            # plt.plot(x,gaussian_y) #filtered values
            plt.plot(latitudes,plottedy,label="spline values") #splien values
            # plt.plot(x,plottedy)
            # Obtain the derivative function
            plt.plot(latitudes,[derivative(xa)*20 for xa in latitudes], color = (0,0,1,1), linestyle = "dotted",label="derivative")
            plt.hlines([0], [0],[image.shape[0]], colors = (0,0,0), linestyles='dashed',label="0 lat")
            plt.grid(True, 'both')
            if len(north_south_mins) != 0:
                plt.vlines(north_south_mins, np.min(average_brightness)/2+np.mean(average_brightness), np.max(average_brightness)/2+np.mean(average_brightness), colors = (0,0,0), linestyles='dashed',label="min")
            if len(north_south_maxes) != 0:
                plt.vlines(north_south_maxes, np.min(average_brightness)/2+np.mean(average_brightness), np.max(average_brightness)/2+np.mean(average_brightness), colors = (1,0,0), linestyles='dashed',label="max")
            plt.legend()
            plt.title("Mean Brightness Data Asymmetry Location")
            plt.xlabel("Latitude")
            plt.ylabel("Brightness")
            # plt.plot([self.polyXPlugin(ax, *fitted_line_derivative) for ax in x])
            self.show()
        
        if len(north_south_mins) + len(north_south_maxes) == 0:
            return image.shape[0]/2
        elif len(north_south_mins) + len(north_south_maxes) == 1:
            north_south_maxes.extend(north_south_mins)
            return float(north_south_maxes[0]) + top_shift
        else:
            north_south_maxes.extend(north_south_mins)
            north_south_maxes = sorted(north_south_maxes, key= lambda x: x-image.shape[1]/2)
            return float(north_south_maxes[0]) + top_shift
    def locate_north_south_boundary_refined(self, image, min_lat_pixel, max_lat_pixel, shift_amount):
        nsa_lats_with_outliers = []
        nsa_lat_pixels_with_outliers = []
        image = image[min_lat_pixel:max_lat_pixel]
        shifted_image, cropped_rows, cropped_columns = self.shift_image(image, 10, True)
        if self.figures["cropped_shifted_image"] == True:
            plt.figure(self.figure_keys["cropped_shifted_image"])
            plt.imshow(shifted_image, cmap = "gray")
            plt.title("Cropped Projection for Refined Processing")
            self.show()
        latitudes = list(range(shifted_image.shape[0]))
        longitudes = list(range(shifted_image.shape[1]))
        # self.convert_array_to_image(self.hc_band, os.path.join(self.save_location, "high_contrasting_band", self.cube_name))
        #column analysis
        # lat_pixel_range = np.array(range(image.shape[0])); bounding_box_latitude_pixels = (lat_pixel_range > min_lat_pixel and lat_pixel_range < max_lat_pixel)
        standard_deviations = [np.std(shifted_image[:,longitude_pixel]) for longitude_pixel in range(shifted_image.shape[1])]
        
        standard_deviations_sorted = np.array(sorted(zip(standard_deviations, range(len(standard_deviations)))))
        quantile = standard_deviations_sorted[int(0.9*len(standard_deviations_sorted))][0]
        if self.figures["nsb_column"] == True:
            plt.figure(self.figure_keys["nsb_column"])
            plt.title("Asymmetry Predictions per Longitude")
            plt.xlabel("Latitude")
            plt.ylabel("Brightness")
            plt.vlines(0,0,0,colors = (0,0,0), label = "prediction")
            plt.legend()
        for longitude_pixel in range(shifted_image.shape[1]):
            deviation = standard_deviations[longitude_pixel]
            if np.all(image[:,longitude_pixel] == image[:,longitude_pixel][0]) or deviation <= quantile:
                nsa_lats_with_outliers.append(np.nan)
                nsa_lat_pixels_with_outliers.append(np.nan)
                continue
            bounding_box_data = shifted_image[:, longitude_pixel]
            gaussian_y = gaussian_filter(bounding_box_data, sigma = 4)
            # relative_height = self.fit_line_and_get_roots(bounding_box_data)
            relative_height = self.cubic_spline_derivative_zero_maximum(latitudes,gaussian_y)
            # Find the zeros of the derivative function
            # sign_changes = np.where(np.diff(np.sign(derivative(x))) != 0)[0]
            if relative_height is np.nan:
                nsa_lats_with_outliers.append(np.nan)
                nsa_lat_pixels_with_outliers.append(np.nan)
                continue
            relative_height = int(np.around(relative_height))

            if self.figures["nsb_column"] == True:
                plt.plot(latitudes, gaussian_y)
                plt.vlines(relative_height, np.min(gaussian_y)/2+np.mean(gaussian_y), np.max(gaussian_y)/2+np.mean(gaussian_y),colors = (0,0,0))

            if self.figures["visualized_plotting"] == True:
                plt.pause(0.01)
            actual_height = relative_height + min_lat_pixel
            north_south_boundary_lat = self.pixel_to_geo(actual_height, self.projected_lat[:,0])
            nsa_lats_with_outliers.append(north_south_boundary_lat)
            nsa_lat_pixels_with_outliers.append(actual_height)
        if self.figures["nsb_column"] == True:
            self.show()
        
        nsa_lat_pixels_outlier_free = self.remove_outliers(nsa_lat_pixels_with_outliers)
        nsa_lat_outlier_free = self.remove_outliers(nsa_lats_with_outliers)

        if self.figures["boundary_vs_longitude"] == True:
            plt.figure(self.figure_keys["boundary_vs_longitude"],figsize=(8,8))
            plt.title("Asymmetry Predictions vs Longitude")
            x = np.isnan(nsa_lat_pixels_outlier_free[0])
            longitude_values = [longitudes[i] for i in range(len(longitudes)) if not np.isnan(nsa_lat_pixels_outlier_free[i])]
            plotted_lats = nsa_lat_pixels_outlier_free[~np.isnan(nsa_lat_pixels_outlier_free)]
            plt.imshow(self.shifted_image, cmap = "gray")
            
            csx = PchipInterpolator(longitude_values, plotted_lats)
            gaussian_longitudes = [long for long in longitudes if long >= np.min(longitude_values) and long <= np.max(longitude_values)]
            gaussian_y = [csx(long) for long in gaussian_longitudes]
            plt.plot(gaussian_longitudes, gaussian_filter(gaussian_y,sigma = 2), color = (0,1,0), label = "gaussian_filter", linewidth = 2)
            plt.plot(longitude_values, plotted_lats, color = (1,0,0), label = "unfiltered (aside for outliers)", linestyle = "dotted", linewidth = 2)
            plt.legend()
            self.show()
        # linear fit and check rscore
        longitude_values = [longitudes[i] for i in range(len(longitudes)) if not np.isnan(nsa_lat_pixels_outlier_free[i])]
        lat_pixels_no_nans = [i for i in nsa_lat_pixels_outlier_free if not np.isnan(i)]
        slope, intercept, r_value, p_value, std_err = linregress(longitude_values, lat_pixels_no_nans)
        r_squared = r_value**2
        if r_squared > 0.7 or np.std(lat_pixels_no_nans)/np.mean(lat_pixels_no_nans) > 0.2:
            print("dev",np.nanstd(nsa_lat_outlier_free), "var", r_squared)
            return False
        return np.nanmean(nsa_lat_outlier_free)
    def albedo_brightness_of_north_south_asymmetry(self, preprojection_image, latitude, lats):
        return "not implemented yet"
    def north_south_flux_ratio(self, preprojection_image, latitude, lats, sampling_area=None):
        if sampling_area is None:
            north, south = self.albedo_brightness_of_north_south_asymmetry(preprojection_image, latitude, lats)
            return np.mean(north) / np.mean(south)
        else:
            return "not implemented yet"
    def get_tilt_from_data(self, nsa_lats, nsa_lons):
        new_lats = gaussian_filter(nsa_lats, sigma=1)
        function = self.linearRegress(nsa_lons, new_lats)
        x = np.array(x, dtype="float64")
        ys = x*function[0]+function[1]
        a = r2_score(nsa_lats, ys)
        return function, self.angle(function[0]), a
    
    # def rotate_image_to_north(self, longitude, latitude, original_image):
    #     # Calculate rotation angle
    #     rotation_angle = np.arctan2(np.max(longitude) - np.min(longitude), np.max(latitude) - np.min(latitude))

    #     # Create rotation matrix
    #     rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
    #                                 [np.sin(rotation_angle), np.cos(rotation_angle)]])

    #     # Apply rotation to coordinates
    #     rotated_coordinates = np.dot(rotation_matrix, np.array([longitude.flatten(), latitude.flatten()]))
    #     rotated_longitude = rotated_coordinates[0].reshape(longitude.shape)
    #     rotated_latitude = rotated_coordinates[1].reshape(latitude.shape)

    #     # Translate the image to match the rotation
    #     translated_image = np.roll(original_image, int(rotation_angle / (2 * np.pi) * original_image.shape[1]), axis=1)

    #     # Plot the rotated image
    #     plt.imshow(translated_image, cmap='gray')
    #     plt.show()
    def complete_image_analysis_from_cube(self, cube, band: int, save_location: str = None):
        self.save_location = save_location

        self.cube_name = cube.img_id

        # self.rotate_image_to_north(cube.lon,cube.lat, cube[band])
        
        if self.figures["original_image"]:
            plt.figure(self.figure_keys["original_image"])
            plt.imshow(cube[band], cmap = "gray")
            plt.title("Cube from "+ cube.flyby.name  +" at " + str(cube.w[band]) + " µm")
            self.show()
        # take image and apply cylindrical projection
        self.projected_image, (self.projected_lon, self.projected_lat), _, _= self.equirectangular_projection(cube, band)

        if self.figures["shifted_unrefined_image"] == True:
            plt.imshow(self.projected_image, cmap = "gray")
            plt.title("Equirectangular Projection from "+ cube.flyby.name  +" at " + str(cube.w[band]) + " µm")
            self.show()
        # after cylindrical projection, remove extraneous longitude data
        
        # use opencv to find north south boundary - shift images, apply gaussian blur, then find line (towards center)
        nsa_bounding_box_pixel = self.locate_north_south_boundary_unrefined(self.projected_image)        # use data from opencv analysis as a bounding box (+- 15 degs) for the data. refer back to cropped imaged (not opencv one), and locate north south asymmetry - derived brightness fit
        self.nsa_bounding_box = self.pixel_to_geo(nsa_bounding_box_pixel,self.projected_lat[:,0])/2
        self.nsa_bounding_box = [self.nsa_bounding_box-30, self.nsa_bounding_box+30]
        lat_res = np.mean(np.diff(self.projected_lat[:,0]))
        nsa_bounding_box_pixels = [np.max((0,int(np.round(nsa_bounding_box_pixel - 30/lat_res)))), int(np.round(nsa_bounding_box_pixel + 30/lat_res))]
        
        # locate north_south_boundary_refined
        refine_attempt = self.locate_north_south_boundary_refined(self.projected_image, *nsa_bounding_box_pixels, 5)
        if refine_attempt == False:
            self.north_south_boundary = self.pixel_to_geo(nsa_bounding_box_pixel,self.projected_lat[:,0])
        else:
            self.north_south_boundary = refine_attempt
        print(self.north_south_boundary)
        #get north south flux ratios (15 deg 30 deg 45 deg 60 deg 90 deg)
        if self.figures["persist_figures"] == "wait_till_end":
            self.show(force =True)
        #get if
        return "not implemented yet"

class complete_cube:
    def __init__(self, parent_directory: str, cube_subdir : str, save_location: None, clear_cache: bool = False):
        if save_location is None:
            save_location = os.path.join(parent_directory, "post_analysis")
        self.cube_subdir = cube_subdir
        self.clear_cache = clear_cache
        self.save_location = save_location
    def save_data_as_json(self, data: dict, save_location: str, overwrite: bool = None):
        if overwrite is None:
            overwrite = self.clear_cache
        if overwrite:
            with open(save_location, "w") as outfile:
                json.dump(data, outfile)
            return "not implemented yet"
    def analyze_dataset(self, cube, overwrite: bool = None):
        

        self.cube_vis = pyvims.VIMS(cube, channel="vis")
        self.cube_ir = pyvims.VIMS(cube, channel = "ir")
        
        # #visual analysis
        # for band in self.cube_vis.bands:
        #     if surface_windows[int(band)]:
        #         continue
        #     analysis = analyze_image()
        #     data = analysis.complete_image_analysis_from_cube(self.cube_vis, int(band), self.save_location)
        
        #Infrared analysis
        for band in self.cube_ir.bands:
            if int(band) in ir_surface_windows:
                continue
            analysis = analyze_image()
            data = analysis.complete_image_analysis_from_cube(self.cube_ir, int(band), self.save_location)
                    

class data:
    def __init__(self, directory, datasetName, shiftDegree, purpose):
        # create all class paths,directories, and variables
        self.createDirectories(directory, datasetName)
        self.createLists(purpose, datasetName)
        self.analysisPrep(shiftDegree)
        # gets purpose condition and executes individual operations based on purpose
        self.conditionals()

    def createDirectories(self, directory, flyby):  # create all file directory paths
        # basic datasets
        self.directoryList = directory
        self.flyby = flyby[0]
        self.masterDirectory = self.directoryList["flyby_parent_directory"]
        self.csvFolder = self.directoryList["flyby_data"]
        self.tFolder = os.path.join(
            self.directoryList["flyby_parent_directory"], self.flyby)
        self.imageFolder = os.path.join(
            self.directoryList["flyby_parent_directory"], self.flyby, self.directoryList["flyby_image_directory"])
        # finding files
        try:
            self.csvFile = [os.path.join(self.tFolder, self.csvFolder, file) for file in os.listdir(
                os.path.join(self.tFolder, self.csvFolder))]
            if len(self.csvFile) == 1:
                self.csvFile = self.csvFile[0]
        except:
            print("no csv file found")
        self.allFiles = [os.path.join(self.imageFolder, e) for e in os.listdir(self.imageFolder)]
        self.resultsFolder = os.path.join(
            self.masterDirectory, self.directoryList["analysis_folder"])
        self.flyby_bg_info = os.path.join(
            self.masterDirectory, self.directoryList["flyby_info"])
        if len(self.allFiles) != 96:
            print("missing 1+ files")

    def createLists(self, purpose, NSA):  # create global variables for data
        self.NSA = []
        self.deviation = []
        self.IF = []
        self.NS = []
        self.lat = []
        self.lon = []
        self.iterations = []
        self.goalNSA = NSA[1]
        self.errorMargin = NSA[2]
        self.leftCrop = NSA[3][0]
        self.rightCrop = NSA[3][1]
        self.purpose = purpose
        if self.purpose[0] == "tilt":
            self.band = []
        elif self.purpose[0] == "if_sh":
            self.if_sh = []
            self.latSh = []
        self.play = False

    def analysisPrep(self, shiftDegree):
        try:
            self.im = plt.imread(self.allFiles[0])[:, :, 0]
        except:
            self.im = plt.imread(self.allFiles[0])[:, :]
        self.height = len(self.im)
        self.width = len(self.im[0])
        self.im = self.im.astype(np.float32)

        if len(self.csvFile) == 2:
            try:
                nans = np.empty((1, self.width))
                nans[:] = np.nan
                if "lat" in self.csvFile[0]:
                    self.lat = np.reshape(np.append(
                        (pd.read_csv(self.csvFile[0])), nans, axis=0), (self.height, self.width))
                    self.lon = np.reshape(np.append(
                        (pd.read_csv(self.csvFile[1])), nans, axis=0), (self.height, self.width))
                else:
                    self.lat = np.reshape(np.append(
                        (pd.read_csv(self.csvFile[1])), nans, axis=0), (self.height, self.width))
                    self.lon = np.reshape(np.append(
                        (pd.read_csv(self.csvFile[0])), nans, axis=0), (self.height, self.width))
            except:
                raise ValueError("error finding csv")
        else:
            try:
                self.lat = self.Jcube(
                    geo_csv=self.csvFile, var='lat', nL=self.height, nS=self.width)
                self.lon = self.Jcube(
                    geo_csv=self.csvFile, var='lon', nL=self.height, nS=self.width)
                x = 0
            except:
                raise ValueError("error finding csv")
        self.columnLat = self.lat[:, 0]
        temp = self.leftCrop
        if self.leftCrop < 0:
            self.leftCrop = [0, abs(self.rightCrop)]
            self.rightCrop = [abs(temp), self.width]
            self.leftCrop[0] = min(range(len(self.lon[0, :])), key=lambda x: abs(
                self.lon[0, x] - self.leftCrop[0]))
            self.rightCrop[0] = min(range(len(self.lon[0, :])), key=lambda x: abs(
                self.lon[0, x]-self.rightCrop[0]))
            self.leftCrop[1] = min(range(len(self.lon[0, :])), key=lambda x: abs(
                self.lon[0, x] - self.leftCrop[1]))
            self.rightCrop[1] = min(range(len(self.lon[0, :])), key=lambda x: abs(
                self.lon[0, x]-self.rightCrop[1]))
        else:
            self.leftCrop = min(range(len(self.lon[0, :])), key=lambda x: abs(
                self.lon[0, x] - self.leftCrop))
            self.rightCrop = min(range(len(self.lon[0, :])), key=lambda x: abs(
                self.lon[0, x]-self.rightCrop))
        if "Ti" in self.flyby:
            self.subset = tuple(
                [(self.columnLat > self.goalNSA - 30) & (self.columnLat < self.goalNSA + 30.0)])
        else:
            self.subset = tuple(
                [(self.columnLat > self.goalNSA - 15) & (self.columnLat < self.goalNSA + 15.0)])
        # subset HC band b/t 30°S to 0°N
        self.lat_sh = self.columnLat[self.subset]
        self.lon_sh = self.lon[self.subset]
        latRange = np.nanmax(self.lat)-np.nanmin(self.lat)
        latTicks = len(self.lat)/latRange
        self.shiftDegree = shiftDegree
        self.num_of_nans = int(latTicks*shiftDegree)
        self.nans = [np.nan]*(self.num_of_nans)
        if self.purpose[0] == "if_sh":
            self.ifSubset = tuple(
                [(self.columnLat > -90.0) & (self.columnLat < 90.0)])

    def createFolder(self, folderPath=None):
        if not folderPath:
            folderPath = os.path.join(
                self.masterDirectory, self.directoryList["analysis_folder"])
        if not os.path.exists(folderPath):
            os.makedirs(folderPath)
        self.resultsFolder = folderPath

    def createFile(self):
        open(self.resultsFolder + self.flyby + "_analytics.csv", "w")
        x = open(self.resultsFolder + self.flyby +
                 "_analytics.csv", "a+", newline='')
        self.analysisFile = x

    def fileWrite(self, x):
        a = csv.writer(self.analysisFile)
        a.writerow(x)

    def fileClose(self):
        self.analysisFile.close()

    def averages(self, x):
        X = []
        for i in range(len(x)):
            X.append(np.mean(x[i]))
        return X

    # Get image to create arrays that help read image data as latitude data
    def Jcube(self, nL, nS, csv=None, geo_csv=None, bands=None, var=None):
        '''Converts any Jcube or geocube csv file to numpy 2D array
        Note: Requires pandas and numpy.

        Parameters
        ----------
        csv: str
            Name of Jcube csv file with raw I/F data 
        geo_csv: str
            Name of csv file from geocubes with geographic info on a VIMS cube (.cub) 
            Can include relative directory location (e.g. 'some_dir/test.csv')
        nL: int
            number of lines or y-dimension of VIMS cube
        nS: int
            number of samples or x-dimension of VIMS cube
        vars: str
            geo_cube parameters that include "I/F", lat", "lon", "lat_res", "lon_res", 
            "phase", "inc", "eme", and "azimuth"
        '''
        # check var is the correct string
        if var not in ["lat", "lon", "lat_res", "lon_res", "phase", "inc", "eme", "azimuth"] and var is not None:
            raise ValueError("Variable string ('var') is not formatted as one of the following options: \n" +
                             '"I/F", "lat", "lon", "lat_res", "lon_res", "phase", "inc", "eme", "azimuth"')
        # create image of Titan
        if geo_csv is None:
            # read csv; can include relative directory
            csv = pd.read_csv(csv, header=None)

            def getMeanImg(csv, bands, nL, nS):
                '''Get the mean I/F image for a set of wavelengths from a Jcube csv file
                Note: a single 'bands' value will return the image at that wavelength.

                Parameters
                ----------
                csv: str 
                    name of csv file for VIMS cube
                bands: int or list of int 
                    band values from 96-352 for near-infrared VIMS windows
                nL: int
                    number of lines 
                nS: int
                    number of samples
                '''
                if isinstance(bands, int):
                    bands = [bands]
                img = []
                for band in bands:
                    cube = np.array(csv)[:, band].reshape(nL, nS)
                    cube[cube < -1e3] = 0
                    img.append(cube)  # [band, :, :])
                return np.nanmean(img, axis=0)
            return getMeanImg(csv=csv, bands=bands, nL=nL, nS=nS)
        # create geocube
        if csv is None:
            # read csv; can include relative directory
            geo = pd.read_csv(geo_csv, header=None)
            # output chosen variable 2D array
            if var == 'lat':
                return np.array(geo)[:, 0].reshape(nL, nS)
            if var == 'lon':
                return np.array(geo)[:, 1].reshape(nL, nS)
            if var == 'lat_res':
                return np.array(geo)[:, 2].reshape(nL, nS)
            if var == 'lon_res':
                return np.array(geo)[:, 3].reshape(nL, nS)
            if var == 'phase':
                return np.array(geo)[:, 4].reshape(nL, nS)
            if var == 'inc':
                return np.array(geo)[:, 5].reshape(nL, nS)
            if var == 'eme':
                return np.array(geo)[:, 6].reshape(nL, nS)
            if var == 'azimuth':
                return np.array(geo)[:, 7].reshape(nL, nS)
        # create geocube
        if csv is None:
            # read csv; can include relative directory
            geo = pd.read_csv(geo_csv, header=None)
            return (np.array(geo)[:, 0].reshape(nL, nS), np.array(geo)[:, 1].reshape(nL, nS))

    def brightness(self, im_file):
        im = Image.open(im_file).convert('L')
        stat = ImageStat.Stat(im)
        return stat.mean[0]

    def polyfit(self, x, y, degree):  # alternate fit for polynomials
        results = {}
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        # calculate r-squared
        yhat = p(x)
        ybar = np.sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)
        sstot = np.sum((y - ybar)**2)
        return ssreg / sstot

    def gaussian(self, x, amplitude, mean, stddev):  # gaussian fit
        return amplitude * np.exp(-((x - mean) / 4 / stddev)**2)

    def poly6(self, x, g, h, i, j, a, b, c):  # sextic function
        return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c

    def poly6Prime(self, x, g, h, i, j, a, b, c):  # derivative of sextic function
        return 6*g*x**5+5*h*x**4+4*i*x**3+3*j*x**2+2*a*x+b

    # derivative of sextic function coefficents
    def poly6Derivative(self, g, h, i, j, a, b, c):
        return [6*g, 5*h, 4*i, 3*j, 2*a, b]

    def running_mean(self, x, N):  # window avearage
        return np.convolve(x, np.ones(N)/N, mode='valid')

    def conditionals(self):
        print("purpose is", self.purpose[0])
        if self.purpose[0] == "data":
            self.getDataControl()
        elif self.purpose[0] == "tilt":
            self.getTiltControl()
        elif self.purpose[0] == "if_sh":
            self.getIf_shControl()
        else:
            print("data output type not understood; ",
            self.purpose[0], " not valid")

    def getDataControl(self):  # iteration over images within flyby
        for self.iter in range(len(self.allFiles)):
            self.a = time.time()
            self.currentFile = self.allFiles[self.iter]
            self.iterations.append(self.iter)
            self.dataAnalysis()
            print("dataset", self.flyby, "     image", "%02d" % (self.iter), "     boundary", format(
                self.NSA[self.iter], '.15f'), "      deviation", format(self.deviation[self.iter], '.15f'), "        N/S", format(self.NS[self.iter], '.10f'))
        # write data
        if "write" in self.purpose[1]:
            # create folder and file
            self.createFolder()
            self.createFile()
            try:
                datasetAverage = self.averages(
                    [self.NSA, np.std(self.NSA), self.NS])
                self.NSA.append(datasetAverage[0])
                self.deviation.append(datasetAverage[1])
                self.NS.append(datasetAverage[2])
                self.iterations.append(self.flyby)
                self.NSA.insert(0, "NSA")
                self.deviation.insert(0, "Deviation")
                self.NS.insert(0, "N/S")
                self.iterations.insert(0, "File Number")
                self.fileWrite(self.NSA)
                self.fileWrite(self.deviation)
                self.fileWrite(self.NS)
                self.fileWrite(self.iterations)
            finally:
                self.fileClose()

    def getIf_shControl(self):
        if len(self.purpose[1]) == 0:
            for self.iter in range(len(self.allFiles)):
                self.currentFile = self.allFiles[self.iter]
                self.iterations.append(self.iter)
                self.ifAnalysis()
        else:
            for self.iter in range(len(self.allFiles)):
                if self.iter in self.purpose[1]:
                    self.currentFile = self.allFiles[self.iter]
                    self.iterations.append(self.iter)
                    self.if_sh.append(self.ifAnalysis())

    def getTiltControl(self):
        # go through each file
        for self.iter in range(len(self.allFiles)):
            self.currentFile = self.allFiles[self.iter]
            if self.iter in self.purpose[1]:
                self.tiltAnalysis()
        # write data
            # print(self.band)
            pass

    def visualizeBrightnessDifferenceSamplingArea(self, im, x, nsaLat):
        if type(self.leftCrop) is list:
            im[x[0]:x[1], self.rightCrop[0]:self.leftCrop[0]] *= 2
            im[(x[1]+1):x[2], self.rightCrop[0]:self.leftCrop[0]] *= 0.5
            im[x[0]:x[1], self.rightCrop[1]:self.leftCrop[1]] *= 2
            im[(x[1]+1):x[2], self.rightCrop[1]:self.leftCrop[1]] *= 0.5
        else:
            im[x[0]:x[1], self.leftCrop:self.rightCrop] *= 2
            im[(x[1]+1):x[2], self.leftCrop:self.rightCrop] *= 0.5
        plt.imshow(im, cmap='Greys')
        plt.show()

    def brightnessDifference(self, im, nsaLat):
        splitY = min(range(len(self.lat[:, 0])),
                     key=lambda x: abs(self.lat[x, 0]-nsaLat))
        horizontalSample = 4
        verticalSample = 30
        northSplit = min(range(len(self.lat[:, 0])), key=lambda x: abs(
            self.lat[x, 0]-verticalSample-nsaLat))
        southSplit = min(range(len(self.lat[:, 0])), key=lambda x: abs(
            self.lat[x, 0]+verticalSample-nsaLat))

        if type(self.leftCrop) is list:
            north = im[northSplit:splitY, self.rightCrop[0]:self.leftCrop[0]]
            south = im[(splitY+1):southSplit,
                       self.rightCrop[0]:self.leftCrop[0]]
            north = np.concatenate(
                (north, im[northSplit:splitY, self.rightCrop[1]:self.leftCrop[1]]), axis=1)
            south = np.concatenate(
                (south, im[(splitY+1):southSplit, self.rightCrop[1]:self.leftCrop[1]]), axis=1)
        else:
            if self.leftCrop > self.rightCrop:
                north = im[northSplit:splitY, self.rightCrop:self.leftCrop]
                south = im[(splitY+1):southSplit, self.rightCrop:self.leftCrop]
            else:
                north = im[northSplit:splitY, self.leftCrop:self.rightCrop]
                south = im[(splitY+1):southSplit, self.leftCrop:self.rightCrop]
        northM = np.mean(north[north != 0.])
        southM = np.mean(south[south != 0.])
        return northM/southM

    def dataAnalysis(self):
        try:  # open image arrays
            self.im = plt.imread(self.currentFile)[:, :, 0]
        except:
            self.im = plt.imread(self.currentFile)[:, :]
        self.im = self.im.astype(np.int16)
        hc_band = np.empty((self.height, self.width), float)
        nsa_lats = []
        nsa_lons = []
        cols = []
        latRange = np.max(self.lat)-np.min(self.lat)
        latTicks = len(self.lat)/latRange
        # get latitude values of each pixel using CSV
        width = []
        subtraction = (np.insert(self.im, 0, np.array(
            self.num_of_nans*[[0]*self.width]), axis=0) - np.concatenate((self.im, self.num_of_nans*[[0]*self.width])))
        hc_band = subtraction[int(
            np.round(self.num_of_nans/2, 0)):-1*int(np.round(self.num_of_nans/2, 0))]
        if True:
            self.createFolder(os.path.join(
                self.masterDirectory, self.flyby, "hc_band"))
            
            # self.convert_array_to_image(hc_band, os.path.join(self.masterDirectory, self.flyby, "hc_band", (os.path.splitext(os.path.relpath(
            #     self.currentFile, os.path.join(self.masterDirectory, self.flyby, self.directoryList["flyby_image_directory"])))[0] + "_band")))
        if_sh = hc_band[self.subset]
        lon_subset = []
        if type(self.leftCrop) is list:
            a = sorted((self.rightCrop[0], self.leftCrop[0]))
            b = sorted((self.rightCrop[1], self.leftCrop[1]))
            lon_subset = np.concatenate((range(*a), range(*b)))
        else:
            if self.leftCrop > self.rightCrop:
                lon_subset = range(self.rightCrop, self.leftCrop)
            else:
                lon_subset = range(self.leftCrop, self.rightCrop)
        for col in range(self.width):
            if col in lon_subset:
                columnHC = hc_band[:, col]
                if_sh = columnHC[self.subset]  # subset HC band b/t 30°S to 0°N
                if np.min(if_sh) != np.max(if_sh):
                    try:
                        # apply sextic regression to data
                        popt, _ = curve_fit(self.poly6, self.lat_sh, if_sh)
                        # get derivative of sextic regression
                        poptD = self.poly6Derivative(*popt)
                        # roots (Real and imaginary) of derivative function
                        derivativeRoot = np.roots(poptD)
                        # remove extraneous soulutions (imaginary)
                        realDerivativeRoots = derivativeRoot[np.isreal(
                            derivativeRoot)]
                        drIndex = min(range(len(realDerivativeRoots)), key=lambda x: abs(
                            realDerivativeRoots[x]-self.goalNSA))  # find value closest to NSA
                        derivativeRoots = realDerivativeRoots[drIndex]
                        if abs(derivativeRoots.real-self.goalNSA) >= self.errorMargin:
                            width.append(False)
                        else:
                            nsa_lats.append(derivativeRoots.real)
                            width.append(True)
                    except:
                        width.append(False)
            else:
                width.append(False)
        self.NSA_Analysis(nsa_lats, self.im, width)
        print(time.time() - self.a)

    def ifAnalysis(self):
        # open image arrays
        try:
            self.im = plt.imread(self.currentFile)[:, :, 0]
        except:
            self.im = plt.imread(self.currentFile)[:, :]
        self.im = self.im.astype(np.float32)
        self.hc_band = np.empty((self.height, self.width), float)
        # get latitude values of each pixel using CSV
        count = 0
        non_zero = np.array(np.any(self.im != 0, axis=1))
        image = self.im[non_zero, :]
        crop = self.im[non_zero, :]
        ab = self.lat[non_zero, 0]
        Result = image[:, ~np.any(crop == 0, axis=0)]
        try:
            b = Result[int(len(Result)*0.25):int(len(Result)*0.75),
                       int(len(Result[0])*0.25):int(len(Result[0])*0.75)]
        except:
            try:
                self.im = plt.imread(self.currentFile)[:, :, 0]
            except:
                self.im = plt.imread(self.currentFile)[:, :]
            b = 0
        b = np.mean(b)*10
        a = np.mean(Result, axis=1)
        ab = ab[abs(a) < abs(b)]
        a = a[abs(a) < abs(b)]
        return [ab, a]

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def tiltAnalysis(self):
        # open image arrays
        if self.iter in self.purpose[1]:
            try:
                self.im = plt.imread(self.currentFile)[:, :, 0]
            except:
                self.im = plt.imread(self.currentFile)[:, :]
            self.im = self.im.astype(np.float32)
            self.hc_band = np.empty((self.height, self.width), float)
            nsa_lats = []
            nsa_lons = []
            cols = []
            columns = []
            lon_shTilt = []
            for col in self.purpose[2]:
                x = self.columnAnalysis(col)
                if x != None:
                    nsa_lats.append(x)
                    lon_shTilt.append(self.lon[0, col])
                    columns.append(col)


    def linearRegress(self, x, y):
        return np.polyfit(x, y, 1)

    def angle(self, slope):
        return 180/math.pi*np.arctan(slope)



    def if_sh_data(self, column):
        subtraction = (np.insert(self.im[:, column], [
                       0]*self.num_of_nans, self.nans) - np.concatenate((self.im[:, column], self.nans)))
        self.hc_band[:, column] = subtraction[int(
            self.num_of_nans/2):int(-self.num_of_nans/2)]
        # hc_band[crop[0]:crop[1],crop[2]:crop[3]]
        columnHC = self.hc_band[:, column]
        if_sh = columnHC  # subset HC band b/t 30°S to 0°N
        # lat_sh = self.columnLat[self.subset]  ## subset HC band b/t 30°S to 0°N
        # lon_sh = self.lon[:,column][self.subset]  ## subset HC band b/t 30°S to 0°N
        return if_sh

    def NSA_Analysis(self, im_nsa_lat, image, x):
        dev = np.std(im_nsa_lat)  # standard deviation
        average = np.nanmean(im_nsa_lat)  # standard average
        combo = 4
        movingAverageList = self.running_mean(
            im_nsa_lat, combo)  # moving average
        # if "showAverage" in self.purpose[1]:
        #     plt.plot(range(len(movingAverageList)),movingAverageList)
        #     plt.show()
        movingAvg = np.mean(movingAverageList)  # moving average
        # difference between north and south
        diff = self.brightnessDifference(image, movingAvg)
        self.NSA.append(movingAvg)
        self.deviation.append(dev)
        self.NS.append(diff)

image = complete_cube("/Users/aadvik/Desktop/NASA/North_South_Asymmetry/cubes", "T62", "/Users/aadvik/Desktop/NASA/North_South_Asymmetry/cubes/results")
image.analyze_dataset("C1634084887_1")
