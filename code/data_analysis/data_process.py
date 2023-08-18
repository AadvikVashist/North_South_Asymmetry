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
import pickle
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
def join_strings(*args):
    return os.path.join(*args).replace("\\", "/")
def check_if_exists_or_write(file : str, base : str = None, prefix : str = None, file_type = None, save = False, data = None, force_write = False, verbose = True):
    full_path = os.path.join(base, prefix, file)
    if file_type is not None:
        file_type = file_type.lower().strip().replace(".", "")
        full_path += "." + file_type
    if save == True:
        if data is None:
            raise ValueError("Data must be provided to save")
        if verbose:
            print("Saving to: ", full_path)
        directory = os.path.dirname(full_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        if force_write == False and os.path.exists(full_path):
            if verbose:
                print("File already exists")
            return False
        if full_path.endswith(".pkl") or full_path.endswith(".pickle"):
            with open(full_path, "wb") as file_object:
                pickle.dump(data, file_object)
            if verbose:
                print("Saved", full_path)
        elif full_path.endswith(".json"):
            with open(full_path, "w") as file_object:
                data = json.dump(file_object)
            if verbose:
                print("Saved", full_path)
        elif full_path.endswith(".csv"):
            with open(full_path, 'w', newline='') as file_object:
                writer = csv.writer(file_object)
                # Write the data to the CSV file
                writer.writerows(data)
            if verbose:
                print("Saved", full_path)
        elif full_path.endswith(".png") or full_path.endswith(".jpg") or full_path.endswith(".jpeg") or full_path.endswith(".tiff") or full_path.endswith(".tif"):
            cv2.imwrite(full_path, data)
            if verbose:
                print("Saved", full_path)
        else:
            if verbose:
                print("File type not recognized")
    else:
        if os.path.exists(full_path):
            if full_path.endswith(".pkl") or full_path.endswith(".pickle"):
                with open(full_path, 'rb') as pickle_file:
                    data = pickle.load(pickle_file)
            elif full_path.endswith(".json"):
                with open(full_path, "r") as file_object:
                    data = json.load(file_object)
            elif full_path.endswith(".csv"):
                data = pd.read_csv(full_path)
            elif full_path.endswith(".png") or full_path.endswith(".jpg") or full_path.endswith(".jpeg") or full_path.endswith(".tiff") or full_path.endswith(".tif"):
                data = Image.open(full_path)
            else:
                data = None
            return data
        else:
            return False
class analyze_image:
    def __init__(self, figures : bool = None, show_figures : bool = False):
        self = self
        if figures == False:
            self.figures = {
                "original_image" : False,
                "shifted_unrefined_image" : False,
                "gaussian_brightness_bands" : False,
                "unrefined_north_south_boundary" : False,
                "cropped_shifted_image" : False,
                "nsb_column" : False,
                "cube_spline_selection_of_boundary": False,
                "visualized_plotting" : False,
                "boundary_vs_longitude" : False,
                "persist_figures" : False
            }
        else:
            self.figures = {
                "original_image" : True,
                "shifted_unrefined_image" : True,
                "gaussian_brightness_bands" : True,
                "unrefined_north_south_boundary" : True,
                "cropped_shifted_image" : True,
                "nsb_column" : True,
                "cube_spline_selection_of_boundary": True,
                "visualized_plotting" : False,
                "boundary_vs_longitude" : True,
                "persist_figures" : 4
            }
        if show_figures == False:
            self.figures["persist_figures"] = False
        self.figure_keys = {key: index for index, (key, value) in enumerate(self.figures.items())}
        self.saved_figures = {key: None for key in self.figures}
    def figure_options(self):
        print("\nCurrent Figure Setttings:\n",*[str(val[0]) + " = " + str(val[1]) for val in self.figures.items()], "\n\n", sep = "\n")
        return self.figures
    def show(self, force = False, duration : float = None):
        if duration is not None:
            plt.pause(duration)
            plt.clf()
            return
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
        elif options == False:
            return
        else:
            plt.pause(2)
            plt.clf()
            plt.close()
    def get_vmin_vmax(self, image):
        return np.quantile(image, [0.1, 0.9])
        return np.quantile(image, [0.1, 0.9])
    
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

        max_min = np.polyfit(x, y, deg= 2)
        if max_min[0] > 0:
            north_south_boundary_index = np.argmin(values_at_zeros)
        else:
            north_south_boundary_index = np.argmax(values_at_zeros)
        if self.figures["cube_spline_selection_of_boundary"]:
            fig = plt.figure(self.figure_keys["cube_spline_selection_of_boundary"])
            plt.plot(x,y)
            plt.plot(x, [max_min[0]*xa**2+max_min[1]*xa+max_min[2] for xa in x])
            plt.vlines(zeros, (np.mean(y) + np.min(y))/2,  (np.mean(y) + np.max(y))/2, color = (0,0,0,1))
            plt.vlines(zeros[north_south_boundary_index], np.min(y), np.max(y), color = (1,0,0,1))
            self.show(duration = 0.05)
        return zeros[north_south_boundary_index]
    
    def locate_north_south_boundary_unrefined(self, image, prejudice : int = None):
        self.shifted_image, cropped_rows, cropped_columns = self.shift_image(image, 20)
        top_shift = cropped_rows.index(True)
        if self.figures["gaussian_brightness_bands"]:
            fig = plt.figure(self.figure_keys["gaussian_brightness_bands"])
            for i in range(0,self.shifted_image.shape[1], 10):
                average_brightness =  gaussian_filter(self.shifted_image[:,i], sigma=3)
                plt.plot(range(self.shifted_image.shape[0]), average_brightness)
            plt.title("Raw Brightness Data with Gaussian Filter")
            plt.xlabel("Latitude")
            plt.ylabel("Brightness")
            self.saved_figures["gaussian_brightness_bands"] = fig
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
            return None
        # remove extraneous soulutions (imaginary)
        max_min = np.polyfit(latitudes, gaussian_y, deg= 2)
        if max_min[0] > 0:
            boundarys_with_right_sign = [approx for approx in zeros if self.min_or_max(derivative, approx, 5, 0) == True]
        else:
            boundarys_with_right_sign = [approx for approx in zeros if self.min_or_max(derivative, approx, 5, 0) == False]
            # north_south_boundary_index = np.argmax(values_at_zeros)
        # north_south_mins = [approx for approx in zeros if self.min_or_max(derivative, approx, 5, 0) == True and self.compare_acceleration_to_flux(approx, True, average_brightness,0)]
        # north_south_maxes = [approx for approx in zeros if self.min_or_max(derivative, approx, 5, 0) == False and self.compare_acceleration_to_flux(approx, False, average_brightness,0)]
        if len(boundarys_with_right_sign) == 0:
            ret_val =  image.shape[0]/2-top_shift
        elif len(boundarys_with_right_sign) == 1:
            ret_val =  float(boundarys_with_right_sign[0])
        else:
            if prejudice:
                boundarys_with_right_sign = sorted(boundarys_with_right_sign, key= lambda x: x-prejudice)
            else:
                boundarys_with_right_sign = sorted(boundarys_with_right_sign, key= lambda x: x-image.shape[1]/2)
            ret_val =  float(boundarys_with_right_sign[0])
        if self.figures["unrefined_north_south_boundary"] == True:
            fig = plt.figure(self.figure_keys["unrefined_north_south_boundary"])
            plt.plot(latitudes,average_brightness,label="shifted data") #values
            # plt.plot(x,gaussian_y) #filtered values
            plt.plot(latitudes,plottedy,label="spline values") #splien values
            # plt.plot(x,plottedy)
            # Obtain the derivative function
            plt.plot(latitudes,[derivative(xa)*20 for xa in latitudes], color = (0,0,1,1), linestyle = "dotted",label="derivative")
            plt.hlines([0], [0],[image.shape[0]], colors = (0,0,0), linestyles='dashed',label="0 lat")
            plt.grid(True, 'both')
            if max_min[0] > 0:
                plt.vlines(boundarys_with_right_sign, np.min(average_brightness)/2+np.mean(average_brightness), np.max(average_brightness)/2+np.mean(average_brightness), colors = (0,0,0), linestyles='dashed',label="min")
            else:
                plt.vlines(boundarys_with_right_sign, np.min(average_brightness)/2+np.mean(average_brightness), np.max(average_brightness)/2+np.mean(average_brightness), colors = (1,0,0), linestyles='dashed',label="max")
            plt.vlines(ret_val, np.min(average_brightness)/2+np.mean(average_brightness), np.max(average_brightness)/2+np.mean(average_brightness), colors = (0,1,0), linestyles='solid',label="lat_selected" )
            plt.legend()
            plt.title("Mean Brightness Data Asymmetry Location")
            plt.xlabel("Latitude")
            plt.ylabel("Brightness")
            self.saved_figures["unrefined_north_south_boundary"] = fig
            # plt.plot([self.polyXPlugin(ax, *fitted_line_derivative) for ax in x])
            self.show()
        
        return ret_val + top_shift
    def locate_north_south_boundary_refined(self, image, min_lat_pixel, max_lat_pixel, shift_amount):
        nsa_lats_with_outliers = []
        nsa_lat_pixels_with_outliers = []
        image = image[min_lat_pixel:max_lat_pixel]
        shifted_image, cropped_rows, cropped_columns = self.shift_image(image, 10, True)
        if self.figures["cropped_shifted_image"] == True:
            fig = plt.figure(self.figure_keys["cropped_shifted_image"])
            plt.imshow(shifted_image, cmap = "gray")
            plt.title("Cropped Projection for Refined Processing")
            self.saved_figures["cropped_shifted_image"] = fig
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
            fig = plt.figure(self.figure_keys["nsb_column"])
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
            self.saved_figures["nsb_column"] = fig
            self.show()
        
        nsa_lat_pixels_outlier_free = self.remove_outliers(nsa_lat_pixels_with_outliers)
        nsa_lat_outlier_free = self.remove_outliers(nsa_lats_with_outliers)

        if self.figures["boundary_vs_longitude"] == True:
            fig = plt.figure(self.figure_keys["boundary_vs_longitude"],figsize=(8,8))
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
            self.saved_figures["boundary_vs_longitude"] = fig
            self.show()
        # linear fit and check rscore
        longitude_values = [longitudes[i] for i in range(len(longitudes)) if not np.isnan(nsa_lat_pixels_outlier_free[i])]
        lat_pixels_no_nans = [i for i in nsa_lat_pixels_outlier_free if not np.isnan(i)]
        slope, intercept, r_value, p_value, std_err = linregress(longitude_values, lat_pixels_no_nans)
        r_squared = r_value**2
        if r_squared > 0.7 or np.std(lat_pixels_no_nans)/np.mean(lat_pixels_no_nans) > 0.2:
            print("deviation",np.nanstd(nsa_lat_outlier_free), "variance", r_squared)
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
    def cube_to_unrefined_prediction(self, cube, band: int, prejudice : int = None):
        self.cube_name = cube.img_id
        self.projected_image, (self.projected_lon, self.projected_lat), _, _= self.equirectangular_projection(cube, band)
        if self.figures["shifted_unrefined_image"] == True:
            fig = plt.figure(self.figure_keys["shifted_unrefined_image"])
            plt.imshow(self.projected_image, cmap = "gray")
            plt.title("Equirectangular Projection from "+ cube.flyby.name  +" at " + str(cube.w[band]) + " µm")
            self.saved_figures["shifted_unrefined_image"] = fig
            self.show()
        nsa_bounding_box_pixel = self.locate_north_south_boundary_unrefined(self.projected_image, prejudice = prejudice)       # use data from opencv analysis as a bounding box (+- 15 degs) for the data. refer back to cropped imaged (not opencv one), and locate north south asymmetry - derived brightness fit
        if nsa_bounding_box_pixel is None:
            return None
        return self.pixel_to_geo(nsa_bounding_box_pixel,self.projected_lat[:,0])
    def complete_image_analysis_from_cube(self, cube, band: int, save_location: str = None, biased_lat : float = 0):
        self.save_location = save_location

        self.cube_name = cube.img_id

        # self.rotate_image_to_north(cube.lon,cube.lat, cube[band])
        
        if self.figures["original_image"]:
            fig = plt.figure(self.figure_keys["original_image"])
            plt.imshow(cube[band], cmap = "gray")
            plt.title("Cube from "+ cube.flyby.name  +" at " + str(cube.w[band]) + " µm")
            self.saved_figures["original_image"] = fig
            self.show()
        # take image and apply cylindrical projection
        self.projected_image, (self.projected_lon, self.projected_lat), _, _= self.equirectangular_projection(cube, band)

        if self.figures["shifted_unrefined_image"] == True:
            fig = plt.figure(self.figure_keys["shifted_unrefined_image"])
            plt.imshow(self.projected_image, cmap = "gray")
            plt.title("Equirectangular Projection from "+ cube.flyby.name  +" at " + str(cube.w[band]) + " µm")
            self.saved_figures["shifted_unrefined_image"] = fig
            self.show()
        # after cylindrical projection, remove extraneous longitude data
        
        # use opencv to find north south boundary - shift images, apply gaussian blur, then find line (towards center)
        nsa_bounding_box_pixel = self.locate_north_south_boundary_unrefined(self.projected_image, biased_lat)
        if nsa_bounding_box_pixel is None:
            return None
        # use data from opencv analysis as a bounding box (+- 15 degs) for the data. refer back to cropped imaged (not opencv one), and locate north south asymmetry - derived brightness fit
        nsa_bounding_box_lat = self.pixel_to_geo(nsa_bounding_box_pixel,self.projected_lat[:,0])
        nsa_bounding_box_lat =  (nsa_bounding_box_lat + 3 * biased_lat) /4
        print("unref boundary latitude:", nsa_bounding_box_lat)
        self.nsa_bounding_box = [nsa_bounding_box_lat-20, nsa_bounding_box_lat+20]
        lat_res = np.mean(np.diff(self.projected_lat[:,0]))
        nsa_bounding_box_pixels = [np.max((0,int(np.round(nsa_bounding_box_pixel - 30/lat_res)))), int(np.round(nsa_bounding_box_pixel + 30/lat_res))]
        
        # locate north_south_boundary_refined
        refine_attempt = self.locate_north_south_boundary_refined(self.projected_image, *nsa_bounding_box_pixels, 5)
        if refine_attempt == False:
            self.north_south_boundary = self.pixel_to_geo(nsa_bounding_box_pixel,self.projected_lat[:,0])
        else:
            self.north_south_boundary = refine_attempt
        print("boundary latitude:", self.north_south_boundary)
        #get north south flux ratios (15 deg 30 deg 45 deg 60 deg 90 deg)
        if self.figures["persist_figures"] == "wait_till_end":
            self.show(force = True)
        #get if
        return "not implemented yet"

class complete_cube:
    def __init__(self, parent_directory : str, cube_subdirectory : str, data_save_folder_name : str = None):
        if data_save_folder_name is None:
            save_location = join_strings(parent_directory, cube_subdirectory, "analysis/nsa/")
        else:
            save_location = join_strings(parent_directory,cube_subdirectory, data_save_folder_name, "nsa/")
        self.cube_name = cube_subdirectory
        self.result_data_base_loc = save_location
        self.parent_directory = parent_directory
        self.cwd = join_strings(self.parent_directory, cube_subdirectory)
    def save_data_as_json(self, data: dict, save_location: str, overwrite: bool = None):
        if overwrite is None:
            overwrite = self.clear_cache
        if overwrite:
            with open(save_location, "w") as outfile:
                json.dump(data, outfile)
            return "not implemented yet"
    def analyze_cube(self, cube : str = None):
        if cube is None:
            print("since no cube was provided, defaulting to cube", self.cube_name)
            cube = self.cube_name
        base_cube = join_strings(self.cwd, cube)
        if not os.path.exists(base_cube + "_vis.cub"):
            print("vis cube not found, checking dir", base_cube + "_vis.cub")
            return None
        if not os.path.exists(base_cube + "_ir.cub"):
            print("ir cube not found, checking dir", base_cube + "_ir.cub")
            return None
        self.cube_vis = pyvims.VIMS(cube  + "_vis.cub", self.cwd, channel="vis")
        self.cube_ir = pyvims.VIMS(cube   + "_ir.cub", self.cwd, channel = "ir")
        # #visual analysis
        
        #Infrared analysis
        lats = check_if_exists_or_write("prejudiced_lats.pkl", base = self.result_data_base_loc, prefix = "cache")
        if lats is False:
            lats = []
            for band in range(1,np.max(self.cube_vis.bands)+1, 5):
                if surface_windows[int(band)-1]:
                    continue
                analysis = analyze_image(False)
                try:
                    if len(lats) > 10:
                        lat = analysis.cube_to_unrefined_prediction(self.cube_vis, int(band), np.mean(lats))
                    else:
                        lat = analysis.cube_to_unrefined_prediction(self.cube_vis, int(band))
                except ValueError:
                    continue
                if lat is not None:
                    lats.append(lat)
                print(lat)
            for band in range(np.min(self.cube_ir.bands),np.max(self.cube_ir.bands)+1, 5):
                if int(band)-1 in ir_surface_windows:
                    continue
                analysis = analyze_image(False)
                try:
                    if len(lats) > 10:
                        lat = analysis.cube_to_unrefined_prediction(self.cube_ir, int(band), np.mean(lats))
                    else:
                        lat = analysis.cube_to_unrefined_prediction(self.cube_ir, int(band))
                except ValueError:
                    continue
                if lat is not None:
                    lats.append(lat)
                print(lat)
            check_if_exists_or_write("prejudiced_lats.pkl", base = self.result_data_base_loc, prefix = "cache", save = True, data = lats)
        prejudice_lat = np.mean(lats)

        analysis_objects = {}
        for band in self.cube_vis.bands:
            if surface_windows[int(band)-1]:
                continue
            analysis = analyze_image(show_figures = False, figures=False)
            data = analysis.complete_image_analysis_from_cube(self.cube_vis, int(band), self.result_data_base_loc, biased_lat = prejudice_lat)
            key = str(self.cube_vis.w[int(band)]) + "µm_" + str(int(band))
            check_if_exists_or_write(key + ".pkl", base = self.result_data_base_loc, prefix = "bands", save = True, data = analysis, force_write=True, verbose=False)
            print(key)
            analysis_objects[key] = analysis
        for band in self.cube_ir.bands:
            if int(band) in ir_surface_windows:
                continue
            analysis = analyze_image(show_figures = False, figures=False)
            data = analysis.complete_image_analysis_from_cube(self.cube_ir, int(band), self.result_data_base_loc, biased_lat = prejudice_lat)
            key = str(self.cube_vis.w[int(band)]) + "_" + str(int(band))
            check_if_exists_or_write(key + ".pkl", base = self.result_data_base_loc, prefix = "bands", save = True, data = analysis, force_write=True)
            print(key)
            analysis_objects[key] = analysis
        check_if_exists_or_write("analysis_objects.pkl", base = self.result_data_base_loc, save = True, data = analysis_objects, force_write=True)
class analyze_complete_dataset:
    def __init__(self, cubes_location, manifest_sublocation) -> None:
        self.cubes_location = cubes_location
        self.manifest_location = os.path.join(cubes_location, manifest_sublocation)

        self.all_cubes = [cubes_location + file for file in os.listdir(cubes_location) if file.endswith(".cub") or file.endsiwth(".jcube")]
        with open(self.manifest_location, "r") as infile:
            self.manifest_contents = json.loads(infile)
        self.figures = {
            "i/f" : True,
            "timeline" : True,
            "lat_vs_wavelength" : True,
            "flux_vs_wavelength" : True,
            "false_color" : True,
            "tilt" : True,
        }
        self.figure_keys = {key: index for index, (key, value) in enumerate(self.figures.items())}
    def figure_options(self):
        print("\nCurrent Figure Setttings:\n",*[str(val[0]) + " = " + str(val[1]) for val in self.figures.items()], "\n\n", sep = "\n")
        return self.figures
    def complete_dataset_analysis(self, save_location = None):
        return "not implemented yet"
        
    
image = complete_cube("C:/Users/aadvi/Desktop/North_South_Asymmetry/cubes", "C1634084887_1" )
image.analyze_cube()
