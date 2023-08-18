import os
import os.path as path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib import cm

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
# surface_windows = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True,
#                    True, True, False, False, False, False, False, False, True, True, True, True, False, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True, True, True, True, False, False, False, False, False, False, False, False, False, False, False, True, True]
# ir_surface_windows = [99, 100, 106, 107, 108, 109, 119, 120, 121, 122, 135, 136, 137, 138, 139, 140, 141, 142, 163, 164,
#                       165, 166, 167, 206, 207, 210, 211, 212, 213, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352]


def join_strings(*args):
    return os.path.join(*args).replace("\\", "/")


def check_if_exists_or_write(file: str, base: str = None, prefix: str = None, file_type=None, save=False, data=None, force_write=False, verbose=True):
    full_path = join_strings(base, prefix, file)
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
                json.dump(data, file_object)
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


class polar_profile:
    def __init__(self, figures: bool = None):
        self = self
        self.figures = {
            "original_image": False,
            "emission_heatmap": False,
            "emission_heatmap_overlay": True,
            "distance_from_center": False,
            "show_slant": True,
            "plot_polar": False,
            "full_cube_slant": False,
            "persist_figures": True
        }
        self.figure_keys = {key: index for index,
                            (key, value) in enumerate(self.figures.items())}
        self.saved_figures = {}

    def figure_options(self):
        print("\nCurrent Figure Setttings:\n", *[str(val[0]) + " = " + str(
            val[1]) for val in self.figures.items()], "\n\n", sep="\n")
        return self.figures

    def show(self, force=False, duration: float = None):
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
            return None
        else:
            plt.pause(2)
            plt.clf()
            plt.close()

    def gauss2(self, x, b, a, x0, sigma):
        return b + (a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)))

    def poly12(self, x, a, b, c, d, e, f, g, h, i, j, k, l, m):
        return a*x**12+b*x**11+c*x**10+d*x**9+e*x**8+f*x**7+g*x**6+h*x**5+i*x**4+j*x**3+k*x**2+l*x+m

    def polyXDerivative(self, *args):  # 5)
        return [(len(args) - index-1)*value for index, value in enumerate(args)]

    def polyXPlugin(self, x,  *args):  # 5)
        y = 0
        for index, value in enumerate(args):
            y += value*x**(len(args) - index-1)
        return y

    def poly6(self, x, g, h, i, j, a, b, c):  # sextic function
        return g*x**6+h*x**5+i*x**4+j*x**3+a*x**2+b*x+c

    def poly6Prime(self, x, g, h, i, j, a, b, c):  # derivative of sextic function
        return 6*g*x**5+5*h*x**4+4*i*x**3+3*j*x**2+2*a*x+b

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

    def equirectangular_projection(self, cube, index):
        # take image and apply cylindrical projection
        # after cylindrical projection, remove extraneous longitude data
        proj = pyvims.projections.equirectangular.equi_cube(cube, index, 3)
        return proj

    def shift_image(self, image, shift, crop: bool = True):
        subtraction = (np.insert(image, 0, np.array(
            shift*2*[[0]*image.shape[1]]), axis=0) - np.concatenate((image, shift*2*[[0]*image.shape[1]])))
        hc_band = subtraction[2*shift:-2*shift]
        nan_rows = [False for i in range(
            shift)] + [True for i in range(hc_band.shape[0])] + [False for i in range(shift)]
        if crop:
            imager = image.filled(fill_value=np.nan)
            nan_columns = np.isnan(imager).sum(axis=0) > 0.15 * image.shape[0]
            hc_band = hc_band[:, ~nan_columns]
            return hc_band, nan_rows, nan_columns

        return hc_band, nan_rows

    def pixel_to_geo(self, pixel, geo_list):  # convert pixel to lat or pixel to lon
        return geo_list[int(np.around(pixel))]

    def pixel_to_geo_interpolated(self, pixel, geo_list):
        return geo_list[int(np.around(pixel))]

    def geo_to_pixel(self, geo, geo_list):  # convert lat to pixel or lon to pixel
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

    def find_center_of_cube(self, cube, band, w_index):
        flattened_array = cube.eme.flatten()

        lowest_indexes = np.argpartition(flattened_array, 5)[:4]
        lowest_indexes_2d = np.unravel_index(lowest_indexes, cube.eme.shape)

        lowest_index = np.argmin(cube.eme)
        lowest_index_2d = np.unravel_index(lowest_index, cube.eme.shape)

        center = [lowest_index_2d, np.mean(lowest_indexes_2d, axis=1)]
        center = np.mean(center, axis=0)
        center_point = [int(np.round(center[0])), int(np.round(center[1]))]

        if self.figures["emission_heatmap"]:
            fig = plt.figure(self.figure_keys["emission_heatmap"])
            plt.imshow(cube.eme)
            plt.scatter(lowest_indexes_2d[1], lowest_indexes_2d[0], s=4)
            plt.scatter(lowest_index_2d[1], lowest_index_2d[0], s=4)
            plt.scatter(center_point[1], center_point[0], c="r")
            self.show()
        if self.figures["emission_heatmap_overlay"]:
            fig = plt.figure(self.figure_keys["emission_heatmap_overlay"])
            plt.imshow(cube[band])
            plt.scatter(lowest_indexes_2d[1], lowest_indexes_2d[0], s=4)
            plt.scatter(lowest_index_2d[1], lowest_index_2d[0], s=4)
            plt.scatter(center_point[1], center_point[0], c="r")
            self.show()
            self.saved_figures["emission_heatmap_overlay"] = fig
        return center, center_point

    def distance_from_center_map(self, cube, band, w_index):
        # find distance from center
        center, center_point = self.find_center_of_cube(cube, band, w_index)
        ret = np.empty(cube.eme.shape)
        for y in range(cube.eme.shape[0]):
            for x in range(cube.eme.shape[1]):
                ret[y, x] = np.sqrt((x-center_point[1]) **
                                    2 + (y-center_point[0])**2)
        if self.figures["distance_from_center"]:
            fig = plt.figure(self.figure_keys["distance_from_center"])
            plt.imshow(ret)
            self.show()
            self.saved_figures["distance_from_center"] = fig
        return ret, center, center_point

    def horizontal_cross_section(self, cube, band, heat_map, center_point, degree):
        if degree == 180:
            start = 0
            end = center_point[1]
        elif degree == 0:
            start = center_point[1]
            end = heat_map.shape[1]
        y_val = center_point[0]
        y = []
        img = cube[band].copy()
        x_values = list(range(start, end))
        distance_x_values = [heat_map[y_val, x] for x in x_values]
        for x in x_values:
            y.append(cube[band][y_val, x])
            img[y_val, x] = 0
        if self.figures["show_slant"]:
            fig = plt.figure(self.figure_keys["show_slant"])
            plt.imshow(img)
            self.show()
            self.saved_figures["show_slant_" + str(degree)] = fig
        if self.figures["plot_polar"]:
            fig = plt.figure(self.figure_keys["plot_polar"])
            plt.plot(distance_x_values, y)
            self.show()
            self.saved_figures["plot_polar_" + str(degree)] = fig
        return distance_x_values, y

    def vertical_cross_section(self, cube, band, heat_map, center_point, degree):
        if degree == 90:
            start = 0
            end = center_point[0]
        elif degree == 270:
            start = center_point[0]
            end = heat_map.shape[0]
        x_val = center_point[1]
        x = []
        img = cube[band].copy()
        brightness_values = list(range(start, end))
        distance_brightness_values = [heat_map[y, x_val]
                                      for y in brightness_values]
        for y in brightness_values:
            x.append(cube[band][y, x_val])
            img[y, x_val] = 0
        if self.figures["show_slant"]:
            fig = plt.figure(self.figure_keys["show_slant"])
            plt.imshow(img)
            self.show()
            self.saved_figures["show_slant_" + str(degree)] = fig
        if self.figures["plot_polar"]:
            fig = plt.figure(self.figure_keys["plot_polar"])
            plt.plot(distance_brightness_values, x)
            self.show()
            self.saved_figures["plot_polar_" + str(degree)] = fig
        return distance_brightness_values, x

    def slanted_cross_section(self, cube, band, heat_map, center_point, degree):
        angle_rad = np.radians(degree)
        start_x = center_point[1]
        start_y = center_point[0]
        dist = np.sqrt((heat_map.shape[0] - center_point[0])
                       ** 2 + (heat_map.shape[1] - center_point[1])**2)
        endpoint_x = start_x + np.sin(angle_rad) * dist
        endpoint_y = start_y - np.cos(angle_rad) * dist
        mask = np.zeros(heat_map.shape)
        cv2.line(mask, (int(start_x), int(start_y)),
                 (int(endpoint_x), int(endpoint_y)), 1, 1)
        line_indexes = np.where(mask == 1)
        line_indexes = np.array(line_indexes).T
        if self.figures["show_slant"]:
            fig = plt.figure(self.figure_keys["show_slant"])
            plt.scatter(line_indexes[:, 1], line_indexes[:, 0], marker="s")
            img = cube[band].copy()
            img[line_indexes[:, 0], line_indexes[:, 1]] = 0
            plt.imshow(img)
            self.show()
            self.saved_figures["show_slant_" + str(degree)] = fig
        try:
            distances = [heat_map[x, y] for x, y in line_indexes]
            distances = np.array(distances)
        except:
            print(line_indexes)
        brightness_values = [cube[band][x, y] for x, y in line_indexes]
        # sort values by distance
        pairs = list(zip(distances, brightness_values, line_indexes))

        # Sort the pairs based on distances
        sorted_pairs = sorted(pairs, key=lambda x: x[0])

        # Unpack the sorted pairs into separate arrays
        sorted_distances, sorted_brightness_values, pixel_indices = zip(
            *sorted_pairs)
        emission_angles = [cube.eme[x, y] for x, y in line_indexes]
        if self.figures["plot_polar"]:
            fig = plt.figure(self.figure_keys["plot_polar"])
            plt.plot(sorted_distances, sorted_brightness_values)
            self.show()
            self.saved_figures["plot_polar_" + str(degree)] = fig
        return distances, emission_angles, brightness_values, pixel_indices

    def cross_section(self, cube, band, heat_map, center_point, degree):
        # if degree == 0 or degree == 180:
        #     distance, brightness = self.horizontal_cross_section(cube, band, heat_map, center_point, degree)
        # elif degree == 90 or degree == 270:
        #     distance, brightness = self.vertical_cross_section(cube, band, heat_map, center_point, degree)
        # else:
        pixel_distance, emission_angles, brightness, pixel_indices = self.slanted_cross_section(
            cube, band, heat_map, center_point, degree)
        # make sure both arrays are lists
        pixel_distance = list(pixel_distance)
        emission_angles = list(emission_angles)
        brightness = list(brightness)

        pixel_indices = [index.tolist() for index in pixel_indices]

        return pixel_distance, emission_angles, brightness, pixel_indices

    def remove_duplicates(self, x, y):
        data_dict = {}
        for x_val, y_val in zip(x, y):
            if x_val in data_dict:
                data_dict[x_val].append(y_val)
            else:
                data_dict[x_val] = [y_val]

        averaged_x = []
        averaged_y = []
        for x_val, y_vals in data_dict.items():
            averaged_x.append(x_val)
            averaged_y.append(sum(y_vals) / len(y_vals))
        return np.array(averaged_x), np.array(averaged_y)

    def complete_image_analysis_from_cube(self, cube,  north_orientation: int, distance_array, center_of_cube, center_point, band: int, band_index: int):
        self.cube_name = cube.img_id
        slants = [0, 15, 30, 45, 60, 90, 120, 135,
                   150, 180, 210, 225, 240, 270, 300, 315, 330]
        degrees = north_orientation + np.array(slants)
        cmap = matplotlib.cm.get_cmap('rainbow')
        fits = {}
        for degree in degrees:
            pixel_distance, emission_angles, brightness, pixel_indices = self.cross_section(
                cube, band, distance_array, center_point, degree)
            fits[degree] = (pixel_distance, emission_angles,
                            brightness, pixel_indices)

        # if self.figures["full_cube_slant"]:
        #     fig = plt.figure(self.figure_keys["full_cube_slant"])
        #     for index, (pixel_distance, emission_angles, brightness, pixel_indices) in enumerate(fits.values()):
        #         plt.plot(emission_angles, brightness, label=str(degrees[index]), color=cmap(
        #             index/len(degrees)*0.75))
        #     plt.title("Cross sections of " + cube.flyby.name +
        #               " at " + str(cube.w[band_index]) + " µm")
        #     plt.xlabel("Distance from center (px)")
        #     plt.ylabel("Brightness")
        #     plt.legend()
        #     self.show()
            # self.saved_figures["full_cube_slant"] = fig

            # for index, (pixel_distance, emission_angles, brightness, pixel_indices) in enumerate(fits.values()):
            #     pixel_distance, emission_angles = self.remove_duplicates(pixel_distance, emission_angles)
            #     sorted_indices = np.argsort(pixel_distance)
            #     sorted_x = pixel_distance[sorted_indices]
            #     sorted_y = emission_angles[sorted_indices]
            #     csx = PchipInterpolator(sorted_x, sorted_y)
            #     xs = np.linspace(np.min(pixel_distance), np.max(pixel_distance), 1000)
            #     plt.plot(xs, gaussian_filter([csx(xss) for xss in xs], sigma=4), label=str(
            #         degrees[index]), color=cmap(index/len(degrees)*0.75))
            # plt.title("Smoothed cross sections of " + cube.flyby.name +
            #           " at " + str(cube.w[band_index]) + " µm")
            # plt.xlabel("Distance from center (px)")
            # plt.ylabel("Brightness")
            # plt.legend()
            # self.show()
        if self.figures["full_cube_slant"]:
            for index, (pixel_distance, emission_angles, brightness, pixel_indices) in enumerate(fits.values()):
                # emission_angles, brightness = self.remove_duplicates(emission_angles, brightness)
                # sorted_indices = np.argsort(emission_angles)
                # sorted_x = emission_angles[sorted_indices]
                # sorted_y = brightness[sorted_indices]
                # csx = PchipInterpolator(sorted_x, sorted_y)
                # xs = np.linspace(np.min(pixel_distance), np.max(pixel_distance), 1000)
                # plt.plot(xs, gaussian_filter([csx(xss) for xss in xs], sigma=4), label=str(degrees[index]), color=cmap(index/len(degrees)*0.75))
                sorted_indices = np.argsort(emission_angles)
                pairs = list(zip(emission_angles, brightness))

                # Sort the pairs based on distances
                sorted_pairs = sorted(pairs, key=lambda x: x[0])

                # Unpack the sorted pairs into separate arrays
                emission_angles, brightness = zip(
                    *sorted_pairs)
                plt.plot(emission_angles, brightness, label=str(degrees[index]), color=cmap(index/len(degrees)*0.75))

            plt.title("Smoothed cross sections of " + cube.flyby.name +
                    " at " + str(cube.w[band_index]) + " µm")
            plt.xlabel("Eme")
            plt.ylabel("Brightness")
            plt.legend()
            self.show()
            

        if self.figures["original_image"]:
            fig = plt.figure(self.figure_keys["original_image"])
            plt.imshow(cube[band], cmap="gray")
            plt.title("Cube from " + cube.flyby.name +
                      " at " + str(cube.w[band_index]) + " µm")
            self.show()
            self.saved_figures["original_image"] = fig

        if self.figures["persist_figures"] == "wait_till_end":
            self.show(force=True)
        plt.close("all")
        return [fits, slants]


class complete_cube:
    def __init__(self, parent_directory: str, cube_subdirectory: str, data_save_folder_name: str = None):
        if data_save_folder_name is None:
            save_location = join_strings(
                parent_directory, cube_subdirectory, "analysis/limb/")
        else:
            save_location = join_strings(
                parent_directory, cube_subdirectory, data_save_folder_name, "limb/")
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

    def find_north_and_south(self, cube):
        objectx = polar_profile()
        distance_array, center_of_cube, center_point = objectx.distance_from_center_map(
            cube, 69, w_index=69)

        center_of_cube, center_point = objectx.find_center_of_cube(
            cube, 69, w_index=69)
        angles = []
        brightnesses = cube.lat.flatten()
        for y in range(cube.lat.shape[0]):
            for x in range(cube.lat.shape[1]):
                angles.append(
                    int(np.degrees(np.arctan2(x - center_point[1], center_point[0] - y))))
        actual_angles = sorted(set(angles))
        br = [np.mean([brightnesses[index] for index, a in enumerate(
            angles) if a == angle]) for angle in actual_angles]
        min_angle = actual_angles[np.argmin(br)]
        if min_angle < 0:
            min_angle += 360
        max_angle = actual_angles[np.argmax(br)]
        if max_angle < 0:
            max_angle += 360
        calculated_rots = np.array([min_angle, max_angle])
        if min_angle > max_angle:
            rot_angle = np.mean([min_angle, max_angle]) - 90
        else:
            rot_angle = np.mean([min_angle, max_angle]) + 90
        if rot_angle > 180:
            rot_angle -= 360
        return rot_angle, distance_array, center_of_cube, center_point

    def run_analysis(self, cube, north_orientation, distance_array, center_of_cube, center_point, band, band_index, key, force_write=False):
        start = time.time()
        analysis = polar_profile()
        data = analysis.complete_image_analysis_from_cube(
            cube, north_orientation, distance_array, center_of_cube, center_point, int(band), band_index=band_index)
        
        data.append({"flyby" : cube.flyby.name, "cube" : cube.img_id, "band" : cube.w[band_index], "center_of_cube" : center_point, "north_orientation" : north_orientation})
        check_if_exists_or_write(key + ".pkl", base=self.result_data_base_loc,
                                 prefix="objects", save=True, data=analysis, force_write=force_write, verbose=False)

        check_if_exists_or_write(key + ".pkl", base=self.result_data_base_loc,
                                 prefix="fits", save=True, data=data, force_write=force_write, verbose=False)
        end = time.time()
        total_time = end - self.start_time
        percentage = (band-self.starting_band) / \
            (self.total_bands-self.starting_band)

        print(key, "took", end - start, "seconds.", "expected time left:",
              total_time/percentage - total_time, "seconds")
        return data, analysis

    def analyze_dataset(self, cube: str = None, force=False):
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
        self.cube_vis = pyvims.VIMS(cube + "_vis.cub", self.cwd, channel="vis")
        self.cube_ir = pyvims.VIMS(cube + "_ir.cub", self.cwd, channel="ir")
        self.total_bands = self.cube_vis.bands.shape[0] + \
            self.cube_ir.bands.shape[0]
        self.start_time = time.time()
        self.starting_band = -1
        north_orientation, distance_array, center_of_cube, center_point = self.find_north_and_south(
            self.cube_vis)
        datas = {}
        objects = {}
        for band in self.cube_vis.bands:
            band_index = int(band)-1
            # if surface_windows[band_index]:
            #     continue
            key = str(self.cube_vis.w[band_index]) + "µm_" + str(int(band))
            check = os.path.exists(join_strings(
                self.result_data_base_loc, "objects", key + ".pkl"))
            if check is False or force is True:
                if self.starting_band == -1:
                    self.starting_band = int(band)
                data, cube_object = self.run_analysis(self.cube_vis, north_orientation, distance_array,
                                         center_of_cube, center_point, band, band_index, key,force)
                datas[key] = data
                objects[key] = cube_object
        for band in self.cube_ir.bands:
            band_index = int(band)-97
            # if int(band) in ir_surface_windows:
            #     continue
            key = str(self.cube_ir.w[band_index]) + "µm_" + str(int(band))
            check = os.path.exists(join_strings(
                
                self.result_data_base_loc, "objects", key + ".pkl"))
            if check is False or force is True:
                if self.starting_band == -1:
                    self.starting_band = int(band)
                data, cube_object = self.run_analysis(self.cube_ir, north_orientation, distance_array,
                                         center_of_cube, center_point, band, band_index, key,force)
                datas[key] = data
                objects[key] = cube_object
        return datas, objects


class analyze_complete_dataset:
    def __init__(self, cubes_location) -> None:
        self.cubes_location = cubes_location
        folders = os.listdir(self.cubes_location)
        self.cubes = [folder for folder in folders if folder.startswith(
            "C") and folder[-2] == ("_")]

        self.figures = {
        }
        self.figure_keys = {key: index for index,
                            (key, value) in enumerate(self.figures.items())}

    def figure_options(self):
        print("\nCurrent Figure Setttings:\n", *[str(val[0]) + " = " + str(
            val[1]) for val in self.figures.items()], "\n\n", sep="\n")
        return self.figures

    def complete_dataset_analysis(self, save_location=None, force=False):
        all_data = {}
        all_objects = {}
        for cub in self.cubes:
            image = complete_cube(base, cub)
            datas, objects = image.analyze_dataset(force=force)
            fit_cube = join_strings(self.cubes_location, cub, "analysis/limb/")
            if force == True:
                check_if_exists_or_write(cub + "_fits.pkl"  , base=fit_cube,
                                    prefix="", save=True, data=datas, force_write=True, verbose=True)
                check_if_exists_or_write(cub + "_objects.pkl"  , base=fit_cube,
                            prefix="", save=True, data=objects, force_write=True, verbose=True)
            all_data[cub] = datas
            all_objects[cub] = objects
        if force == True:
            check_if_exists_or_write("combined_objects.pkl", base=self.cubes_location,
                                    prefix="analysis/limb", save=True, data=all_objects, force_write=True, verbose=True)
            check_if_exists_or_write("combined_fits.pkl", base=self.cubes_location,
                                    prefix="analysis/limb", save=True, data=all_data, force_write=True, verbose=True)

    # def combine_all_bands_in_cube(self):
    #     all_fits = {}
    #     all_objects = {}
    #     for cub in self.cubes:
    #         fits = {}
    #         fit_cube = join_strings(
    #             self.cubes_location, cub, "analysis/limb/fits")
    #         fit_bands = os.listdir(fit_cube)
    #         for band in fit_bands:
    #             with open(join_strings(fit_cube, band), 'rb') as handle:
    #                 fits[band] = pickle.load(handle)
    #             print(cub, band, "fit loaded     ", end="\r")
    #         objects = {}
    #         object_cube = join_strings(
    #             self.cubes_location, cub, "analysis/limb/objects")
    #         fit_bands = os.listdir(object_cube)
    #         print("\n")
    #         for band in fit_bands:
    #             with open(join_strings(object_cube, band), 'rb') as handle:
    #                 objects[band] = pickle.load(handle)
    #             print(cub, band, "object loaded     ", end="\r")

    #         # save objects in respective location
    #         print("\n\n")
    #         check_if_exists_or_write("analysis/limb/combined_objects.pkl", base=self.cubes_location,
    #                                  prefix=cub, save=True, data=objects, force_write=True, verbose=True)
    #         check_if_exists_or_write("analysis/limb/combined_fits.pkl", base=self.cubes_location,
    #                                  prefix=cub, save=True, data=fits, force_write=True, verbose=True)
    #         print("\n\n")

    #         all_fits[cub] = fits
    #         all_objects[cub] = objects
    #     print("\n\n")
    #     check_if_exists_or_write("combined_objects.pkl", base=self.cubes_location,
    #                              prefix="analysis/limb", save=True, data=all_objects, force_write=True, verbose=True)
    #     check_if_exists_or_write("combined_fits.pkl", base=self.cubes_location,
    #                              prefix="analysis/limb", save=True, data=all_fits, force_write=True, verbose=True)



    def timeline_figure(self, band: int = None):
        all_fits = {}
        fig, axs = plt.subplots(4, 3, figsize=(12, 8))
        axs = axs.flatten()
        plt.title(str(band))
        for index, cub in enumerate(self.cubes):
            cube = join_strings(self.cubes_location, cub, "analysis/limb/fits")
            bands = os.listdir(cube)
            band_file = [band_f for band_f in bands if band_f.endswith(
                '_' + str(band) + ".pkl")][0]
            with open(join_strings(cube, band_file), 'rb') as handle:
                all_fits[cub] = pickle.load(handle)
        fig = plt.figure()
        quantity = len(all_fits)
        cmap = matplotlib.cm.get_cmap('rainbow')
        for index, (cube, fit) in enumerate(all_fits.items()):
            list_keys = list(fit[0].keys())

            for ind, deg in enumerate(list_keys):
                emission = fit[0][deg][1]
                brightness = fit[0][deg][2]
                zi = zip(emission, brightness)
                zi = sorted(zi, key=lambda x: x[0])
                emission, brightness = zip(*zi)
                if ind == 0:
                    plt.plot(emission, brightness, label = fit[2]["flyby"] + " " + str(deg),  color= cmap(ind/len(list_keys)*0.75))
                else:
                    plt.plot(emission, brightness, label = fit[2]["flyby"] + " " +  str(deg), color= cmap(ind/len(list_keys)*0.75))

            plt.legend()
            plt.tight_layout()
            plt.waitforbuttonpress()
        return all_fits

    def generate_figures(self, figures: dict = None):
        for band in range(1, 352, 10):
            # if band < 97 and surface_windows[band-1] == True:
            #     continue
            # elif band > 96 and band in ir_surface_windows:
            #     continue
            self.timeline_figure(band)


base = "C:/Users/aadvi/Desktop/North_South_Asymmetry/cubes"
analyze = analyze_complete_dataset(base)
analyze.complete_dataset_analysis(force=True)

analyze.timeline_figure(band = 1)
