import datetime
import numpy as np
import pandas as pd
import pickle
from thefuzz import fuzz
def get_limb(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "limb vis" in h.lower()][0]
    return [header] + [c for c in csv if "yes" in c[header_index].lower()]
def get_percentage(csv,min):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "percentage" in h.lower()][0]
    return [header] + [c for c in csv[1::] if float(c[header_index]) >= min]
def get_vis(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "sampling mode" in h.lower()][0]
    header_split_index = [index for index,h in enumerate(header[header_index].split("|")) if "vis" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index].lower().split("|")[header_split_index]]
def get_ir(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "sampling mode" in h.lower()][0]
    header_split_index = [index for index,hiin enumerate(header[header_index].split("|")) if "ir" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index].lower().split("|")[header_split_index]]
def get_vis_and_ir(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "sampling mode" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index]]

def get_filtered_phase(csv,phase_list : list):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "phase" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and float(c[header_index]) >= min(phase_list) and float(c[header_index]) <= max(phase_list)]
def get_info_cubes(csv):
    header = csv[0]
    header_indexes = set([index for row in csv for index,c in enumerate(row) if c == "_"])
    return [header] + [row for row in csv[1::] if all([col != "_" for col in [row[i] for i in header_indexes]])]
def get_refined_samples(csv,x,y):
    header = csv[0]
    minx = min(x); maxx = max(x); miny = min(y); maxy = max(y)
    header_index = [index for index,h in enumerate(header) if "samples" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and int(c[header_index].split("x")[0]) >= minx and int(c[header_index].split("x")[0]) <= maxx and int(c[header_index].split("x")[1]) >= miny and int(c[header_index].split("x")[1]) <= maxy]
def get_targeted_flyby(csv):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "flyby" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "|" in c[header_index] and "T" in c[header_index].split(" | ")[0]]
def get_mission(csv,mission_ints):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "mission" in h.lower()][0]
    mission_list = [c[header_index] for c in csv[1::]]
    #sort based on index of occurence in mission_list 
    missions = sorted(list(set(mission_list)), key=mission_list.index)
    if type(mission_ints) == list  and len(mission_ints) > 1:
        for index in range(len(mission_ints)):
            if type(mission_ints[index]) == str:
                mission = [i for i,m in enumerate(missions) if fuzz.ratio(m,mission_ints[index]) > 70]
                if len(mission) != 1:
                    raise ValueError("Mission name not found")
                else:
                    mission_ints[index] = mission[0]
        return [header] + [c for c in csv[1::] if any([missions[miss] in c[header_index] for miss in mission_ints])]
    else:
        
        if type(mission_ints) == list and len(mission_ints) == 1:
            mission_ints = mission_ints[0]
        if type(mission_ints) == str:
            mission = [i for i,m in enumerate(missions) if fuzz.ratio(m,mission_ints) > 70]
        if len(mission) != 1:
            raise ValueError("Mission name not found")
        else:
            mission_ints = mission[0]
        return [header] + [c for c in csv[1::] if missions[mission_ints] in c[header_index]]
def filter_dates(csv,start_date,end_date):
    header = csv[0]
    header_index = [index for index,h in enumerate(header) if "date" in h.lower()][0]
    return [header] + [c for c in csv[1::] if "n/a" not in c[header_index] and datetime.strptime(c[header_index],"%Y-%m-%d") >= datetime.strptime(start_date,"%Y-%m-%d") and datetime.strptime(c[header_index],"%Y-%m-%d") <= datetime.strptime(end_date,"%Y-%m-%d")]
if __name__ == "__main__":
    with open("code/VIMS Data Ingestion/data/combined_nantes.pickle", "rb") as f:
        # Use pickle to dump the variable into the file
        data = pickle.load(f)    

    # limb = get_limb(data)
    refined_search = get_info_cubes(data)
    refined_search = get_targeted_flyby(refined_search)
    refined_search = get_vis_and_ir(refined_search)
    refined_search = get_percentage(refined_search, 0.5)
    refined_search = get_filtered_phase(refined_search,[0,40])
    refined_search = get_refined_samples(refined_search,[10,150],[10,150])
    refined_search = get_mission(refined_search, ["equinox","solsticesd"])
    refined_search = filter_dates(refined_search, "2010-01-01", "2020-01-01")
    print(refined_search)