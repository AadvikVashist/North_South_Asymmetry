import pygsheets
import numpy as np
gc = pygsheets.authorize(service_file='C:/Users/aadvi/Desktop/Coding/T62/Frame Images/Code/Vis.Cyl Iterations/key.json')
sh = gc.open_by_key("18UnakpgRde3uqaBS27O7GgAetDJ4JqS4RMzrL2mEVSQ")
wks = sh[0]
wks.update_value('a1',"wassup tim")
print(wks.cell((1,4)))