from PIL import Image
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename  
import tkinter as tk
import tkinter.filedialog as fd
#Read the two images
def combineImage(list, location, types):
    x = len(list)
    aa = [];bb = []; cc = []
    for a in range(0, (x+1)):
        z = 0
        for b in range(0, (x+1)):
            if z == 0 and a*b >= len(list):
                aa.append(a)
                bb.append(b)
                cc.append(abs(a-b))
                z += 1
    x = 0
    for i in cc:
        if i == min(cc):
            print(aa[x], bb[x])
        x+=1
    print("a x b")
    a = (input())
    a = a.split(' ')
    b = int(a[2])
    a = int(a[0])
    #resize, first image

    image1 = Image.open(list[0])
    width, height = image1.size
    i = 0
    new_image = Image.new('RGB',(width*a, b*height), (250,250,2503))
    for x in range(0, width *(a+1), width):
        for y in range(0, height*b, height):
            try:                
                image = Image.open(list[i])
                new_image.paste(image, (x, y))
                print(int(x/742),int(y/742), list[i])
            except: 
                pass
            i+= 1
    new_image.save(location, types)
    new_image.show()

def files():

    list = []
    while(True):
        root = tk.Tk()
        filename = fd.askopenfilenames(parent=root, title='Choose a file')
        if filename == '':
            break
        print(filename)    
        for i in filename:
            list.append(i)
    return list
def main():
    filename = fd.askdirectory()
    file = input("what do you want the file name to be? (include filetype)   ")
    filename += "/" + file
    type =file.split('.')[-1]
    if type == "jpg":
        type = "JPEG"
    elif type == "ico":
        type = "ICO"
    elif type == "png":
        type = "PNG"
    elif type == "tif":
        type = "TIFF"
    combineImage(files(), filename, "JPEG")
main()