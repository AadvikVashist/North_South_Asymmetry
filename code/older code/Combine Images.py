from PIL import Image
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilename  
import tkinter as tk
import tkinter.filedialog as fd
#Read the two images
def combineImage(list, location, types):
    x = len(list)
    aa = [];bb = []; cc = []; aaa = []; bbb = []
    for i in list:
        image1 = Image.open(i)
        aaa.append(image1.width)
        bbb.append(image1.height)
    width = max(aaa)
    height = max(bbb)
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
    i = 0
    new_image = Image.new('RGB',(width*a, b*height), (250,250,2503))
    for x in range(0, width *(a+1), width):
        for y in range(0, height*b, height):
            try:                
                img = Image.open(list[i])
                cWidth, cHeight = img.size
                if cWidth / width > cHeight / height:
                    basewidth = width
                    newheight = cHeight / cWidth * basewidth
                    img = img.resize((basewidth,newheight), Image.ANTIALIAS)
                    new_image.paste(img, (x, y))
                    print(int(x/742),int(y/742), list[i])
                    print(basewidth, newheight)
                elif cWidth == width and cHeight == height:
                    new_image.paste(img, (x, y))
                else:
                    baseHeight = height
                    newWidth = cWidth / cHeight * baseHeight 
                    img = img.resize((newWidth,baseHeight), Image.ANTIALIAS)
                    new_image.paste(img, (x, y))
                    print(int(x/742),int(y/742), list[i])
                    print(newWidth, baseHeight)
            except: 
                print("didn't work")
                pass
            finally:
                i+= 1
    new_image.show()
    new_image.save(location, types)


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
            
    root = Tk()
    #root.attributes("-topmost", True) # this also works
    root.lift()
    root.withdraw()
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
    else:
        print("try again")
        main()
    x = files()
    combineImage(x, filename, type)
main()