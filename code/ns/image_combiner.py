from PIL import Image
import matplotlib.pyplot as plt
a = "C:/Users/aadvi/Desktop/North_South_Asymmetry/nsa_paper_local/figures_updated/IF1.png"
b = "C:/Users/aadvi/Desktop/North_South_Asymmetry/nsa_paper_local/figures_updated/IF2.png"
a =  Image.open(a)
b = Image.open(b)
wa,ha = a.size
wb,hb = b.size
a = a.crop((30, 30, wa-80, ha-50))
b = b.crop((30, 30, wb-80, hb-50))
plt.rcParams["font.family"] = 'monospace'
plt.rcParams["font.weight"] = 'light'

def get_concat_v(im1, im2): 
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
c= get_concat_v(a,b)
plt.imshow(c)
c.save("C:/Users/aadvi/Desktop/North_South_Asymmetry/nsa_paper_local/figures_updated/IF.png")
plt.text(150,500,"  North - South\nBoundary Uncertainty", fontsize = 15)
plt.show()