from PIL import Image
import os

dir =  os.path.dirname(__file__) + '\\image\\'

for _ in range(1,11):
    sample_list = list()
    for img in os.listdir(dir + '{num}\\x\\'):
        imgdata =  Image.open(img)
        sample_list.append(imgdata)
        for y_img in os.listdir(dir+'{num}\\y\\'):
            y_imgdata = Image.open(y_img)
            sample_list.append(y_img)
