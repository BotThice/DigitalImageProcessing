import matplotlib.pyplot as plt
import numpy as np

def readPicture(picPath):
    pic = open(picPath, "rb")
    formats = ""
    line = ""
    width = 0
    height = 0
    maxLevel = 0
    content = ""
    
    formats = pic.readline().decode('utf-8').strip()
    print(formats)
    if (formats != "P5"):
        print("The format of the picture is not P5")
        return
    
    line = pic.readline().decode('utf-8')
    while line[0] == "#":
        line = pic.readline().decode('utf-8')
        
    width, height = line.split()
    maxLevel = pic.readline()
    content = pic.read()
    return width, height, maxLevel, content

def convertToInt(data):
    width, height, maxGreyLevel, content = data
    width = int(width)
    height = int(height)
    maxGreyLevel = int(maxGreyLevel)
    content = [int(pixel) for pixel in content]
    return width, height, maxGreyLevel, content

def histrogram(maxGreyLevel, content) :
    histrogram = {}
    
    for greyLevel in range (maxGreyLevel + 1):
        histrogram[greyLevel] = 0
        
    for pixel in content:
        histrogram[pixel] += 1
           
    return histrogram

InputPicture = "scaled_shapes.pgm"
InputpictureData = readPicture(InputPicture)
InputpictureData = convertToInt(InputpictureData)
InputpictureData = {'width': InputpictureData[0], 
                    'height': InputpictureData[1],
                    'maxGreyLevel': InputpictureData[2],
                    'content': InputpictureData[3]}
histrogramInput = histrogram(InputpictureData['maxGreyLevel'], InputpictureData['content'])
print(histrogramInput)
print(list(histrogramInput.keys()))
print(list(histrogramInput.values()))
plt.plot(list(histrogramInput.keys()), list(histrogramInput.values()))
plt.show()