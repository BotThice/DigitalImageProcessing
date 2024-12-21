def readPicture(picPath):
    pic = open(picPath, "rb")
    format = ""
    line = ""
    width = 0
    height = 0
    maxLevel = 0
    content = ""
    
    format = pic.readline().decode('utf-8') 
    if (format != "P5\n"):
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

def histrogram (maxGreyLevel, content) :
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
print(histrogram(InputpictureData))
