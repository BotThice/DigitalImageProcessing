
def prepareInputData(picPath):
    def readPicture(picPath):
        pic = open(picPath, "rb")
        formats = ""
        line = ""
        width = 0
        height = 0
        maxLevel = 0
        content = ""
        
        formats = pic.readline().decode('utf-8').strip()
        if (formats != "P5"):
            print("The format of the picture is not P5")
            return
        
        line = pic.readline().decode('utf-8').strip()
        while line[0] == "#":
            line = pic.readline().decode('utf-8').strip()
            
        width, height = line.split()
        maxLevel = pic.readline()
        content = pic.read()
        pic.close()
        return width, height, maxLevel, content

    def convertToInt(data):
        width, height, maxGreyLevel, content = data
        width = int(width)
        height = int(height)
        maxGreyLevel = int(maxGreyLevel)
        content = list(content)
        return width, height, maxGreyLevel, content
    
    data = readPicture(picPath)
    data = convertToInt(data)
    data = {'width': data[0], 
            'height': data[1],
            'maxGreyLevel': data[2],
            'content': data[3]}
    return data

def writePicture(picPath, data):
    width = data['width']
    height = data['height']
    maxGreyLevel = data['maxGreyLevel']
    content = data['content']
    
    pic = open(picPath, "wb")
    pic.write("P5\n".encode('utf-8'))
    pic.write((str(width) + " " + str(height) + "\n").encode('utf-8'))
    pic.write((str(maxGreyLevel) + "\n").encode('utf-8'))
    pic.write(bytearray(content))
    pic.close()