import histrogram
import matplotlib.pyplot as plt

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
    
    line = pic.readline().decode('utf-8')
    while line[0] == "#":
        line = pic.readline().decode('utf-8')
        
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
    content = [int(pixel) for pixel in content]
    return width, height, maxGreyLevel, content

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

inputPicture = "Cameraman.pgm"
inputPictureData = readPicture(inputPicture)
inputPictureData = convertToInt(inputPictureData)
inputPictureData = {'width': inputPictureData[0], 
                    'height': inputPictureData[1],
                    'maxGreyLevel': inputPictureData[2],
                    'content': inputPictureData[3]}

inputHistrogram = histrogram.histrogram(inputPictureData['maxGreyLevel'],
                                        inputPictureData['content'])
result = histrogram.histrogramEqualization(inputPictureData)
eqHistrogram, mappedGreyLevel = result

outPicture = "equalizedCameraman.pgm"
outPictureData = {'width': inputPictureData['width'],
                  'height': inputPictureData['height'],
                  'maxGreyLevel': inputPictureData['maxGreyLevel'],
                  'content': [mappedGreyLevel[pixel] for pixel in inputPictureData['content']]}
writePicture(outPicture, outPictureData)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original histogram
ax1.plot(list(inputHistrogram.keys()), list(inputHistrogram.values()))
ax1.set_title('Original Histogram')

# Plot the equalized histogram
ax2.plot(list(eqHistrogram.keys()), list(eqHistrogram.values()))
ax2.set_title('Equalized Histogram')

# Show the plots
plt.show()
