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

def prepareInputData(picPath):
    data = readPicture(picPath)
    data = convertToInt(data)
    data = {'width': data[0], 
            'height': data[1],
            'maxGreyLevel': data[2],
            'content': data[3]}
    return data

def prepareOutputData(inputPictureData, mappedGreyLevel):
    data = {'width': inputPictureData['width'],
                  'height': inputPictureData['height'],
                  'maxGreyLevel': inputPictureData['maxGreyLevel'],
                  'content': [mappedGreyLevel[pixel] for pixel in inputPictureData['content']]}
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

def showBothHistrograms(inputHistrogram, eqHistrogram):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Plot the original histogram
    ax1.plot(list(inputHistrogram.keys()), list(inputHistrogram.values()))
    ax1.set_title('Original Histogram')

    # Plot the equalized histogram
    ax2.plot(list(eqHistrogram.keys()), list(eqHistrogram.values()))
    ax2.set_title('Equalized Histogram')

    # Show the plots
    plt.show()

# picture 1
picture1 = "inputPictures/Cameraman.pgm"
picture1Data = prepareInputData(picture1)

result1 = histrogram.histrogramEqualization(picture1Data)
eqHistrogram1, mappedGreyLevel1 = result1

outPic1 = "outputPictures/equalizedCameraman.pgm"
outPic1Data = prepareOutputData(picture1Data, mappedGreyLevel1)
writePicture(outPic1, outPic1Data)

pic1Histrogram = histrogram.histrogram(picture1Data['maxGreyLevel'],picture1Data['content'])
showBothHistrograms(pic1Histrogram, eqHistrogram1)

# picture 2
picture2 = "inputPictures/SEM256_256.pgm"
picture2Data = prepareInputData(picture2)

result2 = histrogram.histrogramEqualization(picture2Data)
eqHistrogram2, mappedGreyLevel2 = result2

outPic2 = "outputPictures/equalizedSEM256_256.pgm"
outPic2Data = prepareOutputData(picture2Data, mappedGreyLevel2)
writePicture(outPic2, outPic2Data)

pic2Histrogram = histrogram.histrogram(picture2Data['maxGreyLevel'],picture2Data['content'])
showBothHistrograms(pic2Histrogram, eqHistrogram2)

