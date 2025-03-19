import pgmRW
import numpy as np
import math

def nextPowOf2(x):
    return 1 << (x - 1).bit_length()

def centerPadWith0(img, newWidth, newHeight):
    newImg = img.copy()
    newImg['width'] = newWidth
    newImg['height'] = newHeight
    padded = []

    oldImgContent = img['content']
    oldWidth = img['width']
    oldHeight = img['height']

    heightOffset = newHeight - oldHeight
    topOffset = int (heightOffset / 2)

    widthOffset = newWidth - oldWidth
    leftOffset = int (widthOffset / 2)

    for y in range(newHeight):
        for x in range(newWidth):

            if x < leftOffset or y < topOffset:
                padded.append(0)
            elif x >= (leftOffset + oldWidth) or y >= (topOffset + oldHeight):
                padded.append(0)
            else:
                padded.append(oldImgContent[(y - topOffset) * oldWidth + (x - leftOffset)])

    newImg['content'] = padded

    return newImg

def make2Dlist(list1D, row, col):
    if len(list1D) != row * col:
        raise ValueError("The size of the 1D list does not match the given dimensions.")

    image_2d = []  # Create an empty 2D list

    for i in range(row):
        row = []  # Create an empty row
        for j in range(col):
            row.append(list1D[i * col + j])  # Extract element from 1D list
        image_2d.append(row)  # Add row to 2D list

    return image_2d

def make1Dlist(list2D):
    list = []
    for row in list2D:
        for val in row:
            list.append(val)
    return list

def write_2d_list_to_file(filename, data):
    """Write a 2D list to a text file, separating values with spaces."""
    with open(filename, 'w') as file:
        for row in data:
            file.write(" ".join(map(str, row)) + "\n")  # Convert each row to a space-separated string

def init2DWith0(row, col):
    return [[0] * col for _ in range(row)]

def normalize(list2D):
    row = len(list2D)
    col = len(list2D[0])

    # Normalize list2Dlitude data to range [0, 255]
    maxlist2D = max(max(row) for row in list2D)
    minlist2D = min(min(row) for row in list2D)

    # Create a list to hold normalized values
    normalized = init2DWith0(row, col)
    if maxlist2D == minlist2D:  # Avoid division by zero
        return normalize
    
    for i in range(row):
        for j in range(col):
            # Normalize and scale to 0-255
            value = round(255 * (list2D[i][j] - minlist2D) / (maxlist2D - minlist2D))
            normalized[i][j] = value

    return normalized

def amplitude(Complex):
    amp = init2DWith0(len(Complex), len(Complex[0]))

    for row in range(len(Complex)):
        for col in range(len(Complex[row])):
            real = Complex[row][col].real
            im = Complex[row][col].imag
            amp[row][col] = math.sqrt(pow(real, 2) + (pow(im, 2)))

    normAmp = normalize(amp)

    return amp, normAmp

def phase(Complex):
    phase = init2DWith0(len(Complex), len(Complex[0]))

    for row in range(len(Complex)):
        for col in range(len(Complex[row])):
            phase[row][col] = np.angle(Complex[row][col])

    normPhase = normalize(phase)

    return phase, normPhase

def reconstructFFTwith1Infomation(fftImg):

    def reconstructOnlyPhase(phase):
        phases = np.array(phase)
        fft = np.exp(1j * phases)
        return fft
    
    def reconstructOnlyAmp(amp):
        amps = np.array(amp)
        fft = amps * np.exp(1j * 0)
        return fft

    height = len(fftImg)
    width = len(fftImg[0])

    amp, normAmp = amplitude(fftImg)
    phases, normPhase = phase(fftImg)

    onlyAmpImg = np.fft.ifftshift(reconstructOnlyAmp(amp))
    onlyAmpImg = np.fft.ifft2(onlyAmpImg)
    onlyAmpImg = amplitude(onlyAmpImg)[1]
    onlyAmpImg = make1Dlist(onlyAmpImg)
    onlyAmpImg = {
        'width': width,
        'height': height,
        'maxGreyLevel': 255,
        'content': onlyAmpImg
    }

    onlyPhaseImg = np.fft.ifftshift(reconstructOnlyPhase(phases))
    onlyPhaseImg = np.fft.ifft2(onlyPhaseImg)
    onlyPhaseImg = normalize(onlyPhaseImg.real)
    onlyPhaseImg = make1Dlist(onlyPhaseImg)
    onlyPhaseImg = {
        'width': width,
        'height': height,
        'maxGreyLevel': 255,
        'content': onlyPhaseImg
    }

    return onlyAmpImg, onlyPhaseImg

def phasesShift(Fs, shiftX, shftY):
    phases, normPhase = phase(Fs)
    shiftedPhases = phases.copy()
    origin = (len(phases) - 1) / 2
    height = len(phases)
    width = len(phases[0])

    for row in range (height):
        for col in range (width):
            shiftedPhases[row][col] = np.exp(-2j * np.pi * ((shiftX * (col - origin) + shftY * (row - origin)) / len(phases)))

    shiftedImg = shiftedPhases * fftImg 
    shiftedImg = np.fft.ifftshift(shiftedImg)
    shiftedImg = np.fft.ifft2(shiftedImg)
    shiftedImg = normalize(shiftedImg.real)
    shiftedImg = make1Dlist(shiftedImg)
    shiftedImg = {
        'width': width,
        'height': height,
        'maxGreyLevel': 255,
        'content': shiftedImg
    }
    return shiftedImg

def rotateImg(img, degree):
    old2D = make2Dlist(img['content'], img['height'], img['width'])  # Convert 1D to 2D
    newImg = init2DWith0(img['height'], img['width'])  # Initialize new image with zeros
    center = (img['height'] - 1) / 2  # Center of rotation

    cos_theta = math.cos(degree * np.pi / 180)
    sin_theta = math.sin(degree * np.pi / 180)

    for newRow in range(len(newImg)):
        for newCol in range(len(newImg[newRow])):
            # Inverse Mapping: Find corresponding old pixel
            oldRow = int((newRow - center) * cos_theta + (newCol - center) * sin_theta + center)
            oldCol = int(-(newRow - center) * sin_theta + (newCol - center) * cos_theta + center)

            # Check if the old pixel is within bounds
            if 0 <= oldRow < len(old2D) and 0 <= oldCol < len(old2D[0]):
                newImg[newRow][newCol] = old2D[oldRow][oldCol]  # Assign pixel

    newImg = make1Dlist(newImg)  # Convert 2D back to 1D
    return {
        'width': img['width'],
        'height': img['height'],
        'maxGreyLevel': img['maxGreyLevel'],
        'content': newImg
    }

def downSampletoHalf(img):
    old2D = make2Dlist(img['content'], img['height'], img['width'])  # Convert 1D to 2D
    newImg = init2DWith0(int(img['height'] / 2), int(img['width'] / 2))  # Initialize new image with zeros

    for newRow in range(len(newImg)):
        for newCol in range(len(newImg[newRow])):
            oldRow = newRow * 2
            oldCol = newCol * 2
            newImg[newRow][newCol] = old2D[oldRow][oldCol]  # Assign pixel

    newImg = make1Dlist(newImg)  # Convert 2D back to 1D
    return {
        'width': int(img['width'] / 2),
        'height': int(img['height'] / 2),
        'maxGreyLevel': img['maxGreyLevel'],
        'content': newImg
    }

def meanTimeDomainConvolution(img, kernel, kernelOrigin):
    img2D = make2Dlist(img['content'], img['height'], img['width'])
    result = init2DWith0(img['height']-4, img['width']-4)
    sumWeight = sum(sum(row) for row in kernel)
    
    for y in range(img['height']):
        for x in range(img['width']):
            sumKernel = 0
            checkY = y - kernelOrigin
            if checkY < 0 or checkY >= img['height'] - 4:
                break

            checkX = x - kernelOrigin
            if checkX < 0 or checkX >= img['width'] - 4:
                continue

            for ky in range(len(kernel)):
                for kx in range(len(kernel[ky])):
                    imgY = y + ky - kernelOrigin
                    imgX = x + kx - kernelOrigin
                    if imgY >= 0 and imgY < img['height'] and imgX >= 0 and imgX < img['width']:
                        sumKernel += img2D[imgY][imgX] * kernel[ky][kx]

            result[checkY][checkX] = round(sumKernel / sumWeight)
    return {
        'height': img['height']-4,
        'width': img['width']-4,
        'maxGreyLevel': img['maxGreyLevel'],
        'content': make1Dlist(result)
    }

def meanFreqDomainConvolution(img, kernel):
    paddedKernel = init2DWith0(img['height'], img['width'])
    for row in range(len(kernel)):
        for col in range(len(kernel[row])):
            paddedKernel[row][col] = kernel[row][col]

    fftKernel = np.fft.fft2(paddedKernel)
    fftKernel = np.fft.fftshift(fftKernel)

    fftImg = np.fft.fft2(make2Dlist(img['content'], img['height'], img['width']))
    fftImg = np.fft.fftshift(fftImg)

    convolution = fftImg * fftKernel
    convolution = np.fft.ifftshift(convolution)
    convolution = np.fft.ifft2(convolution)
    convolution = normalize(convolution.real)
    convolution = make1Dlist(convolution)

    convoluted = {
        'width': img['width'],
        'height': img['height'],
        'maxGreyLevel': img['maxGreyLevel'],
        'content': convolution
    }
    return convoluted

# main part
crossImg = pgmRW.prepareInputData("./inputPictures/Cross.pgm")
newSize = nextPowOf2(max(crossImg['width'], crossImg['height']))
paddedCross = centerPadWith0(crossImg, newSize, newSize)
pgmRW.writePicture("./outputPictures/centerPaddedCross.pgm", paddedCross)

pixels = make2Dlist(paddedCross['content'], paddedCross['height'], paddedCross['width']) 
fftImg = np.fft.fft2(pixels)
fftImg = np.fft.fftshift(fftImg)

# 1.1 part
# amp, normAmp = amplitude(fftImg)
# amp1D = make1Dlist(normAmp)

# ampImg = {
#     'width': paddedCross['width'],
#     'height': paddedCross['height'],
#     'maxGreyLevel': paddedCross['maxGreyLevel'],
#     'content': amp1D,
# }
# pgmRW.writePicture("./outputPictures/ampCross.pgm", ampImg)

# phases, normPhase = phase(fftImg)
# normPhaseImg = make1Dlist(normPhase)
# phaseImg = {
#     'width': paddedCross['width'],
#     'height': paddedCross['height'],
#     'maxGreyLevel': paddedCross['maxGreyLevel'],
#     'content': normPhaseImg
# }
# pgmRW.writePicture("./outputPictures/phaseCross2.pgm", phaseImg)

# del amp, normAmp, amp1D, ampImg
# del phases, normPhase, normPhaseImg, phaseImg


# # part 1.2
# shiftedImg = phasesShift(fftImg, 20, 30)
# pgmRW.writePicture("./outputPictures/shiftedCross.pgm", shiftedImg)
# del shiftedImg


# # part 1.3 rotate 30 degree
# rotatedImg = rotateImg(paddedCross, 30)
# pgmRW.writePicture("./outputPictures/rotatedCross.pgm", rotatedImg)
# paddedRotated = centerPadWith0(rotatedImg, newSize, newSize)
# pixelsRotated = make2Dlist(paddedRotated['content'], paddedRotated['height'], paddedRotated['width'])

# fftRotatedImg = np.fft.fft2(pixelsRotated)
# fftRotatedImg = np.fft.fftshift(fftRotatedImg)
# ampRotated, normAmpRotated = amplitude(fftRotatedImg)
# amp1DRotated = make1Dlist(normAmpRotated)
# ampImgRotated = {
#     'width': paddedRotated['width'],
#     'height': paddedRotated['height'],
#     'maxGreyLevel': paddedRotated['maxGreyLevel'],
#     'content': amp1DRotated,
# }
# pgmRW.writePicture("./outputPictures/ampRotatedCross.pgm", ampImgRotated)

# phasesRotated, normPhaseRotated = phase(fftRotatedImg)
# normPhaseRotated = make1Dlist(normPhaseRotated)
# phaseImgRotated = {
#     'width': paddedRotated['width'],
#     'height': paddedRotated['height'],
#     'maxGreyLevel': paddedRotated['maxGreyLevel'],
#     'content': normPhaseRotated
# }
# pgmRW.writePicture("./outputPictures/phaseRotatedCross.pgm", phaseImgRotated)

# # part 1.4 down sample to 100*100 and FFT
# downed = downSampletoHalf(crossImg)
# newDownedSize = nextPowOf2(max(downed['width'], downed['height']))
# paddedDowned = centerPadWith0(downed, newDownedSize, newDownedSize)
# pgmRW.writePicture("./outputPictures/paddedDownSampledCross.pgm", paddedDowned)
# pgmRW.writePicture("./outputPictures/downSampledCross.pgm", downed)
# downedPadded = make2Dlist(paddedDowned['content'], paddedDowned['height'], paddedDowned['width'])
# fftDowned = np.fft.fft2(downedPadded)
# fftDowned = np.fft.fftshift(fftDowned)
# ampDowned, normAmpDowned = amplitude(fftDowned)
# amp1DDowned = make1Dlist(normAmpDowned)
# ampImgDowned = {
#     'width': paddedDowned['width'],
#     'height': paddedDowned['height'],
#     'maxGreyLevel': paddedDowned['maxGreyLevel'],
#     'content': amp1DDowned,
# }
# pgmRW.writePicture("./outputPictures/ampDownedCross.pgm", ampImgDowned)

# phases, normPhaseImg = phase(fftDowned)
# normPhaseImg = make1Dlist(normPhaseImg)
# phaseImg = {
#     'width': paddedDowned['width'],
#     'height': paddedDowned['height'],
#     'maxGreyLevel': paddedDowned['maxGreyLevel'],
#     'content': normPhaseImg
# }
# pgmRW.writePicture("./outputPictures/phaseDownedCross.pgm", phaseImg)

# # part 1.5
# onlyAmpImg, onlyPhaseImg = reconstructFFTwith1Infomation(fftImg)
# pgmRW.writePicture("./outputPictures/onlyAmpCross.pgm", onlyAmpImg)
# pgmRW.writePicture("./outputPictures/onlyPhaseCross.pgm", onlyPhaseImg)
# # # part 1.6
# opera = pgmRW.prepareInputData("./inputPictures/OperaHousePGM_256_256.pgm")
# fftOpera = np.fft.fft2(make2Dlist(opera['content'], opera['height'], opera['width']))
# fftOpera = np.fft.fftshift(fftOpera)
# onlyAmpOpera, onlyPhaseOpera = reconstructFFTwith1Infomation(fftOpera)
# pgmRW.writePicture("./outputPictures/onlyAmpOpera.pgm", onlyAmpOpera)
# pgmRW.writePicture("./outputPictures/onlyPhaseOpera.pgm", onlyPhaseOpera)

# # part 1.7 convolution "Chess.pgm"
# # a) blur the image with convolution kernel
# # blur with mean filter 5*5
# kernel = [[1] * 5 for _ in range (5)] # create kernel 5*5 with all 1
# kernelOrigin = (len(kernel) - 1) / 2
# chess = pgmRW.prepareInputData("./inputPictures/Chess.pgm")
# paddedChess = centerPadWith0(chess, chess['width'] + 4, chess['height'] + 4)
# pgmRW.writePicture("./outputPictures/paddedChess.pgm", paddedChess)
# pgmRW.writePicture("./outputPictures/timeConvolutedChess.pgm", meanTimeDomainConvolution(paddedChess, kernel, int(kernelOrigin)))

# # b) filter in freq domain with same kernel
# convolutedFreq = meanFreqDomainConvolution(chess, kernel)
# pgmRW.writePicture("./outputPictures/freqConvolutedChess.pgm", convolutedFreq)

def idealLPF(fftImg, d0, name):

    center = (len(fftImg) - 1) / 2
    height = len(fftImg)
    width = len(fftImg[0])

    for row in range(height):
        for col in range(width):
            d = math.sqrt(pow(row - center, 2) + pow(col - center, 2))
            if d > d0:
                fftImg[row][col] = 0

    idealFiltered = np.fft.ifftshift(fftImg)
    idealFiltered = np.fft.ifft2(idealFiltered)
    idealFiltered = normalize(idealFiltered.real)
    idealFiltered = make1Dlist(idealFiltered)
    filteredImg = {
        'width': width,
        'height': height,
        'maxGreyLevel': 255,
        'content': idealFiltered
    }
    pgmRW.writePicture("./outputPictures/idealLPF/" + name + ".pgm", filteredImg)

    return filteredImg

def guassianLPF(fftImg, d0, name):
    center = (len(fftImg) - 1) / 2
    height = len(fftImg)
    width = len(fftImg[0])

    for row in range(height):
        for col in range(width):
            d = math.sqrt(pow(row - center, 2) + pow(col - center, 2))
            fftImg[row][col] = fftImg[row][col] * np.exp(-pow(d, 2) / (2 * pow(d0, 2)))

    guassianFiltered = np.fft.ifftshift(fftImg)
    guassianFiltered = np.fft.ifft2(guassianFiltered)
    guassianFiltered = normalize(guassianFiltered.real)
    guassianFiltered = make1Dlist(guassianFiltered)
    filteredImg = {
        'width': width,
        'height': height,
        'maxGreyLevel': 255,
        'content': guassianFiltered
    }
    pgmRW.writePicture("./outputPictures/guassian/" + name + ".pgm", filteredImg)

    return filteredImg

def medianFilter(img, kernelSize):
    pixelPulls = []
    kernelRow, kernelCol = kernelSize

    paddedImg = centerPadWith0(img, img['width'] + kernelCol - 1, img['height'] + kernelRow - 1)
    content = make2Dlist(paddedImg['content'], paddedImg['height'], paddedImg['width'])
    result = init2DWith0(img['height'], img['width'])

    for row in range(paddedImg['height'] - kernelRow):
        for col in range(paddedImg['width'] - kernelCol):

            for ky in range(kernelRow):
                for kx in range(kernelCol):
                    pixelPulls.append(content[row + ky][col + kx])

            pixelPulls.sort()
            result[row][col] = pixelPulls[len(pixelPulls) // 2] # floor to int
            pixelPulls.clear()
    
    return {
        'width': img['width'],
        'height': img['height'],
        'maxGreyLevel': img['maxGreyLevel'],
        'content': make1Dlist(result)
    }

def trimList(list, alpha):
    if alpha < len(list) // 2:
        return list[alpha:-alpha] 
    else:
        return []  # alpha is too large

def avg(list):
    return sum(list) / len(list)

def trimMeanFilter(img, kernelSize, alpha):
    pixelPulls = []
    kernelRow, kernelCol = kernelSize

    paddedImg = centerPadWith0(img, img['width'] + kernelCol - 1, img['height'] + kernelRow - 1)
    content = make2Dlist(paddedImg['content'], paddedImg['height'], paddedImg['width'])
    result = init2DWith0(img['height'], img['width'])
    
    for row in range(paddedImg['height'] - kernelRow):
        for col in range(paddedImg['width'] - kernelCol):

            for ky in range(kernelRow):
                for kx in range(kernelCol):
                    pixelPulls.append(content[row + ky][col + kx])

            pixelPulls.sort()
            pixelPulls = trimList(pixelPulls, alpha)

            if len(pixelPulls) == 0:
                result[row][col] = 0
            else:
                result[row][col] = int(avg(pixelPulls))

            pixelPulls.clear()

    return {
        'width': img['width'],
        'height': img['height'],
        'maxGreyLevel': img['maxGreyLevel'],
        'content': make1Dlist(result)
    }

def alphaTrimFilter(img, kernelSize, alpha, name):
    trimmed = trimMeanFilter(img, kernelSize, alpha)
    pgmRW.writePicture("./outputPictures/alphaTrim/" + name + ".pgm", trimmed)
    return trimmed


# 2.1 ideal filter and non-ideal
fftCross = fftImg
idealLPF(fftCross.copy(), 5, "crossLPFD5")
idealLPF(fftCross.copy(), 10, "crossLPFD10")
idealLPF(fftCross.copy(), 25, "crossLPFD25")
idealLPF(fftCross.copy(), 50, "crossLPFD50")

guassianLPF(fftCross.copy(), 5, "crossLPFD5")
guassianLPF(fftCross.copy(), 10, "crossLPFD10")
guassianLPF(fftCross.copy(), 25, "crossLPFD25")
guassianLPF(fftCross.copy(), 50, "crossLPFD50")

def RMS(noiseFree, clean):
    # noise - clean for compare clean picture error with noise picture
    noiseFreeContent = noiseFree['content']
    cleanContent = clean['content']
    n = len(noiseFreeContent)
    sum = 0

    for i in range(n):
        sum += (noiseFreeContent[i] - cleanContent[i]) ** 2

    return math.sqrt(sum) / n

# 2.2 
file = open ("RMS_of_each_filter_Chess.txt", "w")
# Chess picture
noiseChess = pgmRW.prepareInputData("./inputPictures/Chess_noise.pgm")
noiseFreeChess = pgmRW.prepareInputData("./inputPictures/Chess.pgm")

    # median part
medianChess3 = medianFilter(noiseChess.copy(), (3, 3))
medianChess5 = medianFilter(noiseChess.copy(), (5, 5))
medianChess9 = medianFilter(noiseChess.copy(), (9, 9))

pgmRW.writePicture("./outputPictures/median/chess3.pgm", medianChess3)
pgmRW.writePicture("./outputPictures/median/chess5.pgm", medianChess5)
pgmRW.writePicture("./outputPictures/median/chess9.pgm", medianChess9)

file.write("RMS of 3*3 median filter Chess.pgm : " + str(RMS(noiseFreeChess.copy(), medianChess3)))
file.write("\nRMS of 5*5 median filter Chess.pgm : " + str(RMS(noiseFreeChess.copy(), medianChess5)))
file.write("\nRMS of 9*9 median filter Chess.pgm : " + str(RMS(noiseFreeChess.copy(), medianChess9)))

del medianChess3
del medianChess5
del medianChess9

    # guassian part
noiseChess2D = make2Dlist(noiseChess['content'], noiseChess['height'], noiseChess['width'])
fftNoiseChess = np.fft.fft2(noiseChess2D)
fftNoiseChess = np.fft.fftshift(fftNoiseChess)

cutoff20 = guassianLPF(fftNoiseChess.copy(), 20, "chessLPFD20")
cutoff50 = guassianLPF(fftNoiseChess.copy(), 50, "chessLPFD50")
cutoff100 = guassianLPF(fftNoiseChess.copy(), 100, "chessLPFD100")

file.write("\n\nRMS of guassian filter D0 = 20  Chess.pgm : " + str(RMS(noiseFreeChess.copy(), cutoff20)))
file.write("\nRMS of guassian filter D0 = 50  Chess.pgm : " + str(RMS(noiseFreeChess.copy(), cutoff50)))
file.write("\nRMS of guassian filter D0 = 100 Chess.pgm : " + str(RMS(noiseFreeChess.copy(), cutoff100)))

del fftNoiseChess 
del cutoff20
del cutoff50
del cutoff100

    # alpha trim part
        # try with same kernel, difference alpha
alpha553 = alphaTrimFilter(noiseChess.copy(), (5, 5), 3, "chess553")
alpha555 = alphaTrimFilter(noiseChess.copy(), (5, 5), 5, "chess555")
alpha557 = alphaTrimFilter(noiseChess.copy(), (5, 5), 7, "chess557")
alpha5511 = alphaTrimFilter(noiseChess.copy(), (5, 5), 11, "chess5511")

file.write("\n\nRMS of alpha trim kernel 5*5 alpha = 3 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha553)))
file.write("\nRMS of alpha trim kernel 5*5 alpha = 5 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha555)))
file.write("\nRMS of alpha trim kernel 5*5 alpha = 7 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha557)))
file.write("\nRMS of alpha trim kernel 5*5 alpha = 11 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha5511)))

del alpha553
del alpha555
del alpha557
del alpha5511

        # try with difference kernel size, same alpha
alpha559 = alphaTrimFilter(noiseChess.copy(), (5, 5), 9, "chess559")
alpha779 = alphaTrimFilter(noiseChess.copy(), (7, 7), 9, "chess779")
alpha999 = alphaTrimFilter(noiseChess.copy(), (9, 9), 9, "chess999")
alpha11119 = alphaTrimFilter(noiseChess.copy(), (11, 11), 9, "chess11119")

file.write("\n\nRMS of alpha trim kernel 5*5 alpha = 9 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha559)))
file.write("\nRMS of alpha trim kernel 7*7 alpha = 9 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha779)))
file.write("\nRMS of alpha trim kernel 9*9 alpha = 9 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha999)))
file.write("\nRMS of alpha trim kernel 11*11 alpha = 9 Chess.pgm : " 
           + str(RMS(noiseFreeChess.copy(), alpha11119)))

del alpha559
del alpha779
del alpha999
del alpha11119

del noiseChess
del noiseFreeChess
file.close()


# Opera picture
file = open ("RMS_of_each_filter_Opera.txt", "w")
noiseOpera = pgmRW.prepareInputData("./inputPictures/OperaHousePGM_256_256_noise.pgm")
noiseFreeOpera = pgmRW.prepareInputData("./inputPictures/OperaHousePGM_256_256.pgm")

    # median part
medianOpera3 = medianFilter(noiseOpera.copy(), (3, 3))
medianOpera5 = medianFilter(noiseOpera.copy(), (5, 5))
medianOpera9 = medianFilter(noiseOpera.copy(), (9, 9))

pgmRW.writePicture("./outputPictures/median/Opera3.pgm", medianOpera3)
pgmRW.writePicture("./outputPictures/median/Opera5.pgm", medianOpera5)
pgmRW.writePicture("./outputPictures/median/Opera9.pgm", medianOpera9)

file.write("RMS of 3*3 median filter Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), medianOpera3)))
file.write("\nRMS of 5*5 median filter Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), medianOpera5)))
file.write("\nRMS of 9*9 median filter Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), medianOpera9)))

del medianOpera3
del medianOpera5
del medianOpera9

    # guassian part
noiseOpera2D = make2Dlist(noiseOpera['content'], noiseOpera['height'], noiseOpera['width'])
fftNoiseOpera = np.fft.fft2(noiseOpera2D)
fftNoiseOpera = np.fft.fftshift(fftNoiseOpera)

cutoff20Opera = guassianLPF(fftNoiseOpera.copy(), 20, "OperaLPFD20")
cutoff50Opera = guassianLPF(fftNoiseOpera.copy(), 50, "OperaLPFD50")
cutoff100Opera = guassianLPF(fftNoiseOpera.copy(), 100, "OperaLPFD100")

file.write("\n\nRMS of guassian filter D0 = 20  Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), cutoff20Opera)))
file.write("\nRMS of guassian filter D0 = 50  Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), cutoff50Opera)))
file.write("\nRMS of guassian filter D0 = 100 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), cutoff100Opera)))

del fftNoiseOpera 
del cutoff20Opera
del cutoff50Opera
del cutoff100Opera

    # alpha trim part
        # try with same kernel, difference alpha
alpha553Opera = alphaTrimFilter(noiseOpera.copy(), (5, 5), 3, "Opera553")
alpha555Opera = alphaTrimFilter(noiseOpera.copy(), (5, 5), 5, "Opera555")
alpha557Opera = alphaTrimFilter(noiseOpera.copy(), (5, 5), 7, "Opera557")
alpha5511Opera = alphaTrimFilter(noiseOpera.copy(), (5, 5), 11, "Opera5511")

file.write("\n\nRMS of alpha trim kernel 5*5 alpha = 3 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha553Opera)))
file.write("\nRMS of alpha trim kernel 5*5 alpha = 5 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha555Opera)))
file.write("\nRMS of alpha trim kernel 5*5 alpha = 7 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha557Opera)))
file.write("\nRMS of alpha trim kernel 5*5 alpha = 11 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha5511Opera)))

del alpha553Opera
del alpha555Opera
del alpha557Opera
del alpha5511Opera

        # try with difference kernel size, same alpha
alpha559Opera = alphaTrimFilter(noiseOpera.copy(), (5, 5), 9, "Opera559")
alpha779Opera = alphaTrimFilter(noiseOpera.copy(), (7, 7), 9, "Opera779")
alpha999Opera = alphaTrimFilter(noiseOpera.copy(), (9, 9), 9, "Opera999")
alpha11119Opera = alphaTrimFilter(noiseOpera.copy(), (11, 11), 9, "Opera11119")

file.write("\n\nRMS of alpha trim kernel 5*5 alpha = 9 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha559Opera)))
file.write("\nRMS of alpha trim kernel 7*7 alpha = 9 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha779Opera)))
file.write("\nRMS of alpha trim kernel 9*9 alpha = 9 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha999Opera)))
file.write("\nRMS of alpha trim kernel 11*11 alpha = 9 Opera.pgm : " + str(RMS(noiseFreeOpera.copy(), alpha11119Opera)))

del alpha559Opera
del alpha779Opera
del alpha999Opera
del alpha11119Opera

del noiseOpera
del noiseFreeOpera
file.close()