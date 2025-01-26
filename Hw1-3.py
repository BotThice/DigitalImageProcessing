import histrogram as hs
import pgmRW 

def g2_r_b (r, g, b):
    outPic = {'width': r['width'], 'height': r['height'], 'maxGreyLevel': r['maxGreyLevel'], 'content': []}
    totalPixel = len(r['content'])
    
    for i in range (totalPixel):
            level = 2*g['content'][i] - r['content'][i] - b['content'][i]
            
            outPic['content'].append(forceInRange(level))
    
    return outPic

def r_b (r, b):
    outPic = {'width': r['width'], 'height': r['height'], 'maxGreyLevel': r['maxGreyLevel'], 'content': []}
    totalPixel = len(r['content'])
    
    for i in range (totalPixel):
            level = r['content'][i] - b['content'][i]
            
            outPic['content'].append(forceInRange(level))
    
    return outPic

def r_g_b (r, g, b):
    outPic = {'width': r['width'], 'height': r['height'], 'maxGreyLevel': r['maxGreyLevel'], 'content': []}
    totalPixel = len(r['content'])
    
    for i in range (totalPixel):
            greylevel = (r['content'][i] + b['content'][i] + g['content'][i]) / 3.0
            greylevel = round(greylevel)
            outPic['content'].append(forceInRange(greylevel))
    
    return outPic

def g_b (g, b):
    outPic = {'width': g['width'], 'height': g['height'], 'maxGreyLevel': g['maxGreyLevel'], 'content': []}
    totalPixel = len(g['content'])
    
    for i in range (totalPixel):
            greylevel = g['content'][i] + b['content'][i]
            greylevel = round(greylevel)
            outPic['content'].append(forceInRange(greylevel))
    
    return outPic


def forceInRange(level):
    if level < 0 :
        return 0
    elif level > 255 :
        return 255
    else :
        return level
    
# Main part
red = pgmRW.prepareInputData("./inputPictures/SanFranPeak_red.pgm")
blue = pgmRW.prepareInputData("./inputPictures/SanFranPeak_blue.pgm")
green = pgmRW.prepareInputData("./inputPictures/SanFranPeak_green.pgm")


outPic = g2_r_b(red, green, blue)
pgmRW.writePicture("./outputPictures/SanFranPeak_g2-r-b.pgm", outPic)
outPic = r_b(red, blue)
pgmRW.writePicture("./outputPictures/SanFranPeak_r-b.pgm", outPic)
outPic = r_g_b(red, green, blue)
pgmRW.writePicture("./outputPictures/SanFranPeak_r-g-b.pgm", outPic)
outPic = g_b(green, blue)
pgmRW.writePicture("./outputPictures/SanFranPeak_g_b.pgm", outPic)


