def histrogram(maxGreyLevel, content) :
    histrogram = {}
    
    for greyLevel in range (maxGreyLevel + 1):
        histrogram[greyLevel] = 0
        
    for pixel in content:
        histrogram[pixel] += 1
           
    return histrogram

def histrogramEqualization(picture) : 
    width = picture['width']
    height = picture['height']
    maxGreyLevel = picture['maxGreyLevel']
    content = picture['content']
    picHistrogram = histrogram(maxGreyLevel, content)
    mappedGreyLevel = mapGreyLevel(maxGreyLevel, CDF(probMass (width, height, maxGreyLevel, picHistrogram)))
    eqHistrogram = initialDic(maxGreyLevel)
    
    for level in range (maxGreyLevel + 1) :
        outLevel = mappedGreyLevel[level]
        eqHistrogram[outLevel] += picHistrogram[level]
    
    return eqHistrogram, mappedGreyLevel

def probMass(picWidth, picHeight, maxGreyLevel, histrogram) :
    picSize = picHeight * picWidth
    probOfAllLevel = {}
    
    for level in range (maxGreyLevel + 1) :
        probOfAllLevel[level] = probOfLevel(histrogram[level], picSize)
        
    return probOfAllLevel

def probOfLevel(freq, picSize) :
    return freq/picSize

def CDF(PMF) :
    greyLevel = list(PMF.keys())
    probOfEach = list(PMF.values())
    result = initialDic(max(greyLevel))
    
    for level in greyLevel :
        for previousLevel in range (level + 1) :
            result[level] += probOfEach[previousLevel]
    
    return result

def mapGreyLevel(maxGreyLevel, CDF) :
    result = {}
    
    for level in range (maxGreyLevel + 1) :
        result[level] = round(maxGreyLevel * CDF[level])
    
    return result

def initialDic(maxKey) :
    dic = {}
    for key in range (maxKey + 1) :
        dic[key] = 0
    return dic