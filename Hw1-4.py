import pgmRW
import matrix as mtx
import numpy as np
from scipy.optimize import root

def addBorder(img):
    width = img['width'] + 2
    height = img['height'] + 2
    old_width = img['width']
    pic = img['content']
    bordered = []

    for y in range(height):
        for x in range(width):
            if x == 0 or x == (width - 1) or y == 0 or y == (height - 1):
                bordered.append(0)
            else:
                bordered.append(pic[(y - 1) * old_width + (x - 1)])

    newImg = {'width': width, 'height': height, 'maxGreyLevel': img['maxGreyLevel'], 'content': bordered}
    return newImg

def removeBorder(img):
    width = img['width']
    height = img['height']
    pic = img['content']
    unbordered = []
    for y in range (height):
        for x in range(width):
            if (y == 0 or y == height - 1): continue
            elif (x == 0 or x == width - 1): continue
            else:
                unbordered.append(pic[y * width + x])
    newImg = {'width': width - 2, 'height': height - 2, 'maxGreyLevel': img['maxGreyLevel'], 'content': unbordered}
    
    return newImg

def findConstant(gSquare, tSquare):
    tx1 = tSquare[0][0]
    ty1 = tSquare[0][1]
    tx2 = tSquare[1][0]
    ty2 = tSquare[1][1]
    tx3 = tSquare[2][0]
    ty3 = tSquare[2][1]
    tx4 = tSquare[3][0]
    ty4 = tSquare[3][1]
    gx1 = gSquare[0][0]
    gy1 = gSquare[0][1]
    gx2 = gSquare[1][0]
    gy2 = gSquare[1][1]
    gx3 = gSquare[2][0]
    gy3 = gSquare[2][1]
    gx4 = gSquare[3][0]
    gy4 = gSquare[3][1]
    matrixA = [
        [gx1, gy1, gx1*gy1, 1], 
        [gx2, gy2, gx2*gy2, 1], 
        [gx3, gy3, gx3*gy3, 1], 
        [gx4, gy4, gx4*gy4, 1] 
    ]
    matrixTx = [tx1, tx2, tx3, tx4]
    matrixTy = [ty1, ty2, ty3, ty4]
    aInverse = mtx.inverse(matrixA)
    yConst = mtx.multiply_matrix_vector(aInverse, matrixTy)
    xConst = mtx.multiply_matrix_vector(aInverse, matrixTx)
    return [xConst, yConst]

    result = 0
    tx1 = tSquare[0][0]
    gx1 = gSquare[0][0]
    gy1 = gSquare[0][1]
    
    if(mode == 'y'):
        tx1 = tSquare[0][1]
    
    result = tx1 - a*gx1 - b*gy1 - c*gx1*gy1
    
    return result

def findAllConst(gSquares, tSquares):
    xConsts = []
    yConsts = []
    for i in range(len(gSquares)):
            xConst, yConst = findConstant(gSquares[i], tSquares[i])      
            xConsts.append(xConst)
            yConsts.append(yConst)
    return [xConsts, yConsts]

def mapNewPos(xConst, yConst, pairXY):
    def equations(variables, xConst, yConst, pairXY):
        a, b, c, d = xConst
        e, f, g, h = yConst
        m, n = pairXY
        x, y = variables
        eq1 = a * x + b * y + c * x * y + d - m
        eq2 = e * x + f * y + g * x * y + h - n
        return np.array([eq1, eq2])
    
    initial_guess = [pairXY[0], pairXY[1]]
    
    solution = root(equations, initial_guess, args=(xConst, yConst, pairXY))
    x, y = solution.x
    if solution.success and solution.x[0] >= 0 and solution.x[1] >= 0:
        return (x, y)
    else:
        # Log to file in case of failure
        with open("root_log.txt", "w") as log_file:
            log_file.write(f"No valid solution for inputs: newX={x}, NewY={y}, pairXY={pairXY}\n")
        return (x, y)

def custom_round(num):
    integer_part = int(num)
    decimal_part = num - integer_part
    if decimal_part > 0.1:
        return integer_part + 1  # Round up
    else:
        return integer_part  # Round down

def is_point_inside_square(x, y, vertices):
    # Calculate vectors for edges
    def cross_product(v1, v2):
        return v1[0] * v2[1] - v1[1] * v2[0]

    # Check if point (x, y) is on the same side of each edge
    for i in range(4):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 4]
        edge_vector = (p2[0] - p1[0], p2[1] - p1[1])
        point_vector = (x - p1[0], y - p1[1])
        if cross_product(edge_vector, point_vector) < 0:
            return False
    return True

def findPointInSquare(tSquare):
    
    sortByX = sorted(tSquare.copy(), key=lambda x: (x[0], x[1]))
    sortByY = sorted(tSquare.copy(), key=lambda x: (x[1], x[0]))
    leftCorner = sortByX[:2]
    rightCorner = sortByX[2:]

    if (leftCorner[1] in sortByY[2:] and leftCorner[1][0] > sortByY[2:][0][0]): #special case
        temp = leftCorner[1]
        leftCorner[1] = rightCorner[0]
        rightCorner[0] = temp
    
    tl, bl = y1 = sorted(leftCorner, key=lambda x: (x[1], x[0]))
    tr, br = y2 = sorted(rightCorner, key=lambda x: (x[1], x[0]))
    
    square_vertices = [
        tl,  # top left
        tr,  # top right
        br,  # bottom right
        bl,  # bottom left
    ]
    
    min_x = min(tl[0], bl[0])
    max_x = max(tr[0], br[0])
    min_y = min(tl[1], tr[1])
    max_y = max(bl[1], br[1])
    
    points_inside = [
        (x, y)
        for x in range(min_x, max_x + 1)
        for y in range(min_y, max_y + 1)
        if is_point_inside_square(x, y, square_vertices)
    ]
    
    return points_inside

def backWardMapping(twisted, listGridCrossPoint, listTwistCrossPoint):
    gS = []
    tS = []
    for horizonLine in range(len(listGridCrossPoint) - 1):
        for point in range(len(listGridCrossPoint[horizonLine]) - 1):
            gSquare = [listGridCrossPoint[horizonLine][point], 
                        listGridCrossPoint[horizonLine][point + 1], 
                        listGridCrossPoint[horizonLine + 1][point], 
                        listGridCrossPoint[horizonLine + 1][point + 1]]
            gS.append(gSquare)
            tSquare = [listTwistCrossPoint[horizonLine][point], 
                        listTwistCrossPoint[horizonLine][point + 1], 
                        listTwistCrossPoint[horizonLine + 1][point], 
                        listTwistCrossPoint[horizonLine + 1][point + 1]]
            tS.append(tSquare)
            
    # find all constant of all twisted grid
    xConsts, yConsts = findAllConst(gS, tS)
    gS.clear()
    pixels = []
    newImg = twisted
    newCon = newImg['content'].copy()
    width = twisted['width']
    for i in range(len(newCon)):
        newCon[i] = 0
    
    for square in range(len(tS)):
        pixels = findPointInSquare(tS[square])

        for pixel in pixels :
            newX, newY = newPair = mapNewPos(xConsts[square], yConsts[square], pixel)
            newX = custom_round(newX)
            newY = custom_round(newY)
            if (square == 7):
                with open("square7.txt", "a") as log_file:
                    log_file.write(f"newX, NewY={newPair}, pair={pixel}\n")
            newCon[newY*width + newX] = twisted['content'][pixel[1] * width + pixel[0]]

    newImg['content'] = newCon
    return newImg 
    
# Main
grid = pgmRW.prepareInputData("./inputPictures/grid.pgm")
twistedGrid = pgmRW.prepareInputData("./inputPictures/NewDistGrid.pgm")
operaHouse = pgmRW.prepareInputData("./inputPictures/DistOperaHouse.pgm")

borderedGrid = addBorder(grid)
operaHouse = addBorder(operaHouse)
pgmRW.writePicture("./outputPictures/borderedGrid.pgm", borderedGrid)

ftg = twistedGrid
content = ftg['content']
content.pop(0)
content.append(255)
ftg['content'] = content
ftg = addBorder(ftg)
content = ftg['content']
ftg['content'] = content
pgmRW.writePicture("ftg.pgm", ftg)


twistCrossPoint1 = [(0,0),(16,0),(32,0),(48,0),(64,0),(80,0),(96,0),(112,0),(128,0),(144,0),(160,0),(176,0),(192,0),(208,0),(224,0),(240,0),(257,0)]
twistCrossPoint2 = [(0,16),(16,16),(32,16),(48,16),(64,16),(80,16),(98,16),(116,16),(132,16),(147,16),(162,17),(176,16),(192,16),(208,16),(224,16),(240,16),(257,16)]
twistCrossPoint3 = [(0,32),(16,32),(32,32),(48,32),(66,31),(86,29),(106,29),(125,31),(142,33),(156,35),(170,36),(182,35),(194,33),(208,32),(224,32),(240,32),(257,32)]
twistCrossPoint4 = [(0,48),(16,48),(32,48),(50,45),(72,42),(95,41),(117,42),(138,47),(154,52),(168,57),(179,58),(189,57),(199,54),(211,50),(224,48),(240,48),(257,48)]
twistCrossPoint5 = [(0,64),(16,64),(33,62),(54,57),(79,52),(105,52),(128,56),(149,66),(165,75),(176,83),(186,85),(195,83),(204,79),(214,72),(225,66),(240,64),(257,64)]
twistCrossPoint6 = [(0,80),(16,79),(35,74),(58,66),(84,61),(112,63),(137,71),(156,86),(168,100),(176,109),(185,114),(194,111),(205,104),(215,95),(227,86),(240,81),(257,80)]
twistCrossPoint7 = [(0,96),(17,94),(36,86),(59,76),(86,70),(114,72),(139,85),(154,104),(160,122),(164,134),(173,138),(185,137),(200,129),(214,117),(227,106),(241,98),(257,96)]
twistCrossPoint8 = [(0,112),(17,108),(35,99),(57,88),(83,80),(112,81),(134,93),(146,114),(144,134),(144,146),(153,154),(171,156),(191,150),(209,139),(225,126),(240,116),(257,112)]
twistCrossPoint9 = [(0,128),(16,123),(33,113),(53,101),(76,91),(102,89),(123,97),(133,114),(128,130),(124,145),(135,160),(155,168),(180,166),(204,155),(223,142),(240,133),(257,128)]
twistCrossPoint10 = [(0,144),(16,140),(31,129),(47,117),(66,107),(86,100),(105,102),(115,111),(113,124),(111,143),(122,164),(145,176),(173,177),(199,169),(221,157),(239,148),(257,144)]
twistCrossPoint11 = [(0,160),(16,157),(29,149),(43,138),(57,127),(71,119),(86,117),(94,122),(97,134),(102,152),(117,172),(142,185),(170,186),(197,180),(220,170),(239,162),(257,160)]
twistCrossPoint12 = [(0,176),(16,175),(29,169),(41,160),(52,150),(62,144),(72,141),(81,144),(89,154),(100,170),(119,185),(144,194),(171,195),(198,190),(221,182),(240,176),(257,176)]
twistCrossPoint13 = [(0,192),(16,192),(30,189),(42,183),(52,176),(61,171),(70,169),(79,172),(91,180),(105,190),(126,199),(151,205),(176,204),(201,200),(223,194),(240,192),(257,192)]
twistCrossPoint14 = [(0,208),(16,208),(32,208),(45,205),(56,201),(66,197),(76,196),(86,198),(100,203),(116,209),(137,214),(161,216),(183,214),(205,211),(224,208),(240,208),(257,208)]
twistCrossPoint15 = [(0,224),(16,224),(32,224),(48,224),(62,222),(74,220),(85,219),(98,220),(112,222),(129,225),(149,227),(169,227),(189,226),(208,224),(224,224),(240,224),(257,224)]
twistCrossPoint16 = [(0,240),(16,240),(32,240),(48,240),(64,240),(79,240),(93,239),(108,239),(123,240),(140,240),(157,241),(175,240),(192,240),(208,240),(224,240),(240,240),(257,240)]
twistCrossPoint17 = [(0,257),(16,257),(32,257),(48,257),(64,257),(80,257),(96,257),(112,257),(128,257),(144,257),(160,257),(176,257),(192,257),(208,257),(224,257),(240,257),(257,257)]

gridCrossPoint1 = [(0,0),(16,0),(32,0),(48,0),(64,0),(80,0),(96,0),(112,0),(128,0),(144,0),(160,0),(176,0),(192,0),(208,0),(224,0),(240,0),(257,0)]
gridCrossPoint2 = [(0,16),(16,16),(32,16),(48,16),(64,16),(80,16),(96,16),(112,16),(128,16),(144,16),(160,16),(176,16),(192,16),(208,16),(224,16),(240,16),(257,16)]
gridCrossPoint3 = [(0,32),(16,32),(32,32),(48,32),(64,32),(80,32),(96,32),(112,32),(128,32),(144,32),(160,32),(176,32),(192,32),(208,32),(224,32),(240,32),(257,32)]
gridCrossPoint4 = [(0,48),(16,48),(32,48),(48,48),(64,48),(80,48),(96,48),(112,48),(128,48),(144,48),(160,48),(176,48),(192,48),(208,48),(224,48),(240,48),(257,48)]
gridCrossPoint5 = [(0,64),(16,64),(32,64),(48,64),(64,64),(80,64),(96,64),(112,64),(128,64),(144,64),(160,64),(176,64),(192,64),(208,64),(224,64),(240,64),(257,64)]
gridCrossPoint6 = [(0,80),(16,80),(32,80),(48,80),(64,80),(80,80),(96,80),(112,80),(128,80),(144,80),(160,80),(176,80),(192,80),(208,80),(224,80),(240,80),(257,80)]
gridCrossPoint7 = [(0,96),(16,96),(32,96),(48,96),(64,96),(80,96),(96,96),(112,96),(128,96),(144,96),(160,96),(176,96),(192,96),(208,96),(224,96),(240,96),(257,96)]
gridCrossPoint8 = [(0,112),(16,112),(32,112),(48,112),(64,112),(80,112),(96,112),(112,112),(128,112),(144,112),(160,112),(176,112),(192,112),(208,112),(224,112),(240,112),(257,112)]
gridCrossPoint9 = [(0,128),(16,128),(32,128),(48,128),(64,128),(80,128),(96,128),(112,128),(128,128),(144,128),(160,128),(176,128),(192,128),(208,128),(224,128),(240,128),(257,128)]
gridCrossPoint10 = [(0,144),(16,144),(32,144),(48,144),(64,144),(80,144),(96,144),(112,144),(128,144),(144,144),(160,144),(176,144),(192,144),(208,144),(224,144),(240,144),(257,144)]
gridCrossPoint11 = [(0,160),(16,160),(32,160),(48,160),(64,160),(80,160),(96,160),(112,160),(128,160),(144,160),(160,160),(176,160),(192,160),(208,160),(224,160),(240,160),(257,160)]
gridCrossPoint12 = [(0,176),(16,176),(32,176),(48,176),(64,176),(80,176),(96,176),(112,176),(128,176),(144,176),(160,176),(176,176),(192,176),(208,176),(224,176),(240,176),(257,176)]
gridCrossPoint13 = [(0,192),(16,192),(32,192),(48,192),(64,192),(80,192),(96,192),(112,192),(128,192),(144,192),(160,192),(176,192),(192,192),(208,192),(224,192),(240,192),(257,192)]
gridCrossPoint14 = [(0,208),(16,208),(32,208),(48,208),(64,208),(80,208),(96,208),(112,208),(128,208),(144,208),(160,208),(176,208),(192,208),(208,208),(224,208),(240,208),(257,208)]
gridCrossPoint15 = [(0,224),(16,224),(32,224),(48,224),(64,224),(80,224),(96,224),(112,224),(128,224),(144,224),(160,224),(176,224),(192,224),(208,224),(224,224),(240,224),(257,224)]
gridCrossPoint16 = [(0,240),(16,240),(32,240),(48,240),(64,240),(80,240),(96,240),(112,240),(128,240),(144,240),(160,240),(176,240),(192,240),(208,240),(224,240),(240,240),(257,240)]
gridCrossPoint17 = [(0,257),(16,257),(32,257),(48,257),(64,257),(80,257),(96,257),(112,257),(128,257),(144,257),(160,257),(176,257),(192,257),(208,257),(224,257),(240,257),(257,257)]

listGridPoint = []
listGridPoint.append(gridCrossPoint1)
listGridPoint.append(gridCrossPoint2)
listGridPoint.append(gridCrossPoint3)
listGridPoint.append(gridCrossPoint4)
listGridPoint.append(gridCrossPoint5)
listGridPoint.append(gridCrossPoint6)
listGridPoint.append(gridCrossPoint7)
listGridPoint.append(gridCrossPoint8)
listGridPoint.append(gridCrossPoint9)
listGridPoint.append(gridCrossPoint10)
listGridPoint.append(gridCrossPoint11)
listGridPoint.append(gridCrossPoint12)
listGridPoint.append(gridCrossPoint13)
listGridPoint.append(gridCrossPoint14)
listGridPoint.append(gridCrossPoint15)
listGridPoint.append(gridCrossPoint16)
listGridPoint.append(gridCrossPoint17)


listTwistPoint = []
listTwistPoint.append(twistCrossPoint1)
listTwistPoint.append(twistCrossPoint2)
listTwistPoint.append(twistCrossPoint3)
listTwistPoint.append(twistCrossPoint4)
listTwistPoint.append(twistCrossPoint5)
listTwistPoint.append(twistCrossPoint6)
listTwistPoint.append(twistCrossPoint7)
listTwistPoint.append(twistCrossPoint8)
listTwistPoint.append(twistCrossPoint9)
listTwistPoint.append(twistCrossPoint10)
listTwistPoint.append(twistCrossPoint11)
listTwistPoint.append(twistCrossPoint12)
listTwistPoint.append(twistCrossPoint13)
listTwistPoint.append(twistCrossPoint14)
listTwistPoint.append(twistCrossPoint15)
listTwistPoint.append(twistCrossPoint16)
listTwistPoint.append(twistCrossPoint17)

revImg = backWardMapping(ftg, listGridPoint, listTwistPoint)
pgmRW.writePicture("./outputPictures/revGridcustom2.pgm", revImg)
# revImg = removeBorder(revImg)
# pgmRW.writePicture("./outputPictures/borderLessRevWithFsolve.pgm", revImg)
# revImg = backWardMapping(operaHouse, listGridPoint, listTwistPoint)
# pgmRW.writePicture("./outputPictures/OperaWithFsolve.pgm", revImg)
# revImg = removeBorder(revImg)
# pgmRW.writePicture("./outputPictures/borderLessOperaWithFsolve.pgm", revImg)
