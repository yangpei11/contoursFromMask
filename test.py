# *-* coding:utf-8 *-*
import numpy as np
from skimage import measure, draw
import skimage.morphology
import matplotlib.pyplot as plt
from PIL import Image
import math

import random
def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def findStartSmoothPoint(points, start, end):
    while(start != end):
        mid = int( (start+end)/2)
        A = (points[0][mid] - points[0][end], points[1][mid] - points[1][end])
        B = (points[0][start]- points[0][end], points[1][start]- points[1][end])
        #print(A)
        #print(B)
        r = ((A[0] * B[0] + A[1] * B[1]) * 1.0) / (math.sqrt(A[0] ** 2 + A[1] ** 2) * math.sqrt(B[0] ** 2 + B[1] ** 2))
        if( r < 0 and math.fabs(r) > 1.0):
            angle = math.pi
        elif (r > 0 and math.fabs(r) > 1.0):
            angle = 0.0
        else:
            angle = math.acos(((A[0] * B[0] + A[1] * B[1]) * 1.0) / (math.sqrt(A[0] ** 2 + A[1] ** 2) * math.sqrt(B[0] ** 2 + B[1] ** 2)))
        if(angle*180/math.pi < 1.0):
            return start
        if(start < end):
            start += 1
        else:
            start -= 1

    return end
img = Image.open("road.png").convert('1') #make it binary
im = np.array(img)


#轮廓线提取
contours = measure.find_contours(img, 0.5)

points = []

for n, contour in enumerate(contours):
    X = []
    Y = []
    for data in contour:
        X.append(data[0])
        Y.append(data[1])
    points.append( (X,Y) )
    #plt.plot(Y, X, linewidth=1, color=randomcolor())
    #plt.plot(contour[:, 1], contour[:, 0], linewidth=2, color=randomcolor())
'''    
plt.imshow(img)
plt.show()
'''

#细化
selem = skimage.morphology.disk(2)
im = skimage.morphology.binary_dilation(im, selem)
im = skimage.morphology.thin(im)
im = im.astype('uint8')
row, col = im.shape
X = [-1, 0, 1]
Y = [-1, 0, 1]
posX = []
posY = []
pos = []
Degree = []
for i in range(row):
    for j in range(col):
        if(im[i, j] == 1):
            degree = -1
            for oi in X:
                for oj in Y:
                    ni = i + oi
                    nj = j + oj
                    if ni >= 0 and ni < row and nj >=0 and nj < col and im[ni, nj] == 1:
                        degree += 1
            if(degree > 2):
                posX.append(i)
                posY.append(j)
                pos.append((i,j,degree))

initPos = (-10, -10)
truthPos = []
for data in pos:
    x, y, drgree = data
    if( math.sqrt((initPos[0]-x)*(initPos[0]-x)+(initPos[1]-y)*(initPos[1]-y)) > 5.0):
        truthPos.append(data)
    initPos = data
#error 1 3
point = truthPos[9]
detailInfo = {"intersection":point}
info = []
j = 0
for line in points:
    X, Y = line
    min_distance = 1000000
    min_point = (0, 0)
    k = 0
    for i in range( len(X) ):
        xx = X[i] - point[0]
        yy = Y[i] - point[1]
        distance = math.sqrt(xx**2+yy**2)
        if(i != 0 and i != len(X) -1 ):
            pre_distance = math.sqrt( (X[i-1]-point[0])**2+(Y[i-1]-point[1])**2 )
            next_distance = math.sqrt( (X[i+1]-point[0])**2+(Y[i+1]-point[1])**2 )
            if( distance <= pre_distance and distance <= next_distance):
                info.append([distance, (X[i], Y[i]), j, i])
    j += 1
info = sorted(info,key =lambda x:x[0])
#plt.plot([ info[3][1][1] ], [ info[3][1][0] ], "go", markersize = 1)
tmp = [ info[0] ]
for i in range(1, len(info)):
    A = ( info[i][1][0] - point[0], info[i][1][1]-point[1])
    flag = 1
    for data in tmp:
        B = (data[1][0] -point[0], data[1][1]-point[1])
        ang = math.acos(((A[0] * B[0] + A[1] * B[1]) * 1.0) / (math.sqrt(A[0] ** 2 + A[1] ** 2) * math.sqrt(B[0] ** 2 + B[1] ** 2)))
        if(ang < math.pi/6):
            flag = 0
            break
    if(flag):
        tmp.append( info[i] )
        if(len(tmp) == detailInfo["intersection"][2]):
            break

'''
plt.plot([ info[2][1][1] ], [ info[2][1][0] ], "go", markersize = 3)
plt.plot([ info[0][1][1] ], [ info[0][1][0] ], "go", markersize = 3)
plt.plot([ info[1][1][1] ], [ info[1][1][0] ], "go", markersize = 3)
plt.plot([ info[3][1][1] ], [ info[3][1][0] ], "go", markersize = 3)
plt.plot([ info[4][1][1] ], [ info[4][1][0] ], "go", markersize = 3)

plt.plot([ tmp[2][1][1] ], [ tmp[2][1][0] ], "go", markersize = 1)
plt.plot([ tmp[0][1][1] ], [ tmp[0][1][0] ], "go", markersize = 2)
plt.plot([ tmp[1][1][1] ], [ tmp[1][1][0] ], "go", markersize = 3)
'''

#info = info[:detailInfo["intersection"][2]]
info = tmp
print(info)

detailInfo["info"] = info
detailInfo["skipIndex"] = []
angle = 100000
max_index = 0
boundyNumber = 50
i = 0
for data in info:
    min_distance, min_point, j, k = data #j为第几条轮廓，k为min_point的索引
    #plt.plot(points[4][1], points[4][0], "bo", markersize = 1)
    if(k-boundyNumber<0):
        startIndex = 0
    else:
        startIndex = k - boundyNumber
    start = (points[j][0][startIndex], points[j][1][startIndex])
    if(k+boundyNumber >= len(points[j][0]) ):
        endIndex = len(points[j][0]) - 1
    else:
        endIndex = k + boundyNumber
    end = (points[j][0][endIndex], points[j][1][endIndex])

    A = (start[0]-min_point[0], start[1]-min_point[1])
    B = (end[0]-min_point[0], end[1]-min_point[1])

    #print(A)
    #print(B)
    ang = math.acos(((A[0] * B[0] + A[1] * B[1]) * 1.0) / (math.sqrt(A[0] ** 2 + A[1] ** 2) * math.sqrt(B[0] ** 2 + B[1] ** 2)) + 0.001)
    if(angle > ang):
        angle = ang
        max_index = i
    i += 1
detailInfo["skipIndex"].append(max_index) #轮廓线的忽略
start = findStartSmoothPoint(points[ info[max_index][2] ], info[max_index][3], info[max_index][3]-boundyNumber)
end = findStartSmoothPoint( points[info[max_index][2] ], info[max_index][3], info[max_index][3]+boundyNumber)
detailInfo["contours"] = { max_index:{"start":start, "end":end}}

print(max_index)
print(start, end, info[max_index][3])

point_index = detailInfo["contours"][max_index]["start"]
xx = points[info[max_index][2]][0][point_index]
yy = points[info[max_index][2]][1][point_index]
plt.plot( [ yy ],  [xx], "ro", markersize = 3)


#plt.plot( points[5][1], points[5][0], "bo", markersize = 1)

#先把end点求出
minValue = 100000
contour_index = 0
point_index = 0
point = ( points[ info[max_index][2] ][0][end],  points[ info[max_index][2] ][1][end])
plt.plot([point[1]], [point[0]], "bo", markersize = 1)


for i in range(len(info)):
    if(i in detailInfo["skipIndex"]):
        continue
    min_distance, min_point, j, k = info[i]
    X, Y = points[j]
    MIN = 100000
    MIN_INDEX = 0
    for m in range(k-boundyNumber, k+boundyNumber):
        if( MIN > math.sqrt( (X[m]-point[0])**2+(Y[m]-point[1])**2 ) ):
            MIN = math.sqrt( (X[m]-point[0])**2+(Y[m]-point[1])**2)
            MIN_INDEX = m
    if(MIN < minValue):
        minValue = MIN
        contour_index = i
        point_index = MIN_INDEX
print(contour_index)
print(point_index)
xx = points[info[contour_index][2]][0][point_index]
yy = points[info[contour_index][2]][1][point_index]
plt.plot( [ yy ],  [xx], "go", markersize = 1)

if(point_index < info[contour_index][3] ):
    detailInfo["contours"][contour_index] = {"start":point_index}
else:
    detailInfo["contours"][contour_index] = {"end":point_index}

main_index = max_index
pointIndex = start
# 4 1 0
print( info[main_index][2])
fff = 1

while(1):
    minValue = 100000
    contour_index = 0
    point_index = 0
    point = ( points[info[main_index][2]][0][pointIndex], points[info[main_index][2]][1][pointIndex] )
    #plt.plot([ point[1] ], [ point[0] ], "go", markersize = 1)
    for i in range(len(info)):
        if(i == main_index or i in detailInfo["skipIndex"]):
            continue
        min_distance, min_point, j, k = info[i]
        X, Y = points[j]
        MIN = 100000
        MIN_INDEX = 0
        for m in range(k-boundyNumber, k+boundyNumber):
            if (MIN > math.sqrt((X[m] - point[0]) ** 2 + (Y[m] - point[1]) ** 2)):
                MIN = math.sqrt((X[m] - point[0]) ** 2 + (Y[m] - point[1]) ** 2)
                MIN_INDEX = m
        if (MIN < minValue):
            minValue = MIN
            contour_index = i
            point_index = MIN_INDEX

    xx = points[info[contour_index][2]][0][point_index]
    yy = points[info[contour_index][2]][1][point_index]
    plt.plot( [ yy ],  [xx], "ro", markersize = 1)
    if (point_index < info[contour_index][3]):
        if(detailInfo["contours"].get(contour_index) == None):
            detailInfo["contours"][contour_index] = {"start": point_index}
        #求解end
            end = findStartSmoothPoint(points[ info[contour_index][2] ], info[contour_index][3], info[contour_index][3] + boundyNumber )
            detailInfo["contours"][contour_index]["end"] = end
            main_index = contour_index
            pointIndex = end
            detailInfo["skipIndex"].append(main_index)
        else:
            detailInfo["contours"][contour_index]["start"] = point_index
            break
    else:
        if(detailInfo["contours"].get(contour_index) == None):
            detailInfo["contours"][contour_index] = {"end": point_index}
        #求解start
            start = findStartSmoothPoint(points[ info[contour_index][2] ], info[contour_index][3], info[contour_index][3] - boundyNumber )
            detailInfo["contours"][contour_index]["start"] = start
            main_index = contour_index
            pointIndex = start

            '''
            xx = points[info[contour_index][2]][0][start]
            yy = points[info[contour_index][2]][1][start]
            plt.plot([yy], [xx], "ro", markersize=2)
            '''

            detailInfo["skipIndex"].append(main_index)
        else:
            detailInfo["contours"][contour_index]["end"] = point_index
            break


print(detailInfo["contours"])


for key in detailInfo["contours"].keys():
    start = detailInfo["contours"][key]["start"]
    end = detailInfo["contours"][key]["end"]
    if(start < 0 ):
        xx = []
        yy = []
        for i in range(start, end+1):
            xx.append( points[ info[key][2] ][0][i] )
            yy.append( points[ info[key][2] ][1][i] )
        plt.plot(yy, xx, "bo", markersize = 1)
    else:
        plt.plot( points[ info[key][2] ][1][start:end+1], points[ info[key][2] ][0][start:end+1], "bo", markersize = 1)







    

#plt.plot([ points[ info[contour_index][2] ][1][point_index] ], [ points[ info[contour_index][2] ][0][point_index] ], "bo", markersize = 1)





#while(1):





#print(info)
#print(point)

#plt.plot( points[index][1][ (info[0][3]-boundyNumber):(info[0][3]+boundyNumber)], points[index][0][ (info[0][3]-boundyNumber):(info[0][3]+boundyNumber)], "bo", markersize = 1)
print(info[max_index][3], start, end)
#plt.plot( points[max_index][1], points[max_index][0], "bo", markersize = 1)
#plt.plot( points[ info[max_index][2] ][1][start:end+1], points[ info[max_index][2] ][0][start:end+1], "bo", markersize = 1)
#plt.plot(pointsY, pointsX,"go", markersize = 1)
truthX = []
truthY = []
for data in truthPos:
    truthX.append(data[0])
    truthY.append(data[1])
plt.plot(truthY[3:4], truthX[3:4], "ro", markersize = 1)
plt.imshow(img)
plt.show()

