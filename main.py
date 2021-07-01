import cv2
import numpy as np
import os

path = 'ImagesQuery'
orb = cv2.ORB_create(nfeatures=1000)

#### Import Images
images = []
classNames = []
myList = os.listdir(path)
print('Total Classes Detected', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)      #print name of every class


def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList


def findID(img, desList, thres=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1 ##3lshan mybd2sh mn 0
    try: ## 3lshann lw mfessh matchess khalss mykhusshh w y3mll error
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    # print(matchList)
    ##3lshann at2kdd anuu hassl matches kazaa maraa
    if len(matchList) != 0:##3lshann check anu msh fadyy
        if max(matchList) > thres: ## thres hw 3add elmarat ely aseebu yt2kdd anu elrakm atkrr=15
            finalVal = matchList.index(max(matchList)) ##byrag33 el index 3lshan akhdoo ageeb beeh any soraa fy el list
    return finalVal


desList = findDes(images)
print(len(desList))

cap = cv2.VideoCapture(0)
while True:

    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = findID(img2, desList)
    if id != -1:
        cv2.putText(imgOriginal, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('PIXELS IMAGE PROCESSING SHOW', imgOriginal)
    cv2.waitKey(1)

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#
# # cv2.imshow('Kp1',imgKp1)
# # cv2.imshow('Kp2',imgKp2)
# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
# cv2.imshow('img3',img3)
# cv2.waitKey(0)