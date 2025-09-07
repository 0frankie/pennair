import cv2 as cv
import numpy as np

img = cv.imread("../../resources/pennair1/pennair1.png")

hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV) # convert to HSV

# blur multiple times to decrease noise
blur = cv.GaussianBlur(hsv_img, (5, 5), 0)
blur = cv.GaussianBlur(blur, (5, 5), 0)
blur = cv.GaussianBlur(blur, (5, 5), 0)
blur = cv.GaussianBlur(blur, (5, 5), 0)

# use the threshold function to filter out certain colors
# I found that 150 was the best number for this scenario
ret, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)

# filter for just the yellow (255, 255, 0)
mask = cv.inRange(thresh, np.array([0, 0, 0]), np.array([255, 255, 0]))

# use canny to filter out for just the yellow
edged = cv.Canny(mask, 30, 200)
contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

for cnt in contours:
    if cv.contourArea(cnt) < 1000 or cv.contourArea(cnt) > 10000:
        continue
    cv.drawContours(img, cnt, -1, (255, 255, 0), 5)

    M = cv.moments(cnt)
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    cv.circle(img, (cX, cY), 7, (255, 255, 255), -1)

    # x, y, w, h = cv.boundingRect(cnt)
    # cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 5)

save_img = img

cv.imshow('Grayscale', save_img)
k = cv.waitKey(0)

if k == ord("s"):
    cv.imwrite("../../resources/pennair1/pennair1_completed.png", save_img)
