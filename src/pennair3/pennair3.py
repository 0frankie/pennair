import cv2 as cv
import numpy as np

vid = cv.VideoCapture("../../resources/PennAir 2024 App Dynamic Hard.mp4")
fourcc = cv.VideoWriter.fourcc(*"mp4v")
out = cv.VideoWriter('../../resources/pennair3/pennair3_out.mp4', fourcc, 30, (1920, 1080))

frame, success = 0, True
while frame < 300 and success:
    success, img = vid.read()
    if success:
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        blur = cv.GaussianBlur(gray_img, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        ret, thresh = cv.threshold(blur, 70, 255, cv.THRESH_BINARY)
        mask = cv.inRange(thresh, np.array([0, 0, 0]), np.array([255, 255, 0]))
        final_mask_dilate = cv.dilate(mask, np.ones((25, 25), dtype=np.uint8))
        final_mask = cv.erode(final_mask_dilate, np.ones((25, 25), dtype=np.uint8))

        edged = cv.Canny(final_mask, 30, 200)

        structuring_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        edged = cv.morphologyEx(edged, cv.MORPH_CLOSE, structuring_element)

        edged = cv.copyMakeBorder(
            edged,
            top=1,
            bottom=1,
            left=1,
            right=1,
            borderType=cv.BORDER_CONSTANT,
            value=1
        )

        contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for cnt in contours:
            if cv.contourArea(cnt) < 1000 or cv.contourArea(cnt) > 1000000:
                continue
            cv.drawContours(img, cnt, -1, (255, 255, 0), 5)

            M = cv.moments(cnt)
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])

            cv.circle(img, (cX, cY), 7, (255, 255, 255), -1)

            # x, y, w, h = cv.boundingRect(cnt)
            # cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 5)
        out.write(img)
        frame += 1

vid.release()