import cv2 as cv
import numpy as np

vid = cv.VideoCapture("../../resources/PennAir 2024 App Dynamic.mp4")
fourcc = cv.VideoWriter.fourcc(*"mp4v")
out = cv.VideoWriter('../../resources/pennair2/pennair2_out.mp4', fourcc, 30, (1920, 1080))

frame, success = 0, True
while frame < 300 and success: # goes through 300 frames (or 10 seconds, since video is 30 fps)
    success, img = vid.read()
    if success:
        # same algorithm from pennair1.py
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        blur = cv.GaussianBlur(gray_img, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        blur = cv.GaussianBlur(blur, (5, 5), 0)
        ret, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)
        mask = cv.inRange(thresh, np.array([0, 0, 0]), np.array([255, 255, 0]))

        # added dilate to remove small pixels and erode to return shapes to mostly original shape
        final_mask_dilate = cv.dilate(mask, np.ones((20, 20), dtype=np.uint8))
        final_mask = cv.erode(final_mask_dilate, np.ones((20, 20), dtype=np.uint8))

        edged = cv.Canny(mask, 30, 200)

        # get overlapping shapes to form a single contour
        structuring_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        edged = cv.morphologyEx(edged, cv.MORPH_CLOSE, structuring_element)

        # add a small border for shapes that go out of the screen to be detected as a shape
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
            if cv.contourArea(cnt) < 1000 or cv.contourArea(cnt) > 100000:
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


# EXTRANEOUS CODE FOR TESTING
# import cv2 as cv
# import numpy as np
#
# img = cv.imread("frame0.png")
#
# gray_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# blur = cv.GaussianBlur(gray_img, (5, 5), 0)
# blur = cv.GaussianBlur(blur, (5, 5), 0)
# blur = cv.GaussianBlur(blur, (5, 5), 0)
# blur = cv.GaussianBlur(blur, (5, 5), 0)
# blur = cv.GaussianBlur(blur, (5, 5), 0)
# blur = cv.GaussianBlur(blur, (5, 5), 0)
# ret, thresh = cv.threshold(blur, 150, 255, cv.THRESH_BINARY)
# mask = cv.inRange(thresh, np.array([0, 0, 0]), np.array([255, 255, 0]))
# final_mask_dilate = cv.dilate(mask, np.ones((20, 20), dtype=np.uint8))
# final_mask = cv.erode(final_mask_dilate, np.ones((20, 20), dtype=np.uint8))
#
# edged = cv.Canny(final_mask, 30, 200) #
#
# structuring_element = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
# edged = cv.morphologyEx(edged, cv.MORPH_CLOSE, structuring_element)
#
# contours, hierarchy = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
#
# for cnt in contours:
#     if cv.contourArea(cnt) < 1000:
#         continue
#     cv.drawContours(img, cnt, -1, (255, 255, 0), 5)
#
#     M = cv.moments(cnt)
#     cX = int(M['m10'] / M['m00'])
#     cY = int(M['m01'] / M['m00'])
#
#     cv.circle(img, (cX, cY), 7, (255, 255, 255), -1)
#
#     # x, y, w, h = cv.boundingRect(cnt)
#     # cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 5)
#
# cv.imshow('Grayscale', img)
# k = cv.waitKey(0)
#
# if k == ord("s"):
#     cv.imwrite("bruh.png", img)

# import cv2 as cv
# import numpy as np
#
# vid = cv.VideoCapture("PennAir 2024 App Dynamic.mp4")
#
# count, success = 0, True
#
# while count < 30 and success:
#     success, img = vid.read()
#     if success:
#         cv.imwrite(f'frame{count}.png', img)
#         count += 1