# self-testing purposes
# gets frames of video to individually analyze

import cv2 as cv

vid = cv.VideoCapture("../../resources/PennAir 2024 App Dynamic Hard.mp4")

count, success = 0, True

while count < 30 and success:
    success, img = vid.read()
    if success:
        cv.imwrite(f'frame{count}.png', img)
        count += 1
