import numpy as np
boxes = np.load("boxes.npy", allow_pickle=True)
classes = np.load("classes.npy", allow_pickle=True)
i = 0
# for b,c in zip(boxes, classes):
#     i+=1
#     print("frame", i)
#     for j in range(len(b)):
#         print(b[j], c[j])

import cv2
x = cv2.VideoCapture("demo.mp4")
fr = 0
while fr <100:
    fr+=1
    x.read()
_, fr100 = x.read()
_, fr101 = x.read()

for box in boxes[100]:
    box = box.astype(int)
    im100 = cv2.rectangle(fr100, (box[0],box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imshow("w",im100)
    cv2.waitKey(0)

for box in boxes[101]:
    box = box.astype(int)
    im101 = cv2.rectangle(fr101, (box[0],box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imshow("w",im101)
    cv2.waitKey(0)
print(boxes[100][0])
print(boxes[101][0])