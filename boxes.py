import numpy as np
boxes = np.load("boxes.npy", allow_pickle=True)
print(boxes)
classes = np.load("classes.npy", allow_pickle=True)
print(classes)