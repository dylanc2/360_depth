import numpy as np
import matplotlib.pyplot as plt
import os
import kornia
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main():
    b = []
    n = []
    for i in range(30):
        n.append(torch.eye(3))
    K = torch.stack(n,dim=0)
    for filename in sorted(os.listdir("outputs")):
        arr = np.load('outputs/'+filename)
        arr_tensor = torch.tensor(arr)
        x = arr_tensor[None, :]
        b.append(x)
        break
    x = torch.stack(b,dim=0)
    print(x.shape)
    print(K.shape)
    points_batch = kornia.geometry.depth.depth_to_3d(x, K).numpy()
    points = points_batch[0]
    print(points.shape)
    num_pixels = 480 * 960
    xs = points[0].reshape(num_pixels)
    ys = points[1].reshape(num_pixels)
    zs = points[2].reshape(num_pixels)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    plt.show()

if __name__ == "__main__":
    main()