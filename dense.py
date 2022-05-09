import numpy as np
import cv2
from PIL import Image
import glob
import matplotlib.pyplot as plt
import kornia
import torch
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

def run_frames(folder):
    # params for corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    # lk_params = dict( winSize = (15, 15),
    #                 maxLevel = 2,
    #                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    #                             10, 0.03))

    lk_params = dict( winSize = (100, 100),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Take first frame and find corners in it
    old_frame= cv2.imread(folder + '/00001.png')
    old_gray = cv2.cvtColor(old_frame,
                            cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
                                **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    iter = 0

    for filename in glob.glob(folder + '/*.png'):
        print(iter)

        if (filename == folder + '/00001.png'):
            pass

        frame = cv2.imread(filename)

        frame_gray = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                            frame_gray,
                                            p0, None,
                                            **lk_params)
    
        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, 
                                        good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)),
                            color[i].tolist(), 2)
            
            frame = cv2.circle(frame, (int(a), int(b)), 5,
                            color[i].tolist(), -1)

        img = cv2.add(frame, mask)
    
        cv2.imshow('frame', img)

        k = cv2.waitKey(50)
        # if k == 27:
        #     break
    
        # Updating Previous frame and points 
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

        iter += 1
    
    cv2.destroyAllWindows()

def run_mp4(file_name):
    cap = cv2.VideoCapture(file_name)
    
    # params for corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15),
                    maxLevel = 2,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                10, 0.03))
    
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    
    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame,
                            cv2.COLOR_BGR2GRAY)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255
    sample_frames = []
    sample_good_new = []
    sample_good_old = []
    sample_interval = 10
    iter = 0


    ###Get tensor of the unprojected points
    depths = torch.tensor(np.load('depths.npy'))[:, None, :, :]
    print(depths.shape)
    B = depths.shape[0]

    n = []
    for i in range(B):
        n.append(torch.tensor([[2304, 0, 960], [0, 2304, 544], [0, 0, 1]]))
    K = torch.stack(n,dim=0)
    print(K.shape)

    points_batch = kornia.geometry.depth.depth_to_3d(depths, K).numpy()
    ###Unprojected points completed

    while(1):
        
        ret, frame = cap.read()

        if ret == False:
            break

        frame_gray = cv2.cvtColor(frame,
                                cv2.COLOR_BGR2GRAY)
    
        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(old_gray,
                                            frame_gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
    
        
        # prev(y,x)∼next(y+flow(y,x)[1],x+flow(y,x)[0])

        
        # for y in range(frame_gray.shape[1]):
        #     for x in range(frame_gray.shape[0]):
        #         pass
        
        points_batch[iter] #unprojected points of the last frame

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('frame2', bgr)
            
        img = cv2.add(frame, mask)
    
        cv2.imshow('frame', img)
        
        k = cv2.waitKey(10)
        if k == 27:
            break
    
        # # Updating Previous frame and points 
        old_gray = frame_gray.copy()
        # p0 = good_new.reshape(-1, 1, 2)

        iter += 1

    print("Number of sample frames: ", len(sample_frames))
    np.save('data.npy', np.asarray(sample_frames))
    print(np.asarray(sample_frames).shape)
    np.save('predicted_points.npy', np.asarray(sample_good_new))
    print(np.asarray(sample_good_new).shape)

    cv2.destroyAllWindows()
    cap.release()

    return sample_frames, sample_good_new, sample_good_old

def plot_3d():
    # sample_frames, sample_good_new, sample_good_old = run_mp4("demo3.mp4")
    depths = np.load('depths.npy')
    predicted_points = np.load('predicted_points.npy')
    print(depths.shape)
    print(predicted_points.shape)
    predicted_points = predicted_points.astype(int)

    x = []
    y = []
    z = []
    num_frames = depths.shape[0]
    for i in range(num_frames):
        depth = depths[i]
        predicted_img_points = predicted_points[i]
        predicted_img_points = np.transpose(predicted_img_points)
        
        z += list(depths[i][predicted_img_points[1], predicted_img_points[0]])
        x += list(predicted_img_points[1])
        y += list(predicted_img_points[0])
 
    print(len(x), len(y), len(z))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(x, z, y, 'gray')
    ax.set_ylabel("z axis")
    ax.set_xlabel("x axis")

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0 

    mid_x = (x.max()+x.min()) / 2.0
    mid_y = (y.max()+y.min()) / 2.0
    mid_z = (z.max()+z.min()) / 2.0

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def plot_kornia():
    depths = torch.tensor(np.load('depths.npy'))[:, None, :, :]
    print(depths.shape)
    B = depths.shape[0]
    
    n = []
    for i in range(B):
        n.append(torch.tensor([[2304, 0, 960], [0, 2304, 544], [0, 0, 1]]))
    K = torch.stack(n,dim=0)
    print(K.shape)

    points_batch = kornia.geometry.depth.depth_to_3d(depths[100:102], K[100:102]).numpy()
    points = points_batch[0]
    points2 = points_batch[1]
    print("Points shape: ", points.shape)

    b100 = np.array([801.7296,  617.7409, 1142.4205,  782.8949]).astype(int)
    b101 = [784.78357,  616.9811,  1130.3014,   783.1737]


    ##select depth out for b100
    set1 = points[:,b100[1]:b100[3],b100[0]:b100[2]]


    ##and for optical flow ahead points
    old_gray = cv2.cvtColor(cv2.imread("f100.jpg"), cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.cvtColor(cv2.imread("f101.jpg"), cv2.COLOR_BGR2GRAY)
    
    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow.shape)

    # prev(y,x)∼next(y+flow(y,x)[1],x+flow(y,x)[0])
    #points2[:,b100[]]
    y = b100[1]
    x = b100[0]
    y_new = int(y+flow[y][x][1])
    x_new = int(x+flow[y][x][0])

    y = b100[3]
    x = b100[2]
    y_new2 = int(y+flow[y][x][1])
    x_new2 = int(x+flow[y][x][0])

    set2 = points2[:,y_new:y_new2,x_new:x_new2+1]
    set1 = np.moveaxis(set1, 0, -1)
    set2 = np.moveaxis(set2, 0, -1)

    r = RANSACRegressor(min_samples=10)
    res1 = set1.reshape(set1.shape[0]*set1.shape[1],set1.shape[2])
    print(res1.shape)
    res2 = set2.reshape(set2.shape[0]*set2.shape[1],set2.shape[2])
    #res1, t1 = res1[:-10000], res1[-10000:]
    #res2, t2 = res2[:-10000], res2[-10000:]
    r.fit(res1,res2)
    print(r.score(res1,res2))
    






    # num_pixels = depths.shape[2] * depths.shape[3]
    # sample_size = 1000
    # idx = np.random.choice(np.arange(num_pixels), sample_size, replace=False)
    # xs = points[0].reshape(num_pixels)[idx]
    # ys = points[1].reshape(num_pixels)[idx]
    # zs = points[2].reshape(num_pixels)[idx]

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xs, ys, zs)
    # plt.show()

if __name__ == "__main__":
    cc = cv2.VideoCapture("demo.mp4")
    ii = 0
    while ii<100:
        cc.read()
        ii+=1
    _, f100 = cc.read()
    _, f101 = cc.read()
    cv2.imwrite("f100.jpg",f100)
    cv2.imwrite("f101.jpg",f101)
    #run_mp4("demo.mp4")
    # run_frames("frames")
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    # plot_3d()
    plot_kornia()