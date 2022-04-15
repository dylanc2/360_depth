import numpy as np
import cv2
from PIL import Image
import glob
import matplotlib.pyplot as plt
import kornia
import torch

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
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None,
                                **feature_params)
    
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    sample_frames = []
    sample_good_new = []
    sample_good_old = []
    sample_interval = 10
    iter = 0

    while(1):
        
        ret, frame = cap.read()

        if ret == False:
            break

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

        if iter % sample_interval == 0:
            sample_frames.append(frame)
            sample_good_new.append(good_new)
            sample_good_old.append(good_old)
    
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
        
        k = cv2.waitKey(10)
        if k == 27:
            break
    
        # Updating Previous frame and points 
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

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

    points_batch = kornia.geometry.depth.depth_to_3d(depths, K).numpy()
    points = points_batch[0]
    print("Points shape: ", points.shape)

    num_pixels = depths.shape[2] * depths.shape[3]
    sample_size = 1000
    idx = np.random.choice(np.arange(num_pixels), sample_size, replace=False)
    xs = points[0].reshape(num_pixels)[idx]
    ys = points[1].reshape(num_pixels)[idx]
    zs = points[2].reshape(num_pixels)[idx]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs)
    plt.show()

if __name__ == "__main__":
    run_mp4("demo.mp4")
    # run_frames("frames")
    # plot_3d()
    # plot_kornia()