import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import cv2

def get_example(location='experiment8/unprocessed/MONKEY_AAPB2c6iii_20190_21240.npy', 
                shapeLocation='experiment8/unprocessed/MONKEY_AAPB2c6iii_20190_21240_shape.npy'):
    coorLoad = np.loadtxt(location)
    coorShape= np.loadtxt(shapeLocation)
    return coorLoad.reshape(coorLoad.shape[0], coorLoad.shape[1] // int(coorShape[2]), int(coorShape[2]))

def mediapipe_demo(data):
    connect = get_connections_list()
    newData = []
    for frame in data:
        newData.append(frame[0:25])
    data = np.array(newData)
    def updateGraph(num):
        x = data[num].T[0]
        y = data[num].T[1]
        z = data[num].T[2]
        graph._offsets3d = (x, y, z)
        title.set_text('3D Test, time={}'.format(num))
        createLines(num)

    def createLines(num):
        numPoints = data[0].shape[0]
        for i, (p1, p2) in enumerate(connect):
            p1, p2 = int(p1), int(p2)
            line = lines[i]
            if p1 >= numPoints or p2>= numPoints:
                break
            line.set_data([[data[num, p1, 0], data[num, p2, 0]], 
                        [data[num, p1, 1], data[num, p2, 1]]])
            line.set_3d_properties([data[num, p1, 2], data[num, p2, 2]])
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')
    ax.view_init(90, -90)

    x = data[0].T[0]
    y = data[0].T[1]
    z = data[0].T[2]
    graph = ax.scatter(x, y, z)
    lines = [ax.plot([], [], [])[0] for _ in connect]

    ani = FuncAnimation(fig, updateGraph, frames=len(data),repeat=True, interval=100)

    plt.show()


def get_connections_list():
    return np.loadtxt('./experiment8/connections_list.npy')

def play_video(videoLocation='methedology/SYDNEY_MTFB3c9a_317220_319465.mp4'):
    playbackSpeed = int((1/25)*1000)
    frames = []
    cap = cv2.VideoCapture(videoLocation)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frameNum = 0
    while True:
        try:
            cv2.imshow('MediaPipe Pose', frames[frameNum])
        except:
            break
        frameNum += 1
        if frameNum >= len(frames):
            frameNum = 0
        k = cv2.waitKey(playbackSpeed) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()

play_video()

location = 'experiment8/unprocessed/SYDNEY_MTFB3c9a_317220_319465.npy'
locationShape = 'experiment8/unprocessed/SYDNEY_MTFB3c9a_317220_319465_shape.npy'
example = get_example(location, locationShape)
mediapipe_demo(example)