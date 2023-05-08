# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import cv2

# videoLocation = 'D:/BOBSL/BOBSL200G/bobsl/videos'
# videoTitles = os.listdir(videoLocation)

# subtitles = 'D:/BOBSL/BOBSL200G/bobsl/subtitles2/subtitles/audio-aligned-heuristic-correction'
# subTitles = os.listdir(subtitles)

# def removeType(string):
#     return string[:-4]



# videoTitlesEdited = list(map(removeType, videoTitles))
# subTitlesEdited = list(map(removeType, subTitles))

# # print(videoTitles) #5087953980081062580
# # print(subTitles) #508795398008106


# alligned = []
# for videoTitle in videoTitlesEdited:
#     for subTitle in subTitlesEdited:
#         if subTitle == videoTitle:
#             alligned.append(videoTitle)

# subtitles = 'D:/BOBSL/BOBSL200G/bobsl/subtitles2/subtitles/audio-aligned-heuristic-correction'
# videoLocation = 'D:/BOBSL/BOBSL200G/bobsl/videos/'
# videoName = videoTitles[0]
# saveLocation = 'D:/Thesis'

# print('opening: ' + videoLocation + videoName)
# cap = cv2.VideoCapture(videoLocation + videoName)

# if (cap.isOpened()== False): 
#     print("Error opening video stream or file")

# while cap.isOpened():
#     success, image = cap.read()
    
#     # cv2.putText(image, str(cap.get(cv2.CAP_PROP_POS_MSEC)), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 10)
#     # Display image
#     cv2.imshow('subtitle Pose', image)
#     if cv2.waitKey(40) & 0xFF == 27:
#         break

# cap.release()

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd


a = np.random.rand(2000, 3)*10
t = np.array([np.ones(100)*i for i in range(20)]).flatten()
df = pd.DataFrame({"time": t ,"x" : a[:,0], "y" : a[:,1], "z" : a[:,2]})

def update_graph(num):
    data=df[df['time']==num]
    graph._offsets3d = (data.x, data.y, data.z)
    title.set_text('3D Test, time={}'.format(num))


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

data=df[df['time']==0]
graph = ax.scatter(data.x, data.y, data.z)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, 19, 
                               interval=40, blit=False)

plt.show()