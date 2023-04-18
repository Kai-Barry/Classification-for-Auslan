import json
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
 
# Opening JSON file
start = time.time()
with open('dict_spottings.json') as json_file:
    data = json.load(json_file)


videoLocation = 'D:/BOBSL/BOBSL200G/bobsl/videos'
videoTitles = os.listdir(videoLocation)

videoName = videoTitles[0]
print('opening: ' + videoLocation + videoName)