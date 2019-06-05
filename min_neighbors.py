### USAGE ###
# python3 min_neighbors.py inVideo.mp4 outVideo.mp4 outPickle.pickle

### IMPORTS ###

import sys
import cv2
import time
import pickle
import numpy as np

### ARGUMENTS ###

in_fname = sys.argv[1] # in video
out_fname = sys.argv[2] # out video
pickle_fname = sys.argv[3] # pickle file (OUT)

### MODEL - BODY ###

body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')

### MODEL - FACE ###

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

### FUNCTIONS ###

def run_with_neighbors(num_neighbors):   
    
    print('Starting neighbors={}...'.format(num_neighbors))         
    
    ### INPUT ###
    
    print('Input video file: ' + in_fname)
    in_video = cv2.VideoCapture(in_fname)
    
    frame_w = int(in_video.get(3))
    frame_h = int(in_video.get(4))
    print(' - Frame width: {}, height: {}'.format(frame_w, frame_h))
    
    ### OUTPUT ###
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(out_fname, fourcc, int(in_video.get(5)), (int(in_video.get(3)), int(in_video.get(4))))
    print('Output video file: ' + out_fname)
    
    frame_counter = 0
    frame_total = in_video.get(7)
    min_neighbors = []
    
    while True:
    
        # grab single frame of video
        ret, frame = in_video.read()
        
        # if no frame, break out
        if not ret:
            break
    
        # update counter
        frame_counter += 1
        print('{}/{}'.format(frame_counter, frame_total))
        
        start = time.time()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        bodies = body_cascade.detectMultiScale(gray, 1.1, num_neighbors)
#         faces = face_cascade.detectMultiScale(gray, 1.1, num_neighbors)
        
        for (x,y,w,h) in bodies:
#         for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)
        
        # display resulting image
        out_video.write(frame)
    
        end = time.time()
        
        min_neighbors.append(end-start)
    
    # release handle to the videos
    in_video.release()
    out_video.release()
    
    return min_neighbors

with3 = run_with_neighbors(3)
# with4 = run_with_neighbors(4)
with5 = run_with_neighbors(5)
# with6 = run_with_neighbors(6)
# with7 = run_with_neighbors(7)
with8 = run_with_neighbors(8)

neighbors_dictionary = {'3neighbors': with3, '5neighbors': with5, '8neighbors': with8}

# write to pickle file
pickle_out = open(pickle_fname, 'wb')
pickle.dump(neighbors_dictionary, pickle_out)
pickle_out.close()        
