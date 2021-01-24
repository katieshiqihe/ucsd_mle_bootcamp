#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For the capstone, I would like to develope a model that colorizes grayscale images.
Specifically, I thought it would be interesting to "watch" old TV shows in color
by colorizing black and white TV screenshots. Although there are many image 
datasets available, I chose to scrape YouTube to get the early TV color schemes. 
The test will be to colorize a clip from the original Twilight Zone. All data 
will be stored as a HDF5 file.

Some off-the-shelf image datasets:
    - CIFAR: https://www.cs.toronto.edu/~kriz/cifar.html
        60,000 32x32 color images with labels. Categories include animals and 
        vehicles but no people.
    - COCO: https://cocodataset.org/#home
        Many labelled datasets for different challenges. 

@author: khe
"""
from pytube import YouTube
import cv2
import numpy as np
import tables
import os

class TrainTable(tables.IsDescription):
    """Table in hdf5 file for storing color image array."""
    R = tables.UInt8Col(pos=0)
    G = tables.UInt8Col(pos=1)
    B = tables.UInt8Col(pos=2)
    
class TestTable(tables.IsDescription):
    """Table in hdf5 file for storing grayscale image array."""
    L = tables.UInt8Col(pos=0)

def capture_youtube(url, filename, skip_open=0, skip_end=0, mode='RGB'):
    """
    Download a YouTube video, take screenshots, and return them as array.

    Parameters
    ----------
    url : str
        Link to Youtube video.
    filename : str
        Output filename without extension.
    skip_open : int, optional
        Number of seconds to skip at the beginning of video. The default is 0.
    skip_end : int, optional
        Number of seconds to skip at the end of video. The default is 0.
    mode: str, optional
        `RGB` (color) or `L` (grayscale). The default is `RGB`. 

    Returns
    -------
    None.

    """
    assert mode in ('RGB', 'L')
    
    # Download video from YouTube
    video = YouTube(url)
    
    for stream in video.streams.filter(file_extension = "mp4"):
        # Get the 360p video stream
        if stream.mime_type == 'video/mp4':
            if stream.resolution == '360p':
                if not os.path.isfile(filename+'.mp4'):
                    stream.download(filename=filename)
                break
            
    # Extract frames from video
    vid_cap = cv2.VideoCapture(filename+'.mp4')
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    total_frames = vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    frame_count = 0
    if mode == 'RGB':
        data = np.empty((0, 360, 480, 3)).astype('uint8')
    else:
        data = np.empty((0, 360, 480, 1)).astype('uint8')
        
    while vid_cap.isOpened():
        
        success,image = vid_cap.read() 
        
        if not success:
            break
        
        # Skip the openning and title (first 75 seconds)
        if frame_count >= fps*skip_open:
            # Get one image per second of footage
            # The videos are about 50-60 minutes long, this will result in about 3000-3600 images per video
            if frame_count % int(fps) == 0:
                # Trim if aspect ratio is not right
                if image.shape[1] == 640:
                    image = image[:,80:-80,:]
                # Default color scheme in openCV is BGR, covert to RGB
                if mode == 'RGB':
                    image = image[:,:,::-1]
                else: 
                    image = image[:,:,:1]
                data = np.concatenate((data, np.expand_dims(image, axis=0)))
            
        frame_count += 1
        
        # Skip the end
        if frame_count >= (total_frames-fps*skip_end):
            break
    
    vid_cap.release()
    cv2.destroyAllWindows()
    
    return data

###############################################################################
# Create HDF5 file
###############################################################################
mdb = tables.open_file('youtube_data.h5', mode="w")
filters = tables.Filters(complevel=5, complib='blosc')
# Create tables
mdb.create_table('/', 'Train', TrainTable, filters=filters)
mdb.create_table('/', 'Test', TestTable, filters=filters)
mdb.flush()
mdb.close()

###############################################################################
# Training Set
###############################################################################
video_urls = [
    'https://www.youtube.com/watch?v=aRRYIe6hXTQ&list=PLklyfwlKNjxD52EQbChCopHxWdAXBX0cg',
    'https://www.youtube.com/watch?v=NZlBM8hw3cg&list=PLklyfwlKNjxD52EQbChCopHxWdAXBX0cg&index=3',
    'https://www.youtube.com/watch?v=_PhHKIufB4Q&list=PLklyfwlKNjxD52EQbChCopHxWdAXBX0cg&index=4',
    'https://www.youtube.com/watch?v=w4Jfm4J-9tw',
    'https://www.youtube.com/watch?v=PmZP_efIOhQ'
    ]

for i in range(len(video_urls)):
    filename = 'video_%s'%i
    data = capture_youtube(video_urls[i], filename, 75, 60)
    R = data[:,:,:,0].flatten()
    G = data[:,:,:,1].flatten()
    B = data[:,:,:,2].flatten()
    
    mdb = tables.open_file('youtube_data.h5', mode="r+")
    data_table = mdb.root.Train
    data_table.append(np.stack((R,G,B), axis=1))
    data_table.flush()
    mdb.close()
    
###############################################################################
# Test Set
###############################################################################
data = capture_youtube('https://www.youtube.com/watch?v=QSegeI5Qn6A', 'test_data', mode='L')
L = data[:,:,:,0].flatten()

mdb = tables.open_file('youtube_data.h5', mode="r+")
data_table = mdb.root.Test
data_table.append(L)
data_table.flush()
mdb.close()