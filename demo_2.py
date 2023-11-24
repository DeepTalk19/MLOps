# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv
import time
from pphumanseg import PPHumanSeg
from PIL  import Image
import math


# Check OpenCV version
assert cv.__version__ >= "4.8.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

frame_count = 0
start_time = time.time()
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

print("here")
# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=2))

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
]

parser = argparse.ArgumentParser(description='PPHumanSeg (https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/PP-HumanSeg)')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input path to a certain image, omit if using camera.')
parser.add_argument('--model', '-m', type=str, default='human_segmentation_pphumanseg_2023mar.onnx',
                    help='Usage: Set model path, defaults to human_segmentation_pphumanseg_2023mar.onnx.')
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save a file with results. Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()

def background_blur(org, img, mask):   
    out = np.zeros_like(org) 
    img_copy = np.asarray(cv.blur(np.asarray(org),(15, 15)))
    rev_mask = np.where((mask==0)|(mask==1), mask^1, mask)
    remove_mask_resgion = img_copy*rev_mask
    image_array = np.where(remove_mask_resgion == 0, 1, remove_mask_resgion)
    img_new = cv.addWeighted(image_array, 0.5, img, 0.5, 0)
    return img_new
    
def background_replace(mask, blend_img):
    bg = Image.open('./3685070.jpg')
    bg = bg.resize((blend_img.shape[1], blend_img.shape[0]))
    bg = cv.cvtColor(np.asarray(bg), cv.COLOR_BGR2RGB)
    rev_mask = np.where((mask==0)|(mask==1), mask^1, mask)
    remove_mask_resgion = bg*rev_mask
    
    image_array = np.where(remove_mask_resgion == 0, 1, remove_mask_resgion)
    img_new = cv.addWeighted(np.asarray(image_array), 1, blend_img, 1, 0)
    
    return img_new

def face_detection(frame):
    face_cascade = cv.CascadeClassifier("./haarcascade_frontalface_default.xml")
    gray = cv.cvtColor(np.array(frame), cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def face_distortion_fn(img, w1, h1):
    h,w,_ = img.shape
    flex_x = np.zeros((h,w),np.float32)
    flex_y = np.zeros((h,w),np.float32)

    scale_y= 1
    scale_x = 1
    alpha = -1.8
    center_x, center_y = (w1 // 2, h1 // 2)
    radius = h/5

    for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y

            if distance >= (radius * radius):
                flex_x[y, x] = x
                flex_y[y, x] = y
            else:
                theta = np.arctan2(delta_x,delta_y) + alpha*(radius-math.sqrt(distance))/radius
                r_sin = math.sqrt(distance)*np.cos(theta)
                r_cos = math.sqrt(distance)*np.sin(theta)
                flex_x[y, x] = r_cos + center_x
                flex_y[y, x] = r_sin + center_y

    dst = cv.remap(img, flex_x, flex_y, cv.INTER_LINEAR)
    return dst

def face_distort(frame):
    faces = face_detection(frame)
    try:
        x, y, w, h = faces[0]
    except:
        return frame
    roi=frame[y:y+h, x:x+w]
    dis_roi = face_distortion_fn(roi, w, h)
    frame[y:y+h, x:x+w] = dis_roi
    return frame

def face_replace(frame):
    cat_img = Image.open('./cat_head.jpg')

    gray = cv.cvtColor(np.array(cat_img), cv.COLOR_BGR2GRAY)
    _ , img_thresh = cv.threshold(gray, 0, 250, cv.THRESH_BINARY) 
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    # filter contours by size
    big_cntrs = []
    # marked = blend_img.copy();
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 15000:
            big_cntrs.append(contour)
    final = cv.drawContours(np.asarray(cat_img), big_cntrs, 0, (0, 255, 0), 3)
    # final = np.where(final==0, 255, final)
    diff_img = final-cat_img

    mask1 = np.zeros_like(diff_img)
    mask2 = cv.drawContours(mask1, big_cntrs, 0, (1,1,1), -1)

    extract_cat = cat_img*mask2

    frame = np.asarray(frame)
    frame1 = frame.copy()

    faces = face_detection(frame)
    for face in faces:
    #     face_di = face_distortion(frame, w,h)
        x, y, w, h = face
        roi=frame[y:y+h, x:x+w]
        face_mask = cv.resize(mask2,(roi.shape[0], roi.shape[1]))
        extract_cat = cv.resize(extract_cat,(roi.shape[0], roi.shape[1]))
        face_mask = np.where((face_mask==0)|(face_mask==1), face_mask^1, face_mask)
        filter_img = roi*face_mask
        replaced_img = cv.addWeighted(filter_img, 1,extract_cat,1,0)
        frame1[y:y+h, x:x+w] = replaced_img
    return frame1

def custom_filter(frame, count):
    video_path = './gif.mp4'
    cap = cv.VideoCapture(video_path)
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    flag = False
    for frame_num in range(frame_count):
        # Read a frame from the video
        ret, frame1 = cap.read()
        frames.append(frame1)      
    try:
        frame2 = cv.resize(frames[count//2],(frame.shape[1],frame.shape[0]))
    except:
        flag = True
        frame2 = cv.resize(frames[0],(frame.shape[1],frame.shape[0]))
    dstimg = cv.addWeighted(np.array(frame),1,np.array(frame2),1,0)
    return dstimg, flag

if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]
    # Instantiate PPHumanSeg
    model = PPHumanSeg(modelPath='./human_segmentation_pphumanseg_2023mar.onnx', backendId=backend_id, targetId=target_id)
 
    deviceId = 0
    cap = cv.VideoCapture(deviceId)
    # cap = cv.VideoCapture(gstreamer_pipeline(flip_method=2), cv.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv.namedWindow("CSI Camera", cv.WINDOW_AUTOSIZE)
        # Window
        count = 0
        while cv.getWindowProperty("CSI Camera", 0) >= 0:
            ret, frame = cap.read()
            if ret:
                # mask = model.infer(frame)
                # # print(mask[0].shape)
                # mask = np.where(np.asarray(mask[0])!=0,1, np.asarray(mask[0]))
                # mask = np.stack([mask, mask, mask], axis=-1)
                # blend_img = frame * mask
                # img = background_blur(frame, blend_img, mask)
                # img = background_replace(mask, blend_img)
                img = face_distort(frame)
                # img = face_replace(frame)                     
                # count += 1
                # img, flag = custom_filter(frame,count)
                # if flag:
                #     count = 0
                cv.imshow("CSI Camera", img)
            keyCode = cv.waitKey(30)
            if keyCode == ord('q'):
                break
            # Increment frame count
            frame_count += 1

            # Calculate FPS every 10 frames
            if frame_count % 10 == 0:
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
            
        cap.release()
        cv.destroyAllWindows()
    else:
        print("Unable to open camera")
