from genericpath import isfile
from io import StringIO
import os
import subprocess
import shutil
import math
import glob
import random
import logging
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from eyetracker import Scenario

# [10.07] processing후 Tablet과 Smartphone에서 CamAngle이 *.txt라고 가정
def bad_record(fn):
    if os.path.isfile(fn):
        return len(open(fn).readlines()) < 100
    else:
        return True

def process_head(arr, factor):
    arr['roll'] -= np.mean(arr['roll'][5::100])
    arr['pitch'] -= np.mean(arr['pitch'][5::100])
    for field in ['roll', 'pitch', 'yaw']:
        ind = np.where(arr[field]!=0)[0]
        if len(ind) < 2:
            print("No valid points", len(ind))
            break
        elif len(ind) < len(arr):
            f = interp1d(ind, arr[field][ind])
            arr[field] = f(np.linspace(min(ind),max(ind), len(arr)))
            print("Interpolated")

    arr["roll"] *= factor
    arr["pitch"] *= factor
    arr["yaw"] *= factor

    return arr 

def move_csv(targets): 
    for tar in targets:
        fpath = glob.glob(f"processing/S1/*/T1/*/{tar}/*.csv")
        fpath.sort()
        
        for f in fpath:
            dirpath = 'newprocessing/' + f.split(tar)[0] + tar
            if not(os.path.isdir(dirpath)):
                os.makedirs(dirpath)
            savepath = 'newprocessing/' + f
            shutil.copy(f, savepath)
            #print(savepath)

def shorten(fn_csv, length, df=None, col_names=None, **kwargs):
    """Resample 'length' lines from a longer CSV using Pandas"""
    if df is None:
        df = pd.read_csv(fn_csv)
    if col_names is not None:
        df.columns = col_names
    
    #print("Fixing length", length, nrows)
    inds = np.round(np.linspace(0, len(df) - 1, length)).astype(int)
    df.iloc[inds].to_csv(fn_csv, index=False, **kwargs)

    return df

def get_csv_from_vid_path(vidpath, dirname, filename):
    return vidpath.replace('/RGB/',dirname).replace('_rgb_',filename).replace('.mp4','.csv')

def fix_length(vidlist):
    for vidpath in vidlist:
        vd = cv2.VideoCapture(vidpath)
        nframe = int(vd.get(cv2.CAP_PROP_FRAME_COUNT))
        vd.release()

        targets = [('CamAngle', '_cam_'), ('Eye-tracker', '_point_'), ('/DistCam2Face/', '_dcam_'), ('/DistDisp2Face/', '_ddisp_')]
        for ttt in targets:
            # Some filename missmatch expected.
            tpath = get_csv_from_vid_path(vidpath, ttt[0], ttt[1])
            try:
                shorten(tpath, nframe)
            except:
                print(f"Failed to shorten {ttt[0]} file", tpath)

def backup():
    #new_path = org_path.replace("processing", "bck_processing")
    subprocess.call(['rsync', '-a', '--ignore-existing', '--relative', '--exclude=*.mp4', "processing", "bck_processing"])

def restore():
    #new_path = org_path.replace("processing", "bck_processing")
    subprocess.call(['cp', '-r', "bck_processing/processing", "./"])

def get_headangles(image, face_landmarks):
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    # The camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])                    
    # The Distortion Matrix
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    ## Head pose ##########################
    #for face_landmarks in results.multi_face_landmarks:
    for idx, lm in enumerate(face_landmarks.landmark):
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            #if idx == 1:
            #    nose_2d = (lm.x * img_w, lm.y * img_h)
            #    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * img_w)

            x, y = int(lm.x * img_w), int(lm.y * img_h)

            face_2d.append([x, y])
            face_3d.append([x, y, lm.z])       

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    dist_matrix[:,:] = 0

    # Solve PnP
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return angles

def save_iris_dist(result, fn):
    #이미지의 시작부터 iris를 못 찾을 경우 -1로 저장된 부분을 후처리
    indexFirstCheck = np.argmax(result > 0)
    result[:indexFirstCheck] = result[indexFirstCheck]
    indexLastCheck = np.argmax(result < 0)
    result[indexLastCheck:] = result[indexLastCheck]
    
    #csv파일로 저장
    print(fn, "saving..")
    df = pd.DataFrame(result, columns = ['distance'])
    df.to_csv(fn, index=False)
    return df

def get_distance_iris(face_landmarks, irisIndex, deviceName):
    if deviceName in ['Monitor', 'VehicleLCD']:
        distance = -1   #첫 프레임에 없으면 아무 값이나.
        Focal = 3.7E-3  #카메라의 초점거리와 시야각
        Fov = 78        #각 촬영 기기마다 고유의 값을 가짐
    elif deviceName == "Tablet":
        distance = -1   #첫 프레임에 없으면 아무 값이나.
        Focal = 2.0E-3  #카메라의 초점거리와 시야각
        Fov = 120       #각 촬영 기기마다 고유의 값을 가짐
    elif deviceName == 'Smartphone':
        distance = -1   #첫 프레임에 없으면 아무 값이나.
        Focal = 2.2E-3  #카메라의 초점거리와 시야각
        Fov = 80        #각 촬영 기기마다 고유의 값을 가짐
    elif deviceName == 'Laptop':
        distance = -1   #첫 프레임에 없으면 아무 값이나.
        #노트북의 초점거리, 시야각은 임시값임 
        Focal = 3.7E-3  #카메라의 초점거리와 시야각
        Fov = 78        #각 촬영 기기마다 고유의 값을 가짐            
    irisRealSize = 11.7E-3  #사람의 실제 눈동자 크기 (편차 약 ±0.5E-3)

    pixSize = ( Focal * math.tan((Fov/2) * math.pi / 180) ) / 1920 * 2


    irisPoints = []
    for start, end in irisIndex:
        irisx = face_landmarks.landmark[start].x
        irisy = face_landmarks.landmark[start].y
        
        irisxScale = int(irisx * 1920)
        irisyScale = int(irisy * 1080)
        
        irisPoints.append([irisxScale, irisyScale])
    
    #왼쪽 눈의 좌우, 오른쪽 눈의 좌우를 활용하여 두 눈 사이의 거리를 계산   
    left1x, left1y = irisPoints[5]
    left2x, left2y = irisPoints[6]
    
    right1x, right1y = irisPoints[2]
    right2x, right2y = irisPoints[4]
    
    leftIrisSizePix = ( ((left1x-left2x) ** 2) + ((left1y-left2y) ** 2) ) ** 0.5
    rightIrisSizePix = ( ((right1x-right2x) ** 2) + ((right1y-right2y) ** 2) ) ** 0.5
    leftIrisSize = leftIrisSizePix * pixSize
    rightIrisSize = rightIrisSizePix * pixSize
    
    distanceLeft = ( Focal * irisRealSize ) / (leftIrisSize) 
    distanceRight = ( Focal * irisRealSize ) / (rightIrisSize)
    
    distance = ( distanceLeft + distanceRight ) / 2 * 100   #cm로 변환
    return distance

def load_txt_or_csv(fn):
    if os.path.isfile(fn):
        if os.path.getsize(fn) > 0:
            ddispCsv = pd.read_csv(fn)
            return ddispCsv
        else: 
            return None
    elif os.path.isfile(fn.replace(".csv",".txt")):
        if os.path.getsize(fn.replace(".csv",".txt")) > 0:
            ddispCsv = pd.read_csv(fn.replace(".csv",".txt"))
            return ddispCsv
        else:
            return None
    else:
        return None


def generate_eye_tracker(sco):
## Eye-tracker #################
    sco.set_points(frameCount)

    fn_ = videoPath.split("/")[-1]
    _, _, ID, _, scenario, device, imgtype, status, action, orientation = fn_.split("_")
    orientation = orientation.split(".mp4")[0]

    #### Eye-tracker
    # Knee -> lapTop
    #fn_ = fn_.replace("_K_", "_T_")

    #if os.path.isfile(saveName_eye) and os.path.getsize(saveName_eye) > 1000:
    #    print(">>>>>   file exists", saveName_eye)
    #    continue
    with open(saveName_eye, "w") as f:
        f.write("[point] x y \n")
        px, py = random.choice(sco.scenarios[status])()
        for xx, yy in zip(px, py):
            f.write(f"{int(xx)}, {int(yy)}\n")
    #####################################################

def load_quoted_csv(fn):
    with open(fn, "r") as f:
        ss = f.read()
    ss = ss.replace('"',"").replace("\n\n", "\n")
    lines = ss.split("\n")
    ncol = len(lines[0])
    keep = [ll for ll in lines if (len(ll) < ncol+5) * (len(ll) > 5)]
    keep = "\n".join(keep)
    df = pd.read_csv(StringIO(keep), sep=',')
    return df
    