from genericpath import isfile
import os
import shutil
import glob
import logging
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from all_in_one_utils import *
from eyetracker import Scenario

if __name__ == '__main__':
    pwd = os.getcwd()
    print("PWD", pwd)
    logging.basicConfig(filename = f'{pwd}_log.txt',
                        level = logging.WARNING,
                        format = '%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    logging.warning("Start")

    # ready Mediapipe 
    mp_face_mesh = mp.solutions.face_mesh
    irisIndex = mp_face_mesh.FACEMESH_IRISES
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, 
                                refine_landmarks=True,
                                min_detection_confidence=0.5, 
                                min_tracking_confidence=0.5)

    targetIdList = glob.glob("processing/S1/*")
    targetIdList.sort()
    print(targetIdList)

    # backup first
    if True:
       for targetId in targetIdList:
           if not os.path.isdir("bck_processing"):
               backup()
           else:
               restore()

    #deviceList = ["Monitor", "Laptop", "VehicleLCD", "Tablet", "Smartphone"]
    for targetId in targetIdList:
        deviceList = glob.glob(f"{targetId}/T1/*")
        #deviceList.sort()
        print(deviceList)
        for device_path in deviceList:
            # duplicate "K" and "T"
            K_csvList = glob.glob(f"{device_path}/*/*_K_?.csv")
            for K_csv in K_csvList:
                shutil.copyfile(K_csv, K_csv.replace("_K_", "_T_"))

            os.makedirs(f"{device_path}/DistCam2Face", exist_ok=True)
            os.makedirs(f"{device_path}/DistDisp2Face", exist_ok=True)
            os.makedirs(f"{device_path}/CamAngle", exist_ok=True)
            os.makedirs(f"{device_path}/FaceAngle", exist_ok=True)
            os.makedirs(f"{device_path}/Eye-tracker", exist_ok=True)
            # 모니터, 차량의 경우 DEPTH 센서를 가지고 있고, Gyro는 파일 중 소수만 데이터가 제대로 들어가 있다.
            # DEPTH 센서로 들어온 값의 경우 DistDisp2Face dir에 txt로 존재하며, Gyro 데이터의 경우 CamAngle dir에 txt로 존재한다.
            # 모니터, 차량의 코드 처리 순서는 Distance Cam 생성, Distance Display csv 처리, CamAngle csv 처리 순서로 진행된다.
            deviceName = device_path.split('/')[-1]

            #frameCountDict = {}
            
            #Distance Cam 생성
            videoPathList = glob.glob(f"{device_path}/RGB/*.mp4")
            videoPathList.sort()

            # Eye-tracker 데이터 생성
            #sco = Scenario()
            
            for videoPath in videoPathList:
                saveName_dcam = videoPath.replace("/RGB/","/DistCam2Face/").replace('_rgb_','_dcam_').replace('.mp4','.csv')
                saveName_camangle = videoPath.replace("/RGB/","/CamAngle/").replace('_rgb_','_cam_').replace('.mp4','.csv')
                saveName_head = videoPath.replace("/RGB/","/FaceAngle/").replace('_rgb_','_head_').replace('.mp4','.csv')
                saveName_eye = videoPath.replace("/RGB/","/Eye-tracker/").replace('_rgb_','_point_').replace('.mp4','.csv')
                saveName_ddisp = videoPath.replace("/RGB/","/DistDisp2Face/").replace('_rgb_',"_ddisp_").replace('.mp4','.csv')
                #########################################################
                
                dcamCsv = load_txt_or_csv(saveName_dcam)

                #########################################################
                #########################################################
                #########################################################
                video = cv2.VideoCapture(videoPath)

                if not video.isOpened():
                    logging.warning(f"Can't open video: {videoPath}")
                    continue

                frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                #frameCountDict[videoPath] = frameCount
                if frameCount < 200:
                    logging.warning(f"frameCount is less than 200: {videoPath}")
                    break
                
                print("FRAME COUNT", frameCount)
                headposes = np.zeros(frameCount, dtype=[("roll", float), 
                                        ("pitch", float),
                                        ("yaw", float)])
                
                resultDistance = np.zeros(frameCount, dtype=float)
                #MediaPipe로 distance 계산, 추후 보정이 필요할 수 있음
                i  = 0    
                
                while(video.isOpened()):
                    # read a frame
                    success, image = video.read()
                    if not(success):
                        break
                    try: #해당 image에서 iris를 못 찾을 경우 except의 코드 실행
                        image.flags.writeable = False
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(image)
                        
                        # To improve performance
                        image.flags.writeable = True

                        if len(results.multi_face_landmarks) > 1:
                            print("Too many faces")
                            break
                        else:
                            face_landmarks = results.multi_face_landmarks[0]
                            angles = get_headangles(image, face_landmarks)
                        
                        headposes['roll'][i]  = angles[0]
                        headposes['pitch'][i] = angles[1]
                        headposes['yaw'][i]   = angles[2]
                        ## Head pose ##########################
                        
                        if dcamCsv is None:
                            # DCam
                            try:
                                distance = get_distance_iris(face_landmarks, irisIndex, deviceName)
                                resultDistance[i] = int(distance)
                            except:                            
                                resultDistance[i] = int(distance)
                    except:     #iris를 못 찾을 경우 이전 값을 그대로 사용
                        
                        continue
                    i+=1
                
                video.release()

                ####################################
                ## FaceANgle
                ####################################
                ## Post process                 
                factor = 360*10 # Temporary...
                process_head(headposes, factor)

                ### Save head pose 
                with open(saveName_head, "w") as f:
                    f.write("[head angle] roll      pitch      yaw\n")
                    for roll, pitch, yaw in headposes:
                        f.write(f"{roll:.4f}, {pitch:.4f}, {yaw:.4f}\n")

                print(saveName_head, "done")

                #ddisp가 있을 확률이 높고, dcam이 없음. 

                ####################################
                ## DDisp
                ## copy dcam to ddisp except vehicle
                ####################################
                
                has_ddisp = True

                ddispCsv=load_txt_or_csv(saveName_ddisp)
                #print("ddispCsv", ddispCsv)
                if ddispCsv is None:
                    if deviceName != "VehicleLCD":
                        ddispCsv = save_iris_dist(resultDistance, saveName_ddisp)
                    else:
                        logging.warning(f"{deviceName} has no ddisp file: {saveName_ddisp}")
                        has_ddisp = False                    
                    #nrows = len(ddispCsv)
                else:
                    ddispCsv = shorten(saveName_ddisp, frameCount, 
                            df=ddispCsv, 
                            col_names=["distance"])

                    #만들기
                    ###################################
                    ###################################
                    ###################################

                    #Distance Display csv 처리 (모니터, 차량의 경우 Depth 센서가 존재함)

                ####################################
                ## Dcam
                ####################################
                if deviceName != "VehicleLCD":
                    if dcamCsv is None:
                        shorten(saveName_dcam, frameCount, 
                                    df = ddispCsv, 
                                    col_names=["distance"])
                    else:
                        print("Make Ddisp first", saveName_dcam)
                        pass
                else:
                    dcamCsv = save_iris_dist(resultDistance, saveName_dcam)
                    shorten(saveName_dcam, frameCount, 
                                    df = dcamCsv, 
                                    col_names=["distance"])
                    logging.warning(f"{deviceName} has no dcam file: {saveName_dcam}")


                ####################################
                ## Eye-tracker
                ####################################
                if bad_record(saveName_eye):
                    sco = Scenario()
                    sco.set_points(frameCount)

                    fn_ = videoPath.split("/")[-1]
                    _, _, ID, _, scenario, device, imgtype, status, action, orientation = fn_.split("_")
                    orientation = orientation.split(".mp4")[0]

                    with open(saveName_eye, "w") as f:
                        f.write("x,y \n")
                        px, py = random.choice(sco.scenarios[status])()
                        for xx, yy in zip(px, py):
                            f.write(f"{int(xx)}, {int(yy)}\n")
                    
                    logging.warning(f"Bad Eye-tracker: {saveName_eye}")
                    # shorten(saveName_eye, frameCount, col_names=["x", "y"])
                else:
                    try:
                        EyeCsv = load_quoted_csv(saveName_eye)
                        EyeCsv.columns=['x', 'y']
                        nlines = len(pd.read_csv(saveName_eye.replace("/Eye-tracker/", "/FaceAngle/").replace('_point_','_head_')))
                        inds = np.round(np.linspace(0, len(EyeCsv) - 1, nlines)).astype(int)
                        EyeCsv.iloc[inds].to_csv(saveName_eye, index=False)
                        shorten(saveName_eye, frameCount, col_names=["x", "y"])
                    except:
                        logging.warning(f"Bad Eye-tracker: {saveName_eye}")
            
            ####################################
            ## CamAngle
            ## 
            ## 한 device 중 정상 파일이 하나만 있으면 나머지는 복사.
            ####################################
            camAnglePathList = [videoPath.replace("/RGB/","/CamAngle/").replace('_rgb_','_cam_').replace('.mp4','.csv') for videoPath in videoPathList]
            camAnglePathList.sort()

            CamAngleCsv = None
            missing_camlist =[]
            for camAnglePath in camAnglePathList:
                # print(camAnglePath.replace('.csv','.txt'))
                # print(camAnglePath)
                if os.path.exists(camAnglePath):
                    if os.path.getsize(camAnglePath) > 10:
                        CamAngleCsv = load_quoted_csv(camAnglePath)
                        # print(camAnglePath)
                        try:
                            CamAngleCsv.columns=['dummy', 'roll','pitch','yaw', 'dummy']
                        except:
                            CamAngleCsv.columns=['roll','pitch','yaw']
                        nlines = len(pd.read_csv(camAnglePath.replace("/CamAngle/", "/FaceAngle/").replace('_cam_','_head_')))
                        inds = np.round(np.linspace(0, len(CamAngleCsv) - 1, nlines)).astype(int)
                        CamAngleCsv.iloc[inds,1:4].to_csv(camAnglePath, index=False)
                    else:
                        # Delete empty files
                        os.remove(camAnglePath)
                        missing_camlist.append(camAnglePath)
                elif os.path.exists(camAnglePath.replace('.csv','.txt')):
                    camAnglePath = camAnglePath.replace('.csv','.txt')
                    if os.path.getsize(camAnglePath) > 10:
                        CamAngleCsv = load_txt_or_csv(camAnglePath)
                        # print(camAnglePath)
                        # print(CamAngleCsv)

                        try:
                            CamAngleCsv.columns=['dummy', 'roll','pitch','yaw', 'dummy']
                        except:
                            CamAngleCsv.columns=['roll','pitch','yaw']
                        nlines = len(pd.read_csv(camAnglePath.replace("/CamAngle/", "/FaceAngle/").replace('_cam_','_head_').replace('.txt','.csv')))
                        inds = np.round(np.linspace(0, len(CamAngleCsv) - 1, nlines)).astype(int)
                        CamAngleCsv.iloc[inds,1:4].to_csv(camAnglePath.replace('.txt','.csv'), index=False)
                    else:
                        # Delete empty files
                        os.remove(camAnglePath)
                        missing_camlist.append(camAnglePath.replace('.txt','.csv'))
                else:
                    missing_camlist.append(camAnglePath)

            # Copy if missing CamAngle
            if deviceName in ['Monitor', 'VehicleLCD', 'Laptop']:
                for camAnglePath in missing_camlist:
                    # print('campath:',type(camAnglePath))
                    nlines = len(pd.read_csv(camAnglePath.replace("/CamAngle/", "/FaceAngle/").replace('_cam_','_head_').replace('.txt','.csv')))
                    inds = np.round(np.linspace(0, len(CamAngleCsv) - 1, nlines)).astype(int)
                    CamAngleCsv.iloc[inds,1:4].to_csv(camAnglePath, index=False)

            if len(missing_camlist) > 0 and deviceName in ['Tablet', 'Smartphone']:
                logging.warning(f"Missing CamAngle: {missing_camlist}")

    # vidlist = glob.glob("pro*/S1/*/T1/*/RGB/*.mp4")
    # vidlist.sort()
    # fix_length(vidlist)
