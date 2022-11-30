from all_in_one_utils import *
import os

pwd = os.getcwd()
print("PWD", pwd)
targetIdList = glob.glob("processing/S1/*")
targetIdList.sort()
for targetId in targetIdList:
    deviceList = glob.glob(f"{targetId}/T1/*")
    #deviceList.sort()
    print(deviceList)
    for device_path in deviceList[2:]:
        videoPathList = glob.glob(f"{device_path}/RGB/*.mp4")
        videoPathList.sort()    
        camAnglePathList = [videoPath.replace("/RGB/","/CamAngle/").replace('_rgb_','_cam_').replace('.mp4','.csv') for videoPath in videoPathList]
        camAnglePathList.sort()
        CamAngleCsv = None
        missing_camlist =[]
        for camAnglePath in camAnglePathList:
            print(camAnglePath.replace('.csv','.txt'))
            print(camAnglePath)
            if os.path.exists(camAnglePath):
                if os.path.getsize(camAnglePath) > 10:
                    CamAngleCsv = load_quoted_csv(camAnglePath)
                    print(camAnglePath)
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
                    print(camAnglePath)
                    print(CamAngleCsv)

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
            else:
                missing_camlist.append(camAnglePath)
        print(missing_camlist)
        # Copy if missing CamAngle
        # if deviceName in ['Monitor', 'VehicleLCD', 'Laptop']:
        #     for camAnglePath in missing_camlist:
        #         print('campath:',type(camAnglePath))
        #         CamAngleCsv.iloc[inds,1:4].to_csv(camAnglePath, index=False)

        # if len(missing_camlist) > 0 and deviceName in ['Tablet', 'Smartphone']:
        #     logging.warning(f"Missing CamAngle: {missing_camlist}")