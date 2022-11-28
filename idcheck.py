import glob
import pandas as pd
import os

device_list = ["Laptop","Monitor","Smartphone","Tablet","VehicleLCD"]
df_columns = ['id','date','device','scenario','condition','CamAngle','DistCam2Face','DistDisp2Face','Eye-tracker','FaceAngle']
root_dir = "/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/1028/processing/S1/"
id_list = glob.glob(root_dir+'*')
id_list.sort()

for id in id_list:
    df = pd.DataFrame()
    id = id.split("/")[-1]
    for device in device_list:
        target_condition=[]
        target_dir_list = glob.glob(root_dir+f"{id}/T1/{device}/RGB/*.mp4")
        target_dir_list.sort()
        for target_dir in target_dir_list:
            condition_list = target_dir.split("/")[-1]
            condition = "_".join(condition_list.split("_")[-3:]).replace(".mp4",".csv")
            scenario = condition_list.split("_")[4]
            dev = condition_list.split("_")[-5]
            date = target_dir.split("/")[-8]
            df_target = glob.glob(root_dir + f"{id}/T1/{device}/*/*_{id}_*_{scenario}_{dev}_*_{condition}")
            df_target.sort()
            rows = [id,date,device,scenario,condition.replace(".csv","")]
            for i in df_target:
                if os.path.exists(i):
                    row = len(pd.read_csv(i))
                    rows.append(str(row))
                else:
                    row = 0
                    rows.append(str(row))
            print(rows)
            df = df.append(pd.Series(rows),ignore_index=True)
    df.columns = df_columns
    df.to_csv(f"summary/{id}.csv",mode="a",index=None)
        
# for device in device_list:
#     print(device)
#     for category in categories:
#         target_dir_list = glob.glob(f"/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/08??/processing/S1/???/T1/{device}/{category}/*.csv")
#         target_dir_list.sort()
#         for target_dir in target_dir_list:
#             try:
#                 print("now:",target_dir)
#                 category = target_dir.split("/")[-2]
#                 condition_list = target_dir.split("/")[-1]
#                 scenario = condition_list.split("_")[4]
#                 condition = "_".join(condition_list.split("_")[-3:]).replace(".csv","")
#                 id = target_dir.split("/")[-5]
#                 date = target_dir.split("/")[-8]
#                 row = len(pd.read_csv(target_dir))
#                 df = pd.DataFrame(id,date,device,scenario,condition)
#             except:
#                 print(target_dir,"error")
#                 bad.append(target_dir)
# with open(f"bad.csv", "a") as f:
#     for i in bad:
#         f.write(f"{i}\n")
