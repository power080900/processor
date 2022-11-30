import glob
import pandas as pd

table = pd.read_csv("/media/di/data/processor/summary/base_table.csv")
columns = list(table.columns[1:])
root_dir = "/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/????/processing/S1/"
device_list = ["Laptop","Monitor","Smartphone","Tablet","VehicleLCD"]

for id in range(160,168):
    id = str(id).zfill(3)
    data_row = []
    data_row.append(id)
    print(id)
    for column in columns:
        if column == 'RGB' or column == 'IR':
            target_dir = root_dir+ f"{id}/*/*/{column}/*.mp4"
            row = len(glob.glob(target_dir))
            data_row.append(row)
        else:
            target_dir = root_dir+ f"{id}/*/*/{column}/*.csv"
            row = len(glob.glob(target_dir))
            data_row.append(row)
    df = pd.DataFrame(data_row).transpose()
    df.columns = table.columns
    print(df)
    table = pd.concat([table,df])
table.to_csv('/media/di/data/processor/summary/data_count.csv',mode="a",index=None)
    
# for id in range(3,5):
#     id = str(id).zfill(3)
#     for device in device_list:
#         data_row = []
#         data_row.append(id)
#         data_row.append(device)
#         for column in columns:
#             if column == 'RGB' or column == 'IR':
#                 target_dir = root_dir+f"{id}/*/{device}/{column}/*.avi"
#                 # target_dir2 = root_dir+f"{id}/*/{device}/{column}/*.mp4"
#                 row = len(glob.glob(target_dir))
#                 data_row.append(row)
#             else:
#                 target_dir = root_dir+ f"{id}/*/{device}/{column}/*.csv"
#                 row = len(glob.glob(target_dir))
#                 data_row.append(row)
#         df = pd.DataFrame(data_row).transpose()
#         df.columns = table.columns
#         table = pd.concat([table,df])
# table.to_csv('/media/di/data/processor/summary/data_count_detail.csv',mode="a",index=None)