import glob
import pandas as pd
import os

# root_dir = "/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/1122/processing/S1/163/T1/*/*/*_rgb_*"
# dir_list = glob.glob(root_dir)
# dir_list.sort()
target_dir = "/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/????/processing/S1/"
target_id = target_dir+"*"
target_list = glob.glob(target_id)
target_list.sort()
table = pd.read_csv('/media/di/data/processor/summary/data_table.csv')

for id in target_list[3:80]:
    df = pd.read_csv('/media/di/data/processor/summary/data_frame.csv')
    target_date = id.split('/')[8]
    id = id.split('/')[-1] 
    
    print("now",id,"saving")
    cnt = 0
    for device, condition in zip(table['device'] , table['condition']):
        if device == "Smartphone" or device == "Tablet":
            # if int(id) % 10 == 0 :
            #     scenario = "S10"
            # else :
            #     scenario = "S"+str(int(id) % 10).zfill(2)
            scenario_before = scenario
            scenario_dir = f'/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/{target_date}/processing/S1/{id}/*/'
            try :
                scenario = ''.join(glob.glob(scenario_dir+f'{device}/RGB/*{condition}.mp4')).split('_')[-6]
            except:
                scenario = scenario_before
            rows = [id, device, scenario, condition]
            for row_column in table.columns[4:]:
                if row_column == "RGB" or row_column == "IR":
                    target_name = scenario_dir+f'{device}/{row_column}/*_{scenario}_*_{condition}.mp4'
                    target_name = glob.glob(target_name)
                    if os.path.exists("".join(target_name)) and os.path.getsize("".join(target_name)) > 0:
                        target_row = 1
                    else:
                        target_row = 0
                else:
                    target_name = f'/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/{target_date}/processing/S1/{id}/*/{device}/{row_column}/*_{scenario}_*_{condition}.csv'
                    target_name = glob.glob(target_name)
                    if os.path.exists("".join(target_name)) and os.path.getsize("".join(target_name)) > 0:
                        target_file = pd.read_csv("".join(target_name))
                        target_row = len(target_file)
                    else:
                        target_row = 0
                rows.append(target_row)
            rows = pd.DataFrame(rows).transpose()
            rows.columns = df.columns
            df = pd.concat([df,rows])
            cnt += 1
            print(cnt,"/ 453")
        else:
            for i in range(3):
                if (int(id)+i) % 10 == 0 :
                    scenario = "S10"
                else :
                    scenario = "S"+str((int(id)+i) % 10).zfill(2)    
                rows = [id, device, scenario, condition]
                for row_column in table.columns[4:]:
                    if row_column == "RGB" or row_column == "IR":
                        target_name = f'/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/{target_date}/processing/S1/{id}/*/{device}/{row_column}/*_{scenario}_*_{condition}.mp4'
                        target_name = glob.glob(target_name)
                        if os.path.exists("".join(target_name)) and os.path.getsize("".join(target_name)) > 0:
                            target_row = 1
                        else:
                            target_row = 0
                    else:
                        target_name = f'/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/{target_date}/processing/S1/{id}/*/{device}/{row_column}/*_{scenario}_*_{condition}.csv'
                        target_name = glob.glob(target_name)
                        if os.path.exists("".join(target_name)) and os.path.getsize("".join(target_name)) > 0:
                            target_file = pd.read_csv("".join(target_name))
                            target_row = len(target_file)
                        else:
                            target_row = 0
                    rows.append(target_row)
                rows = pd.DataFrame(rows).transpose()
                rows.columns = df.columns
                df = pd.concat([df,rows])
                cnt += 1
                print(cnt,"/ 453")
    df.to_csv(f'/media/di/data/processor/summary/{id}.csv',mode="w",index=None)
    print(id,"save done")