import glob
import pandas as pd
import os

# root_dir = "/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/1122/processing/S1/163/T1/*/*/*_rgb_*"
# dir_list = glob.glob(root_dir)
# dir_list.sort()
target_dir = "/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/????/processing/S1/"
# target_dir = "/media/di/data/????/processing/S1/"
target_id = target_dir+"*"
target_list = glob.glob(target_id)
target_list.sort(reverse=True)
table = pd.read_csv('/media/di/data/summary/statistics_table.csv')

for id in target_list[71:]:
    df = pd.read_csv('/media/di/data/summary/statistics_frame.csv')
    target_date = id.split('/')[-4]
    id = id.split('/')[-1] 
    
    print("now",id,"saving")
    cnt = 0
    for device, condition in zip(table['device'] , table['condition']):
        if device == "Smartphone" or device == "Tablet":
            scenario = ''
            scenario_before = scenario
            scenario_dir = f'/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/{target_date}/processing/S1/{id}/*/'
            # scenario_dir = f'/media/di/data/{target_date}/processing/S1/{id}/*/'
            try :
                scenario = ''.join(glob.glob(scenario_dir+f'{device}/RGB/*{condition}.mp4')).split('_')[-6]
            except:
                scenario = scenario_before
            rows = [id, device, scenario, condition]
            for row_column in table.columns[4:]:
                row_main_column = row_column.split('_')[0]
                row_sub_column = row_column.split('_')[-2]
                target_name = f'/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/{target_date}/processing/S1/{id}/*/{device}/{row_main_column}/*_{scenario}_*_{condition}.csv'
                # target_name = f'/media/di/data/{target_date}/processing/S1/{id}/*/{device}/{row_main_column}/*_{scenario}_*_{condition}.csv'
                target_name = glob.glob(target_name)
                try:
                    if os.path.exists("".join(target_name)) and os.path.getsize("".join(target_name)) > 0:
                        if row_main_column == 'FaceAngle' or row_main_column == 'CamAngle':
                            target_file = pd.read_csv("".join(target_name),names=['roll','pitch','yaw'],skiprows=1)
                        elif row_main_column == 'Eye-tracker':
                            target_file = pd.read_csv("".join(target_name),names=['x','y'],skiprows=1)
                        else:
                            target_file = pd.read_csv("".join(target_name),names=['distance'],skiprows=1)
                        stat_max = max(target_file[row_sub_column])
                        stat_min = min(target_file[row_sub_column])
                        stat_mean = round(target_file[row_sub_column].mean(),2)
                        stat_stad = round(target_file[row_sub_column].std(),2)
                #         target_row = len(target_file)
                    else:
                        stat_max = 'None'
                        stat_min = 'None'
                        stat_mean = 'None'
                        stat_stad = 'None'
                except:
                    stat_max = 'error'
                    stat_min = 'error'
                    stat_mean = 'error'
                    stat_stad = 'error'
                    # target_row = 0
            # print(stat_max,stat_min,stat_mean,stat_stad)
                stats = [stat_max,stat_min,stat_mean,stat_stad]
                rows = rows + stats
            rows = pd.DataFrame(rows).transpose()
            rows.columns = df.columns
            df = pd.concat([df,rows])
            cnt += 1
            print(cnt,"/ 453",device,scenario,condition)
        else:
            for i in range(3):
                if (int(id)+i) % 10 == 0 :
                    scenario = "S10"
                else :
                    scenario = "S"+str((int(id)+i) % 10).zfill(2)    
                rows = [id, device, scenario, condition]
                for row_column in table.columns[4:]:
                    row_main_column = row_column.split('_')[0]
                    row_sub_column = row_column.split('_')[-2]
                    target_name = f'/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/{target_date}/processing/S1/{id}/*/{device}/{row_main_column}/*_{scenario}_*_{condition}.csv'
                    # target_name = f'/media/di/data/{target_date}/processing/S1/{id}/*/{device}/{row_main_column}/*_{scenario}_*_{condition}.csv'
                    target_name = glob.glob(target_name)
                    try:
                        if os.path.exists("".join(target_name)) and os.path.getsize("".join(target_name)) > 0:
                            if row_main_column == 'FaceAngle' or row_main_column == 'CamAngle':
                                target_file = pd.read_csv("".join(target_name),names=['roll','pitch','yaw'],skiprows=1)
                            elif row_main_column == 'Eye-tracker':
                                target_file = pd.read_csv("".join(target_name),names=['x','y'],skiprows=1)
                            else:
                                target_file = pd.read_csv("".join(target_name),names=['distance'],skiprows=1)
                            stat_max = max(target_file[row_sub_column])
                            stat_min = min(target_file[row_sub_column])
                            stat_mean = round(target_file[row_sub_column].mean(),2)
                            stat_stad = round(target_file[row_sub_column].std(),2)
                    #         target_row = len(target_file)
                        else:
                            stat_max = 'None'
                            stat_min = 'None'
                            stat_mean = 'None'
                            stat_stad = 'None'
                    except:
                        stat_max = 'error'
                        stat_min = 'error'
                        stat_mean = 'error'
                        stat_stad = 'error'
                    stats = [stat_max,stat_min,stat_mean,stat_stad]
                    rows = rows + stats
                rows = pd.DataFrame(rows).transpose()
                rows.columns = df.columns
                df = pd.concat([df,rows])
                cnt += 1
                print(cnt,"/ 453",device,scenario,condition)
    df.to_csv(f'/media/di/data/summary/statistics/{id}_statistics.csv',mode="w",index=None)
    print(id,"save done")