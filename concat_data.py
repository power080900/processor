import pandas as pd
import glob

target_dir = '/media/di/data/processor/summary/???.csv'
target_file_list = glob.glob(target_dir)
target_file_list.sort()
df = pd.DataFrame()

for file_name in target_file_list:
    file = pd.read_csv(file_name,index_col=None)
    df = pd.concat([df,file])
    print(df)
df.to_csv('/media/di/data/processor/summary/total_data.csv',mode='w',index=None)