import glob
import pandas as pd

idDirlist = glob.glob('processing/*/*/*/*/*/*')
idDirlist.sort()
count = []
for idDir in idDirlist:
    idDir = idDir.replace(".mp4","").replace(".csv","")
    file = idDir.split('/')[-1]
    nowstat = file.split('_')[2,4,-3:]
    nowstat = "_".join(nowstat)
    if len(nowstat) == 5:
        count.append(nowstat)
    df = pd.DataFrame(count)
df.columns =["status"]
df['count'] = 1
print(df)
print(df.groupby("status").sum())

    