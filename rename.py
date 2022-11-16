import glob
import os
import shutil

idDirlist = glob.glob('processing')
for idDir in idDirlist:
    print(idDirlist)
    ID = idDir.split('_')[0]
    place = 'S1'
    Try = 'T1'
    dirlist = glob.glob(f'*/*/*/*/*/*')
    dirlist.sort()
    dirlist = dirlist[::-1]
    for dirpath in dirlist:
        dirtarget = dirpath.split('/')[-1]
        if dirtarget == "xy":
            txtlist = glob.glob(dirpath + '/*_xy_*.csv')
            for txtname in txtlist:
                txtrename = txtname.replace("_xy_","_point_")
                os.rename(txtname,txtrename)
                print(txtrename,'done')
            dirrename = dirpath.replace("xy","Eye-tracker")
            os.rename (dirpath, dirrename)
            print(dirrename,'replaced')
        elif dirtarget == "DistDisp2Face":
            txtlist = glob.glob(dirpath + '/*_SS*.txt')
            for txtname in txtlist:
                txtrename = txtname.replace("_SS","_S")
                os.rename(txtname,txtrename)
                print(txtrename,'done')
        elif dirtarget == "DistCam2Face":
            txtlist = glob.glob(dirpath + '/*_dcam_*.csv')
            for txtname in txtlist:
                print(txtname)
                txtrename = txtname.replace("_dcam_","_ddisp_")
                os.rename(txtname,txtrename)
                print(txtrename,'done')
            dirrename = dirpath.replace("DistCam2Face","DistDisp2Face")
            os.rename (dirpath, dirrename)
            print(dirrename,'saving')
