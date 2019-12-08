import os
videopath = 'dataset'
pngpath = 'pngs'
for filepath in os.listdir(videopath):
    print(filepath)
    outpngs = os.path.join(pngpath,filepath)
    filepathv = os.path.join(videopath,filepath)
    videonames = sorted(os.listdir(filepathv))
    for i in range(350,420):
        print(videonames[i].split('.')[0])
        clipsname = videonames[i].split('.')[0]
        outpngsclips = os.path.join(outpngs,clipsname)
        os.makedirs(outpngsclips,exist_ok=True)
        videoname = os.path.join(filepathv,videonames[i])
        print('ffmpeg -i ' + videoname + ' -vsync 0 ' + os.path.join(outpngsclips, clipsname + '_%4d.png ') + '-y')
        os.system('ffmpeg -i '+ videoname +' -vsync 0 ' + os.path.join(outpngsclips,clipsname + '_%4d.png ') + '-y')
# testdast2pngs
# videopath = 'testAI4K'
# videonames = sorted(os.listdir(videopath))
# pngpath = 'testpngs'
# for i in range(50):
#     clipsname = videonames[i].split('.')[0]
#     outpngsclips = os.path.join(pngpath,clipsname)
#     os.makedirs(outpngsclips,exist_ok=True)
#     videoname = os.path.join(videopath,videonames[i])
#     print('ffmpeg -i ' + videoname + ' -vsync 0 ' + os.path.join(outpngsclips, clipsname + '_%4d.png ') + '-y')
#     os.system('ffmpeg -i '+ videoname +' -vsync 0 ' + os.path.join(outpngsclips,clipsname + '_%4d.png ') + '-y')
    # print(videonames[0])

    # print(filepathv)
