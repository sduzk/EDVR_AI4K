import os
videopath = '../savevideo'
if not os.path.exists(videopath):
    os.mkdir(videopath)

imgpath = '/input/EDVR-dev/results/'
clip_pngs = sorted(os.listdir(imgpath))
for clip_png_name in clip_pngs:
    clip_png_path = os.path.join(imgpath,clip_png_name)
    os.system('ffmpeg -r 24000/1001 -i ' + clip_png_path + '/{}_%4d.png'.format(clip_png_name) +
              ' -vcodec libx265 -pix_fmt yuv422p -crf 10 ' + videopath + '/{}.mp4'.format(clip_png_name))
