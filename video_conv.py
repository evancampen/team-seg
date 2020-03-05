import cv2

vidcap = cv2.VideoCapture('video2.mp4')
success, image = vidcap.read()
count = 0
while success:
    cv2.imwrite("video_frame2/frame%d.jpg" % count, image)
    success, image = vidcap.read()
    print('READ a new frame: ', success)
    count+=1
