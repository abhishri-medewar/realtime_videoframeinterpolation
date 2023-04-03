import cv2
import os

video_path = './figures/Alley - 39837.mp4'
frame_store_path = "./figures/video5"

print("Video Filename: ", video_path.split('/')[-1])
video_data = cv2.VideoCapture(video_path)
count = 1
success = 1
length = int(video_data.get(cv2.CAP_PROP_FRAME_COUNT))
print("No of Frames: ", length)

while success:
    success, image = video_data.read()
    if not success:
        break
    frame_path = os.path.join(frame_store_path, "000%d.png" % count)
    cv2.imwrite(frame_path, image)
    count += 1
print("Total frames extracted: ", count)
print("Completed")