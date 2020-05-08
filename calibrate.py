

import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image



chess_s = (9,6)







re_points = [] 
im_points = [] 
objp = np.zeros((np.prod(chess_s),3),dtype=np.float32)
objp[:,:2] = np.mgrid[0:chess_s[0], 0:chess_s[1]].T.reshape(-1,2)



img_path = glob.glob('./calibration_images/*')


for image_path in tqdm(img_path):

	#Load image
	image = cv2.imread(image_path)
	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print("looking for chessboard")
	#find chessboard corners
	ret,corners = cv2.findChessboardCorners(gray_image, chess_s, None)

	if ret == True:
		print("Chessboard detected!")
		print(image_path)
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		cv2.cornerSubPix(gray_image, corners, (5,5), (-1,-1), criteria)
		re_points.append(objp)
		im_points.append(corners)


ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(re_points, im_points,gray_image.shape[::-1], None, None)


np.save("./camera_params/ret", ret)
np.save("./camera_params/K", K)
np.save("./camera_params/dist", dist)
np.save("./camera_params/rvecs", rvecs)
np.save("./camera_params/tvecs", tvecs)

exif_img = PIL.Image.open(img_path[0])

exif_data = {
	PIL.ExifTags.TAGS[k]:v
	for k, v in exif_img._getexif().items()
	if k in PIL.ExifTags.TAGS}
focal_length_exif = exif_data['FocalLength']
focal_length = focal_length_exif[0]/focal_length_exif[1]
np.save("./camera_params/FocalLength", focal_length)




mean_error = 0
for i in range(len(re_points)):
	im_points2, _ = cv2.projectPoints(re_points[i],rvecs[i],tvecs[i], K, dist)
	error = cv2.norm(im_points[i], im_points2, cv2.NORM_L2)/len(im_points2)
	mean_error += error

total_error = mean_error/len(re_points)
print (total_error)
