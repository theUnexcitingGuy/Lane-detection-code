import cv2
import numpy as np 
import matplotlib.pyplot as plt 


#loading the image
def process_image(image):
	height = image.shape[0]
	width = image.shape[1]


	#defining the region of interest
	region_of_interest = [
		(0, 290),
		(330, 210), 
		(550, height)
	]	

	gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	canny_image = cv2.Canny(image, 100, 200)

	#crop the image leaving only the region of interest
	cropped_image = region_of_interest_mask(canny_image, np.array([region_of_interest], np.int32))

	#now we have to draw the lines
	lines = cv2.HoughLinesP(cropped_image, rho = 6, 
		theta = np.pi/60, threshold = 170, lines = np.array([]), 
		minLineLength = 40, maxLineGap = 25)

	image_with_lines = draw_the_lines(image, lines)
	return image_with_lines


#now i want to cover all areas in the image that are not the region of interest
def region_of_interest_mask(img, vertices):
	mask = np.zeros_like(img)
	#channel_count = img.shape[2]
	match_mask_color = 255
	cv2.fillPoly(mask, vertices, match_mask_color)
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

#function to draw the lines on the image
def draw_the_lines(img, lines):
	img_copy = np.copy(img)
	blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype = np.uint8)
	for line in lines:
		for x1, y1, x2, y2 in line:
			cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness = 1)
	#now i merge the two images
	img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
	return img



#read the video
cap = cv2.VideoCapture('car_on_road.mp4')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
while (cap.isOpened()):
    ret, frame = cap.read()
    
    if ret == True:
        b = cv2.resize(frame,(660,330),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        b = process_image(b)
        cv2.imshow('Frame',b)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

    
cap.release()

cv2.destroyAllWindows()
