import cv2
import numpy as np 
import matplotlib.pyplot as plt 


#loading the image
image = cv2.imread('image.png')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height = rgb_image.shape[0]
width = rgb_image.shape[1]
print(rgb_image.shape)

#defining the region of interest
region_of_interest = [
	(0, 1200),
	(width/2, 750), 
	(2000, 1200)
]	


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
			cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), thickness = 15)
	#now i merge the two images
	img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
	return img



#create the gray image
gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
canny_image = cv2.Canny(gray_image, 100, 200)

#crop the image leaving only the region of interest
cropped_image = region_of_interest_mask(canny_image, np.array([region_of_interest], np.int32))

#now we have to draw the lines
lines = cv2.HoughLinesP(cropped_image, rho = 6, 
	theta = np.pi/60, threshold = 170, lines = np.array([]), 
	minLineLength = 40, maxLineGap = 25)

image_with_lines = draw_the_lines(rgb_image, lines)


#display the image
plt.imshow(image_with_lines)
plt.show()