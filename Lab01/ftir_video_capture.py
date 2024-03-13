import cv2

# Choose your webcam: 0, 1, ...
cap = cv2.VideoCapture(1)

while(True):
	# Get one frame from the camera
	ret, frame = cap.read()

	# Split RGB channels
	blue_channel, green_channel, red_channel = cv2.split(frame)


	# Perform thresholding to each channel
	ret, thresh_blue = cv2.threshold(blue_channel, 150, 255, cv2.THRESH_BINARY_INV)
	ret, thresh_green = cv2.threshold(green_channel, 150, 255, cv2.THRESH_BINARY)
	ret, thresh_red = cv2.threshold(red_channel, 150, 255, cv2.THRESH_BINARY)

	# Get the final result using bitwise operation
	final_thresh = cv2.bitwise_and(thresh_blue, thresh_green)
	final_thresh = cv2.bitwise_and(final_thresh, thresh_red)

	# Find and draw contours
	
	contours, hierarchy = cv2.findContours(final_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

	# Iterate through each contour, check the area and find the center

	for contour in contours:
		area = cv2.contourArea(contour)
		if area > 100:  # You can adjust the minimum area threshold here
			M = cv2.moments(contour)
			if M["m00"] != 0:
				cX = int(M["m10"] / M["m00"])
				cY = int(M["m01"] / M["m00"])
			else:
				cX, cY = 0, 0
			# Draw the center of the shape on the image
			cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
			cv2.putText(frame, "center", (cX - 20, cY - 20),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	# Show the frame
	cv2.imshow('frame', frame)

	# Press q to quit
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()