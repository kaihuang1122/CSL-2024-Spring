import cv2
import numpy as np

import keras

model = keras.saving.load_model("digit_recognization.keras")

# Choose your webcam: 0, 1, ...
cap = cv2.VideoCapture("2024-03-13 17-33-58.mkv")
fr = int(cap.get(cv2.CAP_PROP_FPS))
plain_size = (256, 256)
plain_digit = np.zeros((*plain_size, 3), dtype=np.uint8)
digit_size = (28, 28)
digit_img = np.zeros(digit_size, dtype=np.uint8)
last_time = 0
current_time = 0

predicted = -1
while True:
	# Get one frame from the camera
	ret, frame = cap.read()
	if not ret:
		break
	current_time += 1
	frame = frame[:, 290:1370, :]
	frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
	frame = cv2.flip(frame, 1)
	frame = cv2.resize(frame, plain_size, interpolation=cv2.INTER_AREA)
	frame = cv2.blur(frame, (3, 3))

	# Split RGB channels
	b, g, r = cv2.split(frame)

	# Perform thresholding to each channel
	_, thresh_b = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY_INV)
	_, thresh_g = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY_INV)
	_, thresh_r = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)

	# Get the final result using bitwise operation
	final_thresh = cv2.bitwise_or(thresh_b, thresh_g)
	final_thresh = cv2.bitwise_and(final_thresh, thresh_r)

	# Find and draw contours
	
	contours, hierarchy = cv2.findContours(final_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	is_valid_mark = False

	# Iterate through each contour, check the area and find the center
	for contour in contours:
		area = cv2.contourArea(contour)
		# You can adjust the minimum area threshold here

		if area >= 50:
			is_valid_mark = True
			cv2.drawContours(plain_digit, [contour], -1, (255, 0, 0), -1)
			cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
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
			
	cv2.imshow("frame", frame)
	digit_img = cv2.resize(plain_digit[..., 0], digit_size, interpolation=cv2.INTER_AREA)

	if is_valid_mark:
		last_time = current_time
	else:
		interval = current_time - last_time
		if interval >= 10 and np.any(plain_digit):
			x = digit_img.astype(np.float32) / 255. 
			x = np.expand_dims(x, (0, -1))
			predicted = np.argmax(model(x)[0])
			last_time = current_time
			plain_digit = np.zeros((*plain_size, 3), dtype=np.uint8)


	out = cv2.resize(digit_img, (512, 512))[..., np.newaxis].repeat(3, axis=-1)
	cv2.putText(out, f"Predict: {predicted}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 2)

	# Show the frame
	
	cv2.imshow("output", out)

	# Press q to quit
	if cv2.waitKey(fr) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()