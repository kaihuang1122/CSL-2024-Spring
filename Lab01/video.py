import cv2

# Choose your webcam: 0, 1, ...
cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter.fourcc(*"mp4v")
out = cv2.VideoWriter("abc.mp4", fourcc, 30.0, (1280, 720))


while(True):
	# Get one frame from the camera
	ret, frame = cap.read()
	cv2.imshow("frame", frame)
	out.write(frame)

	# Press q to quit
	if cv2.waitKey(30) & 0xFF == ord('q'):
		break

# Release the camera
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()