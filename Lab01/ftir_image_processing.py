import cv2

# Read the image
frame = cv2.imread("Lab01/test2.png")
frame  = cv2.blur(frame, (45, 45))

# Split RGB channels
b, g, r = cv2.split(frame)

# Perform thresholding to each channel
# Here we use a binary threshold as an example. You can adjust the threshold value and the max value.
_, thresh_b = cv2.threshold(b, 210, 255, cv2.THRESH_BINARY_INV)
_, thresh_g = cv2.threshold(g, 210, 255, cv2.THRESH_BINARY_INV)
_, thresh_r = cv2.threshold(r, 230, 255, cv2.THRESH_BINARY)

cv2.imshow('b', thresh_b)
cv2.waitKey(0)
cv2.imshow('g', thresh_g)
cv2.waitKey(0)
cv2.imshow('r', thresh_r) 
cv2.waitKey(0)

# Get the final result using bitwise operation
# Here we use bitwise AND operation to combine the thresholded channels
final_thresh = cv2.bitwise_or(thresh_b, thresh_g)
final_thresh = cv2.bitwise_and(final_thresh, thresh_r)


cv2.imshow('frame', final_thresh)
cv2.waitKey(0)
# Find and draw contours
contours, hierarchy = cv2.findContours(final_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

# Iterate through each contour, check the area and find the center
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 400:  # You can adjust the minimum area threshold here
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

# Press any key to quit
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()