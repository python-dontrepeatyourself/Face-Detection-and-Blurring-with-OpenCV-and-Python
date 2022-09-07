import cv2

# load the input image and convert it to grayscale
image = cv2.imread("image1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# load the face detector
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# detect the faces in the grayscale image
face_rects = face_detector.detectMultiScale(gray, 1.04, 5, minSize=(30, 30))

# go through the face bounding boxes
for (x, y, w, h) in face_rects:
    # get the region of the face
    face_roi = image[y:y + h, x:x + w]

    # apply blurring to the part of the face
    blurred = cv2.medianBlur(face_roi, 99)
    # replace the part of the face with the blurred image  
    image[y:y+h, x:x+w] = blurred

cv2.imshow("Image", image)
cv2.waitKey(0)
