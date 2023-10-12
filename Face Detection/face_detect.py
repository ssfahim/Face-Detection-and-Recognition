import cv2 as cv

img = cv.imread("groupOf5.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Verify the path to the Haar Cascade XML file
cascade_path = 'haarcascade_frontalface_default.xml'

# Load the Haar Cascade classifier
haar_cascade = cv.CascadeClassifier(cascade_path)

if haar_cascade.empty():
    raise Exception(f"Error loading cascade classifier: {cascade_path}")

# Adjust the scaleFactor and minNeighbors values
faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=1)


for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

cv.imshow(f'Detected Faces & Number of faces found = {len(faces_rect)}', img)
cv.waitKey(0)
