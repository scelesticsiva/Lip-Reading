import dlib
import cv2
import numpy as np

ROOT_PATH = "/home/scelesticsiva/Documents/lip_reading/"
PRE_TRAINED_MODEL =  ROOT_PATH + "pre_trained_models/shape_predictor_68_face_landmarks.dat"

TEST_IMAGE_PATH = ROOT_PATH + "test.jpg"
VIDEO_NAME = "/media/internal_harddisk/datasets/300VW_Dataset_2015_12_14/160/vid.avi"


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def image_resize(img,width = 500):
    (h,w) = img.shape[:2]
    r = width / float(w)
    resized = cv2.resize(img,(width,int(h*r)),interpolation=cv2.INTER_AREA)
    return resized

def show_facial_landmarks(gray,rects,image):
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PRE_TRAINED_MODEL)

frame_count = 0
cap = cv2.VideoCapture(VIDEO_NAME)

while(cap.isOpened()):
    ret,frame = cap.read()
    if frame_count % 3 == 0:
        image = image_resize(frame, width=320)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    cv2.imshow("frame",image)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# image = cv2.imread(TEST_IMAGE_PATH)
# imgae = image_resize(image,width=500)
# gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#
# rects = detector(gray,1)
#
# for (i,rect) in enumerate(rects):
#     shape = predictor(gray,rect)
#     shape = shape_to_np(shape)
#     for (x, y) in shape:
#         cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
# cv2.imshow("output",image)
# cv2.waitKey(0)