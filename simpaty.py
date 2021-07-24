# usage python simpaty.py --image test.jpg
import numpy as np
from tensorflow.keras.models import load_model
import argparse
import cv2
import helper
import dlib
from datetime import datetime


# construct the argument parse and parse the arguments LOL
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the image to classify.")
ap.add_argument("-s", "--smiling", type=float, default=0.5, help="threshold for classify a smile, must be float.")
args = vars(ap.parse_args())


print('[INFO] loading network')
model = load_model('models/best_at21-07-21 15:40:52.hdf5')
print("[INFO] loading CNN face detector...")
detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

image = cv2.imread(args['image'])
# make clone more darker
clone = image.copy()
clone = cv2.add(clone, np.array([-50.0]))
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = detector(rgb, 1)
faces = [helper.convert_and_trim_bb(image, r.rect) for r in results]
print(f'[INFO] found {len(faces)} faces')
# organize the findings
sympathy_score = 0
# loop over the results and get the coords for bbox
for (startX, startY, w, h) in faces:
    endX = startX + w
    endY = startY + h
    # slice the coords in the original image to get the image
    face_detected = image[startY:endY, startX:endX]
    # preprocess the face for make the prediction
    gray_28 = helper.ImagePP(28, 28, gray=True)
    img2array = helper.Image2ArrayPP()
    face_detected = gray_28.pp(face_detected)
    face_detected = face_detected.astype("float") / 255.0
    face_detected = img2array.pp(face_detected)
    face_detected = np.expand_dims(face_detected, axis=0)
    smile = model.predict(face_detected)
    # copy the clear face to clone
    clone[startY:endY, startX:endX] = image[startY:endY, startX:endX]
    # check if is smiling
    if smile[0][1] >= args['smiling']:
        # this face is smilming
        # add it to simpaty score
        sympathy_score += 1
        # make an bbox green
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 200, 0), 1)
        # draw the label above the box
        # cv2.rectangle(clone, (endY, startX), (endY + 10, endX), (0, 200, 0), -1)
        cv2.rectangle(clone, (startX, endY), (endX, endY + 12), (0, 200, 0), -1)
        # label the bounding box with smiling
        cv2.putText(clone, f'@ {np.round((smile[0][1] * 100),1)}%', (startX + 1, endY + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    else:
        # this face isnt smiling
        # draw a red bbox
        cv2.rectangle(clone, (startX, startY), (endX, endY), (0, 0, 200), 1)
        # cv2.rectangle(clone, (endY, startX), (endY + 10, endX), (0, 200, 0), -1)
        cv2.rectangle(clone, (startX, endY), (endX, endY + 12), (0, 0, 200), -1)
        cv2.putText(clone, f'@ {np.round((smile[0][1] * 100), 1)}%', (startX + 1, endY + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
print(f'[INFO] {sympathy_score} faces of the {len(faces)} recognized smiling')
# calculate the score for the sympathy
sympathy_score = np.round(sympathy_score / len(faces), 2)
# crop the almost upper left of the image, blur it, darkener it
score_space = clone[5:100, 5:120]
score_space = cv2.GaussianBlur(score_space, (71, 71), 71)
score_space = cv2.add(score_space, np.array([-50.0]))
# allocate and create the sympathy score
sympathy_color = (0, 200, 0)
if sympathy_score < 0.5:
    sympathy_color = (0, 0, 200)
cv2.putText(score_space, 'sympathy', (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, sympathy_color, 1)
cv2.putText(score_space, 'score', (20, 35), cv2.FONT_HERSHEY_PLAIN, 1, sympathy_color, 1)
sympathy_score = (np.round((sympathy_score * 100), 0)).astype(int)
cv2.putText(score_space, f'{sympathy_score}%', (20, 80), cv2.FONT_HERSHEY_PLAIN, 2, sympathy_color, 2)
print(f'[INFO] this picture has a sympathy score of {sympathy_score}%')
# draw the score to the image
clone[5:100, 5:120] = score_space

cv2.rectangle(clone, (5, 5), (120, 100), sympathy_color, 1)
# save the image to disk
time = datetime.now().strftime('%y-%m-%d %H:%M:%S')
cv2.imwrite(f'output/score_at_{time}.jpg', clone)
