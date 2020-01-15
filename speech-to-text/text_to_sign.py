import cv2
from speech_to_text import sentence_transcription



sentence = sentence_transcription()



cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.startWindowThread()

for e in sentence:
    if e == ' ':
        letter = "space"
    elif e.lower() in "azertyuiopqsdfghjklmwxcvbn":
        letter = e.capitalize()
    else:
        continue

    img = cv2.imread("asl_alphabet_test/{}_test.jpg".format(letter), 1)
    cv2.imshow('image',img)

    cv2.waitKey(500)

cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)