import cv2

def age_progression(path):

    img = cv2.imread(path)

    aged = cv2.GaussianBlur(img,(9,9),0)

    new_path = path.replace(".jpg","_aged.jpg")

    cv2.imwrite(new_path, aged)

    return new_path