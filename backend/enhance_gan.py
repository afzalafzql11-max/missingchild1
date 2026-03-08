import cv2

def enhance_image(path):

    img = cv2.imread(path)

    enhanced = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    new_path = "uploads/enhanced/" + path.split("/")[-1]

    cv2.imwrite(new_path, enhanced)

    return new_path