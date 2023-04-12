from cv_utils import *


img = cv2.imread('./test_image.png')

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


print(get_red_circles(img_hsv))
print(get_blue_circles(img_hsv))
print(get_yellow_circles(img_hsv))
print(get_green_circles(img_hsv))


cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

