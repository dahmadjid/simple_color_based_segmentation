import cv2
import numpy as np
import pydantic

class Circle(pydantic.BaseModel):
    x: int
    y: int
    surface: int
    color: str


def get_red_circles(img_hsv: cv2.Mat) -> list[Circle]:
    max_red = (25, 255, 255)
    min_red = (0 * 360/255, 255 * 0.75, 255 * 0.75)
    red_mask = cv2.inRange(img_hsv, min_red, max_red)
    img_red = cv2.bitwise_and(img_hsv, img_hsv, mask=red_mask)
    contours,_ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    red_circles: list[Circle] = []
    for i in range(len(contours)):
        cimg = np.zeros_like(img_hsv)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        pts = np.where(cimg == 255)

        red_circles.append(Circle(color="red", surface=len(pts[0]) + len(pts[1]), x=np.average(pts[0]), y=np.average(pts[1])))

    return red_circles


def get_blue_circles(img_hsv: cv2.Mat) -> list[Circle]:

    max_blue = (250 * 255/360, 255, 255)
    min_blue = (150 * 255/360, 255 * 0.6, 255 * 0.6)

    blue_mask = cv2.inRange(img_hsv, min_blue, max_blue)
    img_blue = cv2.bitwise_and(img_hsv, img_hsv, mask=blue_mask)
    contours,_ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    blue_circles: list[Circle] = []
    for i in range(len(contours)):
        cimg = np.zeros_like(img_hsv)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        pts = np.where(cimg == 255)

        blue_circles.append(Circle(color="blue", surface=len(pts[0]) + len(pts[1]), x=np.average(pts[0]), y=np.average(pts[1])))

    return blue_circles

def get_green_circles(img_hsv: cv2.Mat) -> list[Circle]:
    max_green = (130 *  255/360, 255, 255)
    min_green = (80 *  255/360, 255 * 0.75, 255 * 0.75)
    green_mask = cv2.inRange(img_hsv, min_green, max_green)
    img_green = cv2.bitwise_and(img_hsv, img_hsv, mask=green_mask)
    contours,_ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    green_circles: list[Circle] = []
    for i in range(len(contours)):
        cimg = np.zeros_like(img_hsv)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        pts = np.where(cimg == 255)

        green_circles.append(Circle(color="green", surface=len(pts[0]) + len(pts[1]), x=np.average(pts[0]), y=np.average(pts[1])))

    return green_circles

def get_yellow_circles(img_hsv: cv2.Mat) -> list[Circle]:
    max_yellow = (70 *  255/360, 255, 255)
    min_yellow = (40 *  255/360, 255 * 0.75, 255 * 0.75)
    yellow_mask = cv2.inRange(img_hsv, min_yellow, max_yellow)
    img_yellow = cv2.bitwise_and(img_hsv, img_hsv, mask=yellow_mask)
    contours,_ = cv2.findContours(yellow_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    yellow_circles: list[Circle] = []
    for i in range(len(contours)):
        cimg = np.zeros_like(img_hsv)
        cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

        pts = np.where(cimg == 255)

        yellow_circles.append(Circle(color="yellow", surface=len(pts[0]) + len(pts[1]), x=np.average(pts[0]), y=np.average(pts[1])))

    return yellow_circles

