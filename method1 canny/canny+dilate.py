import cv2 as cv
import numpy as np


def find_iron(url, scale=100):
    # for i in range(1,41):
    #     num = i
    #     num = '%d' % num
    img = cv.imread('dataset/img (13).png')
    # img = cv.imread('dataset/img (' + num + ').png')
    # img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.medianBlur(img, 15)
    # img = cv.GaussianBlur(img,(5,5),0)
    # auto paramater
    # md = np.median(img)
    # sigma = 0.33
    # lower_value = int(max(0, (1.0-sigma)*md))
    # upper_value = int(max(255, (1.0+sigma)*md))
    lower_value = 50
    upper_value = 100
    edges = cv.Canny(img, lower_value, upper_value)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv.dilate(edges, kernel, iterations=23)
    edges = cv.erode(edges, kernel, iterations=9)
    cv.bitwise_not(edges, edges)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    json_res = []
    mask = np.zeros(img.shape)
    for c in contours:
        if cv.contourArea(c) < scale ** 2 or cv.arcLength(c, True) < 2800:
            continue
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect)
        box = np.int0(box)
        box = showCoordArray(box)
        lowy = min(box[0][1], box[2][1])
        highy = max(box[0][1], box[2][1])
        if abs(box[2][0] - box[1][0]) < 1300 and abs(box[2][0] - box[0][0]) < 1300 or (lowy > 50 and highy < 2030):
            continue
        cv.drawContours(mask, [c], 0, (0, 255, 255), thickness=5)
        cv.rectangle(mask, (box[1][0], box[1][1]), (box[3][0], box[3][1]), (255, 255, 255), 5)
        json_res.append(box)
    print(json_res)
    cv.circle(img, (box[0][0], box[0][1]), 20, (0, 0, 255), -1)
    cv.circle(img, (box[1][0], box[1][1]), 20, (0, 0, 255), -1)
    cv.circle(img, (box[2][0], box[2][1]), 20, (0, 0, 255), -1)
    cv.circle(img, (box[3][0], box[3][1]), 20, (0, 0, 255), -1)
    cv.imwrite("Afimg.jpg", mask)
    cv.imwrite("Afimg.jpg", img)

    # cv.imwrite('dataset/Afimg(' + num + ').jpg', mask)


##
# point to array
# #
def showCoordArray(box):
    res = []
    for p in box:
        res.append([p[0], p[1]])
    return res


if __name__ == "__main__":
    find_iron('venv/dataset/img (11).png', 100)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
