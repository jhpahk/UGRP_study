import numpy as np
import cv2

def find_x(x1, y1, x2, y2, target_y):
    return x1 + ((x2 - x1) / (y2 - y1)) * (target_y - y1)

def find_cross_point(x11, y11, x12, y12, x21, y21, x22, y22):
    if x12 == x11 or x22 == x21:
        return None
    m1 = (y12 - y11) / (x12 - x11)
    m2 = (y22 - y21) / (x22 - x21)
    if m1 == m2:
        return None
    cx = (x11 * m1 - y11 - x21 * m2 + y21) / (m1 - m2)
    cy = m1 * (cx - x11) + y11

    return cx, cy

imgfile = "slope_test.jpg"
videofile = "car_driving.mp4"

img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
cap = cv2.VideoCapture(videofile)

if cap.isOpened():
    ex_lines = None
    ex_cross_points = None
    while True:
        ret, img = cap.read()   # next frame을 read. 성공적으로 read 했으면 ret = True, img에는 next frame이 들어간다.

        height = len(img)
        width = len(img[0])
        ROI = img[height // 2 : height, : ]

        gray_img = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

        canny_edge = cv2.Canny(blur_img, 50, 150)
        lines = cv2.HoughLines(canny_edge, 1, np.pi/180, 100, min_theta=-np.pi/3, max_theta=np.pi/3)

        x_ground_list = []
        r_theta_list = []
        if lines is not None:
            for line in lines:
                r, theta = line[0]

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = r * a
                y0 = r * b
                x1 = int(x0 + 2500 * (-b))
                y1 = int(y0 + 2500 * a)
                x2 = int(x0 - 2500 * (-b))
                y2 = int(y0 - 2500 * a)

                if theta < 0:
                    x_ground = x0 - np.tan(-theta) * (height // 2 - y0)
                else:
                    x_ground = r / a + (height // 2) * np.tan(theta)

                x_ground_list.append(x_ground)
                r_theta_list.append((r, theta))

            for i in range(len(x_ground_list) - 1):
                min_i = i
                for j in range(i + 1, len(x_ground_list)):
                    if x_ground_list[j] < x_ground_list[min_i]:
                        min_i = j
                x_ground_list[i], x_ground_list[min_i] = x_ground_list[min_i], x_ground_list[i]
                r_theta_list[i], r_theta_list[min_i] = r_theta_list[min_i], r_theta_list[i]

            final_list = []
            for i in range(len(x_ground_list)):
                if i == 0:
                    group = [r_theta_list[i]]
                    continue
                else:
                    if x_ground_list[i] - x_ground_list[i - 1] < 200:
                        group.append(r_theta_list[i])
                    else:
                        index = len(group) // 2
                        final_list.append(group[index])
                        group = [r_theta_list[i]]
                        if i == len(x_ground_list) - 1:
                            final_list.append(r_theta_list[i])
                            break

                    if i == len(x_ground_list) - 1:
                        index = len(group) // 2
                        final_list.append(group[index])

            lines_points = []

            for r, theta in final_list:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = r * a
                y0 = r * b
                x1 = int(x0 + 2500 * (-b))
                y1 = int(y0 + 2500 * a)
                x2 = int(x0 - 2500 * (-b))
                y2 = int(y0 - 2500 * a)

                lines_points.append((x1, y1, x2, y2))

            if len(lines_points) == 2:
                cx, cy = find_cross_point(lines_points[0][0], lines_points[0][1], lines_points[0][2], lines_points[0][3], lines_points[1][0], lines_points[1][1], lines_points[1][2], lines_points[1][3])
                cx, cy = map(int, [cx, cy])
                cv2.line(ROI, (lines_points[0][0], lines_points[0][1]), (cx, cy), (0, 255, 0), 2)
                cv2.line(ROI, (lines_points[1][0], lines_points[1][1]), (cx, cy), (0, 255, 0), 2)
                ex_lines = (lines_points[0], lines_points[1])
                ex_cross_points = (cx, cy)
            else:
                if ex_lines == None:
                    continue
                else:
                    cv2.line(ROI, (ex_lines[0][0], ex_lines[0][1]), (ex_cross_points[0], ex_cross_points[1]), (0, 255, 0), 2)
                    cv2.line(ROI, (ex_lines[1][0], ex_lines[1][1]), (ex_cross_points[0], ex_cross_points[1]), (0, 255, 0), 2)
        else:
            cv2.line(ROI, (ex_lines[0][0], ex_lines[0][1]), (ex_cross_points[0], ex_cross_points[1]), (0, 255, 0), 2)
            cv2.line(ROI, (ex_lines[1][0], ex_lines[1][1]), (ex_cross_points[0], ex_cross_points[1]), (0, 255, 0), 2)

        img[height // 2 : height, : ] = ROI
        cv2.imshow('detection', img)
        cv2.imshow('canny', canny_edge)
        if cv2.waitKey(25) != -1:
            break

cv2.destroyAllWindows()