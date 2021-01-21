import numpy as np
import cv2

# (x11, y11), (x12, y12)를 지나는 직선과 (x21, y21), (x22, y22)를 지나는 직선의 교점을 리턴
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

# imgfile/videofile로 경로를 지정해 이미지/비디오 파일을 읽어들인다
img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
cap = cv2.VideoCapture(videofile)

if cap.isOpened():
    ex_lines = None
    ex_cross_points = None
    while True:
        ret, img = cap.read()   # next frame을 read. 성공적으로 read 했으면 ret = True, img에는 next frame이 들어간다.

        height = len(img)   # 이미지의 세로 size
        width = len(img[0]) # 이미지의 가로 size
        ROI = img[height // 2 : height, : ] # Region Of Interest, 차선에만 집중하기 위해 이미지의 아래쪽 반절만을 관심 영역으로 설정한다.

        gray_img = cv2.cvtColor(ROI, cv2.COLOR_RGB2GRAY)    # 이미지를 흑백으로
        blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)    # 5 X 5 Gaussian Filter로 이미지를 흐리게 만들어 노이즈를 제거

        canny_edge = cv2.Canny(blur_img, 50, 150)       # Canny Edge Detection
        lines = cv2.HoughLines(canny_edge, 1, np.pi/180, 100, min_theta=-np.pi/3, max_theta=np.pi/3)    # Hough Transform -> threshold = 100 이고, 기울기가 수평에 가까운 직선은 검출하지 않기 위해 기울기를 제한.

        # 붙어있는 직선들 중 하나만 검출하기 위해 따로 처리함 -> 특정 y 좌표에서의 x 값이 비슷한 직선들을 서로 근접한 직선으로 판단하고 중앙에 있는 직선 하나만 선택함
        x_ground_list = []      # y 좌표가 이미지의 아래쪽 끝(y = height)일 때의 x 값들을 저장하는 list
        r_theta_list = []       # 해당하는 r과 theta를 저장하는 list
        if lines is not None:
            for line in lines:
                r, theta = line[0]

                # 직선을 그리기 위해 이미지상에 표시될 직선의 두 점 계산
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

            if len(lines_points) == 2:  # 목표는 차선 양쪽 2개만을 검출하는 것 -> 2개의 직선이 성공적으로 검출되었으면 img위에 나타내기
                cx, cy = find_cross_point(lines_points[0][0], lines_points[0][1], lines_points[0][2], lines_points[0][3], lines_points[1][0], lines_points[1][1], lines_points[1][2], lines_points[1][3])
                cx, cy = map(int, [cx, cy])
                cv2.line(ROI, (lines_points[0][0], lines_points[0][1]), (cx, cy), (0, 255, 0), 2)
                cv2.line(ROI, (lines_points[1][0], lines_points[1][1]), (cx, cy), (0, 255, 0), 2)
                ex_lines = (lines_points[0], lines_points[1])
                ex_cross_points = (cx, cy)
            else:   # 직선 2개만 검출하는 데에 실패했으면 이전에 계산한 직선을 그대로 표시 -> 차선이 급격하게 바뀌지 않으므로...
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