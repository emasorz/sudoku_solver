import sys 
import os
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2

from keras.models import load_model

from matplotlib import pyplot as plt

from copy import copy, deepcopy

#functions
def to_black_and_white(img, trigger):
    dim = min(img.shape)
    for row in range(0,dim):
        for pixel in range(0,dim):
            if img[row][pixel] <= trigger:
                img[row][pixel] = 255
            else:
                img[row][pixel] = 0
    return img
def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def area(square):
    x,y = 0,1
    l1 = square[1][x] - square[0][x]
    l2 = square[1][y] - square[2][y]
    return l1*l2


def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
 
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	return rect

def four_point_transform(img_path, pts):
    image = cv2.imread(img_path)
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    height, width, channels = warped.shape
    dim = max(width,height)
    warped = cv2.resize(warped, (dim,dim))
    
    return warped 


         
def find_sudoku_square(img_path):
    img = cv2.imread(img_path, 0)
    blur = cv2.GaussianBlur(img,(5,5),0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,5)
    cv2.bitwise_not(th,th);
    th = cv2.dilate(th, np.ones((3,3), np.uint8), iterations=1)
    _, contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    squares = []
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            if max_cos < 0.2:
                squares.append(cnt)

    squares = sorted(squares, key=cv2.contourArea, reverse=True)
    max_square = squares[0]
    return max_square

def save_digits_img(sudoku_img):
    if not os.path.exists(os.getcwd()+ "\digit_images"):    
        os.mkdir(os.getcwd()+ "\digit_images")
    
    scale = round(sudoku_img.shape[0]/9)
    offset = 2
    for y in range(0,9):
        for x in range(0,9):
            point1 = (x*scale+offset, y*scale+offset)
            point2 = (scale*(x+1)-offset, scale*(y+1)-offset)
            
            digit_img = sudoku_img[point1[0]:point2[0],point1[1]:point2[1]]
            
            digit_img = cv2.bitwise_not(digit_img)
            digit_img = cv2.resize(digit_img, (28,28))
            
            cv2.imwrite("digit_images/messigray{}{}.png".format(x,y),digit_img)

def predict_digit(digit_img_path, model):
    digit_img = cv2.imread("digit_images/" + digit_img_path);
    img_rows, img_cols, img_channel = digit_img.shape
    gray = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,300)
    edges = cv2.dilate(edges, (3,3), iterations=1)
    
    to_black_and_white(gray,255/2)
    _, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    try:
        (x, y, w, h) = cv2.boundingRect(cntsSorted[-2])
        if w*h < 100:
            prediction = 0
        else:
            top_left = (x,y)
            bottom_right = (x+w,y+h)
            
            digit_spot_img = 255 - gray[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
            blank_image = np.zeros((28,28), np.uint8) 
            x_offset= round(blank_image.shape[1]/2-digit_spot_img.shape[1]/2)
            y_offset= round(blank_image.shape[0]/2-digit_spot_img.shape[0]/2)
            
            blank_image[y_offset:y_offset+digit_spot_img.shape[0], x_offset:x_offset+digit_spot_img.shape[1]] = digit_spot_img
#            plt.imshow(blank_image, cmap = plt.get_cmap('gray'))
#            plt.show()
            blank_image = blank_image.reshape(1, img_rows, img_cols, 1)
            blank_image //= 255 
            
            prediction = model.predict(blank_image).argmax()
            if(prediction == 5 or prediction == 6):
                if cntsSorted[-2] is cntsSorted[0]:
                    prediction = 5
                else:
                    prediction = 6
#            print(prediction)
    except:
        prediction = 0
        
    return prediction

def format_sudoku(sudoku):
    new_sudoku = []
    for i in range (0,9):
        row = []
        for j in range(0,9):
            row.append(sudoku[i*9+j])
        new_sudoku.append(row)
    return new_sudoku

def get_sudoku(model):
    onlyfiles = [f for f in listdir(os.getcwd()+ "\digit_images") if isfile(join(os.getcwd()+ "\digit_images", f))]
    sudoku = []
    for img in onlyfiles:
        prediction = predict_digit(img, model)
        sudoku.append(prediction)
        
    return format_sudoku(sudoku)

def solve(bo):
    find = find_empty(bo)
    if not find:
        return True
    else:
        row, col = find

    for i in range(1,10):
        if valid(bo, i, (row, col)):
            bo[row][col] = i

            if solve(bo):
                return True

            bo[row][col] = 0

    return False

def find_empty(bo):
    for i in range(len(bo)):
        for j in range(len(bo[0])):
            if bo[i][j] == 0:
                return (i, j)  # row, col

    return None

def valid(bo, num, pos):
    # Check row
    for i in range(len(bo[0])):
        if bo[pos[0]][i] == num and pos[1] != i:
            return False

    # Check column
    for i in range(len(bo)):
        if bo[i][pos[1]] == num and pos[0] != i:
            return False

    # Check box
    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if bo[i][j] == num and (i,j) != pos:
                return False

    return True
sudoku = []

def print_sudoku(sudoku):
    for i in range(0,9):
        print(sudoku[i])
        
def print_solution_on_image(img, solution, unsolved):
    height, width, channel = img.shape
    scale = round(max(width, height)/9)
    
    font = cv2.FONT_HERSHEY_SIMPLEX  
    fontScale = 1
    color = (0, 0, 255)  
    thickness = 2
    for j in range(0,9):
        for i in range(0,9):
            org = (i*scale+round(scale/2.3), j*scale+round(scale/1.6))
            if unsolved[j][i] == 0:
                img = cv2.putText(img, str(solution[j][i]), org, font, fontScale, color, thickness, cv2.LINE_AA) 
    return img     
            
def main():
    if(len(sys.argv) >1):
        source = sys.argv[1]
    else:
        source = "sudoku00.jpg"
        
    img_path = 'C:\\Users\\Emanuele\\OneDrive\\Desktop\\py\\{}'.format(source)
    start = cv2.imread(img_path)
    sudoku_square = order_points(find_sudoku_square(img_path))
    print(sudoku_square)
    start = cv2.rectangle(start, (sudoku_square[0][1],sudoku_square[0][1]), (sudoku_square[2][0],sudoku_square[2][1]), (255,0,0), 2)
    
    sudoku_img = four_point_transform(img_path,sudoku_square)
    save_digits_img(sudoku_img)
    
    model = load_model("my_digit_classifier.h5")
    
    sudoku = get_sudoku(model)
    unsolved = deepcopy(sudoku)
    
    solve(sudoku)
    
    sudoku_img_soution = print_solution_on_image(sudoku_img.copy(),sudoku, unsolved)
    
    print_sudoku(unsolved)
    print("-----------Soluzione-----------")
    print_sudoku(sudoku)
    
    cv2.imshow("Immagine Iniziale", start)
    cv2.imshow("sudoku", sudoku_img)
    cv2.imshow("sudoku solved", sudoku_img_soution)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


main()
    
    
    
    
    
    
    
    
    
    
    
    
    


