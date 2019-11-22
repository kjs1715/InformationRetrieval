import sys
import os
import thulac
import _thread as thread
import threading
import time


def open_raw_file(filename, start_line, end_line, file_num):

    thu = thulac.thulac(deli='\\')
    file_count = 1
    line_count = 1
    FILE_LIMIT_COUNT = 50000
    temp = ''
    
    f = open(filename, 'r')
    line = f.readline()
    while line:
        if line_count == start_line:
            break
        line_count += 1
        line = f.readline()

    while line:
        # print(line_count)
        # print(line_count, '\n', line, '\n', text, '\n')
        if line_count % FILE_LIMIT_COUNT == 0:
            print(str(line_count))
            if line_count == 500000:
                file_num += 1
            f1 = open(str(file_num) + '.txt', 'a')
            f1.write(temp)
            temp = ''
            f1.close()
        text = thu.cut(line, text=True)
        temp += text + '\n'

        if line_count == end_line:
            f1 = open(str(file_num) + '.txt', 'a')
            f1.write(temp)
            temp = ''
            f1.close()
            break;

        line = f.readline()
        line_count += 1
    f.close()



if __name__ == '__main__':
    # open_raw_file("rmrb1946-2003-delrepeat.all")
    # open_raw_file("Untitled.txt")

    ## parameters
    filename = "/Users/kim/Desktop/rmrb1946-2003-delrepeat.all"
    interval = 1840924
    start_num = 50000 # 3681849 # 5522774 # 5522775 
    end_num = 1840924
    filenum = 1
    count = 0

    try:
        # It seems threads couldn`t work well, so I just used single thread to run
        open_raw_file(filename, start_num, end_num, filenum)

    except:
        print("Error")

    fine = open("/Users/kim/Desktop/rmrb1946-2003-delrepeat.all", 'r')
    count = 1
    line = fine.readline()
    while line:
        print(line)
        if count == 80:
            break
        line = fine.readline()
        count += 1
    fine.close()
    
