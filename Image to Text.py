from tkinter import *
from PIL import Image, ImageTk
import numpy as np
from pytesseract import Output
from xlwt import Workbook
from os import listdir
from os.path import isfile, join
from PIL import Image, ImageFile
import pytesseract
import cv2
import ftfy
import json
import os
import io
import re
from passporteye import read_mrz
import pandas as pd

root = Tk()

def preProcessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the
    # image
#    gray = cv2.threshold(gray, 0, 255,
#                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=2)
    gray = cv2.erode(gray, kernel, iterations=2)

    gray = cv2.medianBlur(gray, 3)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


def getPanCard():
    # from PIL import Image, ImageFile

    x_index = 1
    mypath = 'D:\Personal\Machine Learning\PAN CARD'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files_length = len(onlyfiles)
    print(files_length)

    # Workbook is created
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)

    # Write Headers to sheet
    sheet1.write(0, 0, 'S.No')
    sheet1.write(0, 1, 'PAN Number')
    sheet1.write(0, 2, 'Date of Birth')
    sheet1.write(0, 3, 'Name')
    sheet1.write(0, 4, 'Fathers Name')
    sheet1.write(0, 5, 'File Name')
#    sheet1.write(0, 6, 'Text')

    ixsheet = 0
    # print(files_lenth)
    while ixsheet < files_length:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = "'D:\Personal\Machine Learning\PAN CARD\'"
        dir_path = x.replace("'", "")
        file_path = onlyfiles[ixsheet]
        join_path = join(dir_path, file_path)
        print(join_path)
        im = Image.open(join_path)
        # load the example image and convert it to grayscale
        image = cv2.imread(join_path)
        gray = preProcessing(image)

        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)

        ##############################################################################################################
        ######################################## Section 3: Running PyTesseract ######################################
        ##############################################################################################################

        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        pytesseract.pytesseract.tesseract_cmd = 'D:\\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_string(Image.open(filename), lang='eng')
        # add +hin after eng within the same argument to extract hindi specific text - change encoding to utf-8 while writing
        os.remove(filename)
        # print(text)

        # show the output images
        # cv2.imshow("Image", image)
        # cv2.imshow("Output", gray)
        # cv2.waitKey(0)

        # writing extracted data into a text file
        text_output = open('outputbase.txt', 'w', encoding='utf-8')
        text_output.write(text)
        text_output.close()

        file = open('outputbase.txt', 'r', encoding='utf-8')
        text = file.read()
        # print(text)

        # Cleaning all the gibberish text
        text = ftfy.fix_text(text)
        text = ftfy.fix_encoding(text)

        bad_chars = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', "'", '[', ']', '|', ':', ';',
                     ',', '<', '>', '.', '?', '/', '+', '=', '_']
        for i in bad_chars:
            text = text.replace(i, '')

        '''for god_damn in text:
            if nonsense(god_damn):
                text.remove(god_damn)
            else:
                print(text)'''
        # print(text)

        ############################################################################################################
        ###################################### Section 4: Extract relevant information #############################
        ############################################################################################################

        # Initializing data variable
        name = None
        fname = None
        dob = None
        pan = None
        nameline = []
        dobline = []
        panline = []
        text0 = []
        text1 = []
        text2 = []

        # Searching for PAN
        lines = text.split('\n')
        for lin in lines:
            s = lin.strip()
            s = lin.replace('\n', '')
            s = s.rstrip()
            s = s.lstrip()
            text1.append(s)

        text1 = list(filter(None, text1))
        # print(text1)



        # to remove any text read from the image file which lies before the line 'Income Tax Department'

        lineno = 0  # to start from the first line of the text file.

        for wordline in text1:
            xx = wordline.split('\n')
            if ([w for w in xx if re.search(
                    '(INCOMETAXDEPARWENT @|mcommx|INCOME|TAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$',
                    w)]):
                text1 = list(text1)
                lineno = text1.index(wordline)
                break

        # text1 = list(text1)
        text0 = text1[lineno + 1:]
        print(text)  # Contains all the relevant extracted text in form of a list - uncomment to check

        def findword(textlist, wordstring):
            lineno = -1
            for wordline in textlist:
                xx = wordline.split()
                if ([w for w in xx if re.search(wordstring, w)]):
                    lineno = textlist.index(wordline)
                    textlist = textlist[lineno + 1:]
                    return textlist
            return textlist

        ###############################################################################################################
        ######################################### Section 5: Dishwasher part ##########################################
        ###############################################################################################################

        try:

            # Cleaning first names, better accuracy
            name = text0[0]
            name = name.rstrip()
            name = name.lstrip()
            name = name.replace("8", "B")
            name = name.replace("0", "D")
            name = name.replace("6", "G")
            name = name.replace("1", "I")
            name = re.sub('[^a-zA-Z] +', ' ', name)

            # Cleaning Father's name
            fname = text0[1]
            fname = fname.rstrip()
            fname = fname.lstrip()
            fname = fname.replace("8", "S")
            fname = fname.replace("0", "O")
            fname = fname.replace("6", "G")
            fname = fname.replace("1", "I")
            fname = fname.replace("\"", "A")
            fname = re.sub('[^a-zA-Z] +', ' ', fname)

            # Cleaning DOB
            dob = re.findall(r'\d{2}[-/|-]\d{2}[-/|-]\d{4}', text)

            # Cleaning PAN Card details
            text0 = findword(text1, '(Pormanam|Number|umber|Account|ccount|count|Permanent|ermanent|manent|wumm)$')
            panline = text0[0]
            pan = panline.rstrip()
            pan = pan.lstrip()
            pan = pan.replace(" ", "")
            pan = pan.replace("\"", "")
            pan = pan.replace(";", "")
            pan = pan.replace("%", "L")
        except:
            pass

        # Making tuples of data
        data = {}
        data['Name'] = name
        data['Father Name'] = fname
        data['Date of Birth'] = dob
        data['PAN'] = pan
        data['Path'] = join_path
        data['Text'] = text
        sheet1.write(ixsheet + 1, 0, ixsheet+1)
        sheet1.write(ixsheet + 1, 1, data['PAN'])
        sheet1.write(ixsheet + 1, 3, data['Name'])
        sheet1.write(ixsheet + 1, 4, data['Father Name'])

        if dob:
                    sheet1.write(ixsheet + 1, 2, data['Date of Birth'])

        sheet1.write(ixsheet + 1, 5, data['Path'])
#        sheet1.write(ixsheet + 1, 6, data['Text'])


        ixsheet = ixsheet + 1

    wb.save('PAN CARD DATA.xls')


def getPassport():
    x_index = 1
    mypath = 'D:\Personal\Machine Learning\Passport'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files_length = len(onlyfiles)
    print(files_length)

    # Workbook is created
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)

    # Write Headers to sheet
    sheet1.write(0, 0, 'S.No')
    sheet1.write(0, 1, 'Passport Number')
    sheet1.write(0, 2, 'Country')
    sheet1.write(0, 3, 'Gender')
    sheet1.write(0, 4, 'Nationality')
    sheet1.write(0, 5, 'Names')
    sheet1.write(0, 6, 'Surname')
    sheet1.write(0, 7, 'File Name')
    sheet1.write(0, 8, 'Text')
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Ashish.Gupta\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

    ixsheet = 0
    # print(files_lenth)
    while ixsheet < files_length:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = "'D:\Personal\Machine Learning\Passport\'"
        dir_path = x.replace("'", "")
        file_path = onlyfiles[ixsheet]
        join_path = join(dir_path, file_path)
        print(join_path)
        im = Image.open(join_path)
        # load the example image and convert it to grayscale
        mrz = read_mrz(join_path)
        data = {}
        if mrz is not None:
            mrz_data = mrz.to_dict()

            data['Country'] = mrz_data['country']
            data['Name'] = mrz_data['names']
            data['Surname'] = mrz_data['surname']
            data['Passport Number'] = mrz_data['number']
            data['Gender'] = mrz_data['sex']
            data['Nationality'] = mrz_data['nationality']

            sheet1.write(ixsheet + 1, 1, data['Passport Number'])
            sheet1.write(ixsheet + 1, 2, data['Country'])
            sheet1.write(ixsheet + 1, 3, data['Gender'])
            sheet1.write(ixsheet + 1, 4, data['Nationality'])
            sheet1.write(ixsheet + 1, 5, data['Name'])
            sheet1.write(ixsheet + 1, 6, data['Surname'])

        else:
            image = cv2.imread(join_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 3)
            filename = "{}.png".format(os.getpid())
            cv2.imwrite(filename, gray)
            text = pytesseract.image_to_string(Image.open(filename))

            data['Path'] = join_path
            data['Text'] = text
            sheet1.write(ixsheet + 1, 8, data['Text'])
#        sheet1.write(ixsheet + 1, 7, data['Path'])
        sheet1.write(ixsheet + 1, 0, ixsheet + 1)
        ixsheet = ixsheet + 1

    wb.save('Passport DATA.xls')


def getElection():
    # from PIL import Image, ImageFile

    x_index = 1
    mypath = 'D:\Personal\Machine Learning\Election Commission'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files_length = len(onlyfiles)
    print(files_length)

    # Workbook is created
    wb = Workbook()
    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)

    # Write Headers to sheet
    sheet1.write(0, 0, 'S.No')
    sheet1.write(0, 1, 'Election Number')
    sheet1.write(0, 2, 'Gender')
    sheet1.write(0, 3, 'Age')
    sheet1.write(0, 4, 'Name')
    sheet1.write(0, 5, 'Fathers Name')
    sheet1.write(0, 6, 'File Name')
    sheet1.write(0, 7, 'Text')

    ixsheet = 0
    # print(files_lenth)
    while ixsheet < files_length:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = "'D:\Personal\Machine Learning\Election Commission\'"
        dir_path = x.replace("'", "")
        file_path = onlyfiles[ixsheet]
        join_path = join(dir_path, file_path)
        print(join_path)
        im = Image.open(join_path)
        # load the example image and convert it to grayscale
        image = cv2.imread(join_path)
        gray = preProcessing(image)


        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)


        ##############################################################################################################
        ######################################## Section 3: Running PyTesseract ######################################
        ##############################################################################################################

        # load the image as a PIL/Pillow image, apply OCR, and then delete
        # the temporary file
        pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Ashish.Gupta\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'
        text = pytesseract.image_to_string(Image.open(filename), lang='eng')
        # add +hin after eng within the same argument to extract hindi specific text - change encoding to utf-8 while writing
        os.remove(filename)
        # print(text)

        # show the output images
        # cv2.imshow("Image", image)
        # cv2.imshow("Output", gray)
        # cv2.waitKey(0)

        # writing extracted data into a text file
        text_output = open('outputbase.txt', 'w', encoding='utf-8')
        text_output.write(text)
        text_output.close()

        file = open('outputbase.txt', 'r', encoding='utf-8')
        text = file.read()
        # print(text)

        # Cleaning all the gibberish text
        text = ftfy.fix_text(text)
        text = ftfy.fix_encoding(text)
        '''for god_damn in text:
            if nonsense(god_damn):
                text.remove(god_damn)
            else:
                print(text)'''
        # print(text)

        ############################################################################################################
        ###################################### Section 4: Extract relevant information #############################
        ############################################################################################################

        # Initializing data variable
        name = None
        fname = None
        dob = None
        pan = None
        nameline = []
        dobline = []
        panline = []
        text0 = []
        text1 = []
        text2 = []

        # Searching for PAN
        lines = text.split('\n')
        for lin in lines:
            s = lin.strip()
            s = lin.replace('\n', '')
            s = s.rstrip()
            s = s.lstrip()
            text1.append(s)

        text1 = list(filter(None, text1))
        # print(text1)



        # to remove any text read from the image file which lies before the line 'Income Tax Department'

        lineno = 0  # to start from the first line of the text file.

        for wordline in text1:
            xx = wordline.split('\n')
            if ([w for w in xx if re.search(
                    '(ELECTION|COMMISSION|INDIA|NDIA)$',
                    w)]):
                text1 = list(text1)
                lineno = text1.index(wordline)
                break

        # text1 = list(text1)
        text0 = text1[lineno + 1:]
        print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

        def findword(textlist, wordstring):
            lineno = -1
            for wordline in textlist:
                xx = wordline.split()
                if ([w for w in xx if re.search(wordstring, w)]):
                    lineno = textlist.index(wordline)
                    textlist = textlist[lineno + 1:]
                    return textlist
            return textlist

        ###############################################################################################################
        ######################################### Section 5: Dishwasher part ##########################################
        ###############################################################################################################

        try:

            # Cleaning first names, better accuracy
            name = text0[0]
            name = name.rstrip()
            name = name.lstrip()
            name = name.replace("8", "B")
            name = name.replace("0", "D")
            name = name.replace("6", "G")
            name = name.replace("1", "I")
            name = re.sub('[^a-zA-Z] +', ' ', name)

            # Cleaning Father's name
            fname = text0[1]
            fname = fname.rstrip()
            fname = fname.lstrip()
            fname = fname.replace("8", "S")
            fname = fname.replace("0", "O")
            fname = fname.replace("6", "G")
            fname = fname.replace("1", "I")
            fname = fname.replace("\"", "A")
            fname = re.sub('[^a-zA-Z] +', ' ', fname)

            # Cleaning DOB
            dob = re.findall(r'\d{2}[-/|-]\d{2}[-/|-]\d{4}', text)

            # Cleaning PAN Card details
            text0 = findword(text1, '(Pormanam|Number|umber|Account|ccount|count|Permanent|ermanent|manent|wumm)$')
            electiono = re.findall(r'\w{2}[a-zA-Z]\w{6}[0-9]', text)

        except:
            pass

        # Making tuples of data
        data = {}
        data['Name'] = name
        data['Father Name'] = fname
        data['Election No'] = electiono
        data['Path'] = join_path
        data['Text'] = text


        sheet1.write(ixsheet + 1, 0, ixsheet+1)
        sheet1.write(ixsheet + 1, 1, data['Election No'])
        sheet1.write(ixsheet + 1, 4, data['Name'])
        sheet1.write(ixsheet + 1, 5, data['Father Name'])
        sheet1.write(ixsheet + 1, 6, data['Path'])
        sheet1.write(ixsheet + 1, 7, data['Text'])

        ixsheet = ixsheet + 1

    wb.save('Election Card DATA.xls')


def close_window(root):
    root.destroy()




button1 = Button(root, text="PAN Card", font='Helvetica 18 bold', fg="Red", command=lambda: getPanCard())
button2 = Button(root, text="Passport", font='Helvetica 18 bold', fg="Blue", command=lambda: getPassport())
button3 = Button(root, text="Election Card", font='Helvetica 18 bold', fg="Green", command=lambda: getElection())
button4 = Button(root, text="Exit", font='Helvetica 18 bold', fg="Black", command=lambda: close_window(root))


button1.place(x=20, y=20)
button2.place(x=20, y=100)
button3.place(x=20, y=180)
button4.place(x=20, y=260)

#button1.pack(side=LEFT)
#button2.pack(side=LEFT)
#button3.pack(side=BOTTOM)
root.title("Convert Image to Text")
root.geometry("300x350+300+300")
root.mainloop()