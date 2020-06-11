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
import os
import re
import kivy
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
import pandas as pd

class MyGrid(GridLayout):
    def __init__(self, **kwargs):
        super(MyGrid, self).__init__(**kwargs)

        self.inside = GridLayout()
        self.inside.cols = 2

        self.inside.add_widget(Label(text="First Name"))
        self.name = TextInput(multiline=FALSE)
        self.inside.add_widget(self.name)

        self.add_widget(self.inside)
        self.submit = Button(text="Submit", font_size=40)
        self.submit.bind(on_press=self.pressed)
        self.add_widget(self.submit)

    def pressed(self, instance):
        print("pressed")



class TestApp(App):
    def build(self):
        return MyGrid()

if __name__=="__main__":
    TestApp().run()


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we should apply thresholding to preprocess the
        # image
        gray = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        '''
        What we would like to do is to add some additional preprocessing steps as in most cases, you may need to scale your 
        image to a larger size to recognize small characters. 
        In this case, INTER_CUBIC generally performs better than other alternatives, though it’s also slower than others.

        If you’d like to trade off some of your image quality for faster performance, 
        you may want to try INTER_LINEAR for enlarging images.
        '''
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # make a check to see if blurring should be done to remove noise, first is default median blurring

        '''
        1. Gaussian Blurring works in a similar fashion to Averaging, but it uses Gaussian kernel, 
        instead of a normalized box filter, for convolution. Here, the dimensions of the kernel and standard deviations 
        in both directions can be determined independently. 
        Gaussian blurring is very useful for removing — guess what? — 
        gaussian noise from the image. On the contrary, gaussian blurring does not preserve the edges in the input.

        2. In Median Blurring the central element in the kernel area is replaced with the median of all the pixels under the 
        kernel. Particularly, this outperforms other blurring methods in removing salt-and-pepper noise in the images.

        Median blurring is a non-linear filter. Unlike linear filters, median blurring replaces the pixel values 
        with the median value available in the neighborhood values. So, median blurring preserves edges 
        as the median value must be the value of one of neighboring pixels

        3. Speaking of keeping edges sharp, bilateral filtering is quite useful for removing the noise without 
        smoothing the edges. Similar to gaussian blurring, bilateral filtering also uses a gaussian filter 
        to find the gaussian weighted average in the neighborhood. However, it also takes pixel difference into 
        account while blurring the nearby pixels.

        Thus, it ensures only those pixels with similar intensity to the central pixel are blurred, 
        whereas the pixels with distinct pixel values are not blurred. In doing so, the edges that have larger 
        intensity variation, so-called edges, are preserved.
        '''

        gray = cv2.medianBlur(gray, 3)

        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        filename = "{}.png".format(os.getpid())
        cv2.imwrite(filename, gray)

        '''
        A blurring method may be applied. We apply a median blur when the --preprocess flag is set to blur. 
        Applying a median blur can help reduce salt and pepper noise, again making it easier for Tesseract 
        to correctly OCR the image.

        After pre-processing the image, we use  os.getpid to derive a temporary image filename based on the process ID 
        of our Python script.

        The final step before using pytesseract for OCR is to write the pre-processed image, gray, 
        to disk saving it with the filename  from above
        '''

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
                    '(INCOMETAXDEPARWENT @|mcommx|INCOME|TAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$',
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
        sheet1.write(ixsheet + 1, 0, ixsheet + 1)
        sheet1.write(ixsheet + 1, 1, data['PAN'])
        sheet1.write(ixsheet + 1, 3, data['Name'])
        sheet1.write(ixsheet + 1, 4, data['Father Name'])

        if dob:
            sheet1.write(ixsheet + 1, 2, data['Date of Birth'])

        sheet1.write(ixsheet + 1, 5, join_path)

        ixsheet = ixsheet + 1

    wb.save('PAN CARD DATA.xls')


def getPassport():
    print("This is Second Function")


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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # check to see if we should apply thresholding to preprocess the
        # image
        gray = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
        '''
        What we would like to do is to add some additional preprocessing steps as in most cases, you may need to scale your 
        image to a larger size to recognize small characters. 
        In this case, INTER_CUBIC generally performs better than other alternatives, though it’s also slower than others.

        If you’d like to trade off some of your image quality for faster performance, 
        you may want to try INTER_LINEAR for enlarging images.
        '''
        # gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        # gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # make a check to see if blurring should be done to remove noise, first is default median blurring

        gray = cv2.medianBlur(gray, 3)

        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)

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
        sheet1.write(ixsheet + 1, 0, ixsheet + 1)
        sheet1.write(ixsheet + 1, 1, data['Election No'])
        sheet1.write(ixsheet + 1, 4, data['Name'])
        sheet1.write(ixsheet + 1, 5, data['Father Name'])
        sheet1.write(ixsheet + 1, 6, join_path)
        sheet1.write(ixsheet + 1, 7, text)

        ixsheet = ixsheet + 1

    wb.save('Election Card DATA.xls')


def printmyname4():
    print("This is Fourth Function")



