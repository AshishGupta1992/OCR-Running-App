from kivy.app import App
from kivy.uix.button import Button

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



def prePrcoessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=2)
    gray = cv2.erode(gray, kernel, iterations=2)

    # check to see if we should apply thresholding to preprocess the
    # image
    gray = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    gray = cv2.medianBlur(gray, 3)

    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray


def getPanCard():
    # from PIL import Image, ImageFile

    x_index = 1
    mypath = 'D:\Personal\Machine Learning\PAN CARD'
    print(mypath)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files_length = len(onlyfiles)
    #    print(files_length)

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
        #        print(join_path)
        im = Image.open(join_path)
        # load the example image and convert it to grayscale
        image = cv2.imread(join_path)
        gray = prePrcoessing(image)
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
        # from pytesseract import Output
        # import pytesseract
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
                     ',', '<', '>', '.', '?', '+', '=', '_']
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

        # print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

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
            pan = re.findall(r'\w{2}[a-zA-Z]\w{0}[P,C,H,A,B,G,J,L,F,T]\w{0}[A-Z]\w{3}[0-9]\w{0}[A-Z]', text)

            if pan == []:
                pan = re.findall(r'\w{7}[A-Z]', text)
            finlen = len(pan)
            pan = pan[finlen - 1]
        #            print(pan)
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


def getDrivingLicense():
    x_index = 1
    mypath = 'D:\Personal\Machine Learning\Driving License'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files_length = len(onlyfiles)
    #    print(files_length)

    # Workbook is created
    #    wb = Workbook()
    # add_sheet is used to create sheet.
    #    sheet1 = wb.add_sheet('Sheet 1', cell_overwrite_ok=True)

    # Write Headers to sheet
    #    sheet1.write(0, 0, 'S.No')
    #    sheet1.write(0, 1, 'Election Number')
    #    sheet1.write(0, 2, 'Gender')
    #    sheet1.write(0, 3, 'Age')
    #    sheet1.write(0, 4, 'Name')
    #    sheet1.write(0, 5, 'Fathers Name')
    #    sheet1.write(0, 6, 'File Name')
    #    sheet1.write(0, 7, 'Text')

    ixsheet = 0
    # print(files_lenth)
    while ixsheet < files_length:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = "'D:\Personal\Machine Learning\Driving License\'"
        dir_path = x.replace("'", "")
        file_path = onlyfiles[ixsheet]
        join_path = join(dir_path, file_path)
        #        print(join_path)
        im = Image.open(join_path)
        # load the example image and convert it to grayscale
        image = cv2.imread(join_path)
        gray = prePrcoessing(image)

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

        bad_chars = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', "'", '[', ']', '|', ':', ';',
                     ',', '<', '>', '.', '?', '/', '+', '=', '_']
        for i in bad_chars:
            text = text.replace(i, '')

        print(text)

        ############################################################################################################
        ###################################### Section 4: Extract relevant information #############################
        ############################################################################################################

        ixsheet = ixsheet + 1


#    wb.save('Driving License Data.xls')

def getElection():
    # from PIL import Image, ImageFile

    x_index = 1
    mypath = 'D:\Personal\Machine Learning\Election Commission'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    files_length = len(onlyfiles)
    #    print(files_length)

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
        #        print(join_path)
        im = Image.open(join_path)
        # load the example image and convert it to grayscale
        image = cv2.imread(join_path)
        gray = prePrcoessing(image)

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

        # writing extracted data into a text file
        text_output = open('outputbase.txt', 'w', encoding='utf-8')
        text_output.write(text)
        text_output.close()

        file = open('outputbase.txt', 'r', encoding='utf-8')
        text = file.read()

        # Cleaning all the gibberish text
        text = ftfy.fix_text(text)
        text = ftfy.fix_encoding(text)

        bad_chars = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', "'", '[', ']', '|', ':', ';',
                     ',', '<', '>', '.', '?', '/', '+', '=', '_']
        for i in bad_chars:
            text = text.replace(i, '')

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
        # print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

        ###############################################################################################################
        ######################################### Section 5: Dishwasher part ##########################################
        ###############################################################################################################

        # Cleaning first names, better accuracy
        namelist = text.split(' ')
        while ("" in namelist):
            namelist.remove("")

        name = re.findall(r'\w{0}[N,n]\w{0}[A,a]\w{0}[M,m]\w{0}[E,e]', text)

        for n in name:
            indices = [i for i, x in enumerate(namelist) if x == n]
        #            print(indices)

        list_names = list()
        if name is not None:
            for n in indices:
                list_names.append((namelist[n + 1] + " " + namelist[n + 2]))

            # Cleaning Father's name
        Person_name = ""
        #        print(len(list_names))
        if len(list_names) > 0:
            if list_names[0] is not None:
                Person_name = list_names[0]
            else:
                Person_name = ""

        fname = ""
        if len(list_names) == 2:
            if list_names[1] is not None:
                fname = list_names[1]
            else:
                fname = ""

        gender = re.findall(r'\w{0}[F,f]\w{0}[e,E]\w{0}[M,m]\w{0}[A,a]\w{0}[a-zA-Z]\w{0}[E,e]', text)
        if gender == []:
            gender = re.findall(r'\w{0}[M,m]\w{0}[A,a]\w{0}[a-zA-Z]\w{0}[E,e]', text)

            # Cleaning DOB
        dob = re.findall(r'\d{2}[-/|-]\d{2}[-/|-]\d{4}', text)

        electiono = re.findall(r'\w{2}[a-zA-Z]\w{6}[0-9]', text)
        #        print(electiono)

        # Making tuples of data
        data = {}
        data['Name'] = Person_name
        data['Father Name'] = fname
        data['Gender'] = gender

        data['Election No'] = electiono
        sheet1.write(ixsheet + 1, 0, ixsheet + 1)
        sheet1.write(ixsheet + 1, 1, data['Election No'])
        sheet1.write(ixsheet + 1, 2, data['Gender'])
        sheet1.write(ixsheet + 1, 4, data['Name'])
        sheet1.write(ixsheet + 1, 5, data['Father Name'])
        sheet1.write(ixsheet + 1, 6, join_path)
        sheet1.write(ixsheet + 1, 7, text)

        ixsheet = ixsheet + 1

    wb.save('Election Card DATA.xls')


def close_window(root):
    root.destroy()


class MainApp(App):
    def build(self):
        button1 = Button(text='Hello from Kivy',
                        size_hint=(.5, .5),
                        pos_hint={'center_x': .5, 'center_y': .5})
        button1.bind(on_press=self.getElection)

        return button1

    def getElection(self, instance):
        # from PIL import Image, ImageFile

        x_index = 1
        mypath = 'D:\Personal\Machine Learning\Election Commission'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        files_length = len(onlyfiles)
        #    print(files_length)

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
            #        print(join_path)
            im = Image.open(join_path)
            # load the example image and convert it to grayscale
            image = cv2.imread(join_path)
            gray = prePrcoessing(image)

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

            # writing extracted data into a text file
            text_output = open('outputbase.txt', 'w', encoding='utf-8')
            text_output.write(text)
            text_output.close()

            file = open('outputbase.txt', 'r', encoding='utf-8')
            text = file.read()

            # Cleaning all the gibberish text
            text = ftfy.fix_text(text)
            text = ftfy.fix_encoding(text)

            bad_chars = ['~', '`', '!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '{', '}', "'", '[', ']', '|', ':',
                         ';',
                         ',', '<', '>', '.', '?', '/', '+', '=', '_']
            for i in bad_chars:
                text = text.replace(i, '')

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
            # print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

            ###############################################################################################################
            ######################################### Section 5: Dishwasher part ##########################################
            ###############################################################################################################

            # Cleaning first names, better accuracy
            namelist = text.split(' ')
            while ("" in namelist):
                namelist.remove("")

            name = re.findall(r'\w{0}[N,n]\w{0}[A,a]\w{0}[M,m]\w{0}[E,e]', text)

            for n in name:
                indices = [i for i, x in enumerate(namelist) if x == n]
            #            print(indices)

            list_names = list()
            if name is not None:
                for n in indices:
                    list_names.append((namelist[n + 1] + " " + namelist[n + 2]))

                # Cleaning Father's name
            Person_name = ""
            #        print(len(list_names))
            if len(list_names) > 0:
                if list_names[0] is not None:
                    Person_name = list_names[0]
                else:
                    Person_name = ""

            fname = ""
            if len(list_names) == 2:
                if list_names[1] is not None:
                    fname = list_names[1]
                else:
                    fname = ""

            gender = re.findall(r'\w{0}[F,f]\w{0}[e,E]\w{0}[M,m]\w{0}[A,a]\w{0}[a-zA-Z]\w{0}[E,e]', text)
            if gender == []:
                gender = re.findall(r'\w{0}[M,m]\w{0}[A,a]\w{0}[a-zA-Z]\w{0}[E,e]', text)

                # Cleaning DOB
            dob = re.findall(r'\d{2}[-/|-]\d{2}[-/|-]\d{4}', text)

            electiono = re.findall(r'\w{2}[a-zA-Z]\w{6}[0-9]', text)
            #        print(electiono)

            # Making tuples of data
            data = {}
            data['Name'] = Person_name
            data['Father Name'] = fname
            data['Gender'] = gender

            data['Election No'] = electiono
            sheet1.write(ixsheet + 1, 0, ixsheet + 1)
            sheet1.write(ixsheet + 1, 1, data['Election No'])
            sheet1.write(ixsheet + 1, 2, data['Gender'])
            sheet1.write(ixsheet + 1, 4, data['Name'])
            sheet1.write(ixsheet + 1, 5, data['Father Name'])
            sheet1.write(ixsheet + 1, 6, join_path)
            sheet1.write(ixsheet + 1, 7, text)

            ixsheet = ixsheet + 1

        wb.save('Election Card DATA.xls')


if __name__ == '__main__':
    app = MainApp()
    app.run()

input()