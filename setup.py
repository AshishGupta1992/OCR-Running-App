from cx_Freeze import setup, Executable

base = None    

executables = [Executable("Image2Text.py", base=base)]

packages = ["idna","numpy","pandas","xlwt","os","PIL","ftfy","re","passporteye","tkinter"]
options = {
    'build_exe': {    
        'packages':packages,
    },    
}

setup(
    name = "Image2Text",
    options = options,
    version = "1.1",
    description = 'Converts Image to Text',
    executables = executables
)
