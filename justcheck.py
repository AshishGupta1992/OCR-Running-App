from kivy.app import App
from kivy.uix.button import Button



class MainApp(App):
    def build(self):
        button1 = Button(text='Hello from Kivy',
                        size_hint=(.5, .5),
                        pos_hint={'center_x': .5, 'center_y': .5})
        button1.bind(on_press=self.getElection)

        return button1

    def getElection(self):
        print("hello")


if __name__ == '__main__':
    app = MainApp()
    app.run()
        
input()