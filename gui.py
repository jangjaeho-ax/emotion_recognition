
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter.messagebox import *
import cam_emotion_recognition
import photo_emotion_recognition
# window 기본 설정
window = Tk()
window.title("이모지 만들어주는 프로그램")
window.geometry("640x640+650+200")
window.resizable(False, False)
init_image = "./"

panedwindow1 = PanedWindow(width="300", height="300", relief="sunken", bd=5)
panedwindow1.pack(expand=True)

init_image = Image.open('./camera.png')
imageForInit = ImageTk.PhotoImage(init_image.resize((320, 320)))
imageLabel = Label(panedwindow1, image=imageForInit)
imageLabel.pack()
routeLabel = Label(panedwindow1)

# 이미지 선택을 했는지 체크
IsImageSelected = False


def btn_Cam_emotion():
    cam_emotion_recognition.emotion_recognition()

def btn_Photo_emotion():
    if IsImageSelected == False:
        showerror("오류", "이미지를 선택해야합니다!")
    else:
        photo_emotion_recognition.emotion_recognition(routeLabel["text"])
def open():
    global IsImageSelected
    global my_image  # 함수에서 이미지를 기억하도록 전역변수 선언 (안하면 사진이 안보임)
    global imageLabel
    global routeLabel
    panedwindow1.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
        ('jpg files', '*.jpg'), ('png files', '*.png'), ('all files', '*.*')))

    # 선택을 했을때만 실행
    if panedwindow1.filename != "":
        IsImageSelected = True
        routeLabel["text"] = panedwindow1.filename
        routeLabel.pack()  # 파일경로 view

        # 이미지 사이즈 조정
        init_input_img = Image.open(panedwindow1.filename)
        my_image = ImageTk.PhotoImage(init_input_img.resize((320, 320)))
        imageLabel["image"] = my_image
        imageLabel.pack()  # 사진 view
        # imageLabel.pack_forget()


def open():
    global IsImageSelected
    global my_image  # 함수에서 이미지를 기억하도록 전역변수 선언 (안하면 사진이 안보임)
    global imageLabel
    global routeLabel
    panedwindow1.filename = filedialog.askopenfilename(initialdir='', title='파일선택', filetypes=(
        ('png files', '*.png'), ('jpg files', '*.jpg'), ('all files', '*.*')))

    # 선택을 했을때만 실행
    if panedwindow1.filename != "":
        IsImageSelected = True
        routeLabel["text"] = panedwindow1.filename
        routeLabel.pack()  # 파일경로 view

        # 이미지 사이즈 조정
        init_input_img = Image.open(panedwindow1.filename)
        my_image = ImageTk.PhotoImage(init_input_img.resize((320, 320)))
        imageLabel["image"] = my_image
        imageLabel.pack()  # 사진 view
        # imageLabel.pack_forget()


btn_create = Button(window, text='캠 인식', command=btn_Cam_emotion)
btn_create.pack(side="bottom", padx="10", pady="10", fill="x")

btn_create = Button(window, text='사진 인식', command=btn_Photo_emotion)
btn_create.pack(side="bottom", padx="10", pady="10", fill="x")

btn_load = Button(window, text='사진 불러오기', command=open)
btn_load.pack(side="bottom", padx="15", pady="15", fill="x")

label_create = Label(window, text="## 사진을 골라주세요 ##")
label_create.pack(side="bottom", fill="x")

window.mainloop()
