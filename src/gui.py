import tkinter as tk
from tkinter import ttk
import cv2
import threading
from PIL import Image, ImageTk
from cnn_model import CNN
from connect_model import ConnectModel


class Camera:
    def __init__(self, root) -> None:
        self.root = root
        self.root.title("Hand Gesture GUI")
        
        # setup gui size 
        self.root.geometry("{}x{}".format(self.root.winfo_screenwidth() //2,
                           self.root.winfo_screenheight() //2))
        
        # toolbar for camera opeartion
        self.set_toolbar()
        
        # for video capture 
        self.cap = cv2.VideoCapture(0)
        self.capture_paused = False

        # display the captured frame 
        self.camera_label = tk.Label(self.root)
        self.camera_label.pack(fill="both",expand=True)
        
        # display overlay frame
        self.side_label = tk.Label(self.root)
        self.side_label.pack(fill="both",expand=True)
        
        # menu bar for operations 
        #self.set_menu_bar()

        
    ## Menu bar display 
    def set_menu_bar(self):
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu = self.menu_bar)
        
        # Create menu selections 
        menu_list = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label = "Operation", menu=menu_list)
        
        menu_list.add_command(label = "Sart Camera", command=self.start_camera)
        menu_list.add_separator()
        menu_list.add_command(label="Exit", command=self.stop_camera)
        
    def set_toolbar(self):
        toolbar = tk.Frame(self.root, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        
        # add button on toolbar 
        self.add_toolbar_button(toolbar, "Start Camera", self.start_camera)
        self.add_toolbar_button(toolbar, "Pause/Continue Camera", self.pause_camera)
        self.add_toolbar_button(toolbar, "Exit", self.stop_camera)
        self.add_toolbar_button(toolbar, "ASL Chart", self.open_chart)
        
    def add_toolbar_button(self, parent, button_name, command):
        button = ttk.Button(parent, text=button_name, command=command)
        button.pack(side=tk.LEFT, padx=2, pady=2)

    # update the frame and label on gui
    def update_frame(self):
        if not self.capture_paused:
            ret, frame = self.cap.read()
            if ret:
                # modify text overlay 
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_position = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.1))
                cv2.putText(frame, "camera", text_position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
                
                # resize image 
                w,h =(224, 224)
                center = cv2image.shape
                x = center[1]/2 - w/2
                y = center[0]/2 - h/2

                crop_img = cv2image[int(y):int(y+h), int(x):int(x+w)]
                
                resize_image = cv2.resize(crop_img, (224,224),  interpolation = cv2.INTER_LINEAR)
                resize_image = Image.fromarray(resize_image)
                resize_image_tk = ImageTk.PhotoImage(image=resize_image)
                self.side_label.imgtk =  resize_image_tk
                self.side_label.configure(image=resize_image_tk)
                prediction = model.predict(resize_image)
                prediction_label.config(text=str(prediction))
        self.camera_label.after(10, self.update_frame)
    
    def pause_camera(self):
        self.capture_paused = not self.capture_paused

    # start the camera in a different thread 
    def start_camera(self):
        #self.pause_camera = False
        threading.Thread(target=self.update_frame).start()
        
    def stop_camera(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.quit()

    # open a side window with an chart showing the ASL alphabet
    # image source: https://www.nidcd.nih.gov/sites/default/files/Content%20Images/NIDCD-ASL-hands-2014.jpg
    def open_chart(self):
        chart_window = tk.Toplevel(self.root)
        chart_window.title("ASL Chart")
        chart_window.geometry("290x467")
        chart_window.geometry("+{}+{}".format(self.root.winfo_screenwidth()//2 + 100, 150))
        image = ImageTk.PhotoImage(Image.open("src\ASL_chart.jpg"))
        label = tk.Label(chart_window, image=image)
        label.image = image
        label.pack()
        
if __name__ == "__main__":
    model = ConnectModel(r'output\new_model.pth')
    
    
    
    # start main gui 
    root = tk.Tk()
    
    prediction_label = tk.Label(root, text="Prediction will appear here", font=('Helvetica', 16))
    prediction_label.pack(pady=20)
    app = Camera(root)
    root.mainloop()
    
    
