import tkinter as tk
from tkinter import ttk
import cv2
import threading
from PIL import Image, ImageTk

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

if __name__ == "__main__":
    root = tk.Tk()
    app = Camera(root)
    root.mainloop()