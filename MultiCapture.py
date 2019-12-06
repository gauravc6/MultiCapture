import os
import cv2
import threading
import tkinter as tk
from tkinter import messagebox

def tkFunction():
    def addTracker():
        global selectedTracker, key
        key = ord('t')
        selectedTracker = tracker_selection.get()

    def resetTrackers():
        global key
        reset_response = messagebox.askquestion("Reset Trackers","Are you sure you want to reset all trackers?")
        if reset_response == 'yes':
            messagebox.showinfo("Reset Tracker","Tracker reset successful!")
            key = ord('r')

    def flipFlag():
        global flip
        flip = flip_flag.get()

    def initCapture():
        global num_iters
        if num_objects == 0:
            messagebox.showinfo("No objects to capture","No objects to track. Add atleast 1 ROI to capture!")
        elif n_images.get() == 0:
            messagebox.showinfo("Capture size not specified","Capture size can't be 0. Enter a value > 0 to initiate capture!")
        else:
            num_iters = n_images.get() * num_objects
            for i in range(num_objects):
                os.makedirs(str(f"captures/object{i}"),exist_ok=True)

    def exitApplication():
        global key
        exit_response = messagebox.askquestion("Exit Application","Are you sure you want to exit?")
        if exit_response == 'yes':
            key = 27
            root.destroy()

    root = tk.Tk()

    tracker_selection = tk.StringVar(root)
    flip_flag = tk.IntVar(root)
    n_images = tk.IntVar(root)
    flip_flag.set(1)
    tracker_selection.set("KCF Tracker")

    root.title("MultiCapture")
    root.geometry("300x300")

    frame_track = tk.LabelFrame(root,text="Track",width=300,height=300)
    frame_track.pack()
    tk.OptionMenu(frame_track, tracker_selection, *AVAILABLE_TRACKERS).pack()
    tk.Checkbutton(frame_track, variable=flip_flag, command=flipFlag, text="Flip frame").pack()
    tk.Button(frame_track, command=addTracker, text="Add Tracker").pack()

    frame_capture = tk.LabelFrame(root,text="Capture",width=100,height=100)
    frame_capture.pack()
    tk.Entry(frame_capture,textvariable=n_images,relief="flat",width=5).pack()
    tk.Button(frame_capture,command=initCapture,text="Capture").pack()

    frame_danger = tk.LabelFrame(root,text="Danger zone",width=100,height=100)
    frame_danger.pack()
    tk.Button(frame_danger, command=resetTrackers,text="Reset").pack()
    tk.Button(frame_danger,command=exitApplication,text="Exit").pack()

    root.mainloop()

def cvFunction():
    global key, num_objects

    selection = None

    active_trackers = cv2.MultiTracker_create()

    cap = cv2.VideoCapture(0)

    c = 0
    temp_shape = 0

    shapes = []

    while True:


        _, frame = cap.read(0)

        if flip == 1:
            frame = cv2.flip(frame, 1)

        _, tracked_objects = active_trackers.update(frame)
        for object in tracked_objects:
            (x, y, w, h) = [int(x) for x in object]
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 0, 255), 2)
            img = frame[y:y+h,x:x+w]
            if img.shape not in shapes:
                shapes.append(img.shape)

            if c < num_iters:
                temp_shape = shapes.index(img.shape)
                cv2.imwrite(os.path.join(f"captures/object{temp_shape}", str(len(os.listdir(f"captures/object{temp_shape}")))+".jpg"),img)
                c+=1

        cv2.imshow("MultiCapture", frame)


        # use GUI buttons or press 't' to add trackers
        if key == ord('t'):
            selection = cv2.selectROI("MultiCapture", frame, fromCenter=False, showCrosshair=False)
            tracker = AVAILABLE_TRACKERS[selectedTracker]()
            active_trackers.add(tracker, frame, selection)
            num_objects = len(active_trackers.getObjects())

        # use GUI button or 'r' to reset all trackers
        if key == ord('r'):
            shapes = []
            active_trackers = cv2.MultiTracker_create()

        # use GUI button or 'esc' to release capture
        if key == 27:
            break

        key = cv2.waitKey(1) & 0xFF

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    key = None

    flip = 1
    num_iters = 0
    num_objects = 0

    AVAILABLE_TRACKERS = {
        "KCF Tracker": cv2.TrackerKCF_create,
        "CSRT Tracker": cv2.TrackerCSRT_create,
        "MedianFlow Tracker": cv2.TrackerMedianFlow_create,
        "Boosting Tacker": cv2.TrackerBoosting_create,
        "MIL Tracker": cv2.TrackerMIL_create,
        "TLD Tracker": cv2.TrackerTLD_create,
        "MOSSE Tracker": cv2.TrackerMOSSE_create
    }

    threading.Thread(target=cvFunction).start()
    threading.Thread(target=tkFunction).start()
