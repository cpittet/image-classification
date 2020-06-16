import tkinter as tk
from tkinter import messagebox

import cv2
from PIL import ImageTk, Image
import torch
import numpy as np

from cnn.NetworkUtils import NetworkUtils
import processing.detection as detection


class MainApp(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent

        # Set up the camera capture
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.capture.open()

        # Variables used by multiple components in the application
        self.frame = None
        self.snapshot = None
        self.snapshot_cropped = None
        self.thresh_arr = None

        # Create the 3 main panels
        self.live_feed = LiveFeed(parent=self, root=parent, capture=self.capture)
        self.blob_detection = Detection(self, self.capture)
        self.classification = Classification(self, self.live_feed, self.capture, self.blob_detection)

        # Add them to the grid
        self.live_feed.grid(row=0, column=0, padx=15, pady=15)
        self.blob_detection.grid(row=0, column=1, padx=15, pady=15)
        self.classification.grid(row=1, column=0, padx=15, pady=15)

    # Ensure to release the camera when the object is destroyed
    def __del__(self):
        if self.capture.isOpened():
            self.capture.release()
            print('Camera released')


class LiveFeed(tk.Frame):
    def __init__(self, parent, root, capture):
        tk.Frame.__init__(self, parent,
                          width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)+15,
                          height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)+15)
        self.parent = parent
        self.root = root
        self.capture = capture

        # Dimensions of the video
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.width21 = int(self.width // 2 - 100)
        self.width22 = int(self.width // 2 + 100)
        self.height21 = int(self.height // 2 - 100)
        self.height22 = int(self.height // 2 + 100)

        # Create the canvas for the live feed
        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0, padx=15, pady=15)

        # Delay [ms] after which we read a frame from the capture
        self.delay = 15
        self.frame_rect = None
        self.img = None
        self.img_rect = None

        # Start the update (to register the callback) (before call to mainloop ?)
        self.update()

    def get_frame(self):
        """
        Get a frame from the capture
        :return: (ret, frame) : (boolean, frame in RGB)
        """
        ret, frame = self.capture.read()
        if ret:
            return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, None

    def update(self):
        """
        Get the frame from the capture, display it in the Frame
        """
        ret, frame = self.get_frame()
        if ret:
            # Flip the image to display on screen
            self.parent.frame = cv2.flip(frame, 1)
            self.frame_rect = self.parent.frame.copy()

            # For now, rectangle at the center of the image
            self.img = ImageTk.PhotoImage(image=Image.fromarray(self.parent.frame))
            cv2.rectangle(self.frame_rect, (self.width21, self.height21),
                          (self.width22, self.height22), (255, 0, 0), 2)

            # Important to write SELF.img, if drop the self the frame is not displayed
            self.img_rect = ImageTk.PhotoImage(image=Image.fromarray(self.frame_rect))
            self.canvas.create_image(0, 0, image=self.img_rect, anchor=tk.NW)

        # Finally, register update as a callback
        self.root.after(self.delay, self.update)


class Detection(tk.Frame):
    def __init__(self, parent, capture):
        tk.Frame.__init__(self, parent,
                          width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)+15,
                          height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)+15)
        self.parent = parent
        self.capture = capture

        # Dimensions of the video
        self.width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Label to display thresholded image
        self.thresh_img = None
        self.thresh_label = None

        # Sliders for the threshold values
        self.hue_low = tk.Label(self, text='Hue low', justify=tk.LEFT)
        self.hue_low.grid(row=0, column=0, padx=15, pady=15)
        self.hue_high = tk.Label(self, text='Hue high', justify=tk.LEFT)
        self.hue_high.grid(row=1, column=0, padx=15, pady=15)
        self.sat_low = tk.Label(self, text='Saturation low', justify=tk.LEFT)
        self.sat_low.grid(row=2, column=0, padx=15, pady=15)
        self.sat_high = tk.Label(self, text='Saturation high', justify=tk.LEFT)
        self.sat_high.grid(row=3, column=0, padx=15, pady=15)
        self.bri_low = tk.Label(self, text='Brightness low', justify=tk.LEFT)
        self.bri_low.grid(row=4, column=0, padx=15, pady=15)
        self.bri_high = tk.Label(self, text='Brightness high', justify=tk.LEFT)
        self.bri_high.grid(row=5, column=0, padx=15, pady=15)

        self.hue_low_l = tk.Scale(self, from_=0, to=255, command=self.compute_threshold,
                                  orient=tk.HORIZONTAL)
        self.hue_low_l.grid(row=0, column=1, padx=15, pady=15)
        self.hue_high_l = tk.Scale(self, from_=0, to=255, command=self.compute_threshold,
                                   orient=tk.HORIZONTAL)
        self.hue_high_l.grid(row=1, column=1, padx=15, pady=15)
        self.hue_high_l.set(255)

        self.sat_low_l = tk.Scale(self, from_=0, to=255, command=self.compute_threshold,
                                  orient=tk.HORIZONTAL)
        self.sat_low_l.grid(row=2, column=1, padx=15, pady=15)
        self.sat_high_l = tk.Scale(self, from_=0, to=255, command=self.compute_threshold,
                                   orient=tk.HORIZONTAL)
        self.sat_high_l.grid(row=3, column=1, padx=15, pady=15)
        self.sat_high_l.set(255)

        self.bri_low_l = tk.Scale(self, from_=0, to=255, command=self.compute_threshold,
                                  orient=tk.HORIZONTAL)
        self.bri_low_l.grid(row=4, column=1, padx=15, pady=15)
        self.bri_high_l = tk.Scale(self, from_=0, to=255, command=self.compute_threshold,
                                   orient=tk.HORIZONTAL)
        self.bri_high_l.grid(row=5, column=1, padx=15, pady=15)
        self.bri_high_l.set(255)

    def compute_threshold(self, scale):
        """
        Compute the new blob according to the current snapshot and threshold values and update it
        :param scale: the new scale value from the moved slider (not used)
        """
        if self.parent.snapshot is not None and self.parent.snapshot.size > 0:
            # Ensure the threshold values are consistent
            hue_low = self.hue_low_l.get()
            hue_high = self.hue_high_l.get()
            if hue_low > hue_high:
                hue_high = hue_low
                self.hue_high_l.set(hue_high)

            sat_low = self.sat_low_l.get()
            sat_high = self.sat_high_l.get()
            if sat_low > sat_high:
                sat_high = sat_low
                self.sat_high_l.set(sat_high)

            bri_low = self.bri_low_l.get()
            bri_high = self.bri_high_l.get()
            if bri_low > bri_high:
                bri_high = bri_low
                self.bri_high_l.set(bri_high)

            self.parent.thresh_arr = detection.threshold_hsb(self.parent.snapshot, hue_low, hue_high,
                                                             sat_low, sat_high,
                                                             bri_low, bri_high)
            #self.thresh_arr = detection.adaptive_threshold(self.snap)

            self.thresh_img = ImageTk.PhotoImage(image=Image.fromarray(self.parent.thresh_arr))
            self.thresh_label = tk.Label(self, image=self.thresh_img)
            self.thresh_label.grid(row=6, column=0, columnspan=2, padx=15, pady=15)


class Classification(tk.Frame):
    def __init__(self, parent, live_feed, capture, detect):
        tk.Frame.__init__(self, parent, width=capture.get(cv2.CAP_PROP_FRAME_WIDTH), height=250)
        self.parent = parent
        self.live_feed = live_feed
        self.detect = detect

        # The half size of the cropped image
        self.cote = 100

        # Dimensions of the video
        self.width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.net_utils = NetworkUtils()

        # Button for taking a snapshot
        self.snapshot_b = tk.Button(self, text='Take snapshot', command=self.take_snap)
        self.snapshot_b.grid(row=0, column=0, columnspan=3, padx=15, pady=15)

        # Button for classifying the current snapshot
        self.classify = tk.Button(self, text='Classify', command=self.classify)
        self.classify.grid(row=1, column=0, columnspan=3, padx=15, pady=15)

        # Label for the results
        self.cat1 = tk.Label(self, text='', justify=tk.LEFT)
        self.cat1.grid(row=2, column=0, padx=15, pady=15)
        self.cat2 = tk.Label(self, text='', justify=tk.LEFT)
        self.cat2.grid(row=3, column=0, padx=15, pady=15)
        self.cat3 = tk.Label(self, text='', justify=tk.LEFT)
        self.cat3.grid(row=4, column=0, padx=15, pady=15)

        # Label for the proba of the categories
        self.proba1 = tk.Label(self, text='', justify=tk.LEFT)
        self.proba1.grid(row=2, column=1, padx=15, pady=15)
        self.proba2 = tk.Label(self, text='', justify=tk.LEFT)
        self.proba2.grid(row=3, column=1, padx=15, pady=15)
        self.proba3 = tk.Label(self, text='', justify=tk.LEFT)
        self.proba3.grid(row=4, column=1, padx=15, pady=15)

    def take_snap(self):
        """
        Take a snapshot as the current frame of the live feed, without the rectangle
        """
        # First crop the snapshot
        self.parent.snapshot = detection.crop(self.parent.frame, int(self.width / 2 - self.cote),
                                              int(self.height / 2 - self.cote),
                                              int(self.width / 2 + self.cote),
                                              int(self.height / 2 + self.cote))

        # Change the displayed image in detection panel
        self.detect.compute_threshold(0)

    def classify(self):
        """
        Classify the current snapshot (if not None) and display the results
        """
        if self.parent.thresh_arr is None:
            # Error, first take a snapshot
            tk.messagebox.showerror('Error', 'You must first take a snapshot with the button above !')
            return

        # Have to invert the axis for the model (batch size =1, nb channels, height, width | NCHW)
        # before it is (height, width, nb channels)
        inp = detection.resize(self.parent.thresh_arr, 100, 100)

        inp = inp.transpose(2, 0, 1)
        # Add the axis for the batch size which 1 for us as we want to classify a single image
        inp = inp[np.newaxis, ...]
        # Convert to torch tensor
        in_tensor = torch.from_numpy(inp).float()

        categories, proba = self.net_utils.classify(in_tensor)
        # Display results
        self.cat1['text'] = categories[0]
        self.cat2['text'] = categories[1]
        self.cat3['text'] = categories[2]
        self.proba1['text'] = '{:.2f}'.format(float(proba[0])*100)
        self.proba2['text'] = '{:.2f}'.format(float(proba[1]) * 100)
        self.proba3['text'] = '{:.2f}'.format(float(proba[2]) * 100)


def main():
    # main window
    root = tk.Tk()
    root.title('Fruits and vegetables classification')

    MainApp(root).pack(padx=15, pady=15)

    # Start it
    root.mainloop()


if __name__ == '__main__':
    main()
