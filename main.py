import cv2
import pyttsx3
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.core.image import Image as CoreImage
from kivy.uix.image import Image as UIImage
from kivy.clock import Clock
from kivy.graphics.texture import Texture

class ObjectDetectionApp(App):
    def build(self):
        icon:'khalili.png'
        self.icon= 'khalili.png'
        self.layout = BoxLayout(orientation='vertical')
        self.output_label = Label(text="Object Detection Output")
        self.layout.add_widget(self.output_label)

        self.camera_texture = Texture.create(size=(640, 480), colorfmt='bgr')
        self.camera_image = UIImage(texture=self.camera_texture)
        self.layout.add_widget(self.camera_image)

        self.cam = cv2.VideoCapture(1)
        self.cam.set(3, 740)
        self.cam.set(4, 580)

        self.classNames = []
        classFile = 'coco.names'
        with open(classFile, 'rt') as f:
            self.classNames = f.read().rstrip('\n').split('\n')

        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPth = 'frozen_inference_graph.pb'

        self.net = cv2.dnn_DetectionModel(weightsPth, configPath)
        self.net.setInputSize(320, 230)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # Update frame every 1/30 seconds

        return self.layout

    def update_frame(self, dt):
        success, img = self.cam.read()
        img = cv2.flip(img, 0)  # Flip the image vertically

        classIds, confs, bbox = self.net.detect(img, confThreshold=0.5)

        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, self.classNames[classId - 1], (box[0] + 10, box[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

        self.camera_texture.blit_buffer(img.tostring(), colorfmt='bgr', bufferfmt='ubyte')
        ans = self.classNames[classId - 1]
        rate = 100
        text = pyttsx3.init()
        text.setProperty('rate', rate)
        text.say(ans)
        text.runAndWait()

if __name__ == '__main__':
    ObjectDetectionApp().run()