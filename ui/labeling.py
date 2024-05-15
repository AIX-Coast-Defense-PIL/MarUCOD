import os
import sys
import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from utils import load_json, save_json, DESCRIPTIONS


class LabelingTool(QWidget):
    def __init__(self):
        super().__init__()
        self.boundary_dir = './ood/datasets/boundary_data'
        self.known_data_dir = './ood/datasets/known_data'
        self.update_json_list()
        
        self.current_image_index = 0
        self.img_size = 400
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Labeling Tool")

        self.raw_image = QLabel(self)
        self.raw_image.setFixedSize(self.img_size, self.img_size)
        self.raw_title = QLabel('[ Raw Image ]')
        self.raw_title.setAlignment(Qt.AlignCenter)

        self.patch_image = QLabel(self)
        self.patch_image.setFixedSize(self.img_size, self.img_size)
        self.patch_title = QLabel('[ Enlarged Image of the Box Area ]')
        self.patch_title.setAlignment(Qt.AlignCenter)

        self.no_bnd_image = QLabel('Please run Inference first.')
        self.no_bnd_image.setFixedSize(self.img_size, self.img_size)
        self.no_bnd_image.setAlignment(Qt.AlignCenter)

        self.prev_button = QPushButton("◀", self)
        self.next_button = QPushButton("▶", self)
        self.known_button = QPushButton("Known", self)
        self.unknown_button = QPushButton("Unknown", self)
        self.noise_button = QPushButton("Noise", self)
        self.save_button = QPushButton("Save", self)

        self.prev_button.setFixedWidth(40)
        self.next_button.setFixedWidth(40)
        self.known_button.setFixedWidth(90)
        self.unknown_button.setFixedWidth(90)
        self.noise_button.setFixedWidth(90)
        self.save_button.setFixedWidth(90)

        self.prev_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        self.known_button.clicked.connect(lambda: self.label_image("Known"))
        self.unknown_button.clicked.connect(lambda: self.label_image("Unknown"))
        self.noise_button.clicked.connect(lambda: self.label_image("Noise"))
        self.save_button.clicked.connect(self.save_button_clicked)

        self.description = QLabel(DESCRIPTIONS['labeling'])
        self.description.setLineWidth(3)
        self.description.setFrameShape(QFrame.StyledPanel)
        self.description.setFrameShadow(QFrame.Sunken)
        
        image_layout = QHBoxLayout()
        if len(self.boundary_list):
            raw_layout = QVBoxLayout()
            raw_layout.addWidget(self.raw_image)
            raw_layout.addWidget(self.raw_title)

            patch_layout = QVBoxLayout()
            patch_layout.addWidget(self.patch_image)
            patch_layout.addWidget(self.patch_title)

            image_layout.addLayout(raw_layout)
            image_layout.addLayout(patch_layout)
            self.show_current_image()
        else:
            image_layout.addWidget(self.no_bnd_image)
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.known_button.setEnabled(False)
            self.unknown_button.setEnabled(False)
            self.noise_button.setEnabled(False)
            self.save_button.setEnabled(False)

        move_layout = QHBoxLayout()
        move_layout.addWidget(self.prev_button)
        move_layout.addWidget(self.next_button)

        button_layout = QVBoxLayout()
        button_layout.addStretch(5)
        button_layout.addWidget(self.known_button)
        button_layout.addWidget(self.unknown_button)
        button_layout.addWidget(self.noise_button)
        button_layout.addStretch(1)
        button_layout.addLayout(move_layout)
        button_layout.addStretch(4)
        button_layout.addWidget(self.save_button)

        spacer_item = QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Preferred)
        button_layout.addSpacerItem(spacer_item)

        main_layout = QHBoxLayout()
        main_layout.addLayout(image_layout)
        main_layout.addLayout(button_layout)
        
        text_layout = QVBoxLayout()
        text_layout.addWidget(self.description)
        text_layout.addLayout(main_layout)

        self.setLayout(text_layout)
        self.setWindowIcon(QIcon('./ui/figures/pil_logo_L.jpg'))
        self.center()
    
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
    
    def update_json_list(self):
        self.boundary_list, self.boundary_json_path = load_json(self.boundary_dir)
        self.known_data_list, _ = load_json(self.known_data_dir)

    def show_current_image(self):
        img_id = self.boundary_list[self.current_image_index]["image_id"]
        image_path = os.path.join(self.boundary_dir, "images", img_id + ".jpg")
        bbox_xyxy = self.boundary_list[self.current_image_index]["bbox"]
        bbox_xywh = [bbox_xyxy[0], bbox_xyxy[1], bbox_xyxy[2]-bbox_xyxy[0], bbox_xyxy[3]-bbox_xyxy[1]]

        pixmap = QPixmap(image_path)
        cropped_pixmap = pixmap.copy(*bbox_xywh)

        painter = QPainter(pixmap)
        pen = QPen(Qt.red)
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawRect(*bbox_xywh)
        painter.end()

        self.raw_image.setPixmap(pixmap.scaled(self.img_size, self.img_size))
        self.patch_image.setPixmap(cropped_pixmap.scaled(self.img_size, self.img_size, Qt.KeepAspectRatio))
        self.raw_title.setText(f"[ Raw Image ({img_id + '.jpg'}) ]")
            
        if self.current_image_index == 0:
            self.prev_button.setEnabled(False)
        else:
            self.prev_button.setEnabled(True)

        if self.current_image_index == len(self.boundary_list) - 1:
            self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)
    
    def update_button_state(self):
        is_known = self.boundary_list[self.current_image_index]["is_known"]
        if is_known == 1:
            self.known_button.setDown(True)
            self.unknown_button.setDown(False)
            self.noise_button.setDown(False)
        elif is_known == 0:
            self.known_button.setDown(False)
            self.unknown_button.setDown(True)
            self.noise_button.setDown(False)
        elif is_known == 100:
            self.known_button.setDown(False)
            self.unknown_button.setDown(False)
            self.noise_button.setDown(True)
        else:
            self.known_button.setDown(False)
            self.unknown_button.setDown(False)
            self.noise_button.setDown(False)

    def show_previous_image(self):
        if self.current_image_index == 0:
            QMessageBox.warning(self, " ", "This is the first image.")
        else:
            self.current_image_index -= 1
            self.update_button_state()
            self.show_current_image()

    def show_next_image(self):
        if self.current_image_index == len(self.boundary_list)-1:
            QMessageBox.warning(self, " ", "This is the last image.")
        else:
            self.current_image_index += 1
            self.update_button_state()
            self.show_current_image()

    def save_annotations(self):
        unlabel_imgs = []
        unlabel_items = []

        for item in self.boundary_list:
            img_fname = item["image_id"] + ".jpg"

            if (item["is_known"] == 1) or (item["is_known"] == 100):
                self.known_data_list.append(item)

                image_path = os.path.join(self.boundary_dir, "images", img_fname)
                dest_path = os.path.join(self.known_data_dir, "images", img_fname)
                os.chmod(image_path, 0o644)
                shutil.copy2(image_path, dest_path)
            
            elif (item["is_known"] == -1):
                unlabel_imgs.append(img_fname)
                unlabel_items.append(item)
        
        unlabel_imgs = set(unlabel_imgs)
        self.boundary_list = unlabel_items

        save_json(self.boundary_list, self.boundary_dir, self.boundary_json_path)
        save_json(self.known_data_list, self.known_data_dir)

        for img_fname in os.listdir(os.path.join(self.boundary_dir, "images")):
            if img_fname not in unlabel_imgs:
                img_path = os.path.join(self.boundary_dir, "images", img_fname)
                os.remove(img_path)

        self.current_image_index = 0
        self.update_json_list()
        self.show_current_image()

    def label_image(self, label):
        if len(self.boundary_list):
            if label == "Known":
                self.boundary_list[self.current_image_index]["is_known"] = 1
            elif label == "Unknown":
                self.boundary_list[self.current_image_index]["is_known"] = 0
            elif label == "Noise":
                self.boundary_list[self.current_image_index]["is_known"] = 100

            self.update_button_state()

    def save_button_clicked(self):
        self.save_annotations()
        QMessageBox.warning(self, " ", f"Saved.\nThe current number of Known data saved is {len(self.known_data_list)}.")

    def keyPressEvent(self, e): #키가 눌러졌을 때 실행됨
        if e.key() == Qt.Key_1:   self.label_image("Known")
        elif e.key() == Qt.Key_2: self.label_image("Unknown")
        elif e.key() == Qt.Key_3: self.label_image("Noise")
        elif e.key() == Qt.Key_9: self.show_previous_image()
        elif e.key() == Qt.Key_0: self.show_next_image()
        elif e.key() == Qt.Key_S: self.save_button_clicked()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    labeling_tool = LabelingTool()
    labeling_tool.show()
    sys.exit(app.exec_())
