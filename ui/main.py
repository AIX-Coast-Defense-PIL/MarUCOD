import sys
import time
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QCursor, QIcon, QPixmap
from result import ResultWindow
from labeling import LabelingTool
from utils import *

WIDTH = 800
HEIGHT = 600
HEIGHT2 = 400

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon('./ui/figures/pil_logo_P.png'))
        self._createStatusBar()

        self.show_initial_page()

    def _createStatusBar(self):
        self.statusBar = self.statusBar()
        self.sbText = QLabel('2023, Developed by PIL of SNU')
        self.sbIcon = QLabel()
        self.sbIcon.setPixmap(QPixmap('./ui/figures/pil_logo_PIL.jpg').scaled(48,14))

        self.statusBar.addPermanentWidget(self.sbText)
        self.statusBar.addPermanentWidget(self.sbIcon)

    def show_initial_page(self):
        self.dataset, self.dataset_type = None, None
        self.cb_epoch, self.cb_llmbd, self.cb_segbs = None, None, None
        self.cb_nclst, self.cb_imgsize = None, None
        self.cb_yths, self.cb_fths, self.cb_oths, self.cb_ocls = None, None, None, None

        self.setWindowTitle("MarUCOD")
        self.setFixedSize(WIDTH, HEIGHT)

        initial_widget = QWidget()
        layout = QVBoxLayout()

        layout.addStretch(1)
        for task in ['seg', 'ood', 'test', 'label']:
            layout.addWidget(self.create_groupbox(title=WINDOW_TITLE[task],
                                                 description=DESCRIPTIONS[task],
                                                 add_item=self.create_start_button(task)))
            layout.addStretch(1)

        initial_widget.setLayout(layout)
        self.setCentralWidget(initial_widget)

    def show_page_layout(self, task):
        self.setWindowTitle(WINDOW_TITLE[task.split('_')[0]])
        if task == 'test_params':
            self.setFixedSize(WIDTH, 500)
        else:
            self.setFixedSize(WIDTH, HEIGHT2)

        page_layout = QWidget()
        vbox = QVBoxLayout(page_layout)
        vbox.addWidget(self.create_groupbox(description=DESCRIPTIONS[task]))
        vbox.addSpacing(10)

        specific_layout = self.get_specific_layout(task)
        try:
            vbox.addLayout(specific_layout)
        except:
            vbox.addWidget(specific_layout)
        
        if task == 'ood_params':
            layout = QFormLayout()
            self.explanation_label = QLabel()
            layout.addWidget(self.explanation_label)
            hbox = QHBoxLayout()
            hbox.addWidget(self.create_infer_button())
            hbox.addWidget(self.create_label_button())
            layout.addItem(hbox)

            self.explanation_gb = QGroupBox()
            self.explanation_gb.setLayout(layout)
            self.explanation_gb.hide()
            
            vbox.addStretch(1)
            vbox.addWidget(self.create_error_button())
            vbox.addWidget(self.explanation_gb)
            vbox.addStretch(1)

        vbox.addLayout(self.create_navigation_button(task))

        self.setCentralWidget(page_layout)

    def get_specific_layout(self, task):
        if 'params' in task:
            return self.create_params_layout(task.split('_')[0])
        elif task == 'seg_data':
            return self.create_seg_data_layout()
        elif task == 'test_data':
            return self.create_test_data_layout()
        elif 'weights' in task:
            return self.create_params_layout(task)

    def create_groupbox(self, title='Description', description=None, add_item=None):
        groupbox = QGroupBox(f"  {title}  ")
        groupbox.setStyleSheet("QGroupBox { border: 0.5px solid #808080; border-radius: 5px; font-weight: bold; }")

        layout = QFormLayout()
        hbox = QHBoxLayout()
        text = QLabel(description)
        text.setWordWrap(True)  # Enable automatic word wrap
        hbox.addWidget(text, 5)

        if add_item:
            vbox = QVBoxLayout()
            vbox.addStretch(1)
            vbox.addWidget(add_item)
            vbox.addStretch(0)
            hbox.addLayout(vbox, 1)
        
        layout.addItem(QSpacerItem(0,10))
        layout.addItem(hbox)
        layout.addItem(QSpacerItem(0,10))
        groupbox.setLayout(layout)
        return groupbox

    def create_start_button(self, task):
        start_button = QPushButton("Start")

        if task == 'seg':
            start_button.clicked.connect(lambda: self.show_page_layout('seg_data'))
        elif task == 'ood':
            start_button.clicked.connect(lambda: self.show_page_layout('ood_params'))
        elif task == 'test':
            start_button.clicked.connect(lambda: self.show_page_layout('test_data'))
        elif task == 'label':
            start_button.setText("Run Labeling Tool")
            start_button.clicked.connect(self.label_button_clicked)
        
        return start_button
    
    def create_error_button(self):
        self.error_button = QPushButton("▼ If FileNotFoundError occurs..", self)
        self.error_button.setStyleSheet("color: blue; text-decoration: underline; border: none; background-color: transparent; text-align: left;")
        self.error_button.setCursor(QCursor(Qt.PointingHandCursor))
        self.error_button.clicked.connect(self.error_button_clicked)
        self.error_button.setFixedWidth(self.error_button.sizeHint().width())
        return self.error_button
    
    def error_button_clicked(self):
        if self.explanation_gb.isHidden():
            self.setFixedSize(WIDTH, HEIGHT)
            self.explanation_gb.show()
            self.explanation_label.setText(DESCRIPTIONS['ood_error'])
            self.error_button.setText('▲ If FileNotFoundError occurs..')
        else:
            self.setFixedSize(WIDTH, HEIGHT2)
            self.explanation_gb.hide()
            self.error_button.setText('▼ If FileNotFoundError occurs..')
    
    def create_infer_button(self):
        button = QPushButton("Run Inference", self)
        button.clicked.connect(lambda: self.show_page_layout('test_data'))
        return button
    
    def create_label_button(self):
        button = QPushButton("Run Labeling Tool", self)
        button.clicked.connect(self.label_button_clicked)
        return button
    
    def label_button_clicked(self):
        labeling_tool = LabelingTool()
        labeling_tool.show()
        
    def create_navigation_button(self, task):
        button_layout = QHBoxLayout()
        button_layout.addSpacing(HEIGHT2)
        
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.get_button_action(task)[0])
        button_layout.addWidget(prev_button)
        
        if task in ['test_weights']:
            label = "Start Test"
        elif task in ['seg_params', 'ood_params']:
            label = "Start Training"
        else: label = "Next"
        next_button = QPushButton(label)
        next_button.clicked.connect(self.get_button_action(task)[1])
        button_layout.addWidget(next_button)
        
        if 'data' in task:
            if self.dataset is None:
                next_button.setEnabled(False)
            self.next_button = next_button

        return button_layout
    
    def get_button_action(self, task):
        if task=='seg_data':
            return self.show_initial_page, lambda: self.show_page_layout('seg_params')
        elif task=='seg_params':
            return lambda: self.show_page_layout('seg_data'), lambda: self.start_task_button('seg')
        elif task=='ood_params':
            return self.show_initial_page, lambda: self.start_task_button('ood')
        elif task=='test_data':
            return self.show_initial_page, lambda: self.show_page_layout('test_params')
        elif task=='test_params':
            return lambda: self.show_page_layout('test_data'), lambda: self.show_page_layout('test_weights')
        elif task=='test_weights':
            return lambda: self.show_page_layout('test_params'), lambda: self.start_task_button('test')

    def create_params_layout(self, task):
        groupbox = QGroupBox('Hyperparameter Settings')
        groupbox.setCheckable(True)
        groupbox.setChecked(False)
        layout = QFormLayout()
        
        if 'weights' in task:
            groupbox.setTitle('Weight Settings')
            options = [fname.rstrip('.pkl') if fname!='BisectingKMeans_k30_resnet50_s128_SeaShips.pkl' else fname.rstrip('.pkl')+' (Default)' for fname in os.listdir('./ood/cache/distribution')]
            CB_OPTIONS['test_weights'] = [{'name': 'OOD Classifier', 'options': options}]

        cbAct = {'seg': {'Epochs': self.onActivatedEpoch, 'Loss Lambda': self.onActivatedLossLmbd, 'Batch Size': self.onActivatedBatchSize},
                 'ood': {'The number of clusters': self.onActivatedCluster, 'Input image size': self.onActivatedImgSize},
                 'test': {'YOLO Threshold': self.onActivatedYoloThs, 'Filter Threshold': self.onActivatedFiltThs, 'OOD Threshold': self.onActivatedOodThs},
                 'test_weights': {'OOD Classifier': self.onActivatedOodCls}}
        
        for hpDict in CB_OPTIONS[task]:
            name = hpDict['name']
            cb = QComboBox(self)
            cb.addItems(hpDict['options'])
            cb.setCurrentText([i for i in hpDict['options'] if '(Default)' in i][0])
            cb.activated[str].connect(cbAct[task][name])
            layout.addRow(f"- {name}:", cb)
        
        groupbox.setLayout(layout)
        return groupbox

    def create_seg_data_layout(self):
        hbox = QHBoxLayout()
        label = QLabel("- Folder:")
        label.setAlignment(Qt.AlignCenter)

        vbox = QVBoxLayout()
        vbox.addSpacing(5)
        vbox.addWidget(label, alignment=Qt.AlignTop)
        hbox.addLayout(vbox, 1)
        hbox.addSpacing(10)
        
        folder_button = QPushButton('Open Folder',self)
        folder_button.clicked.connect(lambda: self.dataset_entered('folder'))

        if self.dataset:
            self.dataset_label = QLabel(f"'{self.dataset}'")
            self.dataset_label.setStyleSheet("color: blue;")
        else:
            self.dataset_label = QLabel('')
        
        vbox = QVBoxLayout()
        vbox.addWidget(folder_button)
        vbox.addWidget(self.dataset_label, alignment=Qt.AlignTop)
        hbox.addLayout(vbox, 3)
        hbox.addSpacing(10)
        return hbox

    def create_test_data_layout(self):
        image_label = QLabel("- Image:")
        image_label.setAlignment(Qt.AlignCenter)
        video_label = QLabel("- Video:")
        video_label.setAlignment(Qt.AlignCenter)
        rtsp_label = QLabel("- RTSP:")
        rtsp_label.setAlignment(Qt.AlignCenter)
        text_label = QLabel(" ")
        text_label.setAlignment(Qt.AlignCenter)
        
        folder_button = QPushButton('Open Folder',self)
        folder_button.clicked.connect(lambda: self.dataset_entered('folder'))
        file_button = QPushButton('Open File',self)
        file_button.clicked.connect(lambda: self.dataset_entered('file'))
        self.rtsp_input = QLineEdit()
        self.rtsp_input.setPlaceholderText("Enter RTSP address (ex. rtsp://@:1234)")
        self.rtsp_input.returnPressed.connect(lambda: self.dataset_entered('rtsp'))
        
        if self.dataset:
            self.dataset_label = QLabel(f"'{self.dataset}'")
            self.dataset_label.setStyleSheet("color: blue;")
        else:
            self.dataset_label = QLabel('')

        hbox_1 = QHBoxLayout()
        hbox_1.addWidget(image_label, 1)
        hbox_1.addWidget(folder_button, 3)
        hbox_1.addSpacing(10)

        hbox_2 = QHBoxLayout()
        hbox_2.addWidget(video_label, 1)
        hbox_2.addWidget(file_button, 3)
        hbox_2.addSpacing(10)

        hbox_3 = QHBoxLayout()
        hbox_3.addWidget(rtsp_label, 1)
        hbox_3.addWidget(self.rtsp_input, 3)
        hbox_3.addSpacing(10)

        hbox_4 = QHBoxLayout()
        hbox_4.addWidget(text_label, 1)
        hbox_4.addWidget(self.dataset_label, 3)
        hbox_4.addSpacing(10)

        vbox = QVBoxLayout()
        vbox.addLayout(hbox_1)
        vbox.addLayout(hbox_2)
        vbox.addLayout(hbox_3)
        vbox.addLayout(hbox_4)
        return vbox

    def dataset_entered(self, dataset_type='folder'):
        self.dataset_type = dataset_type
        if self.dataset_type == 'folder':
            self.dataset = QFileDialog.getExistingDirectory(self, "Select Directory")
        elif self.dataset_type == 'file':
            self.dataset, _ = QFileDialog.getOpenFileName(self, "Select File", '..')#, "", "All Files (*);;Text Files (*.txt)", options=options)
        elif self.dataset_type == 'rtsp':
            self.dataset = self.rtsp_input.text()

        if self.dataset:
            self.dataset_label.setText(f"'{self.dataset}'")
            self.dataset_label.setStyleSheet("color: blue;")
            self.next_button.setEnabled(True)
    
    def start_task_button(self, task):
        script_path = SCRIPT_PATH[task]
        self.timestamp = time.strftime('%Y%m%d%H%M%S')

        if task == 'seg':
            script = self.edit_shell_seg(script_path)
        elif task == 'ood':
            script = self.edit_shell_ood(script_path)
        elif task == 'test':
            if self.dataset_type == 'rtsp':
                script_path = script_path.replace('.sh', '_stream.sh')
            script = self.edit_shell_test(script_path)
        
        self.result_window = ResultWindow(task, self.dataset)
        self.result_window.show()
        self.result_window.start_script(script)

    def edit_shell_seg(self, script_path):
        script_list = edit_script(script_path, {"source": self.dataset,
                                                "epochs": self.cb_epoch,
                                                "batch_size": self.cb_segbs})
        return [script_list]
    
    def edit_shell_ood(self, script_path):
        script_list_ref = edit_script(SCRIPT_PATH['refine'])
        script_list_ood = edit_script(script_path, {"timestamp": self.timestamp,
                                                    "patch_size": self.cb_imgsize,
                                                    "num_cluster": self.cb_nclst})
        return [script_list_ref, script_list_ood]
    
    def edit_shell_test(self, script_path):
        dir_name = 'rtsp' if self.dataset_type == 'rtsp' else self.dataset.split('.')[0].split('/')[-1]
        script_list = edit_script(script_path, {"source": self.dataset,
                                                "name": f"{dir_name}_{self.timestamp}",
                                                "conf-thres": self.cb_yths,
                                                "filter-thres": self.cb_fths,
                                                "ood-thres": self.cb_oths,
                                                "threshold_path": f"./ood/cache/threshold/{self.cb_ocls}.json",
                                                "distribution_path": f"./ood/cache/distribution/{self.cb_ocls}.pkl"})
        return [script_list]

    def onActivatedEpoch(self, text):
        self.cb_epoch = text.replace(' (Default)', '')

    def onActivatedLossLmbd(self, text):
        self.cb_llmbd = text.replace(' (Default)', '')
    
    def onActivatedBatchSize(self, text):
        self.cb_segbs = text.replace(' (Default)', '')
    
    def onActivatedCluster(self, text):
        self.cb_nclst = text.replace(' (Default)', '')
    
    def onActivatedImgSize(self, text):
        self.cb_imgsize = text.replace(' (Default)', '')
        
    def onActivatedYoloThs(self, text):
        self.cb_yths = text.replace(' (Default)', '')
        
    def onActivatedFiltThs(self, text):
        self.cb_fths = text.replace(' (Default)', '')

    def onActivatedOodThs(self, text):
        self.cb_oths = text.replace(' (Default)', '')

    def onActivatedOodCls(self, text):
        self.cb_ocls = text.replace(' (Default)', '')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())