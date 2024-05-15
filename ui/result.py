import os
import sys
import glob
import time
import subprocess
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
from utils import WINDOW_TITLE


class ShellScriptThread(QThread):
    progress_updated = pyqtSignal(int)
    output_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    script_finished = pyqtSignal()

    def __init__(self, task, command):
        super().__init__()
        self.task = task
        self.command = command

    def run(self):
        process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.completed_lines = 0
        self.start_train = False
        self.val_min, self.val_max = 0, 5
        self.num_datasets = 0

        while True:
            line = process.stdout.readline()
            if not line:
                break

            ## Update Terminal Output
            line = line.decode().strip()
            self.output_updated.emit(line)
            
            ## Error Message
            if "error" in line.lower():
                if (self.task == 'ood') and ('ood/main.py' in self.command) and \
                    ('FileNotFoundError'.lower() in line.lower()) and ('threshold' in line.lower()):
                        pass
                elif (self.task == 'test') and ('round-off errors' in line.lower()):
                    pass
                elif 'h264' in line.lower():
                    pass
                else:
                    self.error_occurred.emit(line)
                    break
            
            ## Update Progress
            if self.task == 'seg':
                self.update_grogress_train_seg(line)
            elif self.task == 'ood':
                self.update_grogress_train_ood(line)
            elif self.task == 'test':
                self.update_grogress_test(line)
            
        process.wait()
        self.script_finished.emit()
    
    def update_grogress_train_seg(self, line):
        if "'epochs':" in line:
            self.num_epochs = line.split(':')[1].replace(',', '')
        
        if 'start evaluation' in line:
            self.start_train = True
        
        if self.start_train:
            if ('epoch ' in line) and (' mIoU ' in line):
                epoch = line.split('epoch ')[1].split(' mIoU ')[0]
                progress = 5 + int(int(epoch) / int(self.num_epochs) * 95)
                self.progress_updated.emit(progress)
            elif 'Train Finish!' in line:
                self.progress_updated.emit(100)
        else:
            self.completed_lines += 1
            progress = 5 * (1 - (1.6)**(-self.completed_lines/5))
            self.progress_updated.emit(progress)
    
    def update_grogress_train_ood(self, line):
        ## refine_yolo_preds.sh & train_ood_cluster.sh
        if 'YOLO-v7 prediction refining Done!' in line:
            self.progress_updated.emit(10)
        elif 'Loaded thresholds from' in line:
            self.progress_updated.emit(50)
        elif 'ood thresholds : ' in line:
            self.progress_updated.emit(80)
        
        elif 'OOD cluster (K-Means) Done!' in line:
            self.progress_updated.emit(100)

    def update_grogress_test(self, line):
        ## infer_whole.sh
        if 'Model Summary:' in line:
            self.progress_updated.emit(20)
            
        elif 'The number of test datasets:' in line:
            self.num_datasets = int(line.split('  ')[1])

        elif (self.num_datasets != 0) and (f'/{self.num_datasets}' in line):
            t = int(line.split(f'/{self.num_datasets}]')[0].split('[')[-1])
            if t != 0:
                progress = int((t / self.num_datasets) * 100)
                if progress != 100: self.progress_updated.emit(progress)
            
        elif ('The window has closed.' in line) or ('mean time per frame :' in line):
            self.progress_updated.emit(100)


class ResultWindow(QWidget):
    def __init__(self, task, source_path):
        super().__init__()
        self.task = task
        self.source_path = source_path
        self.initUI()

    def initUI(self):
        self.setWindowTitle(WINDOW_TITLE[self.task] + ' Results')

        self.progress_bar = QProgressBar()

        self.toggle_button = QPushButton("▼ Details")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.toggle_button.clicked.connect(self.toggle_output)

        task = 'Test' if self.task == 'test' else 'Train'
        source = QLabel(f"{task} Data: {self.source_path}")
        label = QLabel("Progress:")

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.hide()

        self.complete_text = QLabel('')
        self.complete_text.setAlignment(Qt.AlignCenter)

        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)
        self.close_button.clicked.connect(self.close)

        bar_layout = QHBoxLayout()
        bar_layout.addWidget(self.progress_bar)
        bar_layout.addWidget(self.toggle_button)

        progress_layout = QVBoxLayout()
        progress_layout.setAlignment(Qt.AlignCenter)
        if self.task != 'ood':
            progress_layout.addWidget(source)
        progress_layout.addWidget(label)
        progress_layout.addLayout(bar_layout)
        progress_layout.addWidget(self.output_text)
        progress_layout.addWidget(self.complete_text)

        layout = QVBoxLayout(self)
        layout.addLayout(progress_layout)

        if self.task == 'test':
            tip = QLabel("※ To close the result video window, press the 'q' key; to reset parameters, press the 'r' key.\n※ You can adjust parameters using the control bar below the video.")
            layout.addWidget(tip)
        layout.addWidget(self.close_button)

        self.thread = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_output)
        self.thread_error_occurred = False

        self.setWindowIcon(QIcon('./ui/figures/pil_logo_L.jpg'))
        self.resize(600, 200)

    def center(self, mv=0):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp + QPoint(mv,mv))
        self.move(qr.topLeft())

    def start_script(self, scripts):
        self.script_index = 0
        self.scripts = scripts

        self.execute_script(self.scripts[self.script_index])

    def execute_script(self, script):
        self.script_thread = ShellScriptThread(self.task, script)
        self.script_thread.progress_updated.connect(self.update_progress)
        self.script_thread.output_updated.connect(self.append_output)
        self.script_thread.error_occurred.connect(self.show_error_message)
        self.script_thread.script_finished.connect(self.script_finished)
        self.script_thread.start()
        self.timer.start(100)
        
        QApplication.processEvents()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

        if value == 100:
            time.sleep(3)
            if self.task == 'test':
                self.complete_text.setText("Test is completed.")
            else:
                self.complete_text.setText("Training is completed.")

    def append_output(self, text):
        self.output_text.append(text)
        self.output_text.ensureCursorVisible()

    def update_output(self):
        self.output_text.ensureCursorVisible()
        QApplication.processEvents()

    def toggle_output(self):
        if self.toggle_button.isChecked():
            self.output_text.show()
            self.toggle_button.setText("▲ Collapse")
            self.resize(600, 400)
        else:
            self.output_text.hide()
            self.toggle_button.setText("▼ Details")
            self.resize(600, 100)

    def show_error_message(self, error):
        self.thread_error_occurred = True
        QMessageBox.critical(self, "Error Occurred", error)

    def script_finished(self):
        self.timer.stop()

        if self.thread_error_occurred:
            self.complete_text.setText("An error occurred. Please close the window and try again.")
            self.close_button.setEnabled(True)
        elif self.script_index == len(self.scripts) - 1:
            self.close_button.setEnabled(True)
        else:
            self.script_index += 1
            self.execute_script(self.scripts[self.script_index])
    
    def closeEvent(self, event):
        file_list = glob.glob( "./shell/*_edit.sh")
        for f_path in file_list:
            os.remove(f_path)
        
        event.accept()
            
if __name__ == '__main__':
    app = QApplication(sys.argv)

    result_window = ResultWindow('seg', './shell/train_seg_temp.sh')
    result_window.show()
    result_window.start_script()

    sys.exit(app.exec_())