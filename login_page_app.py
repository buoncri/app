from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from PyQt5.QtCore import Qt

class LoginPage(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login Page")
        self.setGeometry(100, 100, 400, 300)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.username_label = QLabel("Username:")
        layout.addWidget(self.username_label)

        self.username_input = QLineEdit()
        layout.addWidget(self.username_input)

        self.password_label = QLabel("Password:")
        layout.addWidget(self.password_label)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        layout.addWidget(self.password_input)

        self.login_button = QPushButton("Login")
        self.login_button.clicked.connect(self.login)
        layout.addWidget(self.login_button)

        self.register_button = QPushButton("Register")
        self.register_button.clicked.connect(self.register)
        layout.addWidget(self.register_button)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # Perform login logic here
        if username == "admin" and password == "password":
            self.show_main_window()
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def register(self):
        username = self.username_input.text()
        password = self.password_input.text()

        # Perform registration logic here
        # Save the username and password to a database or file

        QMessageBox.information(self, "Registration Successful", "Registration successful. You can now login.")

    def show_main_window(self):
        # Create and show the main window of your app here
        main_window = YourMainWindow()
        main_window.show()
        self.close()

if __name__ == "__main__":
    app = QApplication([])
    login_page = LoginPage()
    login_page.show()
    app.exec_()