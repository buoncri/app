from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QListWidget, QListWidgetItem, 
                             QMessageBox, QTableWidget, QTableWidgetItem, QSplashScreen, QDialog, QAction, 
                             QDoubleSpinBox, QSpinBox, QPushButton, QScrollArea, QInputDialog, QFormLayout, QTextEdit, QToolBar)

from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt
import sys
import os
import time
import sqlite3
import pandas as pd
import uuid
import matplotlib.pyplot as plt
import seaborn as sns
from loan_analyst import Loan
import copy

#TODO: Fare in modo che si possano eseguire più attività in contemporanea su più prestiti in contemporanea. Cioè: devo avere sia la possibilità di effettuare un'operazione per volta su un unico prestito, sia la possibilità di effettuare più di un'operazione su un unico prestito, sia la possibilità di effettuare un'operazione unica su più prestiti, sia di effettuare più operazioni su più prestiti


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, 'assets', relative_path)


class LoanCommand:
    def __init__(self, do_action, undo_action, description):
        self.do_action = do_action
        self.undo_action = undo_action
        self.description = description

    def execute(self):
        self.do_action()

    def undo(self):
        self.undo_action()


class LoanComparisonDialog(QDialog):
    def __init__(self, comparison_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loan Comparison")
        self.setWindowIcon(QIcon(resource_path('loan_icon.ico')))
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        comparison_results = QTextEdit(self)
        comparison_results.setReadOnly(True)
        comparison_results.setText(comparison_text)
        layout.addWidget(comparison_results)


class LoanApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoanManager Pro")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon(resource_path('loan_icon.ico')))
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                font: 10pt "Segoe UI";
            }
            QLineEdit, QComboBox {
                font: 10pt "Segoe UI";
                padding: 5px;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
            }
            QPushButton {
                font: 10pt "Segoe UI";
                background-color: #dcdcdc;
                border: 1px solid #a9a9a9;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #c8c8c8;
            }
            QListWidget {
                font: 10pt "Segoe UI";
                padding: 5px;
                border: 1px solid #c0c0c0;
                border-radius: 4px;
                background-color: #ffffff;
            }
        """)

        self.loans = []
        self.selected_loan = None
        self.undo_stack = []
        self.redo_stack = []

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.loan_listbox = QListWidget()
        self.loan_listbox.itemSelectionChanged.connect(self.select_loan)
        layout.addWidget(self.loan_listbox)

        self.initMenu()
        self.initToolbar()

    def initMenu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu('File')
        new_loan_action = QAction('New Loan', self)
        new_loan_action.triggered.connect(self.new_loan)
        file_menu.addAction(new_loan_action)

        delete_loan_action = QAction('Delete Loan', self)
        delete_loan_action.triggered.connect(self.delete_loan)
        file_menu.addAction(delete_loan_action)

        save_to_db_action = QAction('Save to Database', self)
        save_to_db_action.triggered.connect(self.save_to_database)
        file_menu.addAction(save_to_db_action)

        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Loan menu
        loan_menu = menubar.addMenu('Loan')
        edit_loan_action = QAction('Edit Loan', self)
        edit_loan_action.triggered.connect(self.edit_loan)
        loan_menu.addAction(edit_loan_action)

        show_payment_action = QAction('Show Payment', self)
        show_payment_action.triggered.connect(self.pmt)
        loan_menu.addAction(show_payment_action)

        show_amortization_action = QAction('Show Amortization Table', self)
        show_amortization_action.triggered.connect(self.amort)
        loan_menu.addAction(show_amortization_action)

        show_summary_action = QAction('Show Loan Summary', self)
        show_summary_action.triggered.connect(self.summary)
        loan_menu.addAction(show_summary_action)

        plot_balances_action = QAction('Plot Balances', self)
        plot_balances_action.triggered.connect(self.plot)
        loan_menu.addAction(plot_balances_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')

        set_payment_size_action = QAction('Set Payment Size for Specific Time', self)
        set_payment_size_action.triggered.connect(self.pay_faster)
        tools_menu.addAction(set_payment_size_action)

        show_effect_action = QAction('Show Effect of Extra Payment', self)
        show_effect_action.triggered.connect(self.pay_early)
        tools_menu.addAction(show_effect_action)

        compare_loans_action = QAction('Compare Loans', self)
        compare_loans_action.triggered.connect(self.compare_loans)
        tools_menu.addAction(compare_loans_action)

        search_loan_action = QAction('Search Loan by ID', self)
        search_loan_action.triggered.connect(self.search_loan_by_id)
        tools_menu.addAction(search_loan_action)

        consolidate_loans_action = QAction('Consolidate Loans', self)
        consolidate_loans_action.triggered.connect(self.consolidate_loans)
        tools_menu.addAction(consolidate_loans_action)

        calculate_taeg_action = QAction('TAEG Calculations', self)
        calculate_taeg_action.triggered.connect(self.open_taeg_dialog)
        tools_menu.addAction(calculate_taeg_action)

    def initToolbar(self):
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        # Add Undo button
        undo_icon = QIcon(resource_path('undo_icon.png'))
        undo_action = QAction(undo_icon, "Undo", self)
        undo_action.triggered.connect(self.undo)
        toolbar.addAction(undo_action)

        # Add Redo button
        redo_icon = QIcon(resource_path('redo_icon.png'))
        redo_action = QAction(redo_icon, "Redo", self)
        redo_action.triggered.connect(self.redo)
        toolbar.addAction(redo_action)

    def new_loan(self):
        dialog = LoanDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            loan_data = dialog.get_loan_data()
            try:
                loan = Loan(**loan_data)
                self.loans.append(loan)
                self.loan_listbox.addItem(f"Loan {len(self.loans)} - {loan.loan_id} - {loan.amortization_type} amortization, {loan.frequency} payments")

                # Comando per aggiungere il prestito
                def do_action():
                    self.loans.append(loan)
                    self.update_loan_listbox()

                # Comando per annullare l'aggiunta del prestito
                def undo_action():
                    self.loans.remove(loan)
                    self.update_loan_listbox()

                # Aggiungi il comando allo stack
                self.undo_stack.append(LoanCommand(do_action, undo_action, "Add Loan"))
                self.redo_stack.clear()  # Resetta lo stack di redo

                QMessageBox.information(self, "LoanManager Pro", "Loan initialized successfully.")
            except ValueError:
                QMessageBox.warning(self, "LoanManager Pro", "Please enter valid inputs")

    def select_loan(self):
        selected_items = self.loan_listbox.selectedItems()
        if selected_items:
            selected_text = selected_items[0].text()
            loan_index = int(selected_text.split()[1]) - 1
            self.selected_loan = self.loans[loan_index]
        else:
            self.selected_loan = None

    def edit_loan(self):
        if self.selected_loan:
            old_loan_data = copy.deepcopy(self.selected_loan.__dict__)
            dialog = EditLoanDialog(self.selected_loan, self)
            if dialog.exec_() == QDialog.Accepted:
                new_loan_data = dialog.get_updated_loan_data()
                try:
                    self.selected_loan.edit_loan(
                        new_rate=new_loan_data["rate"],
                        new_term=new_loan_data["term"],
                        new_loan_amount=new_loan_data["loan_amount"],
                        new_downpayment_percent=new_loan_data["downpayment_percent"],
                        new_amortization_type=new_loan_data["amortization_type"],
                        new_frequency=new_loan_data["frequency"]
                    )

                    # Comando per modificare il prestito
                    def do_action():
                        self.selected_loan.edit_loan(**new_loan_data)
                        self.update_loan_listbox()

                    # Comando per annullare la modifica del prestito
                    def undo_action():
                        self.selected_loan.__dict__.update(old_loan_data)
                        self.update_loan_listbox()

                    # Aggiungi il comando allo stack
                    self.undo_stack.append(LoanCommand(do_action, undo_action, "Edit Loan"))
                    self.redo_stack.clear()

                    QMessageBox.information(self, "LoanManager Pro", "Loan parameters updated.")
                except ValueError:
                    QMessageBox.warning(self, "LoanManager Pro", "Please enter valid inputs")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def delete_loan(self):
        if self.selected_loan:
            loan_to_delete = self.selected_loan
            self.loans.remove(loan_to_delete)
            self.update_loan_listbox()

            # Comando per eliminare il prestito
            def do_action():
                self.loans.remove(loan_to_delete)
                self.update_loan_listbox()

            # Comando per annullare l'eliminazione del prestito
            def undo_action():
                self.loans.append(loan_to_delete)
                self.update_loan_listbox()

            # Aggiungi il comando allo stack
            self.undo_stack.append(LoanCommand(do_action, undo_action, "Delete Loan"))
            self.redo_stack.clear()

            QMessageBox.information(self, "LoanManager Pro", "Loan deleted successfully.")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def pmt(self):
        if self.selected_loan:
            if self.selected_loan.amortization_type == "French":
                QMessageBox.information(self, "LoanManager Pro", f"The French payment is {self.selected_loan.pmt_str}")
            elif self.selected_loan.amortization_type == "Italian":
                italian_payment = self.selected_loan.table['Payment'].iloc[0]
                QMessageBox.information(self, "LoanManager Pro", f"The Italian payment is €{italian_payment:,.2f}")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def amort(self):
        if self.selected_loan:
            table_data = self.selected_loan.table

            if "Initial Debt" in table_data.columns:
                dialog = AmortizationDialog(table_data)
                dialog.exec_()
            else:
                QMessageBox.warning(self, "LoanManager Pro", "La colonna 'Initial Debt' non è presente nella tabella.")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def summary(self):
        if self.selected_loan:
            additional_costs_summary = "\n".join([f"{name}: €{amount:,.2f}" for name, amount in self.selected_loan.additional_costs.items()])
            summary_text = f"""
            Payment: {self.selected_loan.pmt_str}
            Payoff Date: {self.selected_loan.table.index.date[-1]}
            Interest Paid: €{self.selected_loan.table["Interest"].cumsum()[-1]:,.2f}
            Downpayment: €{self.selected_loan.downpayment:,.2f} ({self.selected_loan.downpayment_percent}%)
            Additional Costs:
            {additional_costs_summary}
            """
            QMessageBox.information(self, "LoanManager Pro", summary_text)
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def plot(self):
        if self.selected_loan:
            self.selected_loan.plot_balances()
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def pay_early(self):
        if self.selected_loan:
            amt, ok = QInputDialog.getDouble(self, "Input", "Enter extra payment:")
            if ok:
                try:
                    result = self.selected_loan.pay_early(amt)
                    QMessageBox.information(self, "LoanManager Pro", result)
                except ValueError:
                    QMessageBox.warning(self, "LoanManager Pro", "Please enter a valid amount")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def pay_faster(self):
        if self.selected_loan:
            years_to_pay, ok = QInputDialog.getInt(self, "Input", "Enter years to debt free:")
            if ok:
                try:
                    result = self.selected_loan.pay_faster(years_to_pay)
                    QMessageBox.information(self, "LoanManager Pro", result)
                except ValueError:
                    QMessageBox.warning(self, "LoanManager Pro", "Please enter a valid number of years")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def compare_loans(self):
        if len(self.loans) < 2:
            QMessageBox.warning(self, "LoanManager Pro", "Please set at least two loans for comparison.")
            return

        comparison_text = Loan.compare_loans(self.loans)
        if comparison_text:
            dialog = LoanComparisonDialog(comparison_text, self)
            dialog.exec_()

    def consolidate_loans(self):
        if len(self.loans) < 2:
            QMessageBox.warning(self, "LoanManager Pro", "Please set at least two loans for consolidation.")
            return

        dialog = ConsolidateLoansDialog(self.loans, self)
        if dialog.exec_() == QDialog.Accepted:
            self.loans.append(dialog.selected_loans[-1])
            self.update_loan_listbox()
            QMessageBox.information(self, "LoanManager Pro", "Loans consolidated successfully.")

    def delete_loan(self):
        if self.selected_loan:
            loan_to_delete = self.selected_loan
            self.loans.remove(loan_to_delete)
            self.update_loan_listbox()

            # Comando per eliminare il prestito
            def do_action():
                self.loans.remove(loan_to_delete)
                self.update_loan_listbox()

            # Comando per annullare l'eliminazione del prestito
            def undo_action():
                self.loans.append(loan_to_delete)
                self.update_loan_listbox()

            # Aggiungi il comando allo stack
            self.undo_stack.append(LoanCommand(do_action, undo_action, "Delete Loan"))
            self.redo_stack.clear()

            QMessageBox.information(self, "LoanManager Pro", "Loan deleted successfully.")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def save_to_database(self):
        if self.selected_loan:
            self.selected_loan.save_to_db()
            QMessageBox.information(self, "LoanManager Pro", "Loan saved to database.")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def search_loan_by_id(self):
        loan_id, ok = QInputDialog.getText(self, 'Search Loan', 'Enter loan ID:')
        if ok and loan_id:
            loan = Loan.get_loan_by_id(loan_id)
            if loan:
                self.loans.append(loan)
                self.update_loan_listbox()
                QMessageBox.information(self, "LoanManager Pro", f"Loan with ID {loan_id} has been loaded.")
            else:
                QMessageBox.warning(self, "LoanManager Pro", "No loan found with the given ID.")

    def update_loan_listbox(self):
        self.loan_listbox.clear()
        for i, loan in enumerate(self.loans):
            self.loan_listbox.addItem(f"Loan {i + 1} - {loan.loan_id} - {loan.amortization_type} amortization, {loan.frequency} payments")

    def open_taeg_dialog(self):
        if self.selected_loan:
            dialog = TAEGCalculationDialog(self.selected_loan, self)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan selected")

    def undo(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            command.undo()
            self.redo_stack.append(command)
            QMessageBox.information(self, "LoanManager Pro", f"Undo: {command.description}")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No more actions to undo")

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.execute()
            self.undo_stack.append(command)
            QMessageBox.information(self, "LoanManager Pro", f"Redo: {command.description}")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No more actions to redo")


class AmortizationDialog(QDialog):
    def __init__(self, table_data, title="Amortization Table"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon(resource_path('loan_icon.ico')))

        layout = QVBoxLayout()
        self.setLayout(layout)

        table_widget = QTableWidget()
        table_widget.setRowCount(len(table_data))
        table_widget.setColumnCount(len(table_data.columns))
        table_widget.setHorizontalHeaderLabels(table_data.columns)

        for row in range(len(table_data)):
            for col in range(len(table_data.columns)):
                item = QTableWidgetItem(str(table_data.iat[row, col]))
                table_widget.setItem(row, col, item)

        layout.addWidget(table_widget)


class LoanDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Loan")
        self.setGeometry(100, 100, 400, 350)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.form_layout = QFormLayout()

        self.rate_entry = QDoubleSpinBox()
        self.rate_entry.setRange(0, 100)
        self.rate_entry.setDecimals(4)
        self.rate_entry.setSingleStep(0.005)
        self.form_layout.addRow("Interest Rate (%):", self.rate_entry)

        self.term_entry = QSpinBox()
        self.term_entry.setRange(1, 100)
        self.form_layout.addRow("Term (years):", self.term_entry)

        self.pv_entry = QDoubleSpinBox()
        self.pv_entry.setRange(0, 100000000)
        self.form_layout.addRow("Loan Amount (€):", self.pv_entry)

        self.downpayment_entry = QDoubleSpinBox()
        self.downpayment_entry.setRange(0, 100)
        self.form_layout.addRow("Downpayment (%):", self.downpayment_entry)

        self.amortization_combobox = QComboBox()
        self.amortization_combobox.addItems(["French", "Italian"])
        self.form_layout.addRow("Amortization Type:", self.amortization_combobox)

        self.frequency_combobox = QComboBox()
        self.frequency_combobox.addItems(["monthly", "quarterly", "semi-annual", "annual"])
        self.form_layout.addRow("Payment Frequency:", self.frequency_combobox)

        layout.addLayout(self.form_layout)

        self.additional_costs_button = QPushButton("Add Additional Costs")
        self.additional_costs_button.clicked.connect(self.open_additional_costs_dialog)
        layout.addWidget(self.additional_costs_button)

        self.additional_costs = {}

        self.submit_button = QPushButton("Create Loan")
        self.submit_button.clicked.connect(self.accept)
        layout.addWidget(self.submit_button)

    def open_additional_costs_dialog(self):
        dialog = AdditionalCostsDialog()
        if dialog.exec_() == QDialog.Accepted:
            self.additional_costs = dialog.costs
            QMessageBox.information(self, "LoanManager Pro", f"Additional Costs: {self.additional_costs}")

    def get_loan_data(self):
        return {
            "rate": self.rate_entry.value(),
            "term": self.term_entry.value(),
            "loan_amount": self.pv_entry.value(),
            "downpayment_percent": self.downpayment_entry.value(),
            "amortization_type": self.amortization_combobox.currentText(),
            "frequency": self.frequency_combobox.currentText(),
            "additional_costs": self.additional_costs
        }


class AdditionalCostsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Additional Costs")
        self.setWindowIcon(QIcon(resource_path('loan_icon.ico')))
        self.setGeometry(100, 100, 400, 400)  # Aumentata l'altezza per ospitare più campi

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.form_layout = QFormLayout()
        self.cost_entries = []

        self.add_cost_field()

        add_button = QPushButton("Add Another Cost")
        add_button.clicked.connect(self.add_cost_field)
        layout.addLayout(self.form_layout)
        layout.addWidget(add_button)

        # Aggiunta di un campo specifico per le spese periodiche
        self.periodic_cost_entry = QDoubleSpinBox()
        self.periodic_cost_entry.setRange(0, 1000000)
        self.form_layout.addRow(QLabel("Periodic Costs (€):"), self.periodic_cost_entry)

        self.save_button = QPushButton("Save Costs")
        self.save_button.clicked.connect(self.save_costs)
        layout.addWidget(self.save_button)

        self.costs = {}

    def add_cost_field(self):
        cost_name = QLineEdit()
        cost_value = QDoubleSpinBox()
        cost_value.setRange(0, 1000000)
        self.form_layout.addRow(QLabel("Cost Name:"), cost_name)
        self.form_layout.addRow(QLabel("Cost Amount (€):"), cost_value)
        self.cost_entries.append((cost_name, cost_value))

    def save_costs(self):
        for name_entry, value_entry in self.cost_entries:
            name = name_entry.text().strip()
            value = value_entry.value()
            if name and value > 0:
                self.costs[name] = value

        # Salva le spese periodiche
        periodic_cost = self.periodic_cost_entry.value()
        if periodic_cost > 0:
            self.costs['Periodic Costs'] = periodic_cost

        self.accept()


class TAEGCalculationDialog(QDialog):
    def __init__(self, loan, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TAEG Calculations")
        self.setGeometry(100, 100, 400, 250)

        self.loan = loan

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.taeg_periodic_label = QLabel("Calculated TAEG Periodic: N/A")
        layout.addWidget(self.taeg_periodic_label)

        self.taeg_annualized_label = QLabel("Calculated TAEG Annualized: N/A")
        layout.addWidget(self.taeg_annualized_label)

        self.calculate_taeg_button = QPushButton("Calculate TAEG")
        self.calculate_taeg_button.clicked.connect(self.calculate_taeg)
        layout.addWidget(self.calculate_taeg_button)

    def calculate_taeg(self):
        taeg_result = self.loan.calculate_taeg()
        if taeg_result is not None:
            # Aggiorna le etichette con i risultati calcolati
            self.taeg_periodic_label.setText(f"Calculated TAEG Periodic: {self.loan.taeg_periodic:.4f}%")
            self.taeg_annualized_label.setText(f"Calculated TAEG Annualized: {self.loan.taeg_annualized:.4f}%")
        else:
            QMessageBox.warning(self, "LoanManager Pro", "TAEG calculation failed.")


class EditLoanDialog(QDialog):
    def __init__(self, loan, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Loan")
        self.setGeometry(100, 100, 400, 300)

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.form_layout = QFormLayout()

        self.rate_entry = QDoubleSpinBox()
        self.rate_entry.setRange(0, 100)
        self.rate_entry.setDecimals(4)
        self.rate_entry.setSingleStep(0.001)
        self.rate_entry.setValue(loan.initial_rate)
        self.form_layout.addRow("Interest Rate (%):", self.rate_entry)

        self.term_entry = QSpinBox()
        self.term_entry.setRange(1, 100)
        self.term_entry.setValue(loan.initial_term)
        self.form_layout.addRow("Term (years):", self.term_entry)

        self.pv_entry = QDoubleSpinBox()
        self.pv_entry.setRange(0, 100000000)
        self.pv_entry.setValue(loan.loan_amount)
        self.form_layout.addRow("Loan Amount (€):", self.pv_entry)

        self.downpayment_entry = QDoubleSpinBox()
        self.downpayment_entry.setRange(0, 100)
        self.downpayment_entry.setValue(loan.downpayment_percent)
        self.form_layout.addRow("Downpayment (%):", self.downpayment_entry)

        self.amortization_combobox = QComboBox()
        self.amortization_combobox.addItems(["French", "Italian"])
        self.amortization_combobox.setCurrentText(loan.amortization_type)
        self.form_layout.addRow("Amortization Type:", self.amortization_combobox)

        self.frequency_combobox = QComboBox()
        self.frequency_combobox.addItems(["monthly", "quarterly", "semi-annual", "annual"])
        self.frequency_combobox.setCurrentText(loan.frequency)
        self.form_layout.addRow("Payment Frequency:", self.frequency_combobox)

        layout.addLayout(self.form_layout)

        self.submit_button = QPushButton("Update Loan")
        self.submit_button.clicked.connect(self.accept)
        layout.addWidget(self.submit_button)

    def get_updated_loan_data(self):
        return {
            "rate": self.rate_entry.value(),
            "term": self.term_entry.value(),
            "loan_amount": self.pv_entry.value(),
            "downpayment_percent": self.downpayment_entry.value(),
            "amortization_type": self.amortization_combobox.currentText(),
            "frequency": self.frequency_combobox.currentText()
        }


class ConsolidateLoansDialog(QDialog):
    def __init__(self, loans, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Consolidate Loans")
        self.setGeometry(100, 100, 400, 300)

        self.loans = loans
        self.selected_loans = []
        self.consolidated_loan = None
        self.consolidated_summary = ""

        self.frequency_combobox = QComboBox(self)
        self.frequency_combobox.addItems(["monthly", "quarterly", "semi-annual", "annual"])

        layout = QVBoxLayout(self)
        self.setLayout(layout)

        self.loan_list = QListWidget(self)
        self.loan_list.setSelectionMode(QListWidget.MultiSelection)

        for loan in loans:
            item_text = f"Loan ID: {loan.loan_id}, Amount: €{loan.loan_amount:,.2f}, Rate: {loan.initial_rate * 100:.2f}%"
            item = QListWidgetItem(item_text)
            self.loan_list.addItem(item)
        layout.addWidget(self.loan_list)

        layout.addWidget(QLabel("Select Payment Frequency:"))
        layout.addWidget(self.frequency_combobox)

        consolidate_button = QPushButton("Consolidate Loans", self)
        consolidate_button.clicked.connect(self.consolidate_loans)
        layout.addWidget(consolidate_button)

        show_summary_button = QPushButton("Show Consolidated Loan Summary", self)
        show_summary_button.clicked.connect(self.show_summary)
        layout.addWidget(show_summary_button)

    def consolidate_loans(self):
        selected_items = self.loan_list.selectedItems()
        self.selected_loans = [self.loans[self.loan_list.row(item)] for item in selected_items]
        if len(self.selected_loans) < 2:
            QMessageBox.warning(self, "LoanManager Pro", "Please select at least two loans to consolidate.")
            return

        frequency = self.frequency_combobox.currentText()
        try:
            self.consolidated_loan = Loan.consolidate_loans(self.selected_loans, frequency)

            self.consolidated_summary = (
                f"Summary\n"
                f"------------------------------------------------------------\n"
                f"Downpayment: €{self.consolidated_loan.downpayment:,.2f} ({self.consolidated_loan.downpayment_percent}%)\n"
                f"Payment: {self.consolidated_loan.pmt_str}\n"
                f"Payoff Date: {self.consolidated_loan.table.index.date[-1]}\n"
                f"Interest Paid: €{self.consolidated_loan.table['Interest'].cumsum().iloc[-1]:,.2f}\n"
                f"TAEG Periodic: {self.consolidated_loan.taeg.get('periodic', 'N/A'):.4f}%\n"
                f"TAEG Annualized: {self.consolidated_loan.taeg.get('annualized', 'N/A'):.4f}%\n"
            )

            QMessageBox.information(self, "LoanManager Pro", "Loans consolidated successfully.")
        except Exception as e:
            QMessageBox.warning(self, "LoanManager Pro", f"Loan consolidation failed: {str(e)}")

    def show_summary(self):
        if self.consolidated_summary:
            QMessageBox.information(self, "Consolidated Loan Summary", self.consolidated_summary)
        else:
            QMessageBox.warning(self, "LoanManager Pro", "No loan has been consolidated yet.")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    splash_pix = QPixmap(resource_path('loan_splashcreen.png'))
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()

    time.sleep(2)

    main_window = LoanApp()
    main_window.show()
    splash.finish(main_window)

    # Ensure that app.exec_() is called only once
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass