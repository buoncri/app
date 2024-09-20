import numpy as np
import numpy_financial as npf
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import os
import seaborn as sns
from tqdm import tqdm
import uuid
import sqlite3
from scipy.optimize import brentq
import scipy.stats as stats
from IPython.display import display


class Loan:
    loans = []
    db_directory = os.getcwd()  # Default to the current working directory
    
    def __init__(self, rate, term, loan_amount, amortization_type, frequency, downpayment_percent=0, additional_costs=None, periodic_expenses=None, start=dt.date.today().isoformat(), loan_id=None):
        self.loan_id = loan_id or str(uuid.uuid4())
        self.initial_rate = rate 
        self.initial_term = term
        self.loan_amount = loan_amount
        self.downpayment_percent = downpayment_percent
        self.downpayment = self.loan_amount * (self.downpayment_percent / 100)
        self.loan_amount -= self.downpayment
        self.start = dt.datetime.fromisoformat(start).replace(day=1)
        self.frequency = frequency
        self.periods = self.calculate_periods()
        self.rate = self.calculate_rate()
        self.pmt = abs(npf.pmt(self.rate, self.periods, self.loan_amount))
        self.pmt_str = f"€ {self.pmt:,.2f}"
        self.amortization_type = amortization_type
        self.additional_costs = additional_costs or {}
        self.periodic_expenses = periodic_expenses or {}
        self.taeg = {}
        self.table = self.loan_table()
        self.active = False
        Loan.loans.append(self)

    def calculate_periods(self):
        if self.frequency == 'monthly':
            return self.initial_term * 12
        elif self.frequency == 'quarterly':
            return self.initial_term * 4
        elif self.frequency == 'semi-annual':
            return self.initial_term * 2
        elif self.frequency == 'annual':
            return self.initial_term
        else:
            raise ValueError("Unsupported frequency")

    def calculate_rate(self):
        if self.frequency == 'monthly':
            return self.initial_rate / 12
        elif self.frequency == 'quarterly':
            return self.initial_rate / 4
        elif self.frequency == 'semi-annual':
            return self.initial_rate / 2
        elif self.frequency == 'annual':
            return self.initial_rate
        else:
            raise ValueError("Unsupported frequency")

    def loan_table(self):
        if self.frequency == 'monthly':
            periods = [self.start + relativedelta(months=x) for x in range(self.periods)]
        elif self.frequency == 'quarterly':
            periods = [self.start + relativedelta(months=3 * x) for x in range(self.periods)]
        elif self.frequency == 'semi-annual':
            periods = [self.start + relativedelta(months=6 * x) for x in range(self.periods)]
        elif self.frequency == 'annual':
            periods = [self.start + relativedelta(years=x) for x in range(self.periods)]
        else:
            raise ValueError("Unsupported frequency")

        if self.amortization_type == "French":
            interest = [npf.ipmt(self.rate, month, self.periods, -self.loan_amount, when="end")
                        for month in range(1, self.periods + 1)]
            principal = [npf.ppmt(self.rate, month, self.periods, -self.loan_amount)
                        for month in range(1, self.periods + 1)]
            balance = [self.loan_amount - sum(principal[:x]) for x in range(1, self.periods + 1)]
            table = pd.DataFrame({
                'Initial Debt': [self.loan_amount] + balance[:-1],
                'Payment': [self.pmt] * self.periods,
                'Interest': interest,
                'Principal': principal,
                'Balance': balance
            }, index=pd.to_datetime(periods))

        elif self.amortization_type == "Italian":
            principal_payment = self.loan_amount / self.periods
            interest = [(self.loan_amount - principal_payment * x) * self.rate for x in range(self.periods)]
            principal = [principal_payment] * self.periods
            payment = [interest[x] + principal[x] for x in range(self.periods)]
            balance = [self.loan_amount - sum(principal[:x+1]) for x in range(self.periods)]

            table = pd.DataFrame({
                'Initial Debt': [self.loan_amount] + balance[:-1],
                'Payment': payment,
                'Interest': interest,
                'Principal': principal,
                'Balance': balance
            }, index=pd.to_datetime(periods))

        else:
            raise ValueError("Unsupported amortization type")

        return table.round(2)

    def plot_balances(self):
        amort = self.loan_table()
        if self.amortization_type == "French":
            plt.title("French Amortization Interest and Balance")
        elif self.amortization_type == "Italian":
            plt.title("Italian Amortization Interest and Balance")
        else:
            plt.title("Unknown Amortization")
        plt.plot(amort.Balance, label='Balance (€)')
        plt.plot(amort.Interest.cumsum(), label='Interest Paid (€)')
        plt.grid(axis='y', alpha=.5)
        plt.legend(loc=8)
        plt.show()

    def summary(self):
        print("Summary")
        print("-" * 60)
        print(f'Downpayment: €{self.downpayment:,.2f} ({self.downpayment_percent}%)')
        if self.amortization_type == "French":
            print(f'Payment (French amortization): {self.pmt_str:>21}')
        elif self.amortization_type == "Italian":
            italian_payment = self.table['Payment'].iloc[0]
            print(f'Payment (Italian Amortization): €{italian_payment:,.2f}')
        print(f'{"Payoff Date:":19s} {self.table.index.date[-1]}')
        print(f'Interest Paid: €{self.table["Interest"].cumsum().iloc[-1]:,.2f}')

        print("-" * 60)

    def pay_early(self, extra_amt):
        """Calculate the new payoff date and periods reduction with an extra payment."""
        new_periods = npf.nper(self.rate, self.pmt + extra_amt, -self.loan_amount)
        reduced_periods = self.periods - new_periods
        
        # Calculate the new payoff date based on frequency
        if self.frequency == 'monthly':
            payoff_date = self.start + relativedelta(months=int(new_periods))
        elif self.frequency == 'quarterly':
            payoff_date = self.start + relativedelta(months=int(new_periods * 3))
        elif self.frequency == 'semi-annual':
            payoff_date = self.start + relativedelta(months=int(new_periods * 6))
        elif self.frequency == 'annual':
            payoff_date = self.start + relativedelta(years=int(new_periods))
        else:
            raise ValueError("Unsupported frequency")

        # Convert periods reduced to appropriate units (e.g., months, quarters, etc.)
        reduced_periods_in_frequency = self.periods - new_periods
        if self.frequency == 'quarterly':
            reduced_periods_in_frequency /= 3
        elif self.frequency == 'semi-annual':
            reduced_periods_in_frequency /= 6
        elif self.frequency == 'annual':
            reduced_periods_in_frequency /= 12

        new_years = new_periods / 12
        return (f'New payoff date: {payoff_date.date()}, '
                f'Periods reduced by: {int(reduced_periods_in_frequency)} {self.frequency} '
                f'({new_years:.2f} years)')

    def pay_faster(self, years_to_debt_free):
        """Calculate the required extra payment to retire the debt within the desired time frame."""
        desired_periods = years_to_debt_free * 12  # Defaulting to monthly periods
        
        # Adjust periods based on frequency
        if self.frequency == 'quarterly':
            desired_periods //= 3
        elif self.frequency == 'semi-annual':
            desired_periods //= 6
        elif self.frequency == 'annual':
            desired_periods //= 12

        extra_pmt = npf.pmt(self.rate, desired_periods, -self.loan_amount) - self.pmt
        new_periods = npf.nper(self.rate, self.pmt + extra_pmt, -self.loan_amount)
        
        # Convert new periods into appropriate units
        if self.frequency == 'quarterly':
            new_periods /= 3
        elif self.frequency == 'semi-annual':
            new_periods /= 6
        elif self.frequency == 'annual':
            new_periods /= 12

        new_years = new_periods / 12
        return (f'Extra payment required: €{extra_pmt:.2f}, '
                f'Total payment: €{self.pmt + extra_pmt:.2f}, '
                f'New term: {int(new_periods)} {self.frequency} ({new_years:.2f} years)')

    def edit_loan(self, new_rate, new_term, new_loan_amount, new_amortization_type, new_frequency, new_downpayment_percent):
        self.initial_rate = new_rate
        self.initial_term = new_term
        self.downpayment_percent = new_downpayment_percent
        self.downpayment = new_loan_amount * (self.downpayment_percent / 100)
        self.loan_amount = new_loan_amount - self.downpayment
        self.amortization_type = new_amortization_type
        self.frequency = new_frequency
        self.periods = self.calculate_periods()
        self.rate = self.calculate_rate()
        self.pmt = abs(npf.pmt(self.rate, self.periods, self.loan_amount))
        self.pmt_str = f"€ {self.pmt:,.2f}"
        self.table = self.loan_table()
        self.update_in_db()

    def calculate_taeg(self):
        """
        Calcola il TAEG periodico e annualizzato. Il TAEG è il tasso che uguaglia la somma attualizzata
        dei pagamenti periodici (rate lorde) all'importo erogato (prestito netto dopo le spese iniziali).
        """

        # Importo del prestito iniziale al netto delle spese iniziali
        loan_amount = self.loan_amount
        initial_expenses = sum(self.additional_costs.values())
        net_loan_amount = loan_amount - initial_expenses

        # Recupera le spese periodiche dall'attributo periodic_expenses
        total_periodic_expenses = sum(self.periodic_expenses.values()) if self.periodic_expenses else 0

        # Pagamento periodico lordo (inclusi eventuali costi periodici)
        gross_payment = self.pmt + total_periodic_expenses

        # Durata del prestito in anni, tenendo conto della frequenza dei pagamenti
        if self.frequency == 'monthly':
            periods_in_years = np.array([(i + 1) / 12 for i in range(self.periods)])
            periods_per_year = 12
        elif self.frequency == 'quarterly':
            periods_in_years = np.array([(i + 1) / 4 for i in range(self.periods)])
            periods_per_year = 4
        elif self.frequency == 'semi-annual':
            periods_in_years = np.array([(i + 1) / 2 for i in range(self.periods)])
            periods_per_year = 2
        elif self.frequency == 'annual':
            periods_in_years = np.array([(i + 1) for i in range(self.periods)])
            periods_per_year = 1
        else:
            raise ValueError("Unsupported frequency")

        # Funzione da azzerare per calcolare il TAEG periodico
        def taeg_func(r):
            return sum([gross_payment / (1 + r)**t for t in periods_in_years]) - net_loan_amount

        # Trova la radice dell'equazione per ottenere il TAEG periodico
        period_rate = brentq(taeg_func, 0, 1)

        # Calcola il TAEG annualizzato
        annualized_taeg = (1 + period_rate)**periods_per_year - 1

        # Converti i TAEG in percentuale
        period_taeg_percent = period_rate * 100
        annualized_taeg_percent = annualized_taeg * 100

        # Aggiorna gli attributi TAEG del prestito
        self.taeg_periodic = period_taeg_percent
        self.taeg_annualized = annualized_taeg_percent

        # Assegna entrambi i TAEG calcolati al dizionario
        self.taeg = {
            'periodic': period_taeg_percent,
            'annualized': annualized_taeg_percent
        }
        
        return f'TAEG Periodico: {period_taeg_percent:.4f}%, TAEG Annualizzato: {annualized_taeg_percent:.4f}%'

    def update_table_structure(self, table_name='loan_table'):
        db_path = os.path.join(Loan.db_directory, 'loans.db')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Creare la tabella se non esiste
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    Date TEXT,
                    'Initial Debt' REAL,
                    Payment REAL,
                    Interest REAL,
                    Principal REAL,
                    Balance REAL,
                    loan_id TEXT
                )
            """)

            # Creare una nuova tabella temporanea con la colonna "Initial Debt"
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name}_temp (
                    Date TEXT,
                    'Initial Debt' REAL,
                    Payment REAL,
                    Interest REAL,
                    Principal REAL,
                    Balance REAL,
                    loan_id TEXT
                )
            """)

            # Copia i dati dalla vecchia tabella alla nuova
            cursor.execute(f"""
                INSERT INTO {table_name}_temp (Date, Payment, Interest, Principal, Balance, loan_id)
                SELECT Date, Payment, Interest, Principal, Balance, loan_id
                FROM {table_name}
            """)

            # Eliminare la vecchia tabella
            cursor.execute(f"DROP TABLE {table_name}")

            # Rinomina la tabella temporanea a quella originale
            cursor.execute(f"ALTER TABLE {table_name}_temp RENAME TO {table_name}")

            conn.commit()

        print(f"Table {table_name} structure updated successfully.")

    def save_to_db(self, table_name='loan_table'):
        # Verifica se la tabella esiste e, in caso contrario, crea la tabella
        self.update_table_structure(table_name)
        
        db_path = os.path.join(Loan.db_directory, 'loans.db')
        with sqlite3.connect(db_path) as conn:
            self.table['loan_id'] = self.loan_id
            self.table.to_sql(table_name, conn, if_exists='append', index_label='Date')
        print(f"Loan table saved to {db_path} in table {table_name}")

    def update_in_db(self, table_name='loan_table'):
        """
        Aggiorna i dati del prestito nel database, sostituendo la vecchia versione.
        Crea la tabella se non esiste.
        """
        db_path = os.path.join(Loan.db_directory, 'loans.db')
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Crea la tabella se non esiste
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    Date TEXT,
                    'Initial Debt' REAL,
                    Payment REAL,
                    Interest REAL,
                    Principal REAL,
                    Balance REAL,
                    loan_id TEXT
                )
            """)

            try:
                # Elimina i record esistenti per questo prestito
                cursor.execute(f"DELETE FROM {table_name} WHERE loan_id = ?", (self.loan_id,))
                # Aggiungi i nuovi dati del prestito
                self.table['loan_id'] = self.loan_id
                self.table.to_sql(table_name, conn, if_exists='append', index_label='Date')
                print(f"Loan table updated in {db_path} in table {table_name}")
            except sqlite3.OperationalError as e:
                print(f"An error occurred: {e}")
                raise

    @classmethod
    def get_loan_by_id(cls, loan_id, table_name='loan_table'):
        db_path = os.path.join(Loan.db_directory, 'loans.db')
        with sqlite3.connect(db_path) as conn:
            query = f"SELECT * FROM {table_name} WHERE loan_id = ?"
            loan_data = pd.read_sql(query, conn, params=(loan_id,))
        if loan_data.empty:
            print("No loan found with the given ID.")
            return None
        loan_details = loan_data.iloc[0]
        loan = Loan(
            rate=loan_details['Rate'],
            term=loan_details['Term'],
            loan_amount=loan_details['LoanAmount'],
            amortization_type=loan_details['AmortizationType'],
            frequency=loan_details['Frequency'],
            start=loan_details['Start'],
            loan_id=loan_id
        )
        loan.table = loan_data.set_index('Date')
        loan.table.index = pd.to_datetime(loan.table.index)
        Loan.loans.append(loan)
        return loan

    @classmethod
    def compare_loans(cls, loans):
        if len(loans) < 2:
            return "Please set at least two loans for comparison."

        results = []
        
        for i, loan in enumerate(loans):
            if not loan.taeg:
                loan.calculate_taeg()

            if loan.amortization_type == "French":
                monthly_payment = loan.pmt
            elif loan.amortization_type == "Italian":
                monthly_payment = loan.table['Payment'].iloc[0]

            results.append(f"Loan {i + 1} - Monthly Payment: €{monthly_payment:,.2f}")
            results.append(f"Loan {i + 1} - TAEG Periodic: {loan.taeg['periodic']:.2f}%")
            results.append(f"Loan {i + 1} - TAEG Annualized: {loan.taeg['annualized']:.2f}%")
            results.append(f"Loan {i + 1} - Interest Paid: €{loan.table['Interest'].cumsum().iloc[-1]:,.2f}")
            results.append("-" * 60)
        
        # Identificare il prestito più conveniente per il mutuatario
        min_periodic_taeg_loan = min(loans, key=lambda loan: loan.taeg['periodic'])
        min_annualized_taeg_loan = min(loans, key=lambda loan: loan.taeg['annualized'])
        
        results.append(f"Most convenient loan for Borrower based on Periodic TAEG: Loan {loans.index(min_periodic_taeg_loan) + 1} (TAEG Periodic: {min_periodic_taeg_loan.taeg['periodic']:.2f}%)")
        results.append(f"Most convenient loan for Borrower based on Annualized TAEG: Loan {loans.index(min_annualized_taeg_loan) + 1} (TAEG Annualized: {min_annualized_taeg_loan.taeg['annualized']:.2f}%)")
        
        # Identificare il prestito più conveniente per il prestatore
        max_periodic_taeg_loan = max(loans, key=lambda loan: loan.taeg['periodic'])
        max_annualized_taeg_loan = max(loans, key=lambda loan: loan.taeg['annualized'])
        
        results.append(f"Most profitable loan for Lender based on Periodic TAEG: Loan {loans.index(max_periodic_taeg_loan) + 1} (TAEG Periodic: {max_periodic_taeg_loan.taeg['periodic']:.2f}%)")
        results.append(f"Most profitable loan for Lender based on Annualized TAEG: Loan {loans.index(max_annualized_taeg_loan) + 1} (TAEG Annualized: {max_annualized_taeg_loan.taeg['annualized']:.2f}%)")
        
        # Identificare il prestito di equilibrio
        equilibrium_loan = min(loans, key=lambda loan: abs(loan.taeg['periodic'] - loan.taeg['annualized']))
        
        results.append(f"\nOptimal Equilibrium Loan: Loan {loans.index(equilibrium_loan) + 1}")
        results.append(f"Optimal Loan TAEG: {equilibrium_loan.taeg['periodic']:.2f}% (Periodic), {equilibrium_loan.taeg['annualized']:.2f}% (Annualized)")
        
        results.append("-" * 60)
        return "\n".join(results)

    @classmethod
    def display_loans(cls):
        """
        Display all loans with their details for selection.
        """
        if not cls.loans:
            print("No loans available.")
            return

        for idx, loan in enumerate(cls.loans):
            print(f"{idx + 1}: Loan ID: {loan.loan_id}, Amount: €{loan.loan_amount:,.2f}, Rate: {loan.initial_rate * 100:.2f}%, Term: {loan.initial_term} years")

    @classmethod
    def delete_loan(cls, loan_idx, table_name='loan_table'):
        """
        Delete a loan based on its index in the list.
        """
        try:
            loan = cls.loans.pop(loan_idx)
            db_path = os.path.join(Loan.db_directory, 'loans.db')
            with sqlite3.connect(db_path) as conn:
                conn.execute(f"DELETE FROM {table_name} WHERE loan_id = ?", (loan.loan_id,))
            print(f"Loan with ID {loan.loan_id} has been deleted.")
        except IndexError:
            print("Invalid loan index.")

    @classmethod
    def delete_loan_with_confirmation(cls):
        """
        Select a loan to delete with user confirmation.
        """
        cls.display_loans()
        loan_idx = int(input("Enter the loan number you want to delete: ")) - 1
        if 0 <= loan_idx < len(cls.loans):
            loan = cls.loans[loan_idx]
            confirm = input(f"Are you sure you want to delete the loan with ID {loan.loan_id}? (yes/no): ").strip().lower()
            if confirm == 'yes':
                cls.delete_loan(loan_idx)
            else:
                print("Deletion cancelled.")
        else:
            print("Invalid loan number.")

    @classmethod
    def set_db_directory(cls, directory):
        """
        Set the directory where the database will be saved.
        """
        if os.path.isdir(directory):
            cls.db_directory = directory
            print(f"Database directory set to: {directory}")
        else:
            print(f"Invalid directory: {directory}")

    @classmethod
    def consolidate_loans(cls, selected_loans, frequency):
        if not selected_loans or len(selected_loans) < 2:
            raise ValueError("At least two loans must be selected for consolidation.")

        # Calcola il totale dell'importo del prestito
        total_amount = sum(loan.loan_amount for loan in selected_loans)

        def convert_to_frequency(loan, target_frequency):
            conversion_factors = {
                'monthly': 12,
                'quarterly': 4,
                'semi-annual': 2,
                'annual': 1
            }

            current_factor = conversion_factors[loan.frequency]
            target_factor = conversion_factors[target_frequency]

            target_rate = (1 + loan.initial_rate) ** (target_factor / current_factor) - 1
            target_payment = npf.pmt(target_rate, loan.periods * (current_factor / target_factor), loan.loan_amount)
            return target_rate, abs(target_payment)

        def convert_taeg_to_frequency(loan, target_frequency):
            conversion_factors = {
                'monthly': 12,
                'quarterly': 4,
                'semi-annual': 2,
                'annual': 1
            }

            current_factor = conversion_factors[loan.frequency]
            target_factor = conversion_factors[target_frequency]

            if 'periodic' in loan.taeg:
                target_taeg_periodic = (1 + loan.taeg['periodic'] / 100) ** (target_factor / current_factor) - 1
            else:
                target_taeg_periodic = loan.rate  # Fallback to the loan rate if TAEG is not available
            return target_taeg_periodic

        weighted_rate_sum = 0
        weighted_term_sum = 0
        amortization_types = {}
        weighted_payments_sum = 0
        weighted_taeg_sum = 0

        for loan in selected_loans:
            # Ensure TAEG is calculated for each loan
            if not loan.taeg:
                loan.calculate_taeg()

            target_rate, target_payment = convert_to_frequency(loan, frequency)
            target_taeg_periodic = convert_taeg_to_frequency(loan, frequency)

            weighted_rate_sum += target_rate * loan.loan_amount
            weighted_term_sum += loan.initial_term * loan.loan_amount
            weighted_payments_sum += target_payment * loan.loan_amount
            weighted_taeg_sum += target_taeg_periodic * loan.loan_amount
            amortization_types[loan.amortization_type] = amortization_types.get(loan.amortization_type, 0) + loan.loan_amount

        average_rate = weighted_rate_sum / total_amount
        average_term = weighted_term_sum / total_amount
        average_payment = weighted_payments_sum / total_amount
        average_taeg_periodic = weighted_taeg_sum / total_amount

        conversion_factors = {
            'monthly': 12,
            'quarterly': 4,
            'semi-annual': 2,
            'annual': 1
        }
        periods_per_year = conversion_factors[frequency]
        average_taeg_annualized = (1 + average_taeg_periodic) ** periods_per_year - 1

        amortization_type = max(amortization_types, key=amortization_types.get)

        consolidated_loan = cls(
            rate=average_rate,
            term=int(round(average_term, 2)),
            loan_amount=total_amount,
            amortization_type=amortization_type,
            frequency=frequency,
        )

        consolidated_loan.taeg = {
            'periodic': average_taeg_periodic * 100,
            'annualized': average_taeg_annualized * 100
        }

        consolidated_loan.save_to_db(table_name="consolidated_loans")

        return consolidated_loan