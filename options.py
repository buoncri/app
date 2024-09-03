import numpy as np
import pandas as pd
import statistics as st
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, t
from datetime import date


#
#TODO: Considera di prezzare, OLTRE ALLE COSE GIA' ESISTENTI, anche delle opzioni composte, lookback discrete, opzioni barriere con rimborsi continui, o opzioni asiatiche con pesi variabili.
#TODO: Valutazione di prodotti derivati più complessi: gli  interest rate swap, currency swap, e commodity swap; valutazione di futures su indici, materie prime, e tassi d'interesse; nonché dei credit default swap (CDS): Valutazione di CDS per gestire il rischio di credito.
#TODO: Implementa modelli come il modello di Heston per volatilità stocastica, modelli a salto, o modelli a più fattori.
#TODO: Implementa algoritmi di calibrazione per stimare i parametri dei modelli di pricing in base ai dati di mercato.
#TODO: Estendi l'analisi a portafogli di opzioni, calcolando il valore a rischio (VaR) e il valore a rischio condizionale (CVaR). Di questa cosa ho già il codice, devo solo integrarlo nella classe. 
#TODO: Consenti l'importazione e l'esportazione di dati da e verso database
#TODO: Genera grafici interattivi per visualizzare i risultati delle analisi.
#TODO: Inserisci la funzionalità di un "copilot" (un bot basato sull'intelligenza artificiale), per guidare l'utente sulle funzionalità dell'applicazione quando quest'ultimo lo richiede  
 

class Options:
    def __init__(self, S, K, T, r, sigma, y=0):
        """
        Initialize the Options object.
        
        Parameters:
        S (float): Underlying asset price
        K (float): Strike price
        T (float): Time to maturity in years
        r (float): Risk-free interest rate
        sigma (float): Volatility
        """
        self.S = S          # |Underlying price
        self.K = K          # |Strike price
        self.T = T          # |Time to maturity in years
        self.r = r          # |Risk-free interest rate
        self.sigma = sigma  # |Volatility
        self.y = y          # |Dividen yield (annual)

    def download_data(tickers, start_date, end_date):
        """
        Download historical stock data from Yahoo Finance.
        
        Parameters:
        tickers (list): List of ticker symbols
        start_date (str): Start date for data download (YYYY-MM-DD)
        end_date (str): End date for data download (YYYY-MM-DD)
        
        Returns:
        dict: Dictionary with ticker symbols as keys and DataFrames of historical data as values
        """
        data = {}
        for ticker in tickers:
            df = yf.download(ticker, start=start_date, end=end_date)
            data[ticker] = df['Adj Close']
        return data

    def log_returns(prices):
        """
        Calculate log returns of given price series.
        
        Parameters:
        prices (pd.Series): Series of prices
        
        Returns:
        pd.Series: Series of log returns
        """
        return np.log(prices / prices.shift(1)).dropna()

    def test_normality(log_returns):
        """
        Test the normality of log returns and fit distributions to the histogram.
        
        Parameters:
        log_returns (pd.Series): Series of log returns
        
        Returns:
        None
        """
        # Test for normality using Shapiro-Wilk test
        shapiro_test = stats.shapiro(log_returns)
        print(f"Shapiro-Wilk Test: Statistics={shapiro_test.statistic}, p-value={shapiro_test.pvalue}")
        
        # Test for normality using Kolmogorov-Smirnov test
        kstest_result = stats.kstest(log_returns, 'norm', args=(log_returns.mean(), log_returns.std()))
        print(f"Kolmogorov-Smirnov Test: Statistics={kstest_result.statistic}, p-value={kstest_result.pvalue}")
        
        # Determine normality based on p-values
        alpha = 0.05
        is_normal_shapiro = shapiro_test.pvalue > alpha
        is_normal_ks = kstest_result.pvalue > alpha
        is_normal = is_normal_shapiro and is_normal_ks
        
        # Print the result
        if is_normal:
            print("The log returns are normally distributed according to both tests.")
        else:
            print("The log returns are NOT normally distributed according to both tests.")
        
        # Fit distributions and plot
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax[0].hist(log_returns, bins=30, density=True, alpha=0.6, color='g', label='Log Returns')
        
        # Fit normal distribution
        mu, std = stats.norm.fit(log_returns)
        p = stats.norm.pdf(np.linspace(log_returns.min(), log_returns.max(), 100), mu, std)
        ax[0].plot(np.linspace(log_returns.min(), log_returns.max(), 100), p, 'k', linewidth=2, label='Normal Fit')
        
        # Fit t-distribution
        params = stats.t.fit(log_returns)
        t_dist = stats.t.pdf(np.linspace(log_returns.min(), log_returns.max(), 100), *params)
        ax[0].plot(np.linspace(log_returns.min(), log_returns.max(), 100), t_dist, 'r', linewidth=2, label='T-distribution Fit')
        
        ax[0].legend()
        ax[0].set_title('Histogram and Distribution Fits')
        
        # QQ-Plot for normal distribution
        stats.probplot(log_returns, dist="norm", plot=ax[1])
        ax[1].set_title('QQ-Plot vs Normal Distribution')
        
        # Add normality result to the plot
        result_text = "Normal" if is_normal else "Not Normal"
        plt.suptitle(f"Normality Test Result: {result_text}", fontsize=16, y=1.02)
        
        plt.tight_layout()
        plt.show()
    def analyze_tickers(tickers):
        """
        Analyze multiple tickers: download data, calculate log returns, and test normality.
        
        Parameters:
        tickers (list): List of ticker symbols
        
        Returns:
        None
        """
        data = download_data(tickers)
        for ticker, prices in data.items():
            print(f"\nAnalyzing {ticker}")
            log_ret = log_returns(prices)
            test_normality(log_ret)
    def binomial_option_pricing(self, option_type='call', n=100, american=False):
        """
        Binomial option pricing model.
        
        Parameters:
        option_type (str): 'call' or 'put'
        n (int): Number of binomial steps
        american (bool): True for American option, False for European option
        
        Returns:
        float: Option price
        """
        S0 = self.S
        K = self.K
        r = self.r
        T = self.T
        sigma = self.sigma
        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        m = np.exp(r * dt)
        q = (m - d) / (u - d)

        S = np.zeros((n + 1, n + 1))
        S[0, 0] = S0
        for t in range(1, n + 1):
            for j in range(t + 1):
                S[j, t] = S0 * (u ** (t - j)) * (d ** j)

        option = np.zeros_like(S)
        for j in range(n + 1):
            option[j, n] = max(0, (K - S[j, n]) if option_type == 'put' else (S[j, n] - K))

        for t in range(n - 1, -1, -1):
            for j in range(t + 1):
                if american:
                    option_value = (1 / m) * (q * option[j, t + 1] + (1 - q) * option[j + 1, t + 1])
                    exercise_value = max(0, (K - S[j, t]) if option_type == 'put' else (S[j, t] - K))
                    option[j, t] = max(option_value, exercise_value)
                else:
                    option[j, t] = (1 / m) * (q * option[j, t + 1] + (1 - q) * option[j + 1, t + 1])

        return option[0, 0]

    def black_scholes(self, option_type='call', option_kind='european'):
        """
        Black-Scholes option pricing model.
        
        Parameters:
        option_type (str): 'call' or 'put'
        option_kind (str): 'european' or 'american'
        
        Returns:
        dict: Dictionary containing option price and Greeks (delta, gamma, vega, theta, rho, vomma, vanna, charm, vera, veta, speed, zomma, color, ultima, dual delta, dual gamma)
        """
        S = self.S
        K = self.K
        T = self.T
        r = self.r
        sigma = self.sigma
        y = self.y
        
        d1 = (np.log(S / K) + (r - y + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        delta = None
        gamma = None
        vega = None
        theta = None
        rho = None
        vomma = None
        vanna = None
        charm = None
        vera = None
        veta = None
        speed = None
        zomma = None
        color = None
        ultima = None
        dual_delta = None
        dual_gamma = None
        
        if option_kind == 'european':       
            if option_type == 'call':
                price = S * np.exp(-y * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                delta = np.exp(-y * T) * norm.cdf(d1)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                         - r * K * np.exp(-r * T) * norm.cdf(d2)
                         + y * S * np.exp(-y * T) * norm.cdf(d1))

                         
            elif option_type == 'put':
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-y * T) * norm.cdf(-d1)
                delta = -np.exp(-y * T) * norm.cdf(-d1)
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                         + r * K * np.exp(-r * T) * norm.cdf(-d2)
                         - y * S * np.exp(-y * T) * norm.cdf(-d1))
                
            gamma = (norm.pdf(d1) * np.exp(-y * T)) / (S * sigma * np.sqrt(T))
            vega = S * np.exp(-y * T) * norm.pdf(d1) * np.sqrt(T)
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)
            vomma = vega * (d1 * d2) / sigma
            vanna = vega * (1 - d1 / (sigma * np.sqrt(T)))
            charm = -np.exp(-y * T) * (norm.pdf(d1) * (2 * (r - y) * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)))
            vera = S * T * np.exp(-y * T) * norm.pdf(d1) * d1 / sigma
            veta = -S * np.exp(-y * T) * norm.pdf(d1) * d1 * np.sqrt(T)
            speed = -gamma / S * (d1 / (sigma * np.sqrt(T)) + 1)
            zomma = gamma * (d1 * d2 - 1) / sigma
            color = -gamma / (2 * T) * (2 * y * T + 1 + d1 * d2 / sigma * np.sqrt(T))
            ultima = -vega / (sigma ** 2) * (d1 * d2 * (1 - d1 * d2) + d1 ** 2 + d2 ** 2)
            dual_delta = -np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else np.exp(-r * T) * norm.cdf(-d2)
            dual_gamma = np.exp(-r * T) * norm.pdf(d2) / (K * sigma * np.sqrt(T))
        
        elif option_kind == 'american':
            price = self.binomial_tree(option_type)
            (delta, gamma, theta, vega, rho, vomma, vanna, charm, vera, veta, speed, 
             zomma, color, ultima, dual_delta, dual_gamma) = self.numerical_greeks(option_type)
        
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho,
            'vomma': vomma,
            'vanna': vanna,
            'charm': charm,
            'vera': vera,
            'veta': veta,
            'speed': speed,
            'zomma': zomma,
            'color': color,
            'ultima': ultima,
            'dual delta': dual_delta,
            'dual gamma': dual_gamma
        }

    def binomial_tree(self, option_type, steps=100000):
        """
        Binomial tree method for American option pricing.
        """
        S, K, T, r, sigma, y = self.S, self.K, self.T, self.r, self.sigma, self.y
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((r - y) * dt) - d) / (u - d)
        
        # Initialize asset prices at maturity
        ST = np.zeros(steps + 1)
        ST[0] = S * d ** steps
        for i in range(1, steps + 1):
            ST[i] = ST[i - 1] * u / d
        
        # Initialize option values at maturity
        option_values = np.maximum(0, (ST - K) if option_type == 'call' else (K - ST))
        
        # Step backwards through the tree
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j] = np.exp(-r * dt) * (q * option_values[j + 1] + (1 - q) * option_values[j])
                ST[j] = ST[j] * u / d
                option_values[j] = np.maximum(option_values[j], (ST[j] - K) if option_type == 'call' else (K - ST[j]))
        
        return option_values[0]

    def numerical_greeks(self, option_type, epsilon=1e-5):
        """
        Numerical method to calculate Greeks for American options.
        """
        base_price = self.binomial_tree(option_type)
        
        # Delta
        self.S += epsilon
        price_up = self.binomial_tree(option_type)
        self.S -= 2 * epsilon
        price_down = self.binomial_tree(option_type)
        self.S += epsilon
        delta = (price_up - price_down) / (2 * epsilon)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)
        
        # Vega
        self.sigma += epsilon
        price_up = self.binomial_tree(option_type)
        self.sigma -= epsilon
        vega = (price_up - base_price) / epsilon
        
        # Theta
        self.T -= epsilon
        price_down = self.binomial_tree(option_type)
        self.T += epsilon
        theta = (price_down - base_price) / epsilon
        
        # Rho
        self.r += epsilon
        price_up = self.binomial_tree(option_type)
        self.r -= epsilon
        rho = (price_up - base_price) / epsilon
        
        # Vomma
        self.sigma += epsilon
        vega_up = (self.binomial_tree(option_type) - base_price) / epsilon
        self.sigma -= 2 * epsilon
        vega_down = (self.binomial_tree(option_type) - base_price) / epsilon
        self.sigma += epsilon
        vomma = (vega_up - vega_down) / (2 * epsilon)
        
        # Vanna
        self.S += epsilon
        vega_up = (self.binomial_tree(option_type) - base_price) / epsilon
        self.S -= 2 * epsilon
        vega_down = (self.binomial_tree(option_type) - base_price) / epsilon
        self.S += epsilon
        vanna = (vega_up - vega_down) / (2 * epsilon)
        
        # Charm
        self.T += epsilon
        delta_up = (self.binomial_tree(option_type) - base_price) / epsilon
        self.T -= 2 * epsilon
        delta_down = (self.binomial_tree(option_type) - base_price) / epsilon
        self.T += epsilon
        charm = (delta_up - delta_down) / (2 * epsilon)
        
        # Vera
        self.T += epsilon
        vomma_up = (self.binomial_tree(option_type) - base_price) / epsilon
        self.T -= 2 * epsilon
        vomma_down = (self.binomial_tree(option_type) - base_price) / epsilon
        self.T += epsilon
        vera = (vomma_up - vomma_down) / (2 * epsilon)
        
        # Veta
        self.T -= epsilon
        vega_down = (self.binomial_tree(option_type) - base_price) / epsilon
        self.T += epsilon
        veta = (vega_down - base_price) / epsilon
        
        # Speed
        self.S += epsilon
        gamma_up = (self.binomial_tree(option_type) - 2 * base_price + price_down) / (epsilon ** 2)
        self.S -= 2 * epsilon
        gamma_down = (self.binomial_tree(option_type) - 2 * base_price + price_up) / (epsilon ** 2)
        self.S += epsilon
        speed = (gamma_up - gamma_down) / (2 * epsilon)
        
        # Zomma
        zomma = (gamma_up - 2 * gamma + gamma_down) / (epsilon ** 2)
        
        # Color
        color = (gamma_up - gamma_down) / (2 * epsilon ** 2)
        
        # Ultima
        self.sigma += epsilon
        vomma_up = (self.binomial_tree(option_type) - base_price) / epsilon
        self.sigma -= 2 * epsilon
        vomma_down = (self.binomial_tree(option_type) - base_price) / epsilon
        self.sigma += epsilon
        ultima = (vomma_up - vomma_down) / (2 * epsilon)
        
        # Dual Delta
        self.K += epsilon
        price_up = self.binomial_tree(option_type)
        self.K -= 2 * epsilon
        price_down = self.binomial_tree(option_type)
        self.K += epsilon
        dual_delta = (price_up - price_down) / (2 * epsilon)
        
        # Dual Gamma
        dual_gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)
        
        return (delta, gamma, theta, vega, rho, vomma, vanna, charm, vera, veta, speed, 
                zomma, color, ultima, dual_delta, dual_gamma)

    def monte_carlo(self, option_type, simulations=100000):
        """
        Monte Carlo method for American option pricing.
        """
        S, K, T, r, sigma, y = self.S, self.K, self.T, self.r, self.sigma, self.y
        dt = T / simulations
        discount_factor = np.exp(-r * T)
        
        # Simulate end-of-period prices
        Z = np.random.standard_normal(simulations)
        ST = S * np.exp((r - y - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        
        # Payoff at maturity
        if option_type == 'call':
            payoff = np.maximum(ST - K, 0)
        else:
            payoff = np.maximum(K - ST, 0)
        
        return np.mean(payoff) * discount_factor

    def numerical_greeks_monte_carlo(self, option_type, epsilon=1e-5, simulations=100000):
        """
        Numerical method to calculate Greeks for American options using Monte Carlo.
        """
        base_price = self.monte_carlo(option_type, simulations)
        
        # Delta
        self.S += epsilon
        price_up = self.monte_carlo(option_type, simulations)
        self.S -= 2 * epsilon
        price_down = self.monte_carlo(option_type, simulations)
        self.S += epsilon
        delta = (price_up - price_down) / (2 * epsilon)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)
        
        # Vega
        self.sigma += epsilon
        price_up = self.monte_carlo(option_type, simulations)
        self.sigma -= epsilon
        vega = (price_up - base_price) / epsilon
        
        # Theta
        self.T -= epsilon
        price_down = self.monte_carlo(option_type, simulations)
        self.T += epsilon
        theta = (price_down - base_price) / epsilon
        
        # Rho
        self.r += epsilon
        price_up = self.monte_carlo(option_type, simulations)
        self.r -= epsilon
        rho = (price_up - base_price) / epsilon
        
        # Vomma
        self.sigma += epsilon
        vega_up = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.sigma -= 2 * epsilon
        vega_down = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.sigma += epsilon
        vomma = (vega_up - vega_down) / (2 * epsilon)
        
        # Vanna
        self.S += epsilon
        vega_up = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.S -= 2 * epsilon
        vega_down = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.S += epsilon
        vanna = (vega_up - vega_down) / (2 * epsilon)
        
        # Charm
        self.T += epsilon
        delta_up = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.T -= 2 * epsilon
        delta_down = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.T += epsilon
        charm = (delta_up - delta_down) / (2 * epsilon)
        
        # Vera
        self.T += epsilon
        vomma_up = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.T -= 2 * epsilon
        vomma_down = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.T += epsilon
        vera = (vomma_up - vomma_down) / (2 * epsilon)
        
        # Veta
        self.T -= epsilon
        vega_down = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.T += epsilon
        veta = (vega_down - base_price) / epsilon
        
        # Speed
        self.S += epsilon
        gamma_up = (self.monte_carlo(option_type, simulations) - 2 * base_price + price_down) / (epsilon ** 2)
        self.S -= 2 * epsilon
        gamma_down = (self.monte_carlo(option_type, simulations) - 2 * base_price + price_up) / (epsilon ** 2)
        self.S += epsilon
        speed = (gamma_up - gamma_down) / (2 * epsilon)
        
        # Zomma
        zomma = (gamma_up - 2 * gamma + gamma_down) / (epsilon ** 2)
        
        # Color
        color = (gamma_up - gamma_down) / (2 * epsilon ** 2)
        
        # Ultima
        self.sigma += epsilon
        vomma_up = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.sigma -= 2 * epsilon
        vomma_down = (self.monte_carlo(option_type, simulations) - base_price) / epsilon
        self.sigma += epsilon
        ultima = (vomma_up - vomma_down) / (2 * epsilon)
        
        # Dual Delta
        self.K += epsilon
        price_up = self.monte_carlo(option_type, simulations)
        self.K -= 2 * epsilon
        price_down = self.monte_carlo(option_type, simulations)
        self.K += epsilon
        dual_delta = (price_up - price_down) / (2 * epsilon)
        
        # Dual Gamma
        dual_gamma = (price_up - 2 * base_price + price_down) / (epsilon ** 2)
        
        return (delta, gamma, theta, vega, rho, vomma, vanna, charm, vera, veta, speed, 
                zomma, color, ultima, dual_delta, dual_gamma)


    def exotic_option_pricing(self, exotic_type='asian', option_type='call', num_simulations=10000, 
                            barrier=None, chooser_date=None, asian_type='average_price', 
                            print_results=False, plot_results=False):
        """
        Price various exotic options using the Monte Carlo method.
        
        Parameters:
        exotic_type (str): Type of exotic option, e.g., 'asian', 'barrier', 'lookback', 'digital', 'chooser', 'quanto', 'compound'
        option_type (str): 'call' or 'put'
        num_simulations (int): Number of Monte Carlo simulations
        barrier (float): Barrier level for barrier options (used only if exotic_type is 'barrier')
        chooser_date (float): Time (in years) at which the chooser option decides between call and put (used only if exotic_type is 'chooser')
        asian_type (str): Type of Asian option, 'average_price' or 'average_strike' (used only if exotic_type is 'asian')
        print_results (bool): If True, prints the pricing results to the console
        plot_results (bool): If True, plots the payoff distribution
        
        Returns:
        float: Option price
        """
        S, K, T, r, sigma, y = self.S, self.K, self.T, self.r, self.sigma, self.y
        dt = T / num_simulations
        discount_factor = np.exp(-r * T)
        
        # Simulate paths
        Z = np.random.standard_normal(num_simulations)
        paths = np.zeros((num_simulations, int(T/dt)))
        paths[:, 0] = S
        
        for t in range(1, paths.shape[1]):
            Z = np.random.standard_normal(num_simulations)
            paths[:, t] = paths[:, t-1] * np.exp((r - y - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        
        if exotic_type == 'asian':
            avg_price = np.mean(paths, axis=1)
            if asian_type == 'average_price':
                # Average Price Asian Option: Payoff based on average price compared to fixed strike
                payoff = np.maximum(avg_price - K, 0) if option_type == 'call' else np.maximum(K - avg_price, 0)
            elif asian_type == 'average_strike':
                # Average Strike Asian Option: Payoff based on final price compared to average price as strike
                payoff = np.maximum(paths[:, -1] - avg_price, 0) if option_type == 'call' else np.maximum(avg_price - paths[:, -1], 0)
            else:
                raise ValueError("Unsupported asian_type. Use 'average_price' or 'average_strike'.")
        
        elif exotic_type == 'barrier':
            if barrier is None:
                raise ValueError("Barrier level must be provided for barrier options.")
            if option_type == 'call':
                payoff = np.where((paths.min(axis=1) > barrier) if 'up-and-out' else (paths.max(axis=1) < barrier),
                                np.maximum(paths[:, -1] - K, 0),
                                0)
            else:
                payoff = np.where((paths.min(axis=1) > barrier) if 'up-and-out' else (paths.max(axis=1) < barrier),
                                np.maximum(K - paths[:, -1], 0),
                                0)
        
        elif exotic_type == 'lookback':
            if option_type == 'call':
                payoff = np.maximum(paths.max(axis=1) - K, 0)
            else:
                payoff = np.maximum(K - paths.min(axis=1), 0)
        
        elif exotic_type == 'digital':
            if option_type == 'call':
                payoff = np.where(paths[:, -1] > K, 1, 0)
            else:
                payoff = np.where(paths[:, -1] < K, 1, 0)
        
        elif exotic_type == 'chooser':
            if chooser_date is None:
                raise ValueError("Chooser date must be provided for chooser options.")
            chooser_time = int(chooser_date / dt)
            call_payoff = np.maximum(paths[:, -1] - K, 0)
            put_payoff = np.maximum(K - paths[:, -1], 0)
            payoff = np.maximum(call_payoff, put_payoff) if chooser_time <= len(paths[0]) else 0
        
        elif exotic_type == 'quanto':
            exchange_rate = 1.2  # Example fixed exchange rate
            if option_type == 'call':
                payoff = exchange_rate * np.maximum(paths[:, -1] - K, 0)
            else:
                payoff = exchange_rate * np.maximum(K - paths[:, -1], 0)
        
        elif exotic_type == 'compound':
            if option_type == 'call':
                underlying_option_price = self.black_scholes(option_type='call')['price']
                payoff = np.maximum(underlying_option_price - K, 0)
            else:
                underlying_option_price = self.black_scholes(option_type='put')['price']
                payoff = np.maximum(K - underlying_option_price, 0)
        
        else:
            raise ValueError("Unsupported exotic option type.")
        
        # Calculate the present value of the expected payoff
        option_price = np.mean(payoff) * discount_factor
        
        if print_results:
            print(f"Option Type: {exotic_type.capitalize()} {option_type.capitalize()}")
            print(f"Number of Simulations: {num_simulations}")
            print(f"Estimated Option Price: {option_price:.4f}")
        
        if plot_results:
            plt.figure(figsize=(10, 6))
            plt.hist(payoff, bins=50, color='blue', alpha=0.7)
            plt.title(f"{exotic_type.capitalize()} {option_type.capitalize()} Option Payoff Distribution")
            plt.xlabel("Payoff")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.show()
        
        return option_price

    def historical_volatiliIty(self, ticker, period='1y', frequency='1d'):
        """
        Calculate historical volatility of a stock.
        
        Parameters:
        ticker (str): Ticker symbol of the stock
        period (str): Historical data period
        frequency (str): Data frequency
        
        Returns:
        float: Annualized volatility
        """
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period=period, interval=frequency)
        prices = pd.DataFrame(hist_data["Close"])
        prices["LogRet"] = np.log(prices["Close"] / prices["Close"].shift(-1))
        log_ret = prices["LogRet"].dropna()
        vol_day = st.stdev(log_ret)
        vol_year = vol_day * np.sqrt(250)
        return vol_year

    def rolling_historical_volatility(self, ticker, period='5y'):
        """
        Calculate rolling historical volatility of a stock.
        
        Parameters:
        ticker (str): Ticker symbol of the stock
        period (str): Historical data period
        
        Returns:
        list: List of annualized volatilities
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        data['Daily_Return'] = data['Close'].pct_change().dropna()
        days_per_year = 250

        volatility = []
        for i in range(0, len(data) - days_per_year, days_per_year):
            window = data.iloc[i:i + days_per_year]
            annualized_volatility = np.std(window['Daily_Return']) * np.sqrt(days_per_year)
            volatility.append(annualized_volatility)

        return volatility

    def implied_volatility(self, ticker, r, maturity):
        """
        Calculate implied volatility for a stock's options.
        
        Parameters:
        ticker (str): Ticker symbol of the stock
        r (float): Risk-free interest rate
        maturity (str): Option maturity date
        
        Returns:
        tuple: DataFrames containing implied volatilities for call and put options
        """
        stock = yf.Ticker(ticker)
        hist_data = stock.history()
        S0 = hist_data["Close"].iloc[-1]
        T = (date.fromisoformat(maturity) - date.today()).days / 365
        options = stock.option_chain(maturity)
        calls = options.calls
        puts = options.puts
        
        calls = pd.DataFrame(calls[["strike", "lastPrice"]])
        puts = pd.DataFrame(puts[["strike", "lastPrice"]])

        calls.rename(columns={"strike": "K", "lastPrice": "C"}, inplace=True)
        puts.rename(columns={"strike": "K", "lastPrice": "P"}, inplace=True)
        
        calls = calls[(calls["K"] <= S0 * 3) & (calls["K"] >= S0 * 0.7)]
        puts = puts[(puts["K"] <= S0 * 3) & (puts["K"] >= S0 * 0.7)]

        calls["sigma"] = calls.apply(lambda row: self.calculate_implied_volatility(S0, row["K"], T, r, row["C"], 'call'), axis=1)
        puts["sigma"] = puts.apply(lambda row: self.calculate_implied_volatility(S0, row["K"], T, r, row["P"], 'put'), axis=1)

        return calls, puts

    def calculate_implied_volatility_bisection(self, S0, K, T, r, market_price, option_type):
        """
        Calculate implied volatility using the bisection method.
        
        Returns:
        float: Implied volatility
        """
        high = 10.0
        low = 0.0
        tol = 1e-6
        max_iterations = 100
        bisection_vols = []

        for _ in range(max_iterations):
            mid = (high + low) / 2.0
            self.sigma = mid
            model_price = self.black_scholes(option_type=option_type)['price']
            bisection_vols.append(mid)
            if abs(model_price - market_price) < tol:
                break
            elif model_price > market_price:
                high = mid
            else:
                low = mid

        return (high + low) / 2.0, bisection_vols
    
    def calculate_implied_volatility_newton(self, S0, K, T, r, market_price, option_type):
        """
        Calculate implied volatility using the Newton-Raphson method.
        
        Returns:
        float: Implied volatility
        """
        sigma = 0.5  # Initial guess
        tol = 1e-6
        max_iterations = 100
        newton_vols = []
        
        for _ in range(max_iterations):
            self.sigma = sigma
            result = self.black_scholes(option_type=option_type)
            price = result['price']
            vega = self.calculate_vega(S0, K, T, r, sigma, option_type)
            newton_vols.append(sigma)
            
            diff = price - market_price  # Difference between model price and market price
            
            if abs(diff) < tol:
                break
            
            sigma -= diff / vega  # Update sigma using Newton-Raphson formula
        
        return sigma, newton_vols

    def plot_convergence(self, bisection_vols, newton_vols):
        plt.figure(figsize=(10, 6))
        plt.plot(bisection_vols, label='Bisection Method', marker='o')
        plt.plot(newton_vols, label='Newton-Raphson Method', marker='x')
        plt.xlabel('Iteration')
        plt.ylabel('Implied Volatility')
        plt.title('Convergence of Implied Volatility Estimates')
        plt.legend()
        plt.grid(True)
        plt.show()