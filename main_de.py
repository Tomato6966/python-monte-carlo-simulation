"""
Monte-Carlo-Simulator für ein Core-Satellite-Depot bestehend aus MSCI World (Core) 
und Amundi Leveraged MSCI USA (Satellite). Ermöglicht unterschiedliche
Depotgewichtungen, Laufzeiten und Sparraten zu simulieren.

Version 3.0 (UI)

Unterstützt optionales Rebalancing:
- "none" (Standard): kein Rebalancing
- "annual": jährlich am 252. Tag
- "dynamic:<max_satellite_weight>": wenn Satellit > max. Anteil

History:
- 2.0: Initiale Version
- 2.1: Tagesrenditen wahlweise aus Normal- oder t-Verteilung (Fat Tails).
- 2.2: Seed optional über CLI konfigurierbar für vollständig zufällige Simulationen.
- 2.3: Ausgabe von IRR (internem Zinsfuß) pro Perzentil bei Sparplänen
- 2.4: Korrelation zwischen Core- und Satelliten-Asset steuerbar (Standard: 0.96)
- 2.5: Realistische Hebel-ETF-Modellierung: der Satellite-ETF wird als 3x gehebelter Indexpfad simuliert
- 3.0: Keine CLI-Config mehr notwendig, alles im UI nutzbar, Daten via Yahoo Finance, CERT Issues ignoriert. 
       Output-Graph ist interaktiv (bewegbar und zoombar). Output mit Max Drawdown & Recovery Time ergänzt (@tomato6966 (github) / @chrissy8283 (discord))
- 3.1: Added detailed function descriptions and comments for crucial steps

Neuer Pip Install: pip install numpy matplotlib pandas scipy requests
Wichtig: Python mit tkinter installieren: sudo apt-get install python-tk
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import matplotlib.ticker as mticker
import datetime
import json
import requests
import warnings
from scipy.optimize import brentq
from requests.packages.urllib3.exceptions import InsecureRequestWarning

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import threading
import multiprocessing
import time 
import webbrowser

warnings.filterwarnings('ignore', category=InsecureRequestWarning)
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

API_BASE = "https://query1.finance.yahoo.com"
YAHOO_FINANCE_URL = "https://finance.yahoo.com/"

from typing import Optional, List, Dict, Tuple, Callable

def get_historical_data_direct(symbol: str, start_date: datetime.datetime, end_date: datetime.datetime, interval: str = "1d") -> pd.Series:
    """
    Fetches historical adjusted closing prices for a given symbol from Yahoo Finance Chart API.
    
    Args:
        symbol (str): Ticker symbol of the asset (e.g., 'MSCI')
        start_date (datetime): Start date for historical data
        end_date (datetime): End date for historical data
        interval (str): Data interval, default is '1d' (daily)
    
    Returns:
        pd.Series: Time series of adjusted closing prices, indexed by date
    """
    start_timestamp = int(start_date.timestamp())
    end_timestamp = int(end_date.timestamp())

    params = {
        'period1': str(start_timestamp),
        'period2': str(end_timestamp),
        'interval': interval,
        'events': 'history', 
        'includeAdjustedClose': 'true',
        'crumb': ''
    }

    url = f"{API_BASE}/v8/finance/chart/{symbol}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json'
    }

    try:
        # Sending HTTP request to Yahoo Finance API
        response = requests.get(url, params=params, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data or 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
            return pd.Series(dtype=float)

        chart_result = data['chart']['result'][0]
        timestamp = chart_result.get('timestamp', [])
        indicators = chart_result.get('indicators', {})
        adjclose = indicators.get('adjclose', [{}])[0]
        quotes = indicators.get('quote', [{}])[0]

        prices = adjclose.get('adjclose') if adjclose and adjclose.get('adjclose') else quotes.get('close')

        if not timestamp or not prices:
            return pd.Series(dtype=float)

        # Creating pandas Series with dates as index
        dates = [datetime.datetime.fromtimestamp(ts) for ts in timestamp]
        series = pd.Series(prices, index=dates)
        return series.rename(symbol).dropna()

    except requests.exceptions.Timeout:
        print(f"Timeout fetching data for {symbol}.")
        return pd.Series(dtype=float)
    except requests.exceptions.ConnectionError:
        print(f"Connection error fetching data for {symbol}.")
        return pd.Series(dtype=float)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        return pd.Series(dtype=float)
    except json.JSONDecodeError:
        print(f"Error decoding JSON for {symbol}: {response.text[:200]}...")
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"An unexpected error occurred for {symbol}: {e}")
        return pd.Series(dtype=float)

def get_historical_data(portfolio_config: List[Dict], period: str = "10y") -> Tuple[pd.Series, pd.DataFrame]:
    """
    Loads historical data for multiple assets and calculates annual mean returns and covariance matrix.
    
    Args:
        portfolio_config (List[Dict]): List of dictionaries containing ticker symbols and weights
        period (str): Historical data period (e.g., '10y' for 10 years, 'max' for maximum available)
    
    Returns:
        Tuple[pd.Series, pd.DataFrame]: Annual mean returns (Series) and covariance matrix (DataFrame)
    """
    # Extracting ticker symbols from portfolio configuration
    tickers = [item['ticker'] for item in portfolio_config]
    print(f"Lade historische Daten für {tickers} für den Zeitraum '{period}'...")

    # Setting date range for data retrieval
    end_date = datetime.datetime.now()
    if period.endswith('y'):
        years_ago = int(period[:-1])
        start_date = end_date - datetime.timedelta(days=years_ago * 365)
    elif period == 'max':
        start_date = datetime.datetime(1900, 1, 1)
    else:
        raise ValueError(f"Unsupported period format: {period}. Use 'Ny' (e.g., '10y') or 'max'.")

    # Fetching data for each ticker
    all_tickers_data = {}
    for ticker in tickers:
        series = get_historical_data_direct(ticker, start_date, end_date)
        if not series.empty:
            all_tickers_data[ticker] = series
        else:
            print(f"Warnung: Keine Daten für {ticker} erhalten. Dieser Ticker wird ignoriert.")

    if not all_tickers_data:
        raise ValueError("Keine Daten für irgendeinen Ticker von Yahoo Finance erhalten. Bitte Ticker prüfen oder den Zeitraum anpassen.")

    # Creating DataFrame from collected data
    data = pd.DataFrame(all_tickers_data)
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    # Handling missing data
    data.ffill(inplace=True)
    data.bfill(inplace=True)
    data.dropna(axis=1, inplace=True) 

    if data.empty:
        raise ValueError("Nach Bereinigung sind keine vollständigen Daten mehr vorhanden. Bitte Ticker prüfen.")

    # Calculating daily returns and annual metrics
    daily_returns = data.pct_change().dropna()

    if daily_returns.empty:
        raise ValueError("Nicht genügend Datenpunkte nach Berechnung der täglichen Renditen. Bitte Ticker prüfen oder Zeitraum erweitern.")

    mean_annual_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov()

    print("Daten erfolgreich geladen und verarbeitet.")
    return mean_annual_returns, cov_matrix

def _run_single_simulation(params_tuple):
    """
    Runs a single Monte Carlo simulation for the portfolio.
    
    Args:
        params_tuple: Tuple containing simulation parameters:
            - initial_capital (float): Initial investment amount
            - monthly_contribution (float): Monthly savings amount
            - increase_rate (float): Annual increase rate for contributions
            - years (int): Investment horizon in years
            - weights_array (np.ndarray): Portfolio weights
            - mu_daily (np.ndarray): Daily mean returns
            - L_matrix (np.ndarray): Cholesky decomposition of covariance matrix
            - rebalancing (str): Rebalancing strategy
            - dist (str): Distribution type ('normal' or 't')
            - t_df (int): Degrees of freedom for t-distribution
    
    Returns:
        Tuple[float, float, float, float]: Final portfolio value, IRR, max drawdown percentage, and max recovery time
    """
    (initial_capital, monthly_contribution, increase_rate, years,
     weights_array, mu_daily, L_matrix, rebalancing, dist, t_df) = params_tuple

    n_assets = len(weights_array)
    days_per_year = 252
    total_days = years * days_per_year
    deposit_interval = days_per_year // 12

    # Initializing asset values
    asset_values = np.zeros((total_days, n_assets))
    asset_values[0] = initial_capital * weights_array
    
    # Tracking cash flows for IRR calculation
    cashflows = [(-initial_capital, 0)]
    monthly_contribution_local = monthly_contribution

    portfolio_history = np.zeros(total_days)
    portfolio_history[0] = initial_capital

    # Simulating daily returns
    for day in range(1, total_days):
        # Generating random returns based on distribution type
        if dist == "t":
            uncorrelated_randoms = np.random.standard_t(t_df, size=n_assets)
        elif dist == "normal":
            uncorrelated_randoms = np.random.standard_normal(size=n_assets)
        else:
            raise ValueError("Verteilungsmodell unbekannt. Erlaubt: 'normal', 't'")

        # Applying correlation via Cholesky decomposition
        correlated_randoms = L_matrix @ uncorrelated_randoms
        daily_asset_returns = mu_daily + correlated_randoms
        asset_values[day] = asset_values[day - 1] * (1 + daily_asset_returns)

        # Handling annual contribution increases
        if day > 0 and day % days_per_year == 0:
            monthly_contribution_local *= (1 + increase_rate)
        
        # Adding monthly contributions
        if day > 0 and day % deposit_interval == 0:
            time_in_years_for_irr = day / days_per_year
            cashflows.append((-monthly_contribution_local, time_in_years_for_irr))
            asset_values[day] += monthly_contribution_local * weights_array
        
        total_value_at_day = np.sum(asset_values[day])
        portfolio_history[day] = total_value_at_day

        # Applying annual rebalancing if specified
        if rebalancing == "annual" and day > 0 and day % days_per_year == 0:
            asset_values[day] = total_value_at_day * weights_array

    final_value = np.sum(asset_values[-1])
    cashflows.append((final_value, years))
    irr = compute_irr(cashflows)

    # Calculating maximum drawdown and recovery time
    peak_value = portfolio_history[0]
    max_drawdown_percent = 0.0
    max_recovery_time_in_years = 0.0

    current_peak = portfolio_history[0]
    current_peak_day_index = 0

    for i in range(1, total_days):
        if portfolio_history[i] > current_peak:
            current_peak = portfolio_history[i]
            current_peak_day_index = i
        else:
            drawdown = (current_peak - portfolio_history[i]) / current_peak
            if drawdown > max_drawdown_percent:
                max_drawdown_percent = drawdown
                
                # Calculating recovery time
                recovery_start_day = current_peak_day_index
                recovery_end_day = -1
                for j in range(i, total_days):
                    if portfolio_history[j] >= current_peak:
                        recovery_end_day = j
                        break
                
                if recovery_end_day != -1:
                    recovery_duration_days = recovery_end_day - recovery_start_day
                    recovery_duration_years = recovery_duration_days / days_per_year
                    if recovery_duration_years > max_recovery_time_in_years:
                        max_recovery_time_in_years = recovery_duration_years

    return final_value, irr, max_drawdown_percent, max_recovery_time_in_years

def simulate_portfolio(
    initial_capital: float,
    monthly_contribution: float,
    increase_rate: float,
    years: int,
    portfolio_config: List[Dict],
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    rebalancing: str = "none",
    n_simulations: int = 25000,
    seed: Optional[int] = None,
    dist: str = "t",
    t_df: int = 5,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    simulation_pool_ref: List = []
):
    """
    Runs multiple Monte Carlo simulations for the portfolio using multiprocessing.
    
    Args:
        initial_capital (float): Initial investment amount
        monthly_contribution (float): Monthly savings amount
        increase_rate (float): Annual increase rate for contributions
        years (int): Investment horizon in years
        portfolio_config (List[Dict]): Portfolio configuration with tickers and weights
        mean_returns (pd.Series): Annual mean returns for assets
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns
        rebalancing (str): Rebalancing strategy ('none' or 'annual')
        n_simulations (int): Number of simulations to run
        seed (Optional[int]): Random seed for reproducibility
        dist (str): Distribution type ('normal' or 't')
        t_df (int): Degrees of freedom for t-distribution
        progress_callback (Optional[Callable]): Callback for progress updates
        cancel_event (Optional[threading.Event]): Event to cancel simulation
        simulation_pool_ref (List): Reference to store multiprocessing pool
    
    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]: 
            Final values, total invested amount, IRRs, max drawdowns, and recovery times
    """
    if seed is not None:
        np.random.seed(seed)

    # Preparing simulation parameters
    weights_dict = {item['ticker']: item['weight'] for item in portfolio_config}
    weights_array = np.array([weights_dict[ticker] for ticker in mean_returns.index])
    
    mu_daily = mean_returns.values / 252
    L_matrix = np.linalg.cholesky(cov_matrix.values)

    simulation_params = [(
        initial_capital, monthly_contribution, increase_rate, years,
        weights_array, mu_daily, L_matrix, rebalancing, dist, t_df
    ) for _ in range(n_simulations)]

    final_values = []
    irr_values = []
    max_drawdown_percents = []
    recovery_times_in_years = []
    
    # Setting up multiprocessing
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    pool = None
    try:
        pool = multiprocessing.Pool(processes=num_processes)
        if simulation_pool_ref is not None:
            simulation_pool_ref.append(pool)
            
        chunksize = max(1, n_simulations // (num_processes * 10))
        results_iterator = pool.imap_unordered(_run_single_simulation, simulation_params, chunksize=chunksize)
        
        # Processing simulation results
        completed_simulations = 0
        for final_value, irr, max_drawdown, recovery_time in results_iterator:
            if cancel_event and cancel_event.is_set():
                pool.terminate()
                break

            final_values.append(final_value)
            irr_values.append(irr)
            max_drawdown_percents.append(max_drawdown)
            recovery_times_in_years.append(recovery_time)
            
            completed_simulations += 1
            if progress_callback:
                progress_callback(completed_simulations, n_simulations)
    except Exception as e:
        if cancel_event and cancel_event.is_set():
            print("Simulation wurde durch Benutzeraktion abgebrochen.")
        else:
            raise e
    finally:
        if pool is not None:
            pool.close()
            pool.join()
            if simulation_pool_ref:
                simulation_pool_ref.clear()

    # Calculating total invested amount
    invested = initial_capital
    current_rate = monthly_contribution
    for _ in range(years):
        invested += current_rate * 12
        current_rate *= (1 + increase_rate)

    return np.array(final_values), invested, np.array(irr_values), np.array(max_drawdown_percents), np.array(recovery_times_in_years)

def compute_irr(cashflows):
    """
    Computes the Internal Rate of Return (IRR) for a series of cash flows.
    
    Args:
        cashflows: List of tuples (amount, time) representing cash flows
    
    Returns:
        float: IRR value, or NaN if calculation fails
    """
    def npv(rate):
        return sum(cf / (1 + rate) ** t for cf, t in cashflows)
    try:
        return brentq(npv, -0.9999, 5.0)
    except (ValueError, RuntimeError):
        return np.nan

def search_yahoo_finance_tickers(query: str) -> List[Dict]:
    """
    Searches for ticker symbols on Yahoo Finance based on a query string.
    
    Args:
        query (str): Search term (e.g., name, ISIN, or symbol)
    
    Returns:
        List[Dict]: List of dictionaries containing symbol, longname, and exchange
    """
    search_url = f"{API_BASE}/v1/finance/search"
    params = { 'q': query, 'lang': 'en-US', 'quotesCount': 6, 'newsCount': 0, 'enableFuzzyQuery': 'false' }
    headers = { 'User-Agent': 'Mozilla/5.0...', 'Accept': 'application/json' }
    try:
        response = requests.get(search_url, params=params, headers=headers, verify=False, timeout=5)
        response.raise_for_status()
        data = response.json()
        results = []
        if data and 'quotes' in data:
            for quote in data['quotes']:
                if quote.get('quoteType') in ['EQUITY', 'ETF', 'FUND'] and quote.get('longname'):
                    results.append({'symbol': quote['symbol'], 'longname': quote['longname'], 'exchange': quote.get('exchange', '')})
        return results
    except requests.exceptions.RequestException as e:
        print(f"Error searching tickers: {e}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding search JSON: {response.text[:200]}...")
        return []

class PortfolioSimulatorGUI:
    """
    GUI class for the Monte Carlo Portfolio Simulator using tkinter.
    Provides interface for inputting parameters, searching tickers, and displaying results.
    """
    def __init__(self, master):
        """Initializes the GUI with input fields, portfolio configuration, and result display."""
        self.master = master
        master.title("Monte Carlo Portfolio Simulator")
        master.geometry("800x850")

        # Setting up GUI styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#f0f0f0')
        self.style.configure('TLabel', background='#f0f0f0', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'), padding=5)
        self.style.configure('TEntry', font=('Arial', 10))
        self.style.configure('TCombobox', font=('Arial', 10))
        self.style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        self.style.configure('Treeview', font=('Arial', 10))
        self.style.configure('Cancel.TButton', foreground='white', background='#e74c3c')

        self.cancel_event = threading.Event()
        self.simulation_pool_ref = []
        self.create_widgets()
        self.portfolio_entries = []
        self.add_portfolio_row()

    def create_widgets(self):
        """Creates and arranges all GUI widgets including input fields and buttons."""
        # Input parameters frame
        input_frame = ttk.LabelFrame(self.master, text="Simulation Parameters", padding=(20, 10))
        input_frame.pack(padx=10, pady=10, fill="x", expand=False)
        ttk.Label(input_frame, text="Startkapital (EUR):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.initial_entry = ttk.Entry(input_frame)
        self.initial_entry.insert(0, "10000")
        self.initial_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(input_frame, text="Monatliche Sparrate (EUR):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.monthly_entry = ttk.Entry(input_frame)
        self.monthly_entry.insert(0, "500")
        self.monthly_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(input_frame, text="Jährliche Sparraten-Steigerung (%):").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        self.increase_entry = ttk.Entry(input_frame)
        self.increase_entry.insert(0, "2")
        self.increase_entry.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(input_frame, text="Anlagedauer (Jahre):").grid(row=3, column=0, padx=5, pady=2, sticky="w")
        self.years_entry = ttk.Entry(input_frame)
        self.years_entry.insert(0, "15")
        self.years_entry.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(input_frame, text="Rebalancing-Strategie:").grid(row=4, column=0, padx=5, pady=2, sticky="w")
        self.rebalancing_var = tk.StringVar(value="annual")
        self.rebalancing_option = ttk.Combobox(input_frame, textvariable=self.rebalancing_var, values=["none", "annual"], state="readonly")
        self.rebalancing_option.grid(row=4, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(input_frame, text="Verteilung:").grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.dist_var = tk.StringVar(value="t")
        self.dist_option = ttk.Combobox(input_frame, textvariable=self.dist_var, values=["normal", "t"], state="readonly")
        self.dist_option.grid(row=5, column=1, padx=5, pady=2, sticky="ew")
        self.dist_option.bind("<<ComboboxSelected>>", self.toggle_df_entry)
        ttk.Label(input_frame, text="Freiheitsgrade (t-Verteilung):").grid(row=6, column=0, padx=5, pady=2, sticky="w")
        self.df_entry = ttk.Entry(input_frame)
        self.df_entry.insert(0, "5")
        self.df_entry.grid(row=6, column=1, padx=5, pady=2, sticky="ew")
        self.toggle_df_entry()
        ttk.Label(input_frame, text="Zufalls-Seed (optional):").grid(row=7, column=0, padx=5, pady=2, sticky="w")
        self.seed_entry = ttk.Entry(input_frame)
        self.seed_entry.insert(0, "42")
        self.seed_entry.grid(row=7, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(input_frame, text="Historischer Zeitraum (z.B. 10y, max):").grid(row=8, column=0, padx=5, pady=2, sticky="w")
        self.period_entry = ttk.Entry(input_frame)
        self.period_entry.insert(0, "max")
        self.period_entry.grid(row=8, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(input_frame, text="Anzahl Simulationen:").grid(row=9, column=0, padx=5, pady=2, sticky="w")
        self.n_simulations_entry = ttk.Entry(input_frame)
        self.n_simulations_entry.insert(0, "5000")
        self.n_simulations_entry.grid(row=9, column=1, padx=5, pady=2, sticky="ew")
        input_frame.columnconfigure(1, weight=1)

        # Ticker search frame
        search_frame = ttk.LabelFrame(self.master, text="Ticker Suche (Yahoo Finance)", padding=(20, 10))
        search_frame.pack(padx=10, pady=10, fill="x", expand=False)
        ttk.Label(search_frame, text="Suchbegriff (Name/ISIN/Symbol):").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.search_query_entry = ttk.Entry(search_frame)
        self.search_query_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.search_button = ttk.Button(search_frame, text="Suchen", command=self.perform_ticker_search)
        self.search_button.grid(row=0, column=2, padx=5, pady=2)
        self.search_results_tree = ttk.Treeview(search_frame, columns=("Symbol", "Name", "Börse"), show="headings", height=5)
        self.search_results_tree.heading("Symbol", text="Symbol")
        self.search_results_tree.heading("Name", text="Name")
        self.search_results_tree.heading("Börse", text="Börse")
        self.search_results_tree.column("Symbol", width=100, anchor="w")
        self.search_results_tree.column("Name", width=250, anchor="w")
        self.search_results_tree.column("Börse", width=80, anchor="w")
        self.search_results_tree.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="ew")
        self.search_results_tree.bind("<Double-1>", self.on_search_result_double_click)
        search_frame.columnconfigure(1, weight=1)

        # Portfolio configuration frame
        portfolio_frame = ttk.LabelFrame(self.master, text="Portfolio Konfiguration", padding=(20, 10))
        portfolio_frame.pack(padx=10, pady=10, fill="both", expand=True)
        self.portfolio_canvas = tk.Canvas(portfolio_frame, borderwidth=0, background="#f0f0f0")
        self.portfolio_scrollbar = ttk.Scrollbar(portfolio_frame, orient="vertical", command=self.portfolio_canvas.yview)
        self.portfolio_scrollable_frame = ttk.Frame(self.portfolio_canvas, padding=(0,0))
        self.portfolio_scrollable_frame.bind("<Configure>", lambda e: self.portfolio_canvas.configure(scrollregion=self.portfolio_canvas.bbox("all")))
        self.portfolio_canvas.create_window((0, 0), window=self.portfolio_scrollable_frame, anchor="nw")
        self.portfolio_canvas.configure(yscrollcommand=self.portfolio_scrollbar.set)
        self.portfolio_canvas.pack(side="left", fill="both", expand=True)
        self.portfolio_scrollbar.pack(side="right", fill="y")
        ttk.Label(self.portfolio_scrollable_frame, text="Ticker Symbol", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=2)
        ttk.Label(self.portfolio_scrollable_frame, text="Gewichtung (%)", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=2)
        button_frame = ttk.Frame(portfolio_frame)
        button_frame.pack(pady=5)
        ttk.Button(button_frame, text="Ticker hinzufügen", command=self.add_portfolio_row).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Ticker entfernen", command=self.remove_portfolio_row).pack(side="left", padx=5)

        # Action buttons frame
        action_button_frame = ttk.Frame(self.master)
        action_button_frame.pack(pady=10)
        self.cancel_button = ttk.Button(action_button_frame, text="Abbrechen", command=self.cancel_simulation, state="disabled", style='Cancel.TButton')
        self.cancel_button.pack(side="left", padx=10)
        self.calculate_button = ttk.Button(action_button_frame, text="Simulation Starten", command=self.start_simulation_thread)
        self.calculate_button.pack(side="left", padx=10)
        self.status_label = ttk.Label(self.master, text="Bereit", font=('Arial', 10, 'italic'), foreground='blue')
        self.status_label.pack(pady=5)

        # Yahoo Finance link
        yahoo_link_frame = ttk.Frame(self.master)
        yahoo_link_frame.pack(pady=5)
        ttk.Label(yahoo_link_frame, text="Weitere Ticker finden Sie auf: ").pack(side="left")
        self.yahoo_link = ttk.Label(yahoo_link_frame, text="Yahoo Finance", foreground="blue", cursor="hand2")
        self.yahoo_link.pack(side="left")
        self.yahoo_link.bind("<Button-1>", lambda e: webbrowser.open_new(YAHOO_FINANCE_URL))

    def toggle_df_entry(self, event=None):
        """Enables/disables degrees of freedom entry based on distribution selection."""
        if self.dist_var.get() == "t":
            self.df_entry.config(state="normal")
        else:
            self.df_entry.config(state="disabled")

    def add_portfolio_row(self, ticker="", weight=""):
        """Adds a new row for entering ticker symbol and weight."""
        row_num = len(self.portfolio_entries) + 1
        ticker_entry = ttk.Entry(self.portfolio_scrollable_frame)
        ticker_entry.insert(0, ticker)
        ticker_entry.grid(row=row_num, column=0, padx=5, pady=2, sticky="ew")
        weight_entry = ttk.Entry(self.portfolio_scrollable_frame)
        if isinstance(weight, (float, int)):
            weight_entry.insert(0, str(weight * 100))
        else:
            weight_entry.insert(0, weight)
        weight_entry.grid(row=row_num, column=1, padx=5, pady=2, sticky="ew")
        self.portfolio_entries.append((ticker_entry, weight_entry))
        self.portfolio_scrollable_frame.columnconfigure(0, weight=1)
        self.portfolio_scrollable_frame.columnconfigure(1, weight=1)

    def remove_portfolio_row(self):
        """Removes the last portfolio row, ensuring at least one remains."""
        if len(self.portfolio_entries) > 1:
            ticker_entry, weight_entry = self.portfolio_entries.pop()
            ticker_entry.destroy()
            weight_entry.destroy()
        else:
            messagebox.showwarning("Warnung", "Mindestens ein Ticker muss vorhanden sein.")

    def get_portfolio_config_from_ui(self) -> List[Dict]:
        """Retrieves and validates portfolio configuration from GUI inputs."""
        portfolio_config = []
        for ticker_entry, weight_entry in self.portfolio_entries:
            ticker = ticker_entry.get().strip()
            weight_str = weight_entry.get().strip()
            if not ticker or not weight_str:
                continue
            try:
                weight_percent = float(weight_str)
                if not (0 <= weight_percent <= 100):
                    raise ValueError("Gewichtung muss zwischen 0 und 100 liegen.")
                weight_decimal = weight_percent / 100.0
                portfolio_config.append({'ticker': ticker, 'weight': weight_decimal})
            except ValueError as e:
                raise ValueError(f"Ungültige Gewichtung für Ticker '{ticker}': {e}")
        if not portfolio_config:
            raise ValueError("Bitte fügen Sie mindestens einen Ticker und eine Gewichtung hinzu.")
        total_weight = sum(item['weight'] for item in portfolio_config)
        if not np.isclose(total_weight, 1.0):
            response = messagebox.askyesno(
                "Gewichtungs-Warnung",
                f"Die Summe der Gewichtungen ({total_weight*100:.2f}%) ist nicht 100%. Möchten Sie die Gewichte automatisch normalisieren?",
                icon='warning'
            )
            if response:
                for item in portfolio_config:
                    item['weight'] /= total_weight
            else:
                raise ValueError("Die Summe der Gewichtungen muss 100% ergeben oder normalisiert werden.")
        return portfolio_config

    def _update_status_label(self, text: str, color: str = 'blue'):
        """Updates the status label with the given text and color."""
        self.master.after(0, self.status_label.config, {"text": text, "foreground": color})

    def perform_ticker_search(self):
        """Initiates a ticker search based on user input."""
        search_query = self.search_query_entry.get().strip()
        if not search_query:
            messagebox.showwarning("Suche", "Bitte geben Sie einen Suchbegriff ein.")
            return
        self._update_status_label(f"Suche nach '{search_query}'...", "orange")
        for item in self.search_results_tree.get_children():
            self.search_results_tree.delete(item)
        search_thread = threading.Thread(target=self._run_ticker_search_in_thread, args=(search_query,))
        search_thread.daemon = True
        search_thread.start()

    def _run_ticker_search_in_thread(self, query: str):
        """Runs ticker search in a separate thread to avoid freezing GUI."""
        try:
            results = search_yahoo_finance_tickers(query)
            self.master.after(0, self._display_search_results, results)
        except Exception as e:
            self.master.after(0, messagebox.showerror, "Suchfehler", f"Ein Fehler bei der Ticker-Suche ist aufgetreten: {e}")
            self._update_status_label("Suche fehlgeschlagen", "red")
        finally:
            self.master.after(0, self._update_status_label, "Bereit", "blue")

    def _display_search_results(self, results: List[Dict]):
        """Displays ticker search results in the Treeview widget."""
        for item in self.search_results_tree.get_children():
            self.search_results_tree.delete(item)
        if not results:
            self.search_results_tree.insert("", "end", values=("", "Keine Ergebnisse gefunden", ""))
            return
        for result in results:
            self.search_results_tree.insert("", "end", values=(result['symbol'], result['longname'], result['exchange']))

    def on_search_result_double_click(self, event):
        """Handles double-click on search results to add ticker to portfolio."""
        selected_item = self.search_results_tree.selection()
        if selected_item:
            item_values = self.search_results_tree.item(selected_item, 'values')
            ticker_symbol = item_values[0]
            self.add_portfolio_row(ticker=ticker_symbol, weight="10")
            self._update_status_label(f"'{ticker_symbol}' zum Portfolio hinzugefügt.", "green")

    def cancel_simulation(self):
        """Cancels an ongoing simulation and cleans up resources."""
        self._update_status_label("Simulation wird abgebrochen...", "red")
        self.cancel_button.config(state="disabled")
        self.cancel_event.set()

        if self.simulation_pool_ref:
            pool = self.simulation_pool_ref[0]
            try:
                pool.terminate()
                pool.close()
                pool.join()
                print("Multiprocessing Pool wurde beendet und geschlossen.")
            except Exception as e:
                print(f"Fehler beim Beenden des Pools: {e}")
            finally:
                self.simulation_pool_ref.clear()
        self._update_status_label("Simulation abgebrochen. Bereit für neuen Start.", "blue")
        self.calculate_button.config(state="normal")

    def start_simulation_thread(self):
        """Starts the simulation in a separate thread to keep GUI responsive."""
        self.calculate_button.config(state="disabled")
        self.cancel_button.config(state="normal")
        self._update_status_label("Simulation wird gestartet...", 'blue')
        
        self.cancel_event.clear()
        
        simulation_thread = threading.Thread(target=self._run_simulation_in_thread)
        simulation_thread.daemon = True
        simulation_thread.start()

    def _run_simulation_in_thread(self):
        """
        Runs the portfolio simulation in a separate thread, handling all steps from data retrieval to result display.
        """
        is_cancelled = False
        try:
            # Step 1: Retrieving input parameters from GUI
            initial_capital = float(self.initial_entry.get())
            monthly_contribution = float(self.monthly_entry.get())
            increase_rate = float(self.increase_entry.get()) / 100.0
            years = int(self.years_entry.get())
            rebalancing = self.rebalancing_var.get()
            dist = self.dist_var.get()
            t_df = int(self.df_entry.get()) if dist == "t" else None
            seed = int(self.seed_entry.get()) if self.seed_entry.get() else None
            period = self.period_entry.get()
            n_simulations = int(self.n_simulations_entry.get())

            if self.cancel_event.is_set():
                self._update_status_label("Simulation abgebrochen, bevor sie gestartet wurde.", "red")
                return

            # Step 2: Getting portfolio configuration
            portfolio_config = self.get_portfolio_config_from_ui()

            self._update_status_label("Lade historische Daten...", "orange")
            if self.cancel_event.is_set():
                self._update_status_label("Simulation abgebrochen während Datenladens.", "red")
                return

            # Step 3: Fetching historical data
            mean_returns_series, cov_matrix_df = get_historical_data(portfolio_config, period=period)

            if self.cancel_event.is_set():
                self._update_status_label("Simulation abgebrochen nach Datenladung.", "red")
                return

            self._update_status_label("Portfolio wurde definiert und Daten vorbereitet...", "orange")

            # Step 4: Aligning portfolio weights with available data
            successful_tickers = mean_returns_series.index.tolist()
            filtered_portfolio_config = []
            for item in portfolio_config:
                if item['ticker'] in successful_tickers:
                    filtered_portfolio_config.append(item)
            if not filtered_portfolio_config:
                raise ValueError("Keine gültigen Ticker mit Daten für die Simulation übrig.")
            total_filtered_weight = sum(item['weight'] for item in filtered_portfolio_config)
            if not np.isclose(total_filtered_weight, 1.0):
                for item in filtered_portfolio_config:
                    item['weight'] /= total_filtered_weight

            ordered_mean_returns = mean_returns_series.loc[[item['ticker'] for item in filtered_portfolio_config]]
            ordered_cov_matrix = cov_matrix_df.loc[[item['ticker'] for item in filtered_portfolio_config], [item['ticker'] for item in filtered_portfolio_config]]

            def simulation_progress_callback(current: int, total: int):
                if not self.cancel_event.is_set():
                    self._update_status_label(f"Führe {total:,} Simulationen durch: {current:,}/{total:,}", "orange")

            self._update_status_label(f"Führe {n_simulations:,} Simulationen durch: 0/{n_simulations:,}", "orange")

            if self.cancel_event.is_set():
                self._update_status_label("Simulation abgebrochen vor Simulationsstart.", "red")
                return

            # Step 5: Running Monte Carlo simulation
            results, invested, irr_values, max_drawdown_percents, recovery_times_in_years = simulate_portfolio(
                initial_capital=initial_capital,
                monthly_contribution=monthly_contribution,
                increase_rate=increase_rate,
                years=years,
                portfolio_config=filtered_portfolio_config,
                mean_returns=ordered_mean_returns,
                cov_matrix=ordered_cov_matrix,
                rebalancing=rebalancing,
                dist=dist,
                t_df=t_df,
                seed=seed,
                n_simulations=n_simulations,
                progress_callback=simulation_progress_callback,
                cancel_event=self.cancel_event,
                simulation_pool_ref=self.simulation_pool_ref
            )

            # Step 6: Checking for cancellation
            if self.cancel_event.is_set():
                is_cancelled = True
                self._update_status_label("Simulation abgebrochen.", "red")
                return

            self._update_status_label("Zeige Ergebnisfenster an...", "green")
            self.master.after(0, self.show_results_window, results, invested, irr_values, max_drawdown_percents, recovery_times_in_years, years, filtered_portfolio_config)

        except ValueError as e:
            if not self.cancel_event.is_set():
                self.master.after(0, messagebox.showerror, "Eingabefehler", str(e))
                self._update_status_label("Fehler: " + str(e), "red")
        except Exception as e:
            if not self.cancel_event.is_set():
                self.master.after(0, messagebox.showerror, "Simulationsfehler", f"Ein unerwarteter Fehler ist aufgetreten: {e}")
                self._update_status_label("Fehler aufgetreten", "red")
                import traceback
                traceback.print_exc()
        finally:
            self.master.after(0, self.calculate_button.config, {"state": "normal"})
            self.master.after(0, self.cancel_button.config, {"state": "disabled"})
            if not is_cancelled:
                self._update_status_label("Bereit", "blue")
            self.simulation_pool_ref.clear()

    def show_results_window(self, results: np.ndarray, invested: float, irr_values: np.ndarray, max_drawdown_percents: np.ndarray, recovery_times_in_years: np.ndarray, years: int, final_portfolio_config: List[Dict]):
        """Displays simulation results in a new window with portfolio details and histograms."""
        results_window = tk.Toplevel(self.master)
        results_window.title("Simulationsergebnisse")
        results_window.geometry("1000x800")
        results_window.resizable(True, True)

        # Displaying final portfolio configuration
        portfolio_display_frame = ttk.LabelFrame(results_window, text="Verwendete Portfolio Konfiguration", padding=(10, 5))
        portfolio_display_frame.pack(padx=10, pady=5, fill="x", expand=False)
        portfolio_text = ""
        for item in final_portfolio_config:
            portfolio_text += f"{item['ticker']}: {item['weight']*100:.2f}%\n"
        portfolio_label = ttk.Label(portfolio_display_frame, text=portfolio_text, justify=tk.LEFT, font=('Arial', 9))
        portfolio_label.pack(padx=5, pady=2, anchor="w")

        # Displaying summary statistics
        summary_frame = ttk.LabelFrame(results_window, text="Zusammenfassung der Endvermögen", padding=(10, 5))
        summary_frame.pack(padx=10, pady=5, fill="x", expand=False)
        percentiles = [10, 25, 50, 75, 90]
        valid_irr_values = irr_values[~np.isnan(irr_values)]
        valid_max_drawdown_percents = max_drawdown_percents[~np.isnan(max_drawdown_percents)]
        valid_recovery_times_in_years = recovery_times_in_years[~np.isnan(recovery_times_in_years)]
        percentile_values = np.percentile(results, percentiles)
        irr_percentile_values = np.nan * np.ones_like(percentiles, dtype=float)
        if valid_irr_values.size > 0:
            irr_percentile_values = np.percentile(valid_irr_values, percentiles)
        max_drawdown_percentile_values = np.nan * np.ones_like(percentiles, dtype=float)
        if valid_max_drawdown_percents.size > 0:
            max_drawdown_percentile_values = np.percentile(valid_max_drawdown_percents, percentiles)
        recovery_time_percentile_values = np.nan * np.ones_like(percentiles, dtype=float)
        if valid_recovery_times_in_years.size > 0:
            recovery_time_percentile_values = np.percentile(valid_recovery_times_in_years, percentiles)
        summary_data = []
        for i, p in enumerate(percentiles):
            total = percentile_values[i]
            irr = irr_percentile_values[i]
            cagr = ((total / invested) ** (1 / years)) - 1 if invested > 0 and total > 0 else 0
            total_return = (total / invested) - 1 if invested > 0 else 0
            max_dd = max_drawdown_percentile_values[i]
            rec_time = recovery_time_percentile_values[i]
            summary_data.append([
                f"{p}%", f"{total:,.2f} EUR", f"{cagr * 100:.2f}%",
                f"{total_return * 100:.2f}%", f"{irr * 100:.2f}%",
                f"{max_dd * 100:.2f}%", f"{rec_time:.2f} Jahre" if not np.isinf(rec_time) else "N/A"
            ])
        tree_columns = ("Perzentil", "Endvermögen (EUR)", "CAGR (%)", "Total Return (%)", "IRR (%)", "Max Drawdown (%)", "Max Erholungszeit (Jahre)")
        tree = ttk.Treeview(summary_frame, columns=tree_columns, show="headings")
        for col in tree_columns:
            tree.heading(col, text=col)
            if "Perzentil" in col: tree.column(col, width=80, anchor="center")
            elif "Endvermögen" in col: tree.column(col, width=120, anchor="e")
            elif "CAGR" in col or "IRR" in col or "Total Return" in col or "Max Drawdown" in col: tree.column(col, width=100, anchor="e")
            elif "Erholungszeit" in col: tree.column(col, width=120, anchor="e")
        for row in summary_data:
            tree.insert("", "end", values=row)
        tree.pack(fill="both", expand=True)

        # Plotting histogram of final values
        plot_frame = ttk.LabelFrame(results_window, text="Verteilung des Endvermögens", padding=(10, 5))
        plot_frame.pack(padx=10, pady=10, fill="both", expand=True)
        fig = Figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.hist(results, bins=100, alpha=0.75, color='royalblue', label="Verteilung Endvermögen")
        ax.axvline(invested, color='red', linestyle='--', linewidth=2, label=f'Investiert ({invested:,.0f} €)')
        ax.axvline(percentile_values[2], color='gold', linestyle='-', linewidth=2, label=f'Median ({percentile_values[2]:,.0f} €)')
        p25, p75 = percentile_values[1], percentile_values[3]
        ax.axvspan(p25, p75, color='orange', alpha=0.2, label=f'25-75% Perzentil ({p25:,.0f} € - {p75:,.0f} €)')
        ax.set_title(f"Monte-Carlo-Simulation über {years} Jahre", fontsize=14)
        ax.set_xlabel("Endvermögen in EUR", fontsize=10)
        ax.set_ylabel("Häufigkeit", fontsize=10)
        formatter = mticker.ScalarFormatter(useOffset=False, useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 4))
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    root = tk.Tk()
    app = PortfolioSimulatorGUI(root)
    root.mainloop()