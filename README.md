# Monte Carlo Portfolio Simulator

---

[German Version](#german-version)
[English Version](#english-version)

<img width="1919" height="1028" alt="image" src="https://github.com/user-attachments/assets/9fd2143a-f4e3-4500-9e73-cd215e63539d" />
<img width="1919" height="1030" alt="image" src="https://github.com/user-attachments/assets/8c45e8ce-7a6c-47ce-a43d-04043346fad1" />


**_(you might need to put it on FULLSCREEN so you can press the button for starting the simulation)_**

# German Version:

## Überblick

Der Monte Carlo Portfolio Simulator ist eine Python-Anwendung mit einer grafischen Benutzeroberfläche (GUI), die es Benutzern ermöglicht, die zukünftige Performance eines benutzerdefinierten Anlageportfolios zu simulieren. Die Simulation basiert auf historischen Daten, die über die Yahoo Finance API abgerufen werden, und verwendet Monte-Carlo-Methoden, um zukünftige Renditen, Volatilität, maximalen Drawdown und Erholungszeiten zu prognostizieren. Die Anwendung unterstützt flexible Portfolio-Konfigurationen, Rebalancing-Optionen und verschiedene Verteilungsannahmen (Normal- oder t-Verteilung).

### Projekthintergrund

Der Discord-User @philipthecorgis hat sich ein CLI-Monte-Carlo Tool gebaut, ich (@chrissy8283 (discord) - @tomato6966 (github)) habe dieses tool dann mit UI und Performance upgrades und weitere analyse-werten ergänzt und hier auf github gepublished.

### Hauptfunktionen

- **Portfolio-Konfiguration**: Definieren Sie ein Portfolio durch Eingabe von Ticker-Symbolen und Gewichtungen (in Prozent).
- **Historische Daten**: Abruf von historischen Kursdaten über die Yahoo Finance Chart-API.
- **Monte-Carlo-Simulation**: Parallelisierte Simulationen mit konfigurierbarer Anzahl und Verteilungsmodellen (Normal- oder t-Verteilung).
- **Ergebnisdarstellung**: Anzeige von Ergebnissen in Tabellenform (Perzentile, CAGR, IRR, Max Drawdown, Erholungszeit) und als Histogramm.
- **Ticker-Suche**: Integrierte Suche nach Aktien, ETFs und Fonds über die Yahoo Finance API.
- **Abbruchfunktion**: Möglichkeit, laufende Simulationen mit einem "Abbrechen"-Button zu stoppen.
- **Multiprocessing**: Parallelisierung der Simulationen zur Verbesserung der Performance.
- **Benutzerfreundliche GUI**: Intuitive Benutzeroberfläche mit Statusmeldungen und interaktiven Elementen.
- **Erweiterte Metriken**: Berechnung von maximalem Drawdown, Erholungszeit und Gesamtrendite.

## Voraussetzungen

- **Python-Version**: Python 3.9 oder höher
- **Abhängigkeiten**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `requests`
  - `scipy`
  - `tkinter` (meist vorinstalliert mit Python)
- **Internetverbindung**: Erforderlich für den Abruf von Daten über die Yahoo Finance API.
- **Betriebssystem**: Windows, macOS oder Linux

## Installation

1. **Python installieren**:
   Stellen Sie sicher, dass Python 3.9 oder höher auf Ihrem System installiert ist. Sie können Python von [python.org](https://www.python.org/downloads/) herunterladen.

2. **Abhängigkeiten installieren**:
   Installieren Sie die erforderlichen Python-Pakete mit pip:
   ```bash
   pip install numpy pandas matplotlib requests scipy
   ```

3. **Projekt herunterladen**:
   Klonen Sie das Repository oder laden Sie die Quelldatei (`main_de.py` / `main_en.py`) herunter.

4. **Yahoo Finance API**:
   Die Anwendung verwendet inoffizielle Endpunkte der Yahoo Finance API (`query1.finance.yahoo.com`). Beachten Sie, dass diese Endpunkte ohne Vorankündigung geändert werden können.

## Nutzung

1. **Anwendung starten**:
   Führen Sie das Deutsche-Skript aus:
   ```bash
   python main_de.py
   ```
   Execute the English-Script:
   ```bash
   python main_en.py
   ```

2. **GUI-Bedienung**:
   - **Simulation Parameter**:
     - Geben Sie Startkapital, monatliche Sparrate, jährliche Steigerung der Sparrate, Anlagedauer, Rebalancing-Strategie, Verteilungsmodell, Freiheitsgrade (für t-Verteilung), Zufalls-Seed und historischen Zeitraum ein.
   - **Ticker-Suche**:
     - Verwenden Sie das Suchfeld, um nach Aktien, ETFs oder Fonds zu suchen. Doppelklicken Sie auf ein Suchergebnis, um es dem Portfolio hinzuzufügen.
   - **Portfolio-Konfiguration**:
     - Fügen Sie Ticker und Gewichtungen (in %) hinzu. Die Gewichtungen werden automatisch normalisiert, falls sie nicht 100% ergeben.
   - **Simulation starten**:
     - Klicken Sie auf "Simulation Starten", um die Monte-Carlo-Simulation zu initiieren.
     - Der Fortschritt wird in der Statusleiste angezeigt.
   - **Simulation abbrechen**:
     - Klicken Sie auf "Abbrechen", um eine laufende Simulation zu stoppen.
   - **Ergebnisse anzeigen**:
     - Nach Abschluss der Simulation öffnet sich ein Fenster mit einer Tabelle (Perzentile, Endvermögen, CAGR, IRR, Max Drawdown, Erholungszeit) und einem Histogramm der Endvermögen.

3. **Ergebnisse interpretieren**:
   - Die Tabelle zeigt statistische Kennzahlen für verschiedene Perzentile.
   - Das Histogramm visualisiert die Verteilung der simulierten Endvermögen mit Markierungen für das investierte Kapital, den Median und das 25-75%-Perzentil.

## Beispiel

1. Starten Sie die Anwendung.
2. Geben Sie die Parameter ein:
   - Startkapital: 20.000 EUR
   - Monatliche Sparrate: 500 EUR
   - Jährliche Steigerung: 2%
   - Anlagedauer: 15y (wie lange der anlagehorizont ist)
   - Rebalancing: Jährlich
   - Verteilung: t-Verteilung (Freiheitsgrade: 5)
   - Historischer Zeitraum: 10y (für die letzten 10 jahre, oder einfach 'max' um alles zu betrachten)
   - Anzahl Simulationen: 5000 (zw 5 und 10k hat man schon ok-gute ergebnisse aber 25k ist empfehlenswert. doch ab 10k muss man ggf. mehrere Minuten warten)
3. Fügen Sie ein Portfolio hinzu, z. B.:
   - Ticker: IVV, Gewichtung: 50% (S&P)
   - Ticker: ACWI, Gewichtung: 50% (ACWI)
4. Klicken Sie auf "Simulation Starten".
5. Überprüfen Sie die Ergebnisse im neuen Fenster.

## Wichtige Hinweise

- **Yahoo Finance API**: Die API-Endpunkte sind inoffiziell und können sich ändern. Bei Problemen mit der Datenabfrage überprüfen Sie die API-URLs (`API_BASE`) oder verwenden Sie alternative Datenquellen.
- **Abbruchfunktion**: Der "Abbrechen"-Button beendet laufende Simulationen, indem er den Multiprocessing-Pool terminiert. Nach dem Abbruch ist die Anwendung sofort wieder einsatzbereit.
- **Performance**: Die Anzahl der Simulationen und Prozesse (`max(1, cpu_count - 1)`) kann die Laufzeit beeinflussen. Reduzieren Sie die Anzahl der Simulationen für schnellere Ergebnisse.
- **Fehlerbehandlung**: Die Anwendung zeigt Fehlermeldungen in der GUI an, z. B. bei ungültigen Eingaben oder fehlenden Daten.

## Projektstruktur

- `main_de.py`: Hauptskript mit der gesamten Logik, einschließlich GUI, Datenabfrage, Simulation und Visualisierung.

## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe [LICENSE](LICENSE) für Details (falls eine Lizenzdatei hinzugefügt wird).


--

# ENGLISH VERSION:
# Monte Carlo Portfolio Simulator

## Overview

The Monte Carlo Portfolio Simulator is a Python application with a graphical user interface (GUI) that allows users to simulate the future performance of a custom investment portfolio. The simulation is based on historical data retrieved via the Yahoo Finance API and uses Monte Carlo methods to forecast future returns, volatility, maximum drawdown, and recovery times. The application supports flexible portfolio configurations, rebalancing options, and different distribution assumptions (normal or t-distribution).

### Project Background

The Discord user @philipthecorgis built a CLI-based Monte Carlo tool, which I (@chrissy8283 (Discord) - @tomato6966 (GitHub)) enhanced with a UI, performance improvements, and additional analysis metrics, and published here on GitHub.

### Key Features

- **Portfolio Configuration**: Define a portfolio by entering ticker symbols and weightings (in percent).
- **Historical Data**: Retrieve historical price data via the Yahoo Finance Chart API.
- **Monte Carlo Simulation**: Parallelized simulations with configurable counts and distribution models (normal or t-distribution).
- **Results Display**: Results presented in tabular form (percentiles, CAGR, IRR, max drawdown, recovery time) and as a histogram.
- **Ticker Search**: Integrated search for stocks, ETFs, and funds via the Yahoo Finance API.
- **Cancel Function**: Option to stop ongoing simulations with a "Cancel" button.
- **Multiprocessing**: Parallelization of simulations to improve performance.
- **User-Friendly GUI**: Intuitive interface with status messages and interactive elements.
- **Advanced Metrics**: Calculation of maximum drawdown, recovery time, and total return.

## Prerequisites

- **Python Version**: Python 3.9 or higher
- **Dependencies**:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `requests`
  - `scipy`
  - `tkinter` (usually pre-installed with Python)
- **Internet Connection**: Required to fetch data via the Yahoo Finance API.
- **Operating System**: Windows, macOS, or Linux

## Installation

1. **Install Python**:
   Ensure Python 3.9 or higher is installed on your system. Download Python from [python.org](https://www.python.org/downloads/).

2. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install numpy pandas matplotlib requests scipy
   ```

3. **Download the Project**:
   Clone the repository or download the source file (`main_de.py` / `main_en.py`).

4. **Yahoo Finance API**:
   The application uses unofficial Yahoo Finance API endpoints (`query1.finance.yahoo.com`). Note that these endpoints may change without notice.

## Usage

1. **Start the Application**:
   Run the English script:
   ```bash
   python main_en.py
   ```
   Or the German script:
   ```bash
   python main_de.py
   ```

2. **GUI Operation**:
   - **Simulation Parameters**:
     - Enter initial capital, monthly savings rate, annual savings rate increase, investment horizon, rebalancing strategy, distribution model, degrees of freedom (for t-distribution), random seed, and historical data period.
   - **Ticker Search**:
     - Use the search field to find stocks, ETFs, or funds. Double-click a search result to add it to the portfolio.
   - **Portfolio Configuration**:
     - Add tickers and weightings (in %). Weightings are automatically normalized if they do not sum to 100%.
   - **Start Simulation**:
     - Click "Start Simulation" to initiate the Monte Carlo simulation.
     - Progress is displayed in the status bar.
   - **Cancel Simulation**:
     - Click "Cancel" to stop an ongoing simulation.
   - **View Results**:
     - After the simulation completes, a window displays a table (percentiles, final wealth, CAGR, IRR, max drawdown, recovery time) and a histogram of final wealth.

3. **Interpret Results**:
   - The table shows statistical metrics for various percentiles.
   - The histogram visualizes the distribution of simulated final wealth, with markers for invested capital, median, and the 25-75% percentile range.

## Example

1. Start the application.
2. Enter the parameters:
   - Initial capital: €20,000
   - Monthly savings rate: €500
   - Annual increase: 2%
   - Investment horizon: 15 years
   - Rebalancing: Annually
   - Distribution: t-distribution (degrees of freedom: 5)
   - Historical period: 10 years (or "max" for all available data)
   - Number of simulations: 5,000 (5,000–10,000 yields decent results, but 25,000 is recommended, though it may take several minutes)
3. Add a portfolio, e.g.:
   - Ticker: IVV, Weighting: 50% (S&P 500)
   - Ticker: ACWI, Weighting: 50% (ACWI)
4. Click "Start Simulation."
5. Review the results in the new window.

## Important Notes

- **Yahoo Finance API**: The API endpoints are unofficial and may change. If data retrieval fails, check the API URLs (`API_BASE`) or use alternative data sources.
- **Cancel Function**: The "Cancel" button terminates ongoing simulations by stopping the multiprocessing pool. The application is immediately ready for use after cancellation.
- **Performance**: The number of simulations and processes (`max(1, cpu_count - 1)`) can affect runtime. Reduce the number of simulations for faster results.
- **Error Handling**: The application displays error messages in the GUI, e.g., for invalid inputs or missing data.

## Project Structure

- `main_en.py`: Main script containing all logic, including GUI, data retrieval, simulation, and visualization.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details (if a license file is included).
