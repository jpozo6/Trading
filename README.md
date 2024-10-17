# Stock Analysis Project

This project is designed to download, analyze, and recommend stocks based on technical indicators. It primarily focuses on NASDAQ tickers and uses various Python libraries to perform the analysis.


## Files and Directories

- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`analysis_results/`**: Contains the results of the stock analysis.
- **`analyze_tickers.ipynb`**: Jupyter notebook for analyzing tickers.
- **`compra_alfayate.ipynb`**: Jupyter notebook for stock purchase analysis based on Alfayate's criteria.
- **`compra.ipynb`**: Jupyter notebook for stock purchase recommendations.
- **`download_tickers.py`**: Python script to download and analyze stock tickers.
- **`nasdaq_analysis_results/`**: Contains the analysis results for NASDAQ tickers.
- **`venv/`**: Virtual environment directory.

## Setup

1. **Clone the repository**:
    ```sh
    git clone https://github.com/jpozo6/Trading.git
    cd Trading/
    ```

2. **Create and activate a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Download and Analyze Tickers

Run the `download_tickers.py` script to download and analyze the top 50 NASDAQ tickers:

```sh
python [download_tickers.py]

## Analyze Tickers in Jupyter Notebooks

Open the Jupyter notebooks (analyze_tickers.ipynb, compra_alfayate.ipynb, compra.ipynb) to perform further analysis and generate recommendations

Functions
download_tickers.py
  - get_top_nasdaq_tickers(n=50): Fetches the top n NASDAQ tickers.
  - download_and_analyze_ticker(ticker, period='1y', interval='1d'): Downloads and analyzes a specific ticker.
  - main(): Main function to create output directories, get tickers, and process them.
compra.ipynb
  - analyze_tickers(interval='1d'): Analyzes tickers and generates purchase recommendations based on technical indicators.
Results
The analysis results are saved in the nasdaq_analysis_results/ directory, and the recommendations are saved in the analysis_results/ directory.

License
This project is licensed under the MIT License.

