#Reference to @simon.villani  https://medium.com/@simon.villani/harnessing-the-power-of-transformers-and-gpt-4-predicting-stock-prices-with-machine-learning-b067fd3f596
import yfinance as yf
import numbers as np

def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def log_transform(data):
    return np.log(data + 1)  # Adding 1 to avoid taking the log of 0

def inverse_log_transform(log_data):
    return np.exp(log_data) - 1

def preprocess_data(stock_data):
    # Handle missing values
    stock_data = stock_data.dropna()
    for col in stock_data.columns:
      stock_data[col] = log_transform(stock_data[col])
    # Calculate technical indicators
    stock_data = add_all_ta_features(stock_data, open="Open", high="High", low="Low", close="Close", volume="Volume")
    stock_data = stock_data.drop(['trend_psar_up', 'trend_psar_down'],axis=1)
    # Remove initial rows with NaN values created by technical indicators
    largest_indicator_window = 14  # Adjust this value based on the technical indicators being used
    stock_data = stock_data.iloc[largest_indicator_window:]
    stock_data = stock_data.dropna()

    # Scale the data
    scaler = MinMaxScaler()
    stock_data_scaled = pd.DataFrame(scaler.fit_transform(stock_data), columns=stock_data.columns)

    return stock_data_scaled, scaler
# Prepare input and target sequences
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + 1:i + seq_len + 1])
    return np.array(X), np.array(y)