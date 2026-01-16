
import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const getTickers = async (index) => {
    const response = await axios.get(`${API_URL}/tickers/${index}`);
    return response.data;
};

export const getStrategies = async () => {
    const response = await axios.get(`${API_URL}/strategies`);
    return response.data;
};

export const runAnalysis = async (ticker, strategyId) => {
    const response = await axios.post(`${API_URL}/analyze`, { ticker, strategy_id: strategyId });
    return response.data;
};

export const runBacktest = async (ticker, strategyId) => {
    const response = await axios.post(`${API_URL}/backtest`, { ticker, strategy_id: strategyId });
    return response.data;
};
