
import React, { useState, useEffect } from 'react';
import { getTickers, getStrategies, runBacktest } from '../api';
import { Play, TrendingUp, TrendingDown, DollarSign, Percent } from 'lucide-react';
import { Line } from 'react-chartjs-2';

const Backtest = () => {
    const [index, setIndex] = useState('nasdaq');
    const [tickers, setTickers] = useState([]);
    const [strategies, setStrategies] = useState([]);
    const [selectedTicker, setSelectedTicker] = useState(null);
    const [selectedStrategy, setSelectedStrategy] = useState('coppock');
    const [results, setResults] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        loadStrategies();
        loadTickers(index);
    }, [index]);

    const loadTickers = async (idx) => {
        const data = await getTickers(idx);
        setTickers(data);
        if (data.length > 0 && !selectedTicker) setSelectedTicker(data[0]);
    };

    const loadStrategies = async () => {
        const data = await getStrategies();
        setStrategies(data);
        if (data.length > 0) setSelectedStrategy(data[0].id);
    };

    const handleBacktest = async () => {
        if (!selectedTicker || !selectedStrategy) return;
        setLoading(true);
        try {
            const data = await runBacktest(selectedTicker, selectedStrategy);
            setResults(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const getEquityData = () => {
        if (!results || !results.equity_curve) return null;
        const curve = Object.values(results.equity_curve);
        return {
            labels: curve.map(d => new Date(d.date ? d.date : Object.keys(results.equity_curve).find(k => results.equity_curve[k] === d)).toLocaleDateString()),
            datasets: [{
                label: 'Portfolio Equity',
                data: curve.map(d => d.equity),
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.2)',
                fill: true,
                tension: 0.4
            }]
        };
        // Note: The structure of equity_curve from backtester is {index: {date, equity}}. 
        // Need to be careful with keys. Backtester returns: equity_df.to_dict(orient='index')
        // Keys are Timestamps/Strings. Values are objects {date, equity}.
    };

    return (
        <div className="space-y-6">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 shadow-lg">
                <h2 className="text-2xl font-bold mb-4">Historical Backtest</h2>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Index</label>
                        <select value={index} onChange={(e) => setIndex(e.target.value)} className="w-full bg-gray-700 border-gray-600 rounded p-2 text-white">
                            <option value="nasdaq">NASDAQ</option>
                            <option value="sp500">S&P 500</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Ticker</label>
                        <select value={selectedTicker || ''} onChange={(e) => setSelectedTicker(e.target.value)} className="w-full bg-gray-700 border-gray-600 rounded p-2 text-white">
                            {tickers.map(t => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Strategy</label>
                        <select value={selectedStrategy || ''} onChange={(e) => setSelectedStrategy(e.target.value)} className="w-full bg-gray-700 border-gray-600 rounded p-2 text-white">
                            {strategies.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                        </select>
                    </div>
                    <div className="flex items-end">
                        <button onClick={handleBacktest} disabled={loading} className="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded transition-colors flex items-center justify-center gap-2">
                            {loading ? 'Running...' : <><Play size={18} /> Run Backtest</>}
                        </button>
                    </div>
                </div>

                {results && (
                    <div className="space-y-6 animate-in fade-in duration-500">
                        {/* Metrics */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-gray-750 p-4 rounded border border-gray-600">
                                <h3 className="text-sm text-gray-400">Total Return</h3>
                                <div className={`text-2xl font-bold mt-1 ${results.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {results.total_return_pct.toFixed(2)}%
                                </div>
                            </div>
                            <div className="bg-gray-750 p-4 rounded border border-gray-600">
                                <h3 className="text-sm text-gray-400">Win Rate</h3>
                                <div className="text-2xl font-bold mt-1 text-blue-400">
                                    {(results.win_rate * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="bg-gray-750 p-4 rounded border border-gray-600">
                                <h3 className="text-sm text-gray-400">Final Balance</h3>
                                <div className="text-2xl font-bold mt-1 text-white">
                                    ${results.final_balance.toFixed(2)}
                                </div>
                            </div>
                            <div className="bg-gray-750 p-4 rounded border border-gray-600">
                                <h3 className="text-sm text-gray-400">Total Trades</h3>
                                <div className="text-2xl font-bold mt-1 text-white">
                                    {results.trades.length}
                                </div>
                            </div>
                        </div>

                        {/* Chart */}
                        <div className="bg-gray-900 p-4 rounded border border-gray-700 h-80">
                            {getEquityData() && <Line
                                data={getEquityData()}
                                options={{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    scales: {
                                        y: { grid: { color: '#374151' } },
                                        x: { grid: { color: '#374151' } }
                                    },
                                    plugins: { legend: { display: false } }
                                }}
                            />}
                        </div>

                        {/* Trades Table */}
                        <div className="overflow-x-auto">
                            <table className="min-w-full bg-gray-900 text-sm text-left">
                                <thead className="text-gray-400 uppercase bg-gray-800">
                                    <tr>
                                        <th className="px-4 py-3">Type</th>
                                        <th className="px-4 py-3">Entry Date</th>
                                        <th className="px-4 py-3">Exit Date</th>
                                        <th className="px-4 py-3">Return</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {results.trades.slice().reverse().map((trade, idx) => (
                                        <tr key={idx} className="border-b border-gray-800 hover:bg-gray-800">
                                            <td className="px-4 py-3">{trade.type}</td>
                                            <td className="px-4 py-3">{new Date(trade.entry_date).toLocaleDateString()}</td>
                                            <td className="px-4 py-3">{new Date(trade.exit_date).toLocaleDateString()}</td>
                                            <td className={`px-4 py-3 flex items-center gap-1 ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                {trade.pnl >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                                {trade.return_pct ? (trade.return_pct * 100).toFixed(2) : 0}%
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Backtest;
