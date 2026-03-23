
import React, { useState, useEffect } from 'react';
import { getTickers, getStrategies, runBacktest, runBacktestAll } from '../api';
import { Play, TrendingUp, TrendingDown, DollarSign, Percent, Activity, Layers } from 'lucide-react';
import { Line } from 'react-chartjs-2';

const Backtest = () => {
    const [index, setIndex] = useState('nasdaq');
    const [tickers, setTickers] = useState([]);
    const [strategies, setStrategies] = useState([]);
    const [selectedTicker, setSelectedTicker] = useState(null);
    const [selectedStrategy, setSelectedStrategy] = useState('coppock');
    const [years, setYears] = useState(5);
    const [results, setResults] = useState(null);
    const [allResults, setAllResults] = useState(null);
    const [loading, setLoading] = useState(false);
    const [loadingAll, setLoadingAll] = useState(false);

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
        setAllResults(null);
        try {
            const data = await runBacktest(selectedTicker, selectedStrategy, years);
            setResults(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoading(false);
        }
    };

    const handleBacktestAll = async () => {
        if (!selectedTicker) return;
        setLoadingAll(true);
        setResults(null);
        try {
            const data = await runBacktestAll(selectedTicker, years);
            setAllResults(data);
        } catch (error) {
            console.error(error);
        } finally {
            setLoadingAll(false);
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
        <div className="space-y-4">
            <div className="glass-card p-4 md:p-6">
                <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent mb-4">Historical Backtest</h2>

                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-400">Index</label>
                        <div className="relative">
                            <select value={index} onChange={(e) => setIndex(e.target.value)} className="input-field appearance-none cursor-pointer">
                                <option value="nasdaq">NASDAQ</option>
                                <option value="sp500">S&P 500</option>
                            </select>
                            <div className="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none text-gray-400">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                            </div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-400">Ticker</label>
                        <div className="relative">
                            <select value={selectedTicker || ''} onChange={(e) => setSelectedTicker(e.target.value)} className="input-field appearance-none cursor-pointer">
                                {tickers.map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                            <div className="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none text-gray-400">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                            </div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-400">Strategy</label>
                        <div className="relative">
                            <select value={selectedStrategy || ''} onChange={(e) => setSelectedStrategy(e.target.value)} className="input-field appearance-none cursor-pointer">
                                {strategies.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                            </select>
                            <div className="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none text-gray-400">
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                            </div>
                        </div>
                    </div>
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-400">Time (Years)</label>
                        <div className="relative">
                            <input
                                type="number"
                                value={years}
                                onChange={(e) => setYears(parseInt(e.target.value) || 1)}
                                min="1"
                                max="100"
                                className="input-field"
                            />
                        </div>
                    </div>
                    <div className="flex items-end gap-2">
                        <button onClick={handleBacktest} disabled={loading || loadingAll} className="btn-primary flex-1">
                            {loading ? 'Running...' : <><Play size={18} /> Run</>}
                        </button>
                        <button onClick={handleBacktestAll} disabled={loading || loadingAll} className="btn-primary flex-1 bg-purple-600 hover:bg-purple-500 shadow-purple-500/20">
                            {loadingAll ? 'Running...' : <><Layers size={18} /> All</>}
                        </button>
                    </div>
                </div>

                {results && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        {/* Metrics */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            <div className="glass-card p-3">
                                <div className="flex items-center gap-1 mb-1 text-gray-400">
                                    <Percent size={14} />
                                    <h3 className="text-xs font-medium uppercase tracking-wider">Total Return</h3>
                                </div>
                                <div className={`text-xl font-bold ${results.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                    {results.total_return_pct.toFixed(2)}%
                                </div>
                            </div>
                            <div className="glass-card p-3">
                                <div className="flex items-center gap-1 mb-1 text-gray-400">
                                    <TrendingUp size={14} />
                                    <h3 className="text-xs font-medium uppercase tracking-wider">Win Rate</h3>
                                </div>
                                <div className="text-xl font-bold text-brand-400">
                                    {(results.win_rate * 100).toFixed(1)}%
                                </div>
                            </div>
                            <div className="glass-card p-3">
                                <div className="flex items-center gap-1 mb-1 text-gray-400">
                                    <DollarSign size={14} />
                                    <h3 className="text-xs font-medium uppercase tracking-wider">Final Balance</h3>
                                </div>
                                <div className="text-xl font-bold text-white">
                                    ${results.final_balance.toFixed(2)}
                                </div>
                            </div>
                            <div className="glass-card p-3">
                                <div className="flex items-center gap-1 mb-1 text-gray-400">
                                    <Activity size={14} />
                                    <h3 className="text-xs font-medium uppercase tracking-wider">Total Trades</h3>
                                </div>
                                <div className="text-xl font-bold text-white">
                                    {results.trades.length}
                                </div>
                            </div>
                        </div>

                        {/* Chart */}
                        <div className="glass-card p-4 h-[250px]">
                            {getEquityData() && <Line
                                data={getEquityData()}
                                options={{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    scales: {
                                        y: { grid: { color: 'rgba(55, 65, 81, 0.3)' }, ticks: { color: '#9ca3af' } },
                                        x: { grid: { color: 'rgba(55, 65, 81, 0.3)' }, ticks: { color: '#9ca3af' } }
                                    },
                                    plugins: { legend: { display: false } }
                                }}
                            />}
                        </div>

                        {/* Trades Table */}
                        <div className="glass-card overflow-hidden">
                            <div className="px-6 py-4 border-b border-gray-700/50">
                                <h3 className="text-lg font-semibold text-white">Trade History</h3>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-left">
                                    <thead className="text-xs text-gray-400 uppercase bg-gray-900/50">
                                        <tr>
                                            <th className="px-6 py-4 font-medium">Type</th>
                                            <th className="px-6 py-4 font-medium">Entry Date</th>
                                            <th className="px-6 py-4 font-medium">Exit Date</th>
                                            <th className="px-6 py-4 font-medium text-right">Return</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-700/30">
                                        {results.trades.slice().reverse().map((trade, idx) => (
                                            <tr key={idx} className="hover:bg-gray-700/20 transition-colors">
                                                <td className="px-6 py-4">
                                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${trade.type === 'LONG' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                                                        }`}>
                                                        {trade.type}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-gray-300">{new Date(trade.entry_date).toLocaleDateString()}</td>
                                                <td className="px-6 py-4 text-gray-300">{new Date(trade.exit_date).toLocaleDateString()}</td>
                                                <td className={`px-6 py-4 text-right font-mono font-medium ${trade.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    <div className="flex items-center justify-end gap-2">
                                                        {trade.pnl >= 0 ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                                        {trade.return_pct ? (trade.return_pct * 100).toFixed(2) : 0}%
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {allResults && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        <div className="glass-card overflow-hidden">
                            <div className="px-4 py-3 border-b border-gray-700/50 flex items-center justify-between">
                                <h3 className="text-lg font-semibold text-white">All Strategies Comparison - {allResults.ticker}</h3>
                                <span className="text-sm text-gray-400">{allResults.years} years</span>
                            </div>
                            <div className="overflow-x-auto">
                                <table className="w-full text-sm text-left">
                                    <thead className="text-xs text-gray-400 uppercase bg-gray-900/50">
                                        <tr>
                                            <th className="px-4 py-3 font-medium">Strategy</th>
                                            <th className="px-4 py-3 font-medium text-right">Return</th>
                                            <th className="px-4 py-3 font-medium text-right">Win Rate</th>
                                            <th className="px-4 py-3 font-medium text-right">Final Balance</th>
                                            <th className="px-4 py-3 font-medium text-right">Trades</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-gray-700/30">
                                        {allResults.results.map((row, idx) => (
                                            <tr key={idx} className="hover:bg-gray-700/20 transition-colors">
                                                <td className="px-4 py-3 font-medium text-white">{row.strategy_name}</td>
                                                <td className={`px-4 py-3 text-right font-mono font-bold ${row.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                                                    {row.total_return_pct.toFixed(2)}%
                                                </td>
                                                <td className="px-4 py-3 text-right font-mono text-brand-400">
                                                    {(row.win_rate * 100).toFixed(1)}%
                                                </td>
                                                <td className="px-4 py-3 text-right font-mono text-gray-200">
                                                    ${row.final_balance.toFixed(2)}
                                                </td>
                                                <td className="px-4 py-3 text-right text-gray-300">
                                                    {row.total_trades}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Backtest;
