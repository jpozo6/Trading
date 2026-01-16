
import React, { useState, useEffect } from 'react';
import { getTickers, getStrategies, runAnalysis } from '../api';
import { Play, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const Dashboard = () => {
    const [index, setIndex] = useState('nasdaq');
    const [tickers, setTickers] = useState([]);
    const [strategies, setStrategies] = useState([]);
    const [selectedTicker, setSelectedTicker] = useState(null);
    const [selectedStrategy, setSelectedStrategy] = useState('coppock');
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        loadStrategies();
        loadTickers(index);
    }, [index]);

    const loadTickers = async (idx) => {
        try {
            const data = await getTickers(idx);
            setTickers(data);
            if (data.length > 0 && !selectedTicker) setSelectedTicker(data[0]);
        } catch (error) {
            console.error("Error loading tickers:", error);
        }
    };

    const loadStrategies = async () => {
        try {
            const data = await getStrategies();
            setStrategies(data);
            if (data.length > 0) setSelectedStrategy(data[0].id);
        } catch (error) {
            console.error("Error loading strategies:", error);
        }
    };

    const handleAnalyze = async () => {
        if (!selectedTicker || !selectedStrategy) return;
        setLoading(true);
        try {
            const result = await runAnalysis(selectedTicker, selectedStrategy);
            setAnalysis(result);
        } catch (error) {
            console.error("Error running analysis:", error);
        } finally {
            setLoading(false);
        }
    };

    // Chart Data Preparation
    const getChartData = () => {
        if (!analysis || !analysis.chart_data) return null;
        const labels = analysis.chart_data.map(d => new Date(d.Date).toLocaleDateString());
        const prices = analysis.chart_data.map(d => d.Close);

        // Attempt to find indicator keys
        const indicatorKeys = Object.keys(analysis.chart_data[0]).filter(k =>
            !['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'week_key', 'month_key', 'Ticker'].includes(k)
        );

        const datasets = [
            {
                label: 'Price',
                data: prices,
                borderColor: '#10b981',
                backgroundColor: 'rgba(16, 185, 129, 0.1)',
                yAxisID: 'y',
            }
        ];

        indicatorKeys.slice(0, 2).forEach((key, idx) => {
            datasets.push({
                label: key,
                data: analysis.chart_data.map(d => d[key]),
                borderColor: idx === 0 ? '#3b82f6' : '#f59e0b',
                yAxisID: key.includes('dist') || key.includes('risk') || key.includes('slope') ? 'y1' : 'y',
                borderDash: idx === 1 ? [5, 5] : [],
            });
        });

        return { labels, datasets };
    };

    const renderSignal = (val) => {
        if (val === 1) return <div className="flex items-center text-green-400 gap-1"><TrendingUp /> BUY</div>;
        if (val === -1) return <div className="flex items-center text-red-400 gap-1"><TrendingDown /> SELL</div>;
        return <div className="flex items-center text-gray-400 gap-1"><Minus /> NEUTRAL</div>;
    };

    return (
        <div className="space-y-6">
            <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 shadow-lg">
                <h2 className="text-2xl font-bold mb-4">Market Analysis</h2>

                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Index</label>
                        <select
                            value={index}
                            onChange={(e) => setIndex(e.target.value)}
                            className="w-full bg-gray-700 border-gray-600 rounded p-2 text-white focus:ring-2 focus:ring-green-500"
                        >
                            <option value="nasdaq">NASDAQ</option>
                            <option value="sp500">S&P 500</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Ticker</label>
                        <select
                            value={selectedTicker || ''}
                            onChange={(e) => setSelectedTicker(e.target.value)}
                            className="w-full bg-gray-700 border-gray-600 rounded p-2 text-white focus:ring-2 focus:ring-green-500"
                        >
                            {tickers.map(t => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Strategy</label>
                        <select
                            value={selectedStrategy || ''}
                            onChange={(e) => setSelectedStrategy(e.target.value)}
                            className="w-full bg-gray-700 border-gray-600 rounded p-2 text-white focus:ring-2 focus:ring-green-500"
                        >
                            {strategies.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                        </select>
                    </div>

                    <div className="flex items-end">
                        <button
                            onClick={handleAnalyze}
                            disabled={loading}
                            className="w-full bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded transition-colors flex items-center justify-center gap-2"
                        >
                            {loading ? 'Analyzing...' : <><Play size={18} /> Run Analysis</>}
                        </button>
                    </div>
                </div>

                {analysis && (
                    <div className="space-y-6 animate-in fade-in duration-500">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="bg-gray-750 p-4 rounded border border-gray-600">
                                <h3 className="text-sm text-gray-400">Latest Signal</h3>
                                <div className="text-2xl font-bold mt-1">{renderSignal(analysis.latest_signal.signal)}</div>
                                <div className="text-xs text-gray-500 mt-1">{new Date(analysis.latest_signal.date).toLocaleDateString()}</div>
                            </div>
                            {/* Expose details */}
                            {Object.entries(analysis.latest_signal.details)
                                .filter(([k]) => !['Signal', 'Open', 'High', 'Low', 'Volume', 'week_key', 'month_key', 'Ticker'].includes(k))
                                .slice(0, 4)
                                .map(([k, v]) => (
                                    <div key={k} className="bg-gray-750 p-4 rounded border border-gray-600">
                                        <h3 className="text-sm text-gray-400 capitalize">{k.replace(/_/g, ' ')}</h3>
                                        <div className="text-xl font-mono mt-1">{typeof v === 'number' ? v.toFixed(2) : v}</div>
                                    </div>
                                ))}
                        </div>

                        <div className="bg-gray-900 p-4 rounded border border-gray-700 h-96">
                            {getChartData() && <Line
                                data={getChartData()}
                                options={{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    interaction: {
                                        mode: 'index',
                                        intersect: false,
                                    },
                                    scales: {
                                        y: {
                                            type: 'linear',
                                            display: true,
                                            position: 'left',
                                            grid: { color: '#374151' }
                                        },
                                        y1: {
                                            type: 'linear',
                                            display: true,
                                            position: 'right',
                                            grid: { drawOnChartArea: false },
                                        },
                                        x: {
                                            grid: { color: '#374151' }
                                        }
                                    },
                                    plugins: {
                                        legend: { labels: { color: '#9ca3af' } },
                                        tooltip: {
                                            backgroundColor: '#1f2937',
                                            titleColor: '#f3f4f6',
                                            bodyColor: '#d1d5db',
                                            borderColor: '#4b5563',
                                            borderWidth: 1
                                        }
                                    }
                                }}
                            />}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default Dashboard;
