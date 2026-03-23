
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
    const [years, setYears] = useState(5);
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
            const result = await runAnalysis(selectedTicker, selectedStrategy, years);
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
        const labels = analysis.chart_data.map(d => new Date(d.Date || d.date).toLocaleDateString());
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
        <div className="space-y-4">
            <div className="glass-card p-4 md:p-6">
                <h2 className="text-xl font-bold bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent mb-4">Market Analysis</h2>

                <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
                    <div className="space-y-2">
                        <label className="text-sm font-medium text-gray-400">Index</label>
                        <div className="relative">
                            <select
                                value={index}
                                onChange={(e) => setIndex(e.target.value)}
                                className="input-field appearance-none cursor-pointer"
                            >
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
                            <select
                                value={selectedTicker || ''}
                                onChange={(e) => setSelectedTicker(e.target.value)}
                                className="input-field appearance-none cursor-pointer"
                            >
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
                            <select
                                value={selectedStrategy || ''}
                                onChange={(e) => setSelectedStrategy(e.target.value)}
                                className="input-field appearance-none cursor-pointer"
                            >
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

                    <div className="flex items-end">
                        <button
                            onClick={handleAnalyze}
                            disabled={loading}
                            className="btn-primary w-full"
                        >
                            {loading ? (
                                <>
                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Analyzing...
                                </>
                            ) : (
                                <><Play size={18} /> Run Analysis</>
                            )}
                        </button>
                    </div>
                </div>

                {analysis && (
                    <div className="space-y-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
                        {/* Summary Section */}
                        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                            <div className="glass-card p-3 border-l-4 border-l-brand-500">
                                <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">Latest Signal</h3>
                                <div>{renderSignal(analysis.latest_signal.signal)}</div>
                                <div className="text-xs text-gray-500 mt-1 font-mono">
                                    {new Date(analysis.latest_signal.date).toLocaleDateString()}
                                </div>
                            </div>

                            {/* Key Metrics Grid - using the details from backend */}
                            {Object.entries(analysis.latest_signal.details)
                                .filter(([k]) => !['Signal', 'Open', 'High', 'Low', 'Volume', 'week_key', 'month_key', 'Ticker', 'date'].includes(k))
                                .slice(0, 4)
                                .map(([k, v]) => (
                                    <div key={k} className="glass-card p-3">
                                        <h3 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-1">{k.replace(/_/g, ' ')}</h3>
                                        <div className="text-lg font-bold font-mono text-gray-100">
                                            {typeof v === 'number' ? v.toFixed(2) : v}
                                        </div>
                                    </div>
                                ))}
                        </div>

                        {/* Chart Section */}
                        <div className="glass-card p-4 h-[250px]">
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
                                            grid: { color: 'rgba(55, 65, 81, 0.3)' },
                                            ticks: { color: '#9ca3af' }
                                        },
                                        y1: {
                                            type: 'linear',
                                            display: true,
                                            position: 'right',
                                            grid: { drawOnChartArea: false },
                                            ticks: { color: '#9ca3af' }
                                        },
                                        x: {
                                            grid: { color: 'rgba(55, 65, 81, 0.3)' },
                                            ticks: { color: '#9ca3af' }
                                        }
                                    },
                                    plugins: {
                                        legend: {
                                            labels: {
                                                color: '#e5e7eb',
                                                usePointStyle: true,
                                                padding: 20
                                            },
                                            position: 'top'
                                        },
                                        tooltip: {
                                            backgroundColor: 'rgba(17, 24, 39, 0.95)',
                                            titleColor: '#f3f4f6',
                                            bodyColor: '#d1d5db',
                                            borderColor: 'rgba(75, 85, 99, 0.4)',
                                            borderWidth: 1,
                                            padding: 12,
                                            cornerRadius: 8,
                                            displayColors: true
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
