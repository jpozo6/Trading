
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Backtest from './components/Backtest';
import { Activity, BarChart2, TrendingUp } from 'lucide-react';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-white font-sans">
        <nav className="bg-gray-800 border-b border-gray-700 p-4">
          <div className="container mx-auto flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <TrendingUp className="text-green-400" />
              <span className="text-xl font-bold tracking-tight">TradeForce AI</span>
            </div>
            <div className="flex space-x-6">
              <Link to="/" className="hover:text-green-400 transition-colors flex items-center gap-2">
                <Activity size={18} /> Dashboard
              </Link>
              <Link to="/backtest" className="hover:text-green-400 transition-colors flex items-center gap-2">
                <BarChart2 size={18} /> Backtest
              </Link>
            </div>
          </div>
        </nav>

        <main className="container mx-auto p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/backtest" element={<Backtest />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
