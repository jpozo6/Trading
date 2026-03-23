
import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Backtest from './components/Backtest';
import { Activity, BarChart2, TrendingUp } from 'lucide-react';

function App() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <Router>
      <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-gray-900 via-gray-950 to-black text-white font-sans selection:bg-brand-500/30">
        <nav className="fixed top-0 w-full z-50 glass border-b border-gray-800">
          <div className="container mx-auto px-4 lg:px-8">
            <div className="flex items-center justify-between h-16">
              <div className="flex items-center gap-3">
                <div className="bg-brand-500/10 p-2 rounded-lg">
                  <TrendingUp className="text-brand-400 w-6 h-6" />
                </div>
                <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                  TradeForce AI
                </span>
              </div>

              {/* Desktop Nav */}
              <div className="hidden md:flex space-x-8">
                <NavLink to="/" icon={<Activity size={18} />} text="Dashboard" />
                <NavLink to="/backtest" icon={<BarChart2 size={18} />} text="Backtest" />
              </div>

              {/* Mobile Menu Button */}
              <div className="md:hidden">
                <button onClick={() => setIsOpen(!isOpen)} className="text-gray-400 hover:text-white transition-colors">
                  <span className="sr-only">Open menu</span>
                  {isOpen ? (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" /></svg>
                  ) : (
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" /></svg>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Mobile Nav */}
          {isOpen && (
            <div className="md:hidden glass border-t border-gray-800">
              <div className="px-4 pt-2 pb-4 space-y-2">
                <MobileNavLink to="/" text="Dashboard" setIsOpen={setIsOpen} />
                <MobileNavLink to="/backtest" text="Backtest" setIsOpen={setIsOpen} />
              </div>
            </div>
          )}
        </nav>

        <main className="container mx-auto px-4 lg:px-8 pt-24 pb-12 animate-in fade-in duration-700">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/backtest" element={<Backtest />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

const NavLink = ({ to, icon, text }) => (
  <Link to={to} className="group flex items-center gap-2 text-sm font-medium text-gray-400 hover:text-brand-400 transition-colors">
    {icon}
    <span className="group-hover:translate-x-0.5 transition-transform">{text}</span>
  </Link>
);

const MobileNavLink = ({ to, text, setIsOpen }) => (
  <Link
    to={to}
    onClick={() => setIsOpen(false)}
    className="block px-3 py-2 rounded-md text-base font-medium text-gray-300 hover:text-white hover:bg-gray-800 transition-colors"
  >
    {text}
  </Link>
);

export default App;
