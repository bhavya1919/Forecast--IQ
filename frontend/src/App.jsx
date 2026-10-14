import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import HomePage from './pages/HomePage';
import DataExplorerPage from './pages/DataExplorerPage';
import UploadDatasetPage from './pages/UploadDatasetPage';
import GenerateForecastPage from './pages/GenerateForecastPage';
import ModelComparisonPage from './pages/ModelComparisonPage';
import EvaluationPage from './pages/EvaluationPage';
import BusinessInsightsPage from './pages/BusinessInsightsPage';

export default function App() {
  const [activeTab, setActiveTab] = useState('home');
  const [mobileOpen, setMobileOpen] = useState(false);
  const [uploadedDataset, setUploadedDataset] = useState(null);

  const renderActivePage = () => {
    switch (activeTab) {
      case 'home':
        return <HomePage onNavigate={setActiveTab} />;
      case 'explorer':
        return <DataExplorerPage onNavigate={setActiveTab} uploadedDataset={uploadedDataset} />;
      case 'upload':
        return (
          <UploadDatasetPage 
            onNavigate={setActiveTab} 
            onDatasetReady={(ds) => setUploadedDataset(ds)} 
          />
        );
      case 'forecast':
        return (
          <GenerateForecastPage 
            onNavigate={setActiveTab} 
            uploadedDataset={uploadedDataset} 
          />
        );
      case 'comparison':
        return (
          <ModelComparisonPage 
            onNavigate={setActiveTab} 
            uploadedDataset={uploadedDataset} 
          />
        );
      case 'evaluation':
        return (
          <EvaluationPage 
            onNavigate={setActiveTab} 
            uploadedDataset={uploadedDataset} 
          />
        );
      case 'insights':
        return (
          <BusinessInsightsPage 
            onNavigate={setActiveTab} 
            uploadedDataset={uploadedDataset} 
          />
        );
      default:
        return <HomePage onNavigate={setActiveTab} />;
    }
  };

  return (
    <div className="min-h-screen bg-[#F5EFFF] text-[#1F1B2C] flex">
      {/* Fixed Sidebar */}
      <Sidebar 
        activeTab={activeTab} 
        setActiveTab={setActiveTab} 
        mobileOpen={mobileOpen} 
        setMobileOpen={setMobileOpen} 
        uploadedDataset={uploadedDataset}
      />

      {/* Main Content Area (Offset by sidebar width on large screens) */}
      <div className="flex-1 lg:pl-64 flex flex-col min-h-screen">
        {/* Top Header */}
        <Header 
          activeTab={activeTab} 
          onMobileToggle={() => setMobileOpen(true)} 
        />

        {/* Page Content Container with generous padding */}
        <main className="flex-1 p-6 sm:p-8 lg:p-10">
          {renderActivePage()}
        </main>

        {/* Minimal Footer */}
        <footer className="px-8 py-5 border-t border-[#E5D9F2]/70 text-xs text-[#8E83A3] flex flex-col sm:flex-row items-center justify-between gap-3">
          <div>
            <strong>ForecastIQ</strong> — Multi-Horizon Time-Series Forecasting for Enterprise Analytics
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>M5 Accuracy Benchmark</span>
            <span>•</span>
            <span>Darts + PyTorch Engine</span>
            <span>•</span>
            <span className="text-[#A294F9] font-medium">Enterprise Ready</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
