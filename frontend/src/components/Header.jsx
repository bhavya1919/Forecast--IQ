import React from 'react';
import { Menu, Bell, Calendar } from 'lucide-react';

export default function Header({ 
  activeTab, 
  onMobileToggle, 
  uploadedDataset = null, 
  isRestoringDataset = false, 
  restorationError = null 
}) {
  const getPageInfo = (tab) => {
    switch (tab) {
      case 'home':
        return {
          title: 'Overview',
          breadcrumb: 'ForecastIQ / Enterprise Demand Planning'
        };
      case 'explorer':
        return {
          title: 'Data Explorer',
          breadcrumb: 'ForecastIQ / M5 Dataset Understanding'
        };
      case 'upload':
        return {
          title: 'Upload Dataset',
          breadcrumb: 'ForecastIQ / Universal Data Ingestion'
        };
      case 'forecast':
        return {
          title: 'Generate Forecast',
          breadcrumb: 'ForecastIQ / Multi-Horizon Inference'
        };
      case 'comparison':
        return {
          title: 'Model Comparison',
          breadcrumb: 'ForecastIQ / Benchmark Across Horizons'
        };
      case 'evaluation':
        return {
          title: 'Evaluation & Backtesting',
          breadcrumb: 'ForecastIQ / Error Diagnostics (MAPE & RMSE)'
        };
      case 'insights':
        return {
          title: 'Business Insights',
          breadcrumb: 'ForecastIQ / Decision Support & Planning'
        };
      default:
        return {
          title: 'ForecastIQ',
          breadcrumb: 'Multi-Horizon Time-Series Forecasting'
        };
    }
  };

  const { title, breadcrumb } = getPageInfo(activeTab);

  // Compute cutoff display based on active dataset state
  const renderCutoff = () => {
    if (isRestoringDataset) {
      return (
        <div 
          data-testid="header-cutoff-badge"
          className="hidden md:flex items-center gap-2 bg-white px-3 py-1.5 rounded-xl border border-[#E5D9F2] shadow-xs text-xs text-[#5F5670]"
        >
          <div className="w-3 h-3 border-2 border-[#A294F9] border-t-transparent rounded-full animate-spin"></div>
          <span className="font-medium text-[#1F1B2C]">Cutoff:</span>
          <span>Resolving...</span>
        </div>
      );
    }

    if (restorationError) {
      return (
        <div 
          data-testid="header-cutoff-badge"
          className="hidden md:flex items-center gap-2 bg-white px-3 py-1.5 rounded-xl border border-rose-200 shadow-xs text-xs text-rose-600"
        >
          <Calendar className="w-3.5 h-3.5 text-rose-400" />
          <span className="font-medium">Cutoff:</span>
          <span>Unavailable</span>
        </div>
      );
    }

    if (!uploadedDataset) {
      // Default M5 Benchmark
      return (
        <div 
          data-testid="header-cutoff-badge"
          className="hidden md:flex items-center gap-2 bg-white px-3 py-1.5 rounded-xl border border-[#E5D9F2] shadow-xs text-xs text-[#5F5670]"
        >
          <Calendar className="w-3.5 h-3.5 text-[#A294F9]" />
          <span className="font-medium text-[#1F1B2C]">Cutoff: Day 1913</span>
          <span className="text-[#CDC1FF]">•</span>
          <span>2016-04-24</span>
        </div>
      );
    }

    // Active Custom Dataset: derive strictly from dataset's valid historical data
    // Never show M5 date or Day 1913 index for a custom dataset
    const dateRangeEnd = uploadedDataset.profile?.dateRange?.end;
    const splitTestEnd = uploadedDataset.splits?.test?.end_date;
    const cutoffDate = (dateRangeEnd && dateRangeEnd !== 'N/A') ? dateRangeEnd : (splitTestEnd || 'N/A');

    // Only display Day N if genuine benchmark day index is defined on the dataset
    const dayIndex = uploadedDataset.benchmarkDayIndex || uploadedDataset.profile?.dayIndex || null;

    return (
      <div 
        data-testid="header-cutoff-badge"
        className="hidden md:flex items-center gap-2 bg-white px-3 py-1.5 rounded-xl border border-[#E5D9F2] shadow-xs text-xs text-[#5F5670]"
      >
        <Calendar className="w-3.5 h-3.5 text-[#A294F9]" />
        {dayIndex ? (
          <>
            <span className="font-medium text-[#1F1B2C]">Cutoff: Day {dayIndex}</span>
            <span className="text-[#CDC1FF]">•</span>
            <span>{cutoffDate}</span>
          </>
        ) : (
          <>
            <span className="font-medium text-[#1F1B2C]">Cutoff:</span>
            <span>{cutoffDate}</span>
          </>
        )}
      </div>
    );
  };

  return (
    <header className="sticky top-0 z-30 bg-[#F5EFFF]/90 backdrop-blur-md border-b border-[#E5D9F2]/60 px-6 py-4">
      <div className="flex items-center justify-between gap-4">
        {/* Left: Mobile trigger & Titles */}
        <div className="flex items-center gap-3">
          <button
            onClick={onMobileToggle}
            className="p-2 -ml-2 rounded-xl text-[#5F5670] hover:bg-white hover:text-[#1F1B2C] lg:hidden transition-colors"
            aria-label="Toggle Navigation"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div>
            <div className="text-xs font-medium text-[#8E83A3]">
              {breadcrumb}
            </div>
            <h2 className="text-xl sm:text-2xl font-bold font-heading text-[#1F1B2C] tracking-tight">
              {title}
            </h2>
          </div>
        </div>

        {/* Right: Controls & Profile */}
        <div className="flex items-center gap-3">
          {/* Baseline Context selector */}
          {renderCutoff()}

          {/* Quick Notification */}
          <button 
            className="relative p-2 rounded-xl bg-white border border-[#E5D9F2] text-[#5F5670] hover:text-[#1F1B2C] hover:border-[#CDC1FF] shadow-xs transition-colors"
            title="System alerts: 0 critical, models validated"
          >
            <Bell className="w-4 h-4" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 rounded-full bg-[#A294F9]"></span>
          </button>

          {/* User / Planner Avatar */}
          <div className="flex items-center gap-2.5 pl-2">
            <div className="w-8 h-8 rounded-full bg-[#E5D9F2] text-[#1F1B2C] font-semibold text-xs flex items-center justify-center border border-[#CDC1FF]">
              EP
            </div>
            <div className="hidden sm:block text-left">
              <div className="text-xs font-semibold text-[#1F1B2C] leading-none">
                Enterprise Planner
              </div>
              <div className="text-[10px] text-[#8E83A3] mt-0.5">
                Supply Chain Analytics
              </div>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
