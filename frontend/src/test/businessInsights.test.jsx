import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import BusinessInsightsPage from '../pages/BusinessInsightsPage';
import ModelComparisonPage from '../pages/ModelComparisonPage';
import * as api from '../services/api';

vi.mock('../services/api', () => ({
  getInsightsFromBackend: vi.fn().mockResolvedValue(null),
  generateInsightsInBackend: vi.fn().mockResolvedValue(null),
  getMonitoringFromBackend: vi.fn().mockResolvedValue(null),
  getEvaluationComparisonFromBackend: vi.fn().mockResolvedValue(null),
  runBacktestInBackend: vi.fn().mockResolvedValue(null),
  getExplorerDataFromBackend: vi.fn().mockResolvedValue(null),
}));

describe('Business Insights & Monitoring UI Engine', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockUploadedDataset = {
    dataset_id: 'test-dataset-insights-123',
    column_mapping: { series_id: 'series_id', target: 'target', date: 'date' },
    preview: {
      rows: [
        { series_id: 'SERIES_A', date: '2024-01-01', target: 100 },
        { series_id: 'SERIES_B', date: '2024-01-01', target: 200 }
      ]
    }
  };

  const sampleInsightsResponse = {
    dataset_id: 'test-dataset-insights-123',
    series_id: 'SERIES_A',
    model: 'LightGBM',
    horizon: 28,
    demand_outlook: {
      recent_average_demand: 100.0,
      forecast_average_demand: 125.0,
      expected_change_pct: 25.0,
      expected_peak: { value: 140.0, date: '2024-02-15', day_index: 15 },
      expected_low: { value: 110.0, date: '2024-02-02', day_index: 2 },
      forecast_period: '2024-02-01 to 2024-02-28',
      demand_direction: 'INCREASING',
      variability: 0.12,
      uncertainty_level: 'LOW'
    },
    inventory_recommendation: {
      category: 'inventory',
      recommendation: 'CONSIDER STOCKING UP / INCREASING REPLENISHMENT',
      priority: 'HIGH',
      reason: 'Forecast demand is projected to be 25.0% above the recent baseline. Consider reviewing safety stock replenishment.',
      evidence: { recent_average: 100.0, forecast_average: 125.0, expected_change_pct: 25.0, horizon: 28 },
      planning_period: '2024-02-01 to 2024-02-28',
      uncertainty: 'LOW',
      horizon: 28
    },
    staffing_recommendation: {
      category: 'staffing',
      recommendation: 'INCREASE STAFFING / RESOURCE AVAILABILITY',
      priority: 'HIGH',
      reason: 'Demand is projected to remain above baseline (+25.0%). Consider increasing fulfillment capacity coinciding with peak on 2024-02-15.',
      evidence: { recent_average: 100.0, forecast_average: 125.0, horizon: 28 },
      planning_period: '2024-02-01 to 2024-02-28',
      uncertainty: 'LOW',
      horizon: 28
    },
    procurement_recommendation: {
      category: 'procurement',
      recommendation: 'REVIEW PROCUREMENT PLANNING',
      priority: 'HIGH',
      reason: 'Elevated velocity indicates increased replenishment requirements. Review vendor lead times.',
      evidence: { recent_average: 100.0, forecast_average: 125.0, horizon: 28 },
      planning_period: '2024-02-01 to 2024-02-28',
      uncertainty: 'LOW',
      horizon: 28
    },
    demand_planning: {
      action: 'EXPAND CAPACITY / ACCELERATE INVENTORY INFLOW',
      explanation: 'Both recent baseline (100.0) and forward forecast (125.0) indicate expanding velocity (+25.0%).',
      planning_focus: 'Ensure order replenishment buffers are scheduled.'
    },
    horizon_insights: {
      '7D': 'Near-term operational demand projected at average 120.0 units/day.',
      '28D': 'Tactical monthly demand projects 125.0 units/day average.',
      '90D': 'Quarterly strategic trajectory provides run-rate visibility.'
    },
    optional_feature_context: {
      price: {
        available: true,
        recent_avg_price: 3.49,
        insight: 'Historical unit price averaged $3.49. Higher demand is observed alongside promotional price adjustments.'
      }
    }
  };

  it('renders clean empty state for uploaded dataset before computing insights', async () => {
    api.getMonitoringFromBackend.mockResolvedValueOnce({
      dataset_id: 'test-dataset-insights-123',
      status: 'HEALTHY',
      is_alert: false,
      models_tracked: ['LightGBM']
    });
    api.getInsightsFromBackend.mockResolvedValueOnce(null);

    render(
      <BusinessInsightsPage 
        onNavigate={() => {}} 
        uploadedDataset={mockUploadedDataset} 
      />
    );

    expect(screen.getByText(/No Business Insights Computed Yet/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Compute Evidence Insights/i).length).toBeGreaterThan(0);
  });

  it('renders complete business insight cards and monitoring banner when insights data is loaded', async () => {
    api.getMonitoringFromBackend.mockImplementation(async () => ({
      dataset_id: 'test-dataset-insights-123',
      status: 'ACCURACY DRIFT',
      recent_mape: 18.5,
      baseline_mape: 12.0,
      mape_drift_pct: 54.2,
      retraining_alert: 'Model performance has degraded beyond configured error thresholds. Review or retrain the model.',
      is_alert: true,
      models_tracked: ['LightGBM', 'DeepAR']
    }));
    api.getInsightsFromBackend.mockImplementation(async () => sampleInsightsResponse);

    render(
      <BusinessInsightsPage 
        onNavigate={() => {}} 
        uploadedDataset={mockUploadedDataset} 
      />
    );

    // Wait for cached insights to load into DOM
    await waitFor(() => {
      expect(screen.getByText(/1. Demand Outlook/i)).toBeInTheDocument();
    });

    // Check Monitoring Alert
    expect(screen.getByText(/Model Health Status: ACCURACY DRIFT/i)).toBeInTheDocument();
    expect(screen.getByText(/Model performance has degraded beyond configured error thresholds/i)).toBeInTheDocument();

    // Check Inventory card
    expect(screen.getByText(/2. Inventory Recommendation/i)).toBeInTheDocument();
    expect(screen.getByText(/CONSIDER STOCKING UP \/ INCREASING REPLENISHMENT/i)).toBeInTheDocument();

    // Check Staffing card
    expect(screen.getByText(/3. Staffing & Resource Planning/i)).toBeInTheDocument();
    expect(screen.getByText(/INCREASE STAFFING \/ RESOURCE AVAILABILITY/i)).toBeInTheDocument();

    // Check Procurement card
    expect(screen.getByText(/4. Procurement & Vendor Planning/i)).toBeInTheDocument();
    expect(screen.getByText(/REVIEW PROCUREMENT PLANNING/i)).toBeInTheDocument();

    // Check Strategic Planning card
    expect(screen.getByText(/5. Strategic Demand Planning/i)).toBeInTheDocument();

    // Check Optional Feature Context (Price)
    expect(screen.getByText(/6. External Feature Context/i)).toBeInTheDocument();
    expect(screen.getByText(/observed alongside/i)).toBeInTheDocument();
  });

  it('handles empty evaluation state in Model Comparison page cleanly when uploaded dataset is present', () => {
    api.getEvaluationComparisonFromBackend.mockResolvedValueOnce(null);

    render(
      <ModelComparisonPage 
        _onNavigate={() => {}} 
        uploadedDataset={mockUploadedDataset} 
      />
    );

    expect(screen.getByText(/No Backtesting Results Computed for This Dataset Yet/i)).toBeInTheDocument();
    expect(screen.getAllByText(/Run Backtesting Evaluation/i).length).toBeGreaterThan(0);
  });

  it('renders Strategic Demand Planning card non-empty even with minimal backend data and handles unavailable price & holiday', async () => {
    const minimalInsights = {
      ...sampleInsightsResponse,
      demand_planning: {
        state: 'STABLE DEMAND',
        action: 'MAINTAIN STEADY ALLOCATION / STABLE VELOCITY',
        explanation: 'Forecast average (1.8) projects stable velocity vs baseline.',
        planning_focus: 'Standard replenishment buffers apply.'
      },
      optional_feature_context: {
        price: {
          available: false,
          recent_avg_price: null,
          insight: 'Historical unit price unavailable (no valid non-zero price records found for this scope).'
        },
        holiday: {
          available: false,
          events_detected: 0,
          insight: 'No calendar holiday events recorded in historical period.'
        }
      }
    };

    api.getMonitoringFromBackend.mockResolvedValueOnce({
      dataset_id: 'test-dataset-insights-123',
      status: 'HEALTHY',
      is_alert: false,
      models_tracked: ['SARIMA']
    });
    api.getInsightsFromBackend.mockImplementation(async () => minimalInsights);

    render(
      <BusinessInsightsPage 
        onNavigate={() => {}} 
        uploadedDataset={mockUploadedDataset} 
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/5. Strategic Demand Planning/i)).toBeInTheDocument();
    });

    // Strategic Demand Planning card is NOT blank
    expect(screen.getByText(/MAINTAIN STEADY ALLOCATION \/ STABLE VELOCITY/i)).toBeInTheDocument();
    expect(screen.getByText(/Forecast average \(1.8\) projects stable velocity vs baseline./i)).toBeInTheDocument();

    // Price is marked Unavailable and does NOT display $0.0
    expect(screen.getAllByText(/Unavailable/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/no valid non-zero price records found/i)).toBeInTheDocument();
    expect(screen.queryByText(/\$0\.0/i)).not.toBeInTheDocument();

    // Holiday is marked Unavailable without claiming detected events
    expect(screen.getByText(/No calendar holiday events recorded/i)).toBeInTheDocument();
  });

  it('renders holiday summary with "and N more" and complete event_names tags for >5 unique event types', async () => {
    const eventNames = [
      'Halloween', 'VeteransDay', 'Thanksgiving', 'Chanukah End', 'Christmas',
      'NewYear', 'OrthodoxChristmas', 'MartinLutherKingDay', 'SuperBowl', 'LentStart'
    ];
    const holidayInsights = {
      ...sampleInsightsResponse,
      optional_feature_context: {
        holiday: {
          available: true,
          event_records: 16,
          events_detected: 16,
          event_names: eventNames,
          insight: '16 unique calendar event type(s) recorded across 16 observation(s) in the historical period (e.g. Halloween, VeteransDay, Thanksgiving, Chanukah End, Christmas, and 11 more). Demand movements near these dates are noted; no causal relationship is implied.'
        }
      }
    };

    api.getMonitoringFromBackend.mockResolvedValueOnce({
      dataset_id: 'test-dataset-insights-123',
      status: 'HEALTHY',
      is_alert: false,
      models_tracked: ['SARIMA']
    });
    api.getInsightsFromBackend.mockImplementation(async () => holidayInsights);

    render(
      <BusinessInsightsPage 
        onNavigate={() => {}} 
        uploadedDataset={mockUploadedDataset} 
      />
    );

    await waitFor(() => {
      expect(screen.getByText(/6. External Feature Context/i)).toBeInTheDocument();
    });

    // Check summary text with "and 11 more"
    expect(screen.getByText(/16 unique calendar event type\(s\) recorded across 16 observation\(s\)/i)).toBeInTheDocument();
    expect(screen.getByText(/and 11 more/i)).toBeInTheDocument();

    // Check that all 10 event-name tags are individually rendered as tags in the UI
    eventNames.forEach((name) => {
      expect(screen.getByText(name)).toBeInTheDocument();
    });
  });
});

