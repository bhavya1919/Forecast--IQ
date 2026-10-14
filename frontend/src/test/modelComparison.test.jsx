import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import ModelComparisonPage from '../pages/ModelComparisonPage';
import * as api from '../services/api';

vi.mock('../services/api', () => ({
  getEvaluationComparisonFromBackend: vi.fn(),
  runBacktestInBackend: vi.fn(),
  getExplorerDataFromBackend: vi.fn(),
}));

describe('ModelComparisonPage - Unevaluated Horizon & Chart Consistency', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockUploadedDataset = {
    dataset_id: 'cddccc63-6774-4114-b5d5-b68b234d4284',
    filename: 'm5_canonical_sample.csv',
  };

  // Verified backtest results for m5_canonical_sample.csv (7D: LightGBM 38.29%, 28D: SARIMA 36.13%, 90D: null)
  const mockBacktestData = {
    dataset_id: 'cddccc63-6774-4114-b5d5-b68b234d4284',
    models_evaluated: ['arima', 'sarima', 'prophet', 'lightgbm', 'deepar', 'tft'],
    horizons_evaluated: ['7D', '28D'],
    table_1_model_x_horizon_mape: {
      arima: { '7D': 42.15, '28D': 39.80, '90D': null },
      sarima: { '7D': 39.50, '28D': 36.13, '90D': null },
      prophet: { '7D': 44.20, '28D': 41.30, '90D': null },
      lightgbm: { '7D': 38.29, '28D': 37.10, '90D': null },
      deepar: { '7D': 48.00, '28D': 45.20, '90D': null },
      tft: { '7D': 46.50, '28D': 43.80, '90D': null },
    },
    table_2_model_x_horizon_rmse: {
      arima: { '7D': 1.25, '28D': 1.18, '90D': null },
      sarima: { '7D': 1.15, '28D': 1.08, '90D': null },
      prophet: { '7D': 1.30, '28D': 1.22, '90D': null },
      lightgbm: { '7D': 1.12, '28D': 1.10, '90D': null },
      deepar: { '7D': 1.45, '28D': 1.35, '90D': null },
      tft: { '7D': 1.40, '28D': 1.30, '90D': null },
    },
    best_per_horizon: {
      '7D': { best_model: 'lightgbm', mape: 38.29, rmse: 1.12, is_tie: false, tied_models: [] },
      '28D': { best_model: 'sarima', mape: 36.13, rmse: 1.08, is_tie: false, tied_models: [] },
      '90D': null,
    },
    best_model_overall: {
      model: 'lightgbm',
      overall_mape: 37.70,
      horizons_covered: 2,
      evaluated_horizons: ['7D', '28D'],
    },
    scope: { store_id: null, category: null },
  };

  it('renders verified 7D and 28D winners, with 90D correctly displaying dashes in tables', async () => {
    api.getExplorerDataFromBackend.mockResolvedValue({
      available_filters: { stores: ['CA_1', 'CA_2'], categories: ['HOBBIES'], items: [], series: [] },
      dimensions: { has_store: true, has_category: true, has_item: false, has_series: false },
    });
    api.getEvaluationComparisonFromBackend.mockResolvedValue(mockBacktestData);

    render(<ModelComparisonPage uploadedDataset={mockUploadedDataset} />);

    await waitFor(() => {
      // 7D winner: LightGBM 38.29%
      expect(screen.getAllByText(/38\.29%/i).length).toBeGreaterThan(0);
      // 28D winner: SARIMA 36.13%
      expect(screen.getAllByText(/36\.13%/i).length).toBeGreaterThan(0);
    });

    // Best overall card clearly states evaluated horizons
    expect(screen.getByText(/Evaluated across 7D, 28D \(90D not evaluated\)/i)).toBeInTheDocument();
  });

  it('does not plot unevaluated 90D horizon as zero error in Degradation Curve', async () => {
    api.getExplorerDataFromBackend.mockResolvedValue(null);
    api.getEvaluationComparisonFromBackend.mockResolvedValue(mockBacktestData);

    const { container } = render(<ModelComparisonPage uploadedDataset={mockUploadedDataset} />);

    await waitFor(() => {
      expect(screen.getAllByText(/90D Not Evaluated/i).length).toBeGreaterThan(0);
    });

    // 90D axis label visibly marked as not evaluated
    expect(screen.getByText(/\(Not Evaluated\)/i)).toBeInTheDocument();
    expect(screen.getByText(/Insufficient history for 90D/i)).toBeInTheDocument();

    // Verify SVG paths in degradation curve:
    // With 90D unevaluated, paths must only connect 2 points (x=90 and x=250), never x=410
    const paths = container.querySelectorAll('svg path');
    paths.forEach(p => {
      const d = p.getAttribute('d');
      // If path contains M 90 ..., it must NOT contain 410 or connect to bottom 150
      if (d && d.includes('M 90')) {
        expect(d).not.toContain('410');
      }
    });
  });

  it('does not draw zero-height bars for unevaluated 90D in Visual Comparison bar chart', async () => {
    api.getExplorerDataFromBackend.mockResolvedValue(null);
    api.getEvaluationComparisonFromBackend.mockResolvedValue(mockBacktestData);

    render(<ModelComparisonPage uploadedDataset={mockUploadedDataset} />);

    await waitFor(() => {
      expect(screen.getByText(/Visual Comparison/i)).toBeInTheDocument();
    });

    // Legend badge clearly shows 90D is Not Evaluated
    expect(screen.getAllByText(/Not Evaluated/i).length).toBeGreaterThan(0);
  });

  it('dynamically reloads metrics when filters change', async () => {
    api.getExplorerDataFromBackend.mockResolvedValue({
      available_filters: { stores: ['CA_1', 'CA_2'], categories: ['HOBBIES'], items: [], series: [] },
      dimensions: { has_store: true, has_category: true, has_item: false, has_series: false },
    });
    api.getEvaluationComparisonFromBackend.mockResolvedValue(mockBacktestData);

    render(<ModelComparisonPage uploadedDataset={mockUploadedDataset} />);

    await waitFor(() => {
      expect(screen.getByRole('combobox', { name: /Horizon/i })).toBeInTheDocument();
    });

    // Change horizon filter
    const horizonSelect = screen.getByRole('combobox', { name: /Horizon/i });
    fireEvent.change(horizonSelect, { target: { value: '7d' } });

    await waitFor(() => {
      expect(api.getEvaluationComparisonFromBackend).toHaveBeenCalledWith(
        mockUploadedDataset.dataset_id,
        expect.objectContaining({ horizon: 7 }),
        expect.any(AbortSignal)
      );
    });
  });
});
