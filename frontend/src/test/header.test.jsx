import React from 'react';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import Header from '../components/Header';

describe('Header Dataset Cutoff Dynamics', () => {
  it('renders default M5 benchmark cutoff when no dataset is active', () => {
    render(<Header activeTab="home" onMobileToggle={() => {}} uploadedDataset={null} />);
    
    const badge = screen.getByTestId('header-cutoff-badge');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent(/Cutoff:\s*Day 1913/i);
    expect(badge).toHaveTextContent(/2016-04-24/i);
  });

  it('renders resolving state when dataset is actively restoring', () => {
    render(
      <Header 
        activeTab="home" 
        onMobileToggle={() => {}} 
        uploadedDataset={null} 
        isRestoringDataset={true} 
      />
    );
    
    const badge = screen.getByTestId('header-cutoff-badge');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent(/Resolving\.\.\./i);
    expect(badge).not.toHaveTextContent(/Day 1913/i);
    expect(badge).not.toHaveTextContent(/2016-04-24/i);
  });

  it('renders unavailable state when dataset restoration encountered an error', () => {
    render(
      <Header 
        activeTab="home" 
        onMobileToggle={() => {}} 
        uploadedDataset={null} 
        isRestoringDataset={false}
        restorationError="Network timeout reconnecting to persistence engine"
      />
    );
    
    const badge = screen.getByTestId('header-cutoff-badge');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent(/Unavailable/i);
    expect(badge).not.toHaveTextContent(/Day 1913/i);
  });

  it('renders custom dataset latest valid historical cutoff without Day N index', () => {
    const customDataset = {
      dataset_id: 'custom-ds-01',
      filename: '01_valid_retail_multiseries.csv',
      profile: {
        dateRange: { start: '2024-01-01', end: '2025-06-30' }
      },
      splits: {
        test: { end_date: '2025-06-30' }
      }
    };

    render(
      <Header 
        activeTab="explorer" 
        onMobileToggle={() => {}} 
        uploadedDataset={customDataset} 
      />
    );
    
    const badge = screen.getByTestId('header-cutoff-badge');
    expect(badge).toBeInTheDocument();
    expect(badge).toHaveTextContent(/2025-06-30/i);
    expect(badge).not.toHaveTextContent(/Day 1913/i);
    expect(badge).not.toHaveTextContent(/2016-04-24/i);
  });

  it('updates cutoff date when switching between two custom datasets', () => {
    const datasetA = {
      dataset_id: 'ds-a',
      profile: { dateRange: { end: '2024-12-31' } }
    };
    const datasetB = {
      dataset_id: 'ds-b',
      profile: { dateRange: { end: '2025-11-08' } }
    };

    const { rerender } = render(
      <Header activeTab="home" onMobileToggle={() => {}} uploadedDataset={datasetA} />
    );
    expect(screen.getByTestId('header-cutoff-badge')).toHaveTextContent(/2024-12-31/i);

    rerender(
      <Header activeTab="home" onMobileToggle={() => {}} uploadedDataset={datasetB} />
    );
    expect(screen.getByTestId('header-cutoff-badge')).toHaveTextContent(/2025-11-08/i);
    expect(screen.getByTestId('header-cutoff-badge')).not.toHaveTextContent(/2024-12-31/i);
  });
});
