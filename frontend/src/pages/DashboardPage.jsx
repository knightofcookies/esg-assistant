import React, { useState, useEffect, useCallback } from 'react';
import ReportUpload from '../components/ReportUpload';
import ReportList from '../components/ReportList';
import { getReports } from '../services/api';

const DashboardPage = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchReports = useCallback(async () => {
    setLoading(true);
    try {
      const response = await getReports();
      setReports(response.data);
    } catch (err) {
      setError('Failed to fetch reports.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);


  useEffect(() => {
    fetchReports();
  }, [fetchReports]);

  const handleUploadSuccess = (newReport) => {
    // Add to list or re-fetch
    setReports(prevReports => [newReport, ...prevReports.filter(r => r.id !== newReport.id)]);
    // Or better: fetchReports(); for consistency if backend modifies status quickly
  };

  if (loading) return <p>Loading reports...</p>;
  if (error) return <p style={{ color: 'red' }}>{error}</p>;

  return (
    <div>
      <h2>ESG Dashboard</h2>
      <ReportUpload onUploadSuccess={handleUploadSuccess} />
      <ReportList reports={reports} />
    </div>
  );
};

export default DashboardPage;