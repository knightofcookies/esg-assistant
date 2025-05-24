import React, { useState, useEffect, useCallback, useRef } from 'react'; // Added useRef
import ReportUpload from '../components/ReportUpload';
import ReportList from '../components/ReportList';
import { getReports } from '../services/api';

const POLLING_INTERVAL = 5000; // 5 seconds

const DashboardPage = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const pollingIntervalRef = useRef(null); // To store interval ID

  const fetchReports = useCallback(async (isPolling = false) => {
    if (!isPolling) setLoading(true); // Only show full loading state on initial load
    setError('');
    try {
      const response = await getReports();
      setReports(response.data);
    } catch (err) {
      setError('Failed to fetch reports.');
      console.error(err);
    } finally {
      if (!isPolling) setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchReports(); // Initial fetch
  }, [fetchReports]);

  // Effect for polling
  useEffect(() => {
    const clearPolling = () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };

    const activeProcessing = reports.some(
      (r) => r.status === 'nlp_queued' || r.status === 'nlp_processing'
    );

    if (activeProcessing) {
      if (!pollingIntervalRef.current) { // Start polling only if not already polling
        pollingIntervalRef.current = setInterval(() => fetchReports(true), POLLING_INTERVAL);
      }
    } else {
      clearPolling(); // Stop polling if no reports are actively processing
    }

    return () => clearPolling(); // Cleanup on component unmount
  }, [reports, fetchReports]); // Re-run when reports array changes (to check activeProcessing)


  const handleUploadSuccess = (newReport) => {
    // New report uploaded, NLP task is queued. Fetch reports to update list and start polling if needed.
    fetchReports();
  };

  const handleReportDeleted = (deletedReportId) => {
    setReports(prevReports => prevReports.filter(report => report.id !== deletedReportId));
  };

  if (loading && reports.length === 0) return <p>Loading reports...</p>; // Show loading only on initial load and if no reports yet

  return (
    <div>
      <h2>ESG Dashboard</h2>
      {error && <p style={{ color: 'red', textAlign: 'center', padding: '10px', border: '1px solid red', borderRadius: '4px' }}>{error}</p>}
      <ReportUpload onUploadSuccess={handleUploadSuccess} />
      <ReportList reports={reports} onReportDeleted={handleReportDeleted} />
    </div>
  );
};

export default DashboardPage;