import React, { useState, useEffect, useCallback, useRef } from 'react';
import ReportUpload from '../components/ReportUpload';
import ReportList from '../components/ReportList';
import { getReports } from '../services/api';
import { Box, Typography, Alert, CircularProgress, Paper } from '@mui/material';

const POLLING_INTERVAL = 5000; // 5 seconds

const DashboardPage = () => {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const pollingIntervalRef = useRef(null);

  const fetchReports = useCallback(async (isPolling = false) => {
    if (!isPolling) setLoading(true);
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
    fetchReports();
  }, [fetchReports]);

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
      if (!pollingIntervalRef.current) {
        pollingIntervalRef.current = setInterval(() => fetchReports(true), POLLING_INTERVAL);
      }
    } else {
      clearPolling();
    }

    return () => clearPolling();
  }, [reports, fetchReports]);

  const handleUploadSuccess = (newReport) => {
    fetchReports();
  };

  const handleReportDeleted = (deletedReportId) => {
    setReports(prevReports => prevReports.filter(report => report.id !== deletedReportId));
  };

  if (loading && reports.length === 0) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="50vh">
        <CircularProgress />
        <Typography variant="h6" sx={{ ml: 2 }}>Loading reports...</Typography>
      </Box>
    );
  }

  return (
    <Box sx={{ my: 2 }}>
      <Typography variant="h4" component="h1" gutterBottom>
        ESG Dashboard
      </Typography>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <ReportUpload onUploadSuccess={handleUploadSuccess} />
      </Paper>
      <Paper elevation={2} sx={{ p: 3 }}>
        <ReportList reports={reports} onReportDeleted={handleReportDeleted} />
      </Paper>
    </Box>
  );
};

export default DashboardPage;