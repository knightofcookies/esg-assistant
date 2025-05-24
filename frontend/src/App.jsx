import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import ReportScoringPage from './pages/ReportScoringPage';
import useAuth from './hooks/useAuth';
import { ThemeProvider, useThemeMode } from './context/ThemeContext';

import { CssBaseline, Container, CircularProgress, Box } from '@mui/material';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';

function ProtectedRoute({ children }) {
  const { authToken, loading } = useAuth();
  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }
  return authToken ? children : <Navigate to="/login" replace />;
}

function AppContent() {
  const { theme } = useThemeMode();
  
  return (
    <MuiThemeProvider theme={theme}>
      <CssBaseline />
      <Navbar />
      <Routes>
        <Route path="/login" element={
          <Container component="main" sx={{ mt: 4, mb: 1 }}>
            <LoginPage />
          </Container>
        } />
        <Route path="/register" element={
          <Container component="main" sx={{ mt: 4, mb: 1 }}>
            <RegisterPage />
          </Container>
        } />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Container component="main" sx={{ mt: 4, mb: 1 }}>
                <DashboardPage />
              </Container>
            </ProtectedRoute>
          }
        />
        <Route
          path="/report/:reportId"
          element={
            <ProtectedRoute>
              <ReportScoringPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/"
          element={<ProtectedRoute><Navigate to="/dashboard" /></ProtectedRoute>}
        />
      </Routes>
    </MuiThemeProvider>
  );
}

function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}

export default App;