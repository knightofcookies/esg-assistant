import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import ReportScoringPage from './pages/ReportScoringPage';
import useAuth from './hooks/useAuth';

import { CssBaseline, Container, CircularProgress, Box } from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';

// Optional: Create a basic theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', // Example primary color
    },
    secondary: {
      main: '#dc004e', // Example secondary color
    },
  },
});

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

function App() {
  return (
    <ThemeProvider theme={theme}>
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
    </ThemeProvider>
  );
}

export default App;