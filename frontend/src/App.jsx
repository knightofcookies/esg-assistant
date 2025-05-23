import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import ReportScoringPage from './pages/ReportScoringPage';
import useAuth from './hooks/useAuth';

function ProtectedRoute({ children }) {
  const { authToken, loading } = useAuth();
  if (loading) return <div>Loading...</div>; // Or a spinner
  return authToken ? children : <Navigate to="/login" replace />;
}

function App() {
  return (
    <>
      <Navbar />
      <div className="container">
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route 
            path="/dashboard" 
            element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} 
          />
          <Route 
            path="/report/:reportId" 
            element={<ProtectedRoute><ReportScoringPage /></ProtectedRoute>} 
          />
          <Route 
            path="/" 
            element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} // Default to dashboard if logged in
          />
           {/* Fallback for non-logged in users visiting root */}
          <Route path="/" element={<Navigate to="/login" />} />
        </Routes>
      </div>
    </>
  );
}

export default App;