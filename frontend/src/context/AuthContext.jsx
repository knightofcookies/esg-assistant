import React, { createContext, useState, useEffect } from 'react';
import { getCurrentUser } from '../services/api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [authToken, setAuthToken] = useState(localStorage.getItem('authToken'));
  const [currentUser, setCurrentUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUser = async () => {
      if (authToken) {
        try {
          const response = await getCurrentUser();
          setCurrentUser(response.data);
        } catch (error) {
          console.error('Failed to fetch user, clearing token', error);
          localStorage.removeItem('authToken');
          setAuthToken(null);
          setCurrentUser(null);
        }
      }
      setLoading(false);
    };
    fetchUser();
  }, [authToken]);

  // Handle OAuth callback with token from URL
  useEffect(() => {
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get('token');
    
    if (token && window.location.pathname === '/auth/success') {
      localStorage.setItem('authToken', token);
      setAuthToken(token);
      // Clear token from URL and redirect to dashboard
      window.history.replaceState({}, document.title, '/dashboard');
    }
  }, []);

  const login = (token, user = null) => {
    localStorage.setItem('authToken', token);
    setAuthToken(token);
    if (user) {
      setCurrentUser(user);
    }
    // User will be fetched by useEffect if not provided
  };

  const logout = () => {
    localStorage.removeItem('authToken');
    setAuthToken(null);
    setCurrentUser(null);
  };

  return (
    <AuthContext.Provider value={{ authToken, currentUser, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;