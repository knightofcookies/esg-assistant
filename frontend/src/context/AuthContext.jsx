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

  const login = (token) => {
    localStorage.setItem('authToken', token);
    setAuthToken(token);
    // User will be fetched by useEffect
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