import React, { createContext, useContext, useState, useEffect } from 'react';
import { createTheme } from '@mui/material/styles';

const ThemeContext = createContext();

export const useThemeMode = () => {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useThemeMode must be used within a ThemeProvider');
  }
  return context;
};

export const ThemeProvider = ({ children }) => {
  const [darkMode, setDarkMode] = useState(() => {
    // Check localStorage for saved preference, default to false
    const saved = localStorage.getItem('darkMode');
    return saved ? JSON.parse(saved) : false;
  });

  // Save preference to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(prev => !prev);
  };

  const theme = createTheme({
    palette: {
      mode: darkMode ? 'dark' : 'light',
      primary: {
        main: '#1976d2',
      },
      secondary: {
        main: '#dc004e',
      },
      ...(darkMode
        ? {
            // Dark mode specific colors
            background: {
              default: '#121212',
              paper: '#1e1e1e',
            },
            text: {
              primary: '#ffffff',
              secondary: '#b3b3b3',
            },
          }
        : {
            // Light mode specific colors
            background: {
              default: '#fafafa',
              paper: '#ffffff',
            },
            text: {
              primary: '#000000',
              secondary: '#666666',
            },
          }),
    },
    components: {
      // Customize component styles for better dark mode support
      MuiPaper: {
        styleOverrides: {
          root: {
            ...(darkMode && {
              backgroundImage: 'none',
            }),
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            ...(darkMode && {
              backgroundColor: '#1e1e1e',
            }),
          },
        },
      },
    },
  });

  return (
    <ThemeContext.Provider value={{ darkMode, toggleDarkMode, theme }}>
      {children}
    </ThemeContext.Provider>
  );
};