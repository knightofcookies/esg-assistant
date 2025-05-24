import React from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import useAuth from '../hooks/useAuth';
import { useThemeMode } from '../context/ThemeContext';
import { AppBar, Toolbar, Typography, Button, Box, IconButton, Tooltip } from '@mui/material';
import BarChartIcon from '@mui/icons-material/BarChart';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';

const Navbar = () => {
  const { authToken, currentUser, logout } = useAuth();
  const { darkMode, toggleDarkMode } = useThemeMode();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <AppBar position="static">
      <Toolbar sx={{ px: { xs: 1, sm: 3 } }}>
        <BarChartIcon sx={{ mr: 1 }} />
        <Typography 
          variant="h6" 
          component="div" 
          sx={{ 
            flexGrow: 1,
            fontSize: { xs: '1rem', sm: '1.25rem' }
          }}
        >
          <RouterLink to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            ESG Scorer
          </RouterLink>
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <Tooltip title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}>
            <IconButton color="inherit" onClick={toggleDarkMode} sx={{ mr: 1 }}>
              {darkMode ? <Brightness7Icon /> : <Brightness4Icon />}
            </IconButton>
          </Tooltip>
          {authToken ? (
            <>
              <Typography 
                variant="subtitle1" 
                component="span" 
                sx={{ 
                  mr: 2,
                  display: { xs: 'none', sm: 'inline' },
                  fontSize: { sm: '0.9rem', md: '1rem' }
                }}
              >
                Welcome, {currentUser?.username || 'User'}!
              </Typography>
              <Button 
                color="inherit" 
                component={RouterLink} 
                to="/dashboard"
                size="small"
                sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}
              >
                Dashboard
              </Button>
              <Button 
                color="inherit" 
                onClick={handleLogout} 
                sx={{ 
                  ml: 1,
                  fontSize: { xs: '0.8rem', sm: '0.875rem' }
                }}
                size="small"
              >
                Logout
              </Button>
            </>
          ) : (
            <>
              <Button 
                color="inherit" 
                component={RouterLink} 
                to="/login"
                size="small"
                sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}
              >
                Login
              </Button>
              <Button 
                color="inherit" 
                component={RouterLink} 
                to="/register"
                size="small"
                sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}
              >
                Register
              </Button>
            </>
          )}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;