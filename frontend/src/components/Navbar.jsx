import React from 'react';
import { Link as RouterLink, useNavigate } from 'react-router-dom';
import useAuth from '../hooks/useAuth';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';
import BarChartIcon from '@mui/icons-material/BarChart'; // Example Icon

const Navbar = () => {
  const { authToken, currentUser, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <BarChartIcon sx={{ mr: 1 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          <RouterLink to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
            ESG Scorer
          </RouterLink>
        </Typography>
        <Box>
          {authToken ? (
            <>
              <Typography variant="subtitle1" component="span" sx={{ mr: 2 }}>
                Welcome, {currentUser?.username || 'User'}!
              </Typography>
              <Button color="inherit" component={RouterLink} to="/dashboard">
                Dashboard
              </Button>
              <Button color="inherit" onClick={handleLogout} sx={{ ml: 1 }}>
                Logout
              </Button>
            </>
          ) : (
            <>
              <Button color="inherit" component={RouterLink} to="/login">
                Login
              </Button>
              <Button color="inherit" component={RouterLink} to="/register">
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