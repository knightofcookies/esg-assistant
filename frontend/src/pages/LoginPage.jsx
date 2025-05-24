import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getGoogleAuthUrl } from '../services/api';
import useAuth from '../hooks/useAuth';
import {
  Container, 
  Paper, 
  Typography, 
  Button, 
  Box, 
  Alert, 
  CircularProgress,
  Divider
} from '@mui/material';
import GoogleIcon from '@mui/icons-material/Google';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Avatar from '@mui/material/Avatar';

const LoginPage = () => {
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { authToken } = useAuth();

  // Redirect if already authenticated
  useEffect(() => {
    if (authToken) {
      navigate('/dashboard');
    }
  }, [authToken, navigate]);

  const handleGoogleLogin = async () => {
    setError('');
    setLoading(true);
    try {
      const response = await getGoogleAuthUrl();
      const authUrl = response.data.auth_url;
      // Redirect to Google OAuth
      window.location.href = authUrl;
    } catch (err) {
      setError('Failed to initiate Google login. Please try again.');
      console.error(err);
      setLoading(false);
    }
  };

  return (
    <Container component="section" maxWidth="xs" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ padding: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5" gutterBottom>
          Sign in to ESG Scorer
        </Typography>
        
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3, textAlign: 'center' }}>
          Use your Google account to access the ESG Scoring Assistant
        </Typography>

        <Box sx={{ mt: 1, width: '100%' }}>
          {error && (
            <Alert severity="error" sx={{ mb: 2, width: '100%' }}>
              {error}
            </Alert>
          )}
          
          <Button
            fullWidth
            variant="outlined"
            startIcon={loading ? <CircularProgress size={20} /> : <GoogleIcon />}
            onClick={handleGoogleLogin}
            disabled={loading}
            sx={{ 
              mt: 2, 
              mb: 2, 
              py: 1.5,
              textTransform: 'none',
              fontSize: '1rem'
            }}
          >
            {loading ? 'Redirecting...' : 'Continue with Google'}
          </Button>

          <Divider sx={{ my: 2 }} />

          <Typography variant="caption" color="text.secondary" sx={{ display: 'block', textAlign: 'center' }}>
            By signing in, you agree to our terms of service and privacy policy.
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default LoginPage;