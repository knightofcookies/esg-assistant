import React, { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import { loginUser } from '../services/api';
import useAuth from '../hooks/useAuth';
import {
  Container, Paper, Typography, TextField, Button, Box, Link, Alert, CircularProgress
} from '@mui/material';
import LockOutlinedIcon from '@mui/icons-material/LockOutlined';
import Avatar from '@mui/material/Avatar';

const LoginPage = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const response = await loginUser({ username, password });
      login(response.data.access_token);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed. Please check your credentials.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container component="section" maxWidth="xs" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ padding: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
          <LockOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Login
        </Typography>
        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 1, width: '100%' }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="username"
            label="Username"
            name="username"
            autoComplete="username"
            autoFocus
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password"
            type="password"
            id="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={loading}
          />
          {error && (
            <Alert severity="error" sx={{ mt: 2, width: '100%' }}>
              {error}
            </Alert>
          )}
          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ mt: 3, mb: 2 }}
            disabled={loading}
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Login'}
          </Button>
          <Box textAlign="right">
            <Link component={RouterLink} to="/register" variant="body2">
              Don't have an account? Register here
            </Link>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
};

export default LoginPage;