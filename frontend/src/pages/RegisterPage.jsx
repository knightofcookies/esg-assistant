import React, { useState } from 'react';
import { useNavigate, Link as RouterLink } from 'react-router-dom';
import { registerUser } from '../services/api';
import {
  Container, Paper, Typography, TextField, Button, Box, Link, Alert, CircularProgress
} from '@mui/material';
import PersonAddOutlinedIcon from '@mui/icons-material/PersonAddOutlined';
import Avatar from '@mui/material/Avatar';

const RegisterPage = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate(); // Keep for potential future use

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');
    if (password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }
    setLoading(true);
    try {
      await registerUser({ username, email, password });
      setSuccess('Registration successful! You can now log in.');
      // navigate('/login'); // Optionally navigate to login directly
    } catch (err) {
      setError(err.response?.data?.detail || 'Registration failed. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container component="section" maxWidth="xs" sx={{ mt: 8 }}>
      <Paper elevation={3} sx={{ padding: 4, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <Avatar sx={{ m: 1, bgcolor: 'secondary.main' }}>
          <PersonAddOutlinedIcon />
        </Avatar>
        <Typography component="h1" variant="h5">
          Register
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
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            disabled={loading}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            label="Email Address"
            name="email"
            autoComplete="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={loading}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            label="Password (min 8 characters)"
            type="password"
            id="password"
            autoComplete="new-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            inputProps={{ minLength: 8 }}
            disabled={loading}
          />
          {error && (
            <Alert severity="error" sx={{ mt: 2, width: '100%' }}>{error}</Alert>
          )}
          {success && (
            <Alert severity="success" sx={{ mt: 2, width: '100%' }}>
              {success} <Link component={RouterLink} to="/login" sx={{ fontWeight: 'bold' }}>Login here</Link>
            </Alert>
          )}
          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{ mt: 3, mb: 2 }}
            disabled={loading || !!success} // Disable if successful to prevent re-submission
          >
            {loading ? <CircularProgress size={24} color="inherit" /> : 'Register'}
          </Button>
          <Box textAlign="right">
            <Link component={RouterLink} to="/login" variant="body2">
              Already have an account? Login here
            </Link>
          </Box>
        </Box>
      </Paper>
    </Container>
  );
};

export default RegisterPage;