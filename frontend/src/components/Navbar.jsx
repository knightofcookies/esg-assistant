import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import useAuth from '../hooks/useAuth';

const Navbar = () => {
  const { authToken, currentUser, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav>
      <Link to="/">ESG Scorer</Link>
      <div>
        {authToken ? (
          <>
            <span>Welcome, {currentUser?.username || 'User'}!</span>
            <Link to="/dashboard">Dashboard</Link>
            <button onClick={handleLogout} style={{ marginLeft: '1rem' }}>Logout</button>
          </>
        ) : (
          <>
            <Link to="/login">Login</Link>
            <Link to="/register">Register</Link>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;