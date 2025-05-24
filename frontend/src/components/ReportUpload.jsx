import React, { useState, useRef } from 'react';
import { uploadReport } from '../services/api';
import { Box, Typography, TextField, Button, Alert, CircularProgress, Stack } from '@mui/material';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const ReportUpload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [companyName, setCompanyName] = useState('');
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef(null); // To reset file input

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
        if (selectedFile.type !== "application/pdf") {
            setError("Please select a PDF file.");
            setFile(null);
            if(fileInputRef.current) fileInputRef.current.value = ""; // Clear the input
            return;
        }
        setError(''); // Clear previous error if any
        setFile(selectedFile);
    } else {
        setFile(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a PDF file.');
      return;
    }
    setError('');
    setUploading(true);

    const formData = new FormData();
    formData.append('file', file);
    if (companyName) {
      formData.append('company_name', companyName);
    }

    try {
      const response = await uploadReport(formData);
      onUploadSuccess(response.data);
      setFile(null);
      setCompanyName('');
      if (fileInputRef.current) {
        fileInputRef.current.value = ""; // Reset file input more reliably
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed.');
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" component="h3" gutterBottom>
        Upload New Report
      </Typography>
      <Stack spacing={2} component="form" onSubmit={handleSubmit}>
          <TextField
            fullWidth
            label="Company Name (Optional)"
            id="companyName"
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
            disabled={uploading}
            variant="outlined"
            size="small"
          />
          <TextField
            fullWidth
            type="file"
            id="file"
            InputLabelProps={{ shrink: true }}
            inputProps={{ accept: ".pdf" }}
            onChange={handleFileChange}
            required
            disabled={uploading}
            variant="outlined"
            size="small"
            inputRef={fileInputRef}
            helperText={file ? file.name : "Select a PDF file"}
          />
          {error && <Alert severity="error">{error}</Alert>}
          <Button
            type="submit"
            variant="contained"
            startIcon={uploading ? <CircularProgress size={20} color="inherit" /> : <CloudUploadIcon />}
            disabled={uploading || !file}
            fullWidth
            sx={{ mt: 2 }}
          >
            {uploading ? 'Uploading...' : 'Upload Report'}
          </Button>
        </Stack>
    </Box>
  );
};

export default ReportUpload;