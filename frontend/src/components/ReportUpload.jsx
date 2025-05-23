import React, { useState } from 'react';
import { uploadReport } from '../services/api';

const ReportUpload = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null);
  const [companyName, setCompanyName] = useState('');
  const [error, setError] = useState('');
  const [uploading, setUploading] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
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
      onUploadSuccess(response.data); // Pass uploaded report data to parent
      setFile(null);
      setCompanyName('');
      e.target.reset(); // Reset file input
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed.');
      console.error(err);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="card">
      <h3>Upload New Report</h3>
      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="companyName">Company Name (Optional):</label>
          <input
            type="text"
            id="companyName"
            value={companyName}
            onChange={(e) => setCompanyName(e.target.value)}
          />
        </div>
        <div>
          <label htmlFor="file">Report PDF:</label>
          <input
            type="file"
            id="file"
            accept=".pdf"
            onChange={handleFileChange}
            required
          />
        </div>
        {error && <p style={{ color: 'red' }}>{error}</p>}
        <button type="submit" disabled={uploading}>
          {uploading ? 'Uploading...' : 'Upload Report'}
        </button>
      </form>
    </div>
  );
};

export default ReportUpload;