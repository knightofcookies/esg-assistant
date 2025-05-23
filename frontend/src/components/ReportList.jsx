import React from 'react';
import { Link } from 'react-router-dom';

const ReportList = ({ reports }) => {
  if (!reports || reports.length === 0) {
    return <p>No reports uploaded yet.</p>;
  }

  return (
    <div className="card">
      <h3>Uploaded Reports</h3>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {reports.map((report) => (
          <li key={report.id} style={{ padding: '10px', borderBottom: '1px solid #eee' }}>
            <Link to={`/report/${report.id}`}>
              <strong>{report.filename}</strong>
            </Link>
            <p>Company: {report.company_name || 'N/A'}</p>
            <p>Uploaded: {new Date(report.upload_timestamp).toLocaleString()}</p>
            <p>Status: <span style={{ fontWeight: 'bold', color: report.status === 'processed' ? 'green' : 'orange'}}>{report.status}</span></p>
            <p>Score: {report.final_score !== null ? report.final_score : 'N/A'} / {report.annotations?.length || 'N/A'}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ReportList;