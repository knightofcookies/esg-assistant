import React from 'react';
import { Link } from 'react-router-dom';
import { deleteReport } from '../services/api';

const ReportList = ({ reports, onReportDeleted }) => {
  if (!reports || reports.length === 0) {
    return <p>No reports uploaded yet.</p>;
  }

  const handleDelete = async (reportId, reportFilename) => {
    // ... (keep existing delete logic) ...
    if (window.confirm(`Are you sure you want to delete report: ${reportFilename}? ...`)) {
        try {
            await deleteReport(reportId);
            onReportDeleted(reportId);
        } catch (error) {
            console.error("Failed to delete report:", error);
            alert(`Failed to delete report: ${error.response?.data?.detail || error.message}`);
        }
    }
  };

  return (
    <div className="card">
      <h3>Uploaded Reports</h3>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        {reports.map((report) => (
          <li key={report.id} style={{ padding: '10px', borderBottom: '1px solid #eee', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div>
              <Link to={`/report/${report.id}`}>
                <strong>{report.filename}</strong>
              </Link>
              <p>Company: {report.company_name || 'N/A'}</p>
              <p>Uploaded: {new Date(report.upload_timestamp).toLocaleString()}</p>
              <p>
                Status: <span style={{ fontWeight: 'bold', color: report.status === 'processed' ? 'green' : (report.status === 'nlp_failed' ? 'red' : 'orange')}}>{report.status}</span>
                {/* Display NLP Progress */}
                {(report.status === 'nlp_queued' || report.status === 'nlp_processing') && (
                  <span style={{ marginLeft: '10px', fontStyle: 'italic' }}>
                    (Processing: {report.nlp_progress || 0}%)
                  </span>
                )}
              </p>
              <p>Score: {report.final_score !== null ? report.final_score : 'N/A'} / {report.annotations?.length || 69}</p>
            </div>
            <button
              onClick={() => handleDelete(report.id, report.filename)}
              style={{ backgroundColor: '#dc3545', color: 'white', padding: '5px 10px', border: 'none', borderRadius: '4px', cursor: 'pointer', alignSelf: 'flex-start' }}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default ReportList;