import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor to add the auth token to requests
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('authToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => {
  return Promise.reject(error);
});

export const registerUser = (userData) => {
  return apiClient.post('/users/', userData);
};

export const loginUser = (credentials) => {
  // FastAPI token endpoint expects form data
  const formData = new URLSearchParams();
  formData.append('username', credentials.username);
  formData.append('password', credentials.password);
  return apiClient.post('/token', formData, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });
};

export const getCurrentUser = () => {
  return apiClient.get('/users/me');
};

export const getEsgTopics = () => {
  return apiClient.get('/esg_topics/');
};

export const uploadReport = (formData) => {
  // company_name and file are part of formData
  return apiClient.post('/reports/', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};

export const getReports = () => {
  return apiClient.get('/reports/');
};

export const getReportDetails = (reportId) => {
  return apiClient.get(`/reports/${reportId}/`);
};

export const annotateTopic = (reportId, topicId, annotationData) => {
  return apiClient.post(`/reports/${reportId}/topics/${topic_id}/annotate/`, annotationData);
};

export const getReportScore = (reportId) => {
  return apiClient.get(`/reports/${reportId}/score/`);
};

export const getNlpSuggestions = (reportId, topicId, threshold) => {
  return apiClient.get(`/reports/${reportId}/topics/${topic_id}/suggestions/?threshold=${threshold}`);
};

// Function to get the original PDF for a report
export const getReportPdfUrl = (reportId) => {
    // This assumes your backend has an endpoint to serve the PDF.
    // Let's say your backend serves it from /reports/{report_id}/original_pdf/
    // The backend code you provided doesn't have this endpoint yet. You'd need to add it.
    // For now, we'll construct a URL, but it won't work until the backend supports it.
    // A placeholder endpoint in the backend would be:
    // @app.get("/reports/{report_id}/original_pdf/")
    // async def serve_original_pdf(report_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_active_user)):
    //     report = db.query(Report).filter(Report.id == report_id, Report.user_id == current_user.id).first()
    //     if not report or not report.original_filepath:
    //         raise HTTPException(status_code=404, detail="Report or PDF file not found")
    //     if not os.path.exists(report.original_filepath):
    //         raise HTTPException(status_code=404, detail="PDF file not found on server")
    //     return FileResponse(report.original_filepath, media_type='application/pdf', filename=report.filename)
    // This is a simplified example. For now, this function helps construct the URL.
    // If your backend serves static files from a specific path, adjust accordingly.
    // Assuming the backend is updated to serve the PDF via a direct link:
    return `${API_BASE_URL}/reports/${reportId}/original_pdf/`;
};


export default apiClient;