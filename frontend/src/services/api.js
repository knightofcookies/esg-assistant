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

// New function to fetch PDF as a blob
export const fetchPdfBlob = (reportId) => {
  return apiClient.get(`/reports/${reportId}/pdf`, {
    responseType: 'blob', // Crucial for file download through axios
  });
};

export const deleteReport = (reportId) => {
  return apiClient.delete(`/reports/${reportId}/`);
};

export const annotateTopic = (reportId, topicId, annotationData) => {
  console.log('🌐 API Call - annotateTopic:', {
    url: `/reports/${reportId}/topics/${topicId}/annotate/`,
    data: annotationData,
    headers: apiClient.defaults.headers
  });
  return apiClient.post(`/reports/${reportId}/topics/${topicId}/annotate/`, annotationData);
};

export const getReportScore = (reportId) => {
  // This endpoint now returns richer data
  return apiClient.get(`/reports/${reportId}/score/`);
};

export const getNlpSuggestions = (reportId, topicId, threshold, topK = 20) => {
  // Ensure the template string uses the correct parameter name 'topicId'
  return apiClient.get(`/reports/${reportId}/topics/${topicId}/suggestions/?threshold=${threshold}&top_k_semantic=${topK}`);
};

// Optional: if you need to fetch a single topic's details
export const getSingleEsgTopic = (topicId) => {
  return apiClient.get(`/esg_topics/${topicId}/`);
};

// Optional: if you need to manually update report status
export const updateReportStatus = (reportId, status) => {
    // status should be one of the literals defined in backend's ReportStatusUpdate
    return apiClient.patch(`/reports/${reportId}/status`, { status });
};


export default apiClient;