import React, { useState, useEffect, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import {
  getReportDetails,
  getEsgTopics,
  getNlpSuggestions,
  annotateTopic,
  getReportPdfUrl,
} from '../services/api';
import PDFViewer from '../components/PDFViewer';
import TopicList from '../components/TopicList';
import SuggestionList from '../components/SuggestionList';
import useAuth from '../hooks/useAuth'; // For token to construct PDF URL if needed

const ReportScoringPage = () => {
  const { reportId } = useParams();
  const { authToken } = useAuth(); // Get auth token
  const [report, setReport] = useState(null);
  const [esgTopics, setEsgTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [pdfUrl, setPdfUrl] = useState('');
  const [pdfPageNumber, setPdfPageNumber] = useState(1);


  const [probabilityThreshold, setProbabilityThreshold] = useState(0.7); // Default 70%
  const [loadingReport, setLoadingReport] = useState(true);
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [error, setError] = useState('');

  const fetchReportData = useCallback(async () => {
    setLoadingReport(true);
    try {
      const reportRes = await getReportDetails(reportId);
      setReport(reportRes.data);
      // Construct PDF URL - this requires the backend to serve the PDF and include the token in the request
      // The getReportPdfUrl function in api.js already appends the token via interceptor if fetching via apiClient
      // If it's a direct link, the backend might need to handle auth differently for file serving or use presigned URLs.
      // For now, let's assume the interceptor handles the token for the PDF request if it goes through apiClient.
      // If getReportPdfUrl just returns a string URL, the browser's fetch for the PDF won't have the auth token unless the backend allows it or the endpoint is public (not ideal).
      // For a secure setup, the backend should serve the PDF via an authenticated endpoint.
      // The api.js getReportPdfUrl has been updated to be a string, but the backend needs an endpoint.
      // If your backend endpoint for PDF requires auth, and you use a direct <Document file={pdfUrl}>,
      // you might need to fetch the PDF as a blob and then create an object URL, or ensure cookies handle auth.
      // A simpler way: The FileResponse in FastAPI doesn't inherently handle JWT from Authorization headers for direct browser requests.
      // One workaround is to pass the token as a query parameter if the backend supports it (less secure).
      // Or, fetch the PDF via axios which adds the header, get a blob, then use URL.createObjectURL(blob).
      // For now, the getReportPdfUrl in api.js will just give the direct URL.
      const url = getReportPdfUrl(reportId); // This is just the URL string
      setPdfUrl(url);

    } catch (err) {
      setError('Failed to load report details.');
      console.error(err);
    } finally {
      setLoadingReport(false);
    }
  }, [reportId]);

  const fetchTopics = useCallback(async () => {
    setLoadingTopics(true);
    try {
      const topicsRes = await getEsgTopics();
      setEsgTopics(topicsRes.data);
    } catch (err) {
      setError('Failed to load ESG topics.');
      console.error(err);
    } finally {
      setLoadingTopics(false);
    }
  }, []);

  useEffect(() => {
    fetchReportData();
    fetchTopics();
  }, [reportId, fetchReportData, fetchTopics]);

  const handleSelectTopic = (topic) => {
    setSelectedTopic(topic);
    setSuggestions([]); // Clear old suggestions
  };

  const fetchSuggestions = useCallback(async () => {
    if (!selectedTopic || !report || report.status !== 'processed') {
        if (report && report.status !== 'processed') {
            setSuggestions([]);
            // console.log("Suggestions can only be fetched after report is processed.");
        }
        return;
    }
    setLoadingSuggestions(true);
    try {
      const suggestionsRes = await getNlpSuggestions(reportId, selectedTopic.id, probabilityThreshold);
      setSuggestions(suggestionsRes.data);
    } catch (err) {
      setError(`Failed to load suggestions for ${selectedTopic.name}.`);
      console.error(err);
      setSuggestions([]);
    } finally {
      setLoadingSuggestions(false);
    }
  }, [reportId, selectedTopic, probabilityThreshold, report]);

  useEffect(() => {
    fetchSuggestions();
  }, [fetchSuggestions]);

  const handleAnnotation = async (status) => {
    if (!selectedTopic) return;
    try {
      const updatedAnnotation = await annotateTopic(reportId, selectedTopic.id, { status });
      // Update report state with new annotation and score
      setReport(prevReport => {
        const newAnnotations = prevReport.annotations.map(ann =>
          ann.topic.id === selectedTopic.id ? { ...ann, status: updatedAnnotation.data.status, timestamp: updatedAnnotation.data.timestamp } : ann
        );
        // Calculate new score
        const newScore = newAnnotations.filter(a => a.status === 'answered').length;
        return { ...prevReport, annotations: newAnnotations, final_score: newScore };
      });
    } catch (err) {
      setError('Failed to save annotation.');
      console.error(err);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    // When a suggestion is clicked, set the PDF page number
    if (suggestion.page_number) {
        setPdfPageNumber(suggestion.page_number);
    }
    // TODO: Implement highlighting logic if coordinates are available
    console.log("Clicked suggestion:", suggestion);
  };


  if (loadingReport || loadingTopics) return <p>Loading scoring page...</p>;
  if (error) return <p style={{ color: 'red' }}>{error}</p>;
  if (!report) return <p>Report not found.</p>;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr 1fr', gap: '20px', alignItems: 'flex-start' }}>
      <TopicList
        topics={esgTopics}
        currentReport={report}
        onSelectTopic={handleSelectTopic}
        activeTopicId={selectedTopic?.id}
      />

      <div> {/* Center column for PDF and actions */}
        <div className="card">
            <h3>Scoring: {report.filename}</h3>
            {selectedTopic && (
                <div>
                    <h4>Selected Topic: {selectedTopic.topic_number}. {selectedTopic.name}</h4>
                    <p>{selectedTopic.description}</p>
                    <p>Hypothesis: <i>"{selectedTopic.hypothesis_template}"</i></p>
                    <div>
                    <label htmlFor="threshold">Suggestion Threshold: </label>
                    <input
                        type="range"
                        id="threshold"
                        min="0.1"
                        max="1.0"
                        step="0.05"
                        value={probabilityThreshold}
                        onChange={(e) => setProbabilityThreshold(parseFloat(e.target.value))}
                    />
                    <span> {(probabilityThreshold * 100).toFixed(0)}%</span>
                    <button onClick={fetchSuggestions} disabled={loadingSuggestions || !selectedTopic || report.status !== 'processed'} style={{marginLeft: '10px'}}>
                        Refresh Suggestions
                    </button>
                    </div>
                     {report.status !== 'processed' && <p style={{color: 'orange', fontSize: '0.9em'}}>Report status: {report.status}. Suggestions will be available once 'processed'.</p>}
                    <div style={{ marginTop: '10px' }}>
                    <button onClick={() => handleAnnotation('answered')} style={{ marginRight: '10px', backgroundColor: 'green' }}>Mark as Answered</button>
                    <button onClick={() => handleAnnotation('unanswered')} style={{ backgroundColor: 'red' }}>Mark as Unanswered</button>
                    </div>
                </div>
            )}
        </div>
        <SuggestionList suggestions={suggestions} isLoading={loadingSuggestions} onSuggestionClick={handleSuggestionClick} />
      </div>

      <PDFViewer pdfUrl={pdfUrl} pageNumberProp={pdfPageNumber}/>
    </div>
  );
};

export default ReportScoringPage;