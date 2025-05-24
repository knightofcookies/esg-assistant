import React, { useState, useEffect, useCallback, useRef } from 'react';
import { useParams } from 'react-router-dom';
import {
  getReportDetails,
  getEsgTopics,
  getNlpSuggestions,
  annotateTopic,
  fetchPdfBlob,
  getReportScore,
} from '../services/api';
import PDFViewer from '../components/PDFViewer';
import TopicList from '../components/TopicList';
import SuggestionList from '../components/SuggestionList';

// Interface for active highlight (matches what PDFViewer will expect)
// interface ActiveHighlight {
//   pageNumber: number;
//   coordinates: { x0: number; y0: number; x1: number; y1: number; };
// }

const ReportScoringPage = () => {
  const { reportId } = useParams();
  const [report, setReport] = useState(null);
  const [esgTopics, setEsgTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [pdfObjectUrl, setPdfObjectUrl] = useState('');
  const [currentPdfPage, setCurrentPdfPage] = useState(1); // Page user is viewing
  const [pdfTotalPages, setPdfTotalPages] = useState(null);
  
  // State for the active highlight
  const [activeHighlight, setActiveHighlight] = useState(null); // Stores { pageNumber, coordinates }

  const [probabilityThreshold, setProbabilityThreshold] = useState(0.7);
  const [topK, setTopK] = useState(20);
  const [scoreDetails, setScoreDetails] = useState(null);

  const [loadingReport, setLoadingReport] = useState(true);
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [error, setError] = useState('');

  const objectUrlRef = useRef(pdfObjectUrl);

  useEffect(() => {
    objectUrlRef.current = pdfObjectUrl;
  }, [pdfObjectUrl]);

  const fetchReportAndScore = useCallback(async () => {
    if (!reportId) return;
    setLoadingReport(true);
    setError('');
    try {
      const reportRes = await getReportDetails(reportId);
      setReport(reportRes.data);

      const pdfBlobRes = await fetchPdfBlob(reportId);
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const newObjectUrl = URL.createObjectURL(pdfBlobRes.data);
      setPdfObjectUrl(newObjectUrl);

      const scoreRes = await getReportScore(reportId);
      setScoreDetails(scoreRes.data);

    } catch (err) {
      setError('Failed to load report details or PDF.');
      console.error(err);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      setPdfObjectUrl('');
      setReport(null);
      setScoreDetails(null);
    } finally {
      setLoadingReport(false);
    }
  }, [reportId]);

  const fetchTopicsData = useCallback(async () => {
    setLoadingTopics(true);
    try {
      const topicsRes = await getEsgTopics();
      setEsgTopics(topicsRes.data);
    } catch (err) {
      setError(prev => prev + '\nFailed to load ESG topics.');
      console.error(err);
    } finally {
      setLoadingTopics(false);
    }
  }, []);

  useEffect(() => {
    fetchReportAndScore();
    fetchTopicsData();
    return () => {
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, [reportId, fetchReportAndScore, fetchTopicsData]);

  const handleSelectTopic = (topic) => {
    setSelectedTopic(topic);
    setSuggestions([]);
    setActiveHighlight(null); // Clear highlight when topic changes
  };

  const fetchSuggestionsForTopic = useCallback(async () => {
    if (!selectedTopic || !report || report.status !== 'processed') {
      setSuggestions([]);
      setActiveHighlight(null);
      return;
    }
    setLoadingSuggestions(true);
    setError('');
    try {
      const suggestionsRes = await getNlpSuggestions(reportId, selectedTopic.id, probabilityThreshold, topK);
      setSuggestions(suggestionsRes.data);
    } catch (err) {
      setError(`Failed to load suggestions for ${selectedTopic.name}. ${err.response?.data?.detail || err.message}`);
      console.error(err);
      setSuggestions([]);
    } finally {
      setLoadingSuggestions(false);
    }
  }, [reportId, selectedTopic, probabilityThreshold, topK, report]);

  useEffect(() => {
    if (selectedTopic && report?.status === 'processed') {
      fetchSuggestionsForTopic();
    } else if (report && report.status !== 'processed') {
        setSuggestions([]); // Clear suggestions if report not processed
        setActiveHighlight(null);
    }
  }, [selectedTopic, probabilityThreshold, topK, report?.status, fetchSuggestionsForTopic]);

  const handleAnnotationUpdate = async (statusToSet) => {
    if (!selectedTopic) return;
    setError('');
    try {
      const updatedAnnotation = await annotateTopic(reportId, selectedTopic.id, { status: statusToSet });
      setReport(prevReport => {
        if (!prevReport) return null;
        const newAnnotations = prevReport.annotations.map(ann =>
          ann.topic.id === selectedTopic.id
            ? { ...ann, status: updatedAnnotation.data.status, timestamp: updatedAnnotation.data.timestamp }
            : ann
        );
        return { ...prevReport, annotations: newAnnotations };
      });
      const scoreRes = await getReportScore(reportId);
      setScoreDetails(scoreRes.data);
    } catch (err) {
      setError(`Failed to save annotation: ${err.response?.data?.detail || err.message}`);
      console.error(err);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    if (suggestion.page_number) {
      setCurrentPdfPage(suggestion.page_number); // Navigate PDF to the suggestion's page
    }
    if (suggestion.coordinates) {
      setActiveHighlight({
        pageNumber: suggestion.page_number,
        coordinates: suggestion.coordinates, // These are PDF points {x0, y0, x1, y1}
        id: suggestion.chunk_id // Unique ID for the highlight
      });
      console.log("Set active highlight for suggestion:", suggestion);
    } else {
      setActiveHighlight(null); // Clear if no coordinates
      console.log("No coordinates for suggestion:", suggestion);
    }
  };

  const handlePdfDocumentLoad = (numPages) => {
    setPdfTotalPages(numPages);
    if (currentPdfPage > numPages) setCurrentPdfPage(1); // Reset if out of bounds
  };

  // Callback for PDFViewer to inform parent about page changes
  const handlePageChangeByViewer = (newPage) => {
    setCurrentPdfPage(newPage);
    // setActiveHighlight(null); // Optionally clear highlight when user manually changes page
  };


  if (loadingReport || loadingTopics) return <p>Loading scoring page data...</p>;
  if (error && !report && !loadingReport) return <p style={{ color: 'red' }}>{error}</p>;
  if (!report) return <p>Report not found or failed to load.</p>;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(300px, 1fr) 2.5fr minmax(300px, 1fr)', gap: '20px', alignItems: 'flex-start', maxHeight: 'calc(100vh - 100px)', overflow: 'hidden' }}>
      <div style={{overflowY: 'auto', height: 'calc(100vh - 120px)'}}>
        <TopicList
            topics={esgTopics}
            currentReport={report}
            onSelectTopic={handleSelectTopic}
            activeTopicId={selectedTopic?.id}
        />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px', overflowY: 'auto', height: 'calc(100vh - 120px)' }}>
        <div className="card">
            <h3>Scoring: {report.filename}</h3>
            <p>Report Overall NLP Status: <span style={{fontWeight: 'bold'}}>{report.status}</span>
                {(report.status === 'nlp_queued' || report.status === 'nlp_processing') && (
                    <span style={{ fontStyle: 'italic' }}> (Processing: {report.nlp_progress || 0}%)</span>
                )}
            </p>
            {scoreDetails && ( /* Display richer score */
                <div>
                    <p>Score: <strong>{scoreDetails.score} / {scoreDetails.max_score} ({scoreDetails.percentage}%)</strong></p>
                    <p>Auditor Status: Answered: {scoreDetails.answered_count} | Unanswered: {scoreDetails.unanswered_count} | Pending: {scoreDetails.pending_count}</p>
                </div>
            )}
            {error && !loadingSuggestions && <p style={{ color: 'red', fontSize: '0.9em' }}>{error}</p>}
            {selectedTopic && (
                <div style={{marginTop: '15px'}}>
                    <h4>Selected Topic: {selectedTopic.topic_number}. {selectedTopic.name}</h4>
                    <p style={{fontSize: '0.9em', color: '#555'}}>{selectedTopic.description}</p>
                    <div style={{ margin: '10px 0' }}>
                        <button onClick={() => handleAnnotationUpdate('answered')} style={{ marginRight: '10px', backgroundColor: 'green' }}>Mark Answered</button>
                        <button onClick={() => handleAnnotationUpdate('unanswered')} style={{ backgroundColor: 'red', marginRight: '10px' }}>Mark Unanswered</button>
                        <button onClick={() => handleAnnotationUpdate('pending')} style={{ backgroundColor: 'grey' }}>Mark Pending</button>
                    </div>
                    <hr/>
                    <div style={{display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap', marginTop: '10px'}}>
                        <label htmlFor="threshold" style={{whiteSpace: 'nowrap'}}>Threshold: </label>
                        <input type="range" id="threshold" min="0.1" max="1.0" step="0.05" value={probabilityThreshold} onChange={(e) => setProbabilityThreshold(parseFloat(e.target.value))} style={{flexGrow: 1, minWidth: '100px'}}/>
                        <span> {(probabilityThreshold * 100).toFixed(0)}%</span>
                        <label htmlFor="topK" style={{whiteSpace: 'nowrap', marginLeft: '10px'}}>Top-K:</label>
                        <input type="number" id="topK" min="5" max="50" step="5" value={topK} onChange={(e) => setTopK(parseInt(e.target.value))} style={{width: '60px'}}/>
                        <button onClick={fetchSuggestionsForTopic} disabled={loadingSuggestions || !selectedTopic || report.status !== 'processed'} style={{marginLeft: 'auto'}}>
                            {loadingSuggestions ? 'Fetching...' : 'Refresh Suggestions'}
                        </button>
                    </div>
                     {report.status !== 'processed' && selectedTopic && <p style={{color: 'orange', fontSize: '0.9em', marginTop: '10px'}}>Report status: {report.status}. Suggestions will be available once 'processed'.</p>}
                </div>
            )}
        </div>
        <SuggestionList suggestions={suggestions} isLoading={loadingSuggestions} onSuggestionClick={handleSuggestionClick} />
      </div>

      <div style={{overflowY: 'hidden', height: 'calc(100vh - 120px)'}}>
        <PDFViewer
            pdfUrl={pdfObjectUrl}
            pageNumberToDisplay={currentPdfPage} // Renamed prop for clarity
            onDocumentLoadSuccess={handlePdfDocumentLoad}
            onPageChange={handlePageChangeByViewer} // Pass callback
            activeHighlight={activeHighlight} // Pass the active highlight object
        />
      </div>
    </div>
  );
};

export default ReportScoringPage;