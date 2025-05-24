import React, { useState, useEffect, useCallback, useRef } from "react";
import { useParams } from "react-router-dom";
import {
  getReportDetails,
  getEsgTopics,
  getNlpSuggestions,
  annotateTopic,
  fetchPdfBlob,
  getReportScore,
} from "../services/api";
import PDFViewer from "../components/PDFViewer"; // Assuming PDFViewer is adapted or works with MUI
import TopicList from "../components/TopicList"; // Assuming TopicList is adapted
import SuggestionList from "../components/SuggestionList"; // Assuming SuggestionList is adapted

import {
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Box,
  Button,
  Alert,
  Slider,
  TextField,
  ButtonGroup,
  Divider,
  Chip,
  Stack,
  FormControl,
  FormLabel,
  Tooltip,
} from "@mui/material";
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline";
import HighlightOffIcon from "@mui/icons-material/HighlightOff";
import HelpOutlineIcon from "@mui/icons-material/HelpOutline";
import RefreshIcon from "@mui/icons-material/Refresh";
import ScoreboardIcon from "@mui/icons-material/Scoreboard";
import InfoIcon from "@mui/icons-material/Info";

const ReportScoringPage = () => {
  const { reportId } = useParams();
  const [report, setReport] = useState(null);
  const [esgTopics, setEsgTopics] = useState([]);
  const [selectedTopic, setSelectedTopic] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [pdfObjectUrl, setPdfObjectUrl] = useState("");
  const [currentPdfPage, setCurrentPdfPage] = useState(1);
  const [pdfTotalPages, setPdfTotalPages] = useState(null);
  const [activeHighlight, setActiveHighlight] = useState(null);
  const [probabilityThreshold, setProbabilityThreshold] = useState(0.7);
  const [topK, setTopK] = useState(20);
  const [scoreDetails, setScoreDetails] = useState(null);

  const [loadingReport, setLoadingReport] = useState(true);
  const [loadingTopics, setLoadingTopics] = useState(true);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [annotationLoading, setAnnotationLoading] = useState(false);
  const [error, setError] = useState("");

  const objectUrlRef = useRef(pdfObjectUrl);

  useEffect(() => {
    objectUrlRef.current = pdfObjectUrl;
  }, [pdfObjectUrl]);

  const fetchReportAndScore = useCallback(async () => {
    if (!reportId) return;
    setLoadingReport(true);
    setError("");
    try {
      const reportRes = await getReportDetails(reportId);
      setReport(reportRes.data);

      const pdfBlobRes = await fetchPdfBlob(reportId);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      const newObjectUrl = URL.createObjectURL(pdfBlobRes.data);
      setPdfObjectUrl(newObjectUrl);

      const scoreRes = await getReportScore(reportId);
      setScoreDetails(scoreRes.data);
    } catch (err) {
      setError("Failed to load report details or PDF.");
      console.error(err);
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
      setPdfObjectUrl("");
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
      setError((prev) => `${prev}\nFailed to load ESG topics.`.trim());
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
    setActiveHighlight(null);
  };

  useEffect(() => {
    if (!selectedTopic || !report || report.status !== "processed") {
      setSuggestions([]);
      setActiveHighlight(null);
      if (report && report.status !== "processed" && selectedTopic) {
        setError(
          `Suggestions unavailable: Report status is "${report.status}". Needs to be "processed".`
        );
      }
      return;
    }

    let isCancelled = false;

    const fetchSuggestionsForTopic = async () => {
      setLoadingSuggestions(true);
      setError("");
      try {
        const suggestionsRes = await getNlpSuggestions(
          reportId,
          selectedTopic.id,
          probabilityThreshold,
          topK
        );
        console.log("Suggestions fetched:", suggestionsRes.data);

        if (!isCancelled) {
          if (Array.isArray(suggestionsRes.data)) {
            setSuggestions(suggestionsRes.data);
          } else {
            console.error(
              "Received non-array suggestions:",
              suggestionsRes.data
            );
            setSuggestions([]);
            setError(
              `Received malformed suggestions for ${selectedTopic.name}.`
            );
          }
        }
      } catch (err) {
        if (!isCancelled) {
          setError(
            `Failed to load suggestions for ${selectedTopic.name}. ${
              err.response?.data?.detail || err.message
            }`
          );
          console.error(err);
          setSuggestions([]);
        }
      } finally {
        if (!isCancelled) {
          setLoadingSuggestions(false);
        }
      }
    };

    fetchSuggestionsForTopic();

    return () => {
      isCancelled = true;
    };
  }, [selectedTopic?.id, report?.status, reportId, probabilityThreshold, topK]);

  // Create a separate function for manual refresh
  const handleRefreshSuggestions = useCallback(async () => {
    if (!selectedTopic || !report || report.status !== "processed") {
      return;
    }
    setLoadingSuggestions(true);
    setError("");
    try {
      const suggestionsRes = await getNlpSuggestions(
        reportId,
        selectedTopic.id,
        probabilityThreshold,
        topK
      );
      console.log("Suggestions refreshed:", suggestionsRes.data);
      if (Array.isArray(suggestionsRes.data)) {
        setSuggestions(suggestionsRes.data);
      } else {
        console.error("Received non-array suggestions:", suggestionsRes.data);
        setSuggestions([]);
        setError(`Received malformed suggestions for ${selectedTopic.name}.`);
      }
    } catch (err) {
      setError(
        `Failed to load suggestions for ${selectedTopic.name}. ${
          err.response?.data?.detail || err.message
        }`
      );
      console.error(err);
      setSuggestions([]);
    } finally {
      setLoadingSuggestions(false);
    }
  }, [reportId, selectedTopic?.id, probabilityThreshold, topK, report?.status]);

  const handleAnnotationUpdate = async (statusToSet) => {
    if (!selectedTopic) return;
    setError("");
    setAnnotationLoading(true);
    try {
      const updatedAnnotation = await annotateTopic(
        reportId,
        selectedTopic.id,
        { status: statusToSet }
      );
      setReport((prevReport) => {
        if (!prevReport) return null;
        const newAnnotations = prevReport.annotations.map((ann) =>
          ann.topic.id === selectedTopic.id
            ? {
                ...ann,
                status: updatedAnnotation.data.status,
                timestamp: updatedAnnotation.data.timestamp,
              }
            : ann
        );
        // Also update the selectedTopic's annotation status if it's derived from the report
        if (selectedTopic) {
          setSelectedTopic((prevTopic) => ({
            ...prevTopic,
            annotation_status: updatedAnnotation.data.status,
          }));
        }
        return { ...prevReport, annotations: newAnnotations };
      });
      const scoreRes = await getReportScore(reportId); // Re-fetch score after annotation
      setScoreDetails(scoreRes.data);
    } catch (err) {
      setError(
        `Failed to save annotation: ${
          err.response?.data?.detail || err.message
        }`
      );
      console.error(err);
    } finally {
      setAnnotationLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    console.log('🔍 Suggestion clicked:', suggestion);
    console.log('📄 Current PDF page before:', currentPdfPage);
    console.log('🎯 Active highlight before:', activeHighlight);

    if (suggestion.page_number) {
      console.log('📍 Setting page to:', suggestion.page_number);
      setCurrentPdfPage(suggestion.page_number);
    }
    
    if (suggestion.coordinates) {
      const highlight = {
        pageNumber: suggestion.page_number,
        coordinates: suggestion.coordinates,
        id: suggestion.chunk_id,
      };
      console.log('✨ Setting highlight:', highlight);
      setActiveHighlight(highlight);
    } else {
      console.log('❌ No coordinates, clearing highlight');
      setActiveHighlight(null);
    }
  };

  const handlePdfDocumentLoad = (numPages) => {
    setPdfTotalPages(numPages);
    if (currentPdfPage > numPages && numPages > 0) setCurrentPdfPage(1);
  };

  const handlePageChangeByViewer = (newPage) => setCurrentPdfPage(newPage);

  if (loadingReport || loadingTopics) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="80vh"
      >
        <CircularProgress size={50} />
        <Typography variant="h6" sx={{ ml: 2 }}>
          Loading scoring page data...
        </Typography>
      </Box>
    );
  }

  if (error && !report && !loadingReport) {
    // Only show full page error if report failed to load entirely
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }
  if (!report)
    return (
      <Typography sx={{ m: 2 }}>Report not found or failed to load.</Typography>
    );

  const reportStatusChip = (status) => {
    if (!status) return <Chip label="unknown" color="default" size="small" />; // Handle null/undefined
    let color = "default";
    let progress = null;
    if (status === "processed") color = "success";
    else if (status === "nlp_failed") color = "error";
    else if (status === "nlp_queued" || status === "nlp_processing") {
      color = "info";
      progress = report.nlp_progress || 0;
    }
    return (
      <Box display="flex" alignItems="center">
        <Chip
          label={status.replace("_", " ") || "unknown"}
          color={color}
          size="small"
        />
        {progress !== null && (
          <CircularProgress
            size={16}
            sx={{ ml: 1 }}
            variant="determinate"
            value={progress}
          />
        )}
        {progress !== null && (
          <Typography variant="caption" sx={{ ml: 0.5 }}>
            ({progress}%)
          </Typography>
        )}
      </Box>
    );
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: { xs: "column", md: "row" },
        height: "calc(100vh - 64px)", // Subtract navbar height (64px is default MUI AppBar height)
        width: "100vw",
        overflow: "hidden",
        mt: 0,
        pt: 1,
        pr: 2,
      }}
    >
      {/* Column 1: Topic List */}
      <Box
        sx={{
          width: { xs: "100%", md: "25%" },
          height: "100%", // Use 100% of parent height instead of 100vh
          overflowY: "auto",
        }}
      >
        <Paper
          elevation={2}
          sx={{
            p: 2,
            height: "100%",
            display: "flex",
            flexDirection: "column",
          }}
        >
          <TopicList
            topics={esgTopics}
            currentReport={report}
            onSelectTopic={handleSelectTopic}
            activeTopicId={selectedTopic?.id}
          />
        </Paper>
      </Box>

      {/* Column 2: Scoring Controls & Suggestions - Wrapped in scrollable container */}
      <Box
        sx={{
          width: { xs: "100%", md: "35%" },
          height: "100%", // Use 100% of parent height
          p: 2,
        }}
      >
        <Paper
          elevation={2}
          sx={{
            height: "100%",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <Box
            sx={{
              height: "100%",
              overflowY: "auto",
              p: 2,
            }}
          >
            {/* Controls Section */}
            <Box sx={{ mb: 3 }}>
              {/* Report Information */}
              <Typography variant="h6" gutterBottom>
                Report: {report.company_name}
              </Typography>
              <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 2 }}>
                <Typography variant="body2">Status:</Typography>
                {reportStatusChip(report.status)}
              </Stack>

              {/* Score Display */}
              {scoreDetails && (
                <Box sx={{ mb: 2 }}>
                  <Typography
                    variant="h6"
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                    }}
                  >
                    <ScoreboardIcon /> ESG Score:{" "}
                    {scoreDetails.overall_score?.toFixed(1) || "N/A"}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Based on {scoreDetails.total_annotations || 0} annotations
                  </Typography>
                </Box>
              )}

              {/* Topic Selection */}
              {selectedTopic && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Selected Topic: {selectedTopic.name}
                  </Typography>

                  {/* Annotation Controls */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" gutterBottom>
                      Mark as:
                    </Typography>
                    <ButtonGroup size="small" disabled={annotationLoading}>
                      <Button
                        startIcon={<CheckCircleOutlineIcon />}
                        onClick={() => handleAnnotationUpdate("answered")}
                        color="success"
                      >
                        Disclosed
                      </Button>
                      <Button
                        startIcon={<HighlightOffIcon />}
                        onClick={() => handleAnnotationUpdate("unanswered")}
                        color="error"
                      >
                        Not Disclosed
                      </Button>
                      <Button
                        startIcon={<HelpOutlineIcon />}
                        onClick={() => handleAnnotationUpdate("pending")}
                        color="warning"
                      >
                        Uncertain
                      </Button>
                    </ButtonGroup>
                    {annotationLoading && <CircularProgress size={16} sx={{ ml: 1 }} />}
                  </Box>
                </Box>
              )}

              {/* Controls for suggestions */}
              {selectedTopic && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Suggestion Controls
                  </Typography>

                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <FormLabel>Probability Threshold: {probabilityThreshold}</FormLabel>
                    <Slider
                      value={probabilityThreshold}
                      onChange={(e, value) => setProbabilityThreshold(value)}
                      min={0.1}
                      max={1.0}
                      step={0.1}
                      marks
                      valueLabelDisplay="auto"
                    />
                  </FormControl>

                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <FormLabel>Max Results: {topK}</FormLabel>
                    <Slider
                      value={topK}
                      onChange={(e, value) => setTopK(value)}
                      min={5}
                      max={50}
                      step={5}
                      marks
                      valueLabelDisplay="auto"
                    />
                  </FormControl>

                  <Button
                    onClick={handleRefreshSuggestions}
                    disabled={loadingSuggestions}
                    startIcon={<RefreshIcon />}
                    variant="outlined"
                    size="small"
                  >
                    Refresh Suggestions
                  </Button>
                </Box>
              )}
            </Box>

            {/* Suggestions Section */}
            <Box>
              <SuggestionList
                suggestions={suggestions}
                isLoading={loadingSuggestions}
                onSuggestionClick={handleSuggestionClick}
                reportStatus={report.status}
              />
            </Box>
          </Box>
        </Paper>
      </Box>

      {/* Column 3: PDF Viewer */}
      <Box
        sx={{
          width: { xs: "100%", md: "40%" },
          height: "100%", // Use 100% of parent height instead of 100vh
          pr: 2, // Increase right padding to give PDF viewer more breathing room
        }}
      >
        <Paper
          elevation={2}
          sx={{
            height: "100%",
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <PDFViewer
            pdfUrl={pdfObjectUrl}
            pageNumberToDisplay={currentPdfPage}
            onDocumentLoadSuccess={handlePdfDocumentLoad}
            onPageChange={handlePageChangeByViewer}
            activeHighlight={activeHighlight}
          />
        </Paper>
      </Box>
    </Box>
  );
};

export default ReportScoringPage;
