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
  const [pdfTotalPages, setPdfTotalPages] = useState(null);
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
  };

  useEffect(() => {
    if (!selectedTopic || !report || report.status !== "processed") {
      setSuggestions([]);
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

        if (!isCancelled) {
          if (Array.isArray(suggestionsRes.data)) {
            setSuggestions(suggestionsRes.data);
          } else {
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
      if (Array.isArray(suggestionsRes.data)) {
        setSuggestions(suggestionsRes.data);
      } else {
        setSuggestions([]);
        setError(`Received malformed suggestions for ${selectedTopic.name}.`);
      }
    } catch (err) {
      setError(
        `Failed to load suggestions for ${selectedTopic.name}. ${
          err.response?.data?.detail || err.message
        }`
      );
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
        if (selectedTopic) {
          setSelectedTopic((prevTopic) => ({
            ...prevTopic,
            annotation_status: updatedAnnotation.data.status,
          }));
        }
        return { ...prevReport, annotations: newAnnotations };
      });
      const scoreRes = await getReportScore(reportId);
      setScoreDetails(scoreRes.data);
    } catch (err) {
      setError(
        `Failed to save annotation: ${
          err.response?.data?.detail || err.message
        }`
      );
    } finally {
      setAnnotationLoading(false);
    }
  };

  const handlePdfDocumentLoad = (numPages) => {
    setPdfTotalPages(numPages);
  };

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
        height: { xs: "auto", md: "calc(100vh - 64px)" },
        minHeight: { xs: "100vh", md: "calc(100vh - 64px)" },
        width: "100%",
        overflow: { xs: "visible", md: "hidden" },
        mt: 0,
        pt: 1,
        px: { xs: 1, sm: 2 },
      }}
    >
      {/* Column 1: Topic List */}
      <Box
        sx={{
          width: { xs: "100%", md: "25%" },
          height: { xs: "300px", md: "100%" },
          mb: { xs: 2, md: 0 },
        }}
      >
        <Paper
          elevation={2}
          sx={{
            p: { xs: 1, sm: 2 },
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
            loadingSuggestionsForTopicId={loadingSuggestions ? selectedTopic?.id : null}
          />
        </Paper>
      </Box>

      {/* Column 2: Scoring Controls & Suggestions */}
      <Box
        sx={{
          width: { xs: "100%", md: "35%" },
          height: { xs: "auto", md: "100%" },
          mb: { xs: 2, md: 0 },
          px: { xs: 0, md: 2 },
        }}
      >
        <Paper
          elevation={2}
          sx={{
            height: { xs: "auto", md: "100%" },
            minHeight: { xs: "400px", md: "auto" },
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <Box
            sx={{
              height: "100%",
              overflowY: "auto",
              p: { xs: 1, sm: 2 },
            }}
          >
            {/* Controls Section */}
            <Box sx={{ mb: 3 }}>
              {/* Report Information */}
              <Typography variant="h6" gutterBottom sx={{ fontSize: { xs: '1.1rem', sm: '1.25rem' } }}>
                Report: {report.company_name}
              </Typography>
              <Stack direction={{ xs: "column", sm: "row" }} spacing={1} alignItems={{ xs: "flex-start", sm: "center" }} sx={{ mb: 2 }}>
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
                      fontSize: { xs: '1.1rem', sm: '1.25rem' },
                      flexWrap: { xs: "wrap", sm: "nowrap" }
                    }}
                  >
                    <ScoreboardIcon /> ESG Score:{" "}
                    {scoreDetails.percentage || "N/A"}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Based on {scoreDetails.max_score - scoreDetails.pending_count || 0} annotations
                  </Typography>
                </Box>
              )}

              {/* Topic Selection */}
              {selectedTopic && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom sx={{ fontSize: { xs: '0.9rem', sm: '1rem' } }}>
                    Selected Topic: {selectedTopic.name}
                  </Typography>

                  {/* Annotation Controls */}
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" gutterBottom>
                      Mark as:
                    </Typography>
                    <Stack 
                      direction={{ xs: "column", sm: "row" }}
                      spacing={1}
                    >
                      <Button
                        startIcon={<CheckCircleOutlineIcon />}
                        onClick={() => handleAnnotationUpdate("answered")}
                        color="success"
                        disabled={annotationLoading}
                        size="small"
                        fullWidth={{ xs: true, sm: false }}
                      >
                        Disclosed
                      </Button>
                      <Button
                        startIcon={<HighlightOffIcon />}
                        onClick={() => handleAnnotationUpdate("unanswered")}
                        color="error"
                        disabled={annotationLoading}
                        size="small"
                        fullWidth={{ xs: true, sm: false }}
                      >
                        Not Disclosed
                      </Button>
                      <Button
                        startIcon={<HelpOutlineIcon />}
                        onClick={() => handleAnnotationUpdate("pending")}
                        color="warning"
                        disabled={annotationLoading}
                        size="small"
                        fullWidth={{ xs: true, sm: false }}
                      >
                        Uncertain
                      </Button>
                    </Stack>
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
                    <FormLabel sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
                      Probability Threshold: {probabilityThreshold}
                    </FormLabel>
                    <Slider
                      value={probabilityThreshold}
                      onChange={(e, value) => setProbabilityThreshold(value)}
                      min={0.1}
                      max={1.0}
                      step={0.1}
                      marks
                      valueLabelDisplay="auto"
                      size="small"
                    />
                  </FormControl>

                  <FormControl fullWidth sx={{ mb: 2 }}>
                    <FormLabel sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
                      Max Results: {topK}
                    </FormLabel>
                    <Slider
                      value={topK}
                      onChange={(e, value) => setTopK(value)}
                      min={5}
                      max={50}
                      step={5}
                      marks
                      valueLabelDisplay="auto"
                      size="small"
                    />
                  </FormControl>

                  <Button
                    onClick={handleRefreshSuggestions}
                    disabled={loadingSuggestions}
                    startIcon={<RefreshIcon />}
                    variant="outlined"
                    size="small"
                    fullWidth={{ xs: true, sm: false }}
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
          height: { xs: "500px", md: "100%" },
          px: { xs: 0, md: 2 },
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
            onDocumentLoadSuccess={handlePdfDocumentLoad}
          />
        </Paper>
      </Box>
    </Box>
  );
};

export default ReportScoringPage;
