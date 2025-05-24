import React from "react";
import {
  Paper,
  Typography,
  List,
  ListItemButton,
  ListItemText,
  Box,
  CircularProgress,
  Alert,
  Chip,
  Divider,
} from "@mui/material";
import PageviewIcon from "@mui/icons-material/Pageview"; // For suggestion icon
import InsightsIcon from "@mui/icons-material/Insights"; // For score icon

const SuggestionList = ({
  suggestions,
  isLoading,
  reportStatus,
}) => {
  if (isLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        sx={{ py: 3 }}
      >
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>Loading suggestions...</Typography>
      </Box>
    );
  }

  if (reportStatus && reportStatus !== "processed") {
    return (
      <Box sx={{ py: 2 }}>
        <Typography variant="h6" gutterBottom>
          Potential Disclosures
        </Typography>
        <Alert severity="info" variant="outlined">
          Suggestions are generated after the report is fully processed. Current
          status: <Chip label={reportStatus} size="small" color="info" />
        </Alert>
      </Box>
    );
  }

  if (!suggestions || suggestions.length === 0) {
    return (
      <Box sx={{ py: 2 }}>
        <Typography variant="h6" gutterBottom>
          Potential Disclosures
        </Typography>
        <Typography color="text.secondary">
          No suggestions found for the current topic and settings.
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Typography
        variant="h6"
        gutterBottom
        sx={{ 
          pb: 2,
          fontSize: { xs: '1rem', sm: '1.25rem' }
        }}
      >
        Potential Disclosures ({suggestions.length})
      </Typography>
      <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {suggestions.map((suggestion, index) => (
          <Paper
            key={suggestion.chunk_id || index}
            elevation={1}
            sx={{
              borderRadius: 1,
              border: "1px solid",
              borderColor: "divider",
              p: { xs: 1, sm: 2 },
            }}
          >
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: { xs: "flex-start", sm: "center" },
                flexDirection: { xs: "column", sm: "row" },
                gap: { xs: 0.5, sm: 0 },
                mb: 0.5,
              }}
            >
              <Typography
                variant="subtitle2"
                color="primary"
                sx={{ 
                  display: "flex", 
                  alignItems: "center",
                  fontSize: { xs: '0.8rem', sm: '0.875rem' }
                }}
              >
                <PageviewIcon fontSize="small" sx={{ mr: 0.5 }} /> Page:{" "}
                {suggestion.page_number}
              </Typography>
              <Chip
                icon={<InsightsIcon />}
                label={`Score: ${(
                  suggestion.entailment_score * 100
                ).toFixed(1)}%`}
                size="small"
                color={
                  suggestion.entailment_score > 0.85
                    ? "success"
                    : suggestion.entailment_score > 0.6
                    ? "warning"
                    : "default"
                }
                variant="outlined"
                sx={{
                  fontSize: { xs: '0.7rem', sm: '0.75rem' }
                }}
              />
            </Box>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ 
                whiteSpace: "pre-line",
                fontSize: { xs: '0.8rem', sm: '0.875rem' },
                lineHeight: { xs: 1.3, sm: 1.4 }
              }}
            >
              {suggestion.chunk_text}
            </Typography>
          </Paper>
        ))}
      </Box>
    </Box>
  );
};

export default SuggestionList;