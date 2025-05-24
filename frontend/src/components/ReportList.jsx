import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { deleteReport } from '../services/api';
import {
  Typography, List, ListItem, ListItemText, IconButton, Box, Chip, Tooltip, LinearProgress, Button, Divider
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import ArticleIcon from '@mui/icons-material/Article';
import BusinessIcon from '@mui/icons-material/Business';
import AccessTimeIcon from '@mui/icons-material/AccessTime';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty';
import ScoreboardIcon from '@mui/icons-material/Scoreboard';


const ReportList = ({ reports, onReportDeleted }) => {
  if (!reports || reports.length === 0) {
    return <Typography variant="subtitle1" color="text.secondary">No reports uploaded yet.</Typography>;
  }

  const handleDelete = async (reportId, reportFilename) => {
    if (window.confirm(`Are you sure you want to delete report: ${reportFilename}? This action cannot be undone.`)) {
      try {
        await deleteReport(reportId);
        onReportDeleted(reportId);
      } catch (error) {
        console.error("Failed to delete report:", error);
        alert(`Failed to delete report: ${error.response?.data?.detail || error.message}`);
      }
    }
  };

  const getStatusChip = (report) => {
    let color = "default";
    let icon = <HourglassEmptyIcon />;
    if (report.status === 'processed') {
      color = "success";
      icon = <CheckCircleIcon />;
    } else if (report.status === 'nlp_failed') {
      color = "error";
      icon = <ErrorIcon />;
    } else if (report.status === 'nlp_queued' || report.status === 'nlp_processing') {
      color = "info";
    }

    return <Chip icon={icon} label={report.status.replace('_', ' ').toUpperCase()} color={color} size="small" />;
  };

  return (
    <Box>
      <Typography variant="h5" component="h3" gutterBottom>
        Uploaded Reports
      </Typography>
      <List>
        {reports.map((report, index) => (
          <React.Fragment key={report.id}>
            <ListItem
              secondaryAction={
                <Tooltip title="Delete Report">
                  <IconButton
                    edge="end"
                    aria-label="delete"
                    onClick={() => handleDelete(report.id, report.filename)}
                    color="error"
                  >
                    <DeleteIcon />
                  </IconButton>
                </Tooltip>
              }
              sx={{
                alignItems: 'flex-start',
                '&:hover': { backgroundColor: 'action.hover' }
              }}
            >
              <IconButton component={RouterLink} to={`/report/${report.id}`} sx={{ mr: 1.5, mt:0.5 }} color="primary">
                  <ArticleIcon fontSize="large"/>
              </IconButton>
              <ListItemText
                primary={
                  <Typography variant="h6" component={RouterLink} to={`/report/${report.id}`} sx={{ textDecoration: 'none', color: 'primary.main' }}>
                    {report.filename}
                  </Typography>
                }
                secondary={
                  <>
                    <Typography variant="body2" color="text.secondary" component="div" sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                      <BusinessIcon fontSize="small" sx={{ mr: 0.5 }} /> Company: {report.company_name || 'N/A'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" component="div" sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                      <AccessTimeIcon fontSize="small" sx={{ mr: 0.5 }} /> Uploaded: {new Date(report.upload_timestamp).toLocaleString()}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', mt: 1, mb:0.5 }}>
                        Status: &nbsp; {getStatusChip(report)}
                    </Box>
                    {(report.status === 'nlp_queued' || report.status === 'nlp_processing') && (
                      <Box sx={{ width: '100%', mt: 0.5 }}>
                        <LinearProgress variant="determinate" value={report.nlp_progress || 0} />
                        <Typography variant="caption" display="block" textAlign="right">{report.nlp_progress || 0}%</Typography>
                      </Box>
                    )}
                     <Typography variant="body2" color="text.secondary" component="div" sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                      <ScoreboardIcon fontSize="small" sx={{ mr: 0.5 }} /> Score: {report.final_score !== null ? report.final_score : 'N/A'} / {report.annotations?.length || 'N/A'}
                    </Typography>
                  </>
                }
                // Add this prop
                secondaryTypographyProps={{ component: 'div' }}
              />
            </ListItem>
            {index < reports.length - 1 && <Divider variant="inset" component="li" />}
          </React.Fragment>
        ))}
      </List>
    </Box>
  );
};

export default ReportList;