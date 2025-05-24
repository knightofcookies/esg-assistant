import React from 'react';
import { Paper, Typography, List, ListItemButton, ListItemText, Chip, Box, CircularProgress, Tooltip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import CancelIcon from '@mui/icons-material/Cancel';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';

const TopicList = ({ topics, currentReport, onSelectTopic, activeTopicId, loadingSuggestionsForTopicId }) => {
  if (!topics || topics.length === 0) {
    return <Typography color="text.secondary" sx={{p:2}}>No ESG topics loaded.</Typography>;
  }

  const getAnnotationStatus = (topicId) => {
    if (!currentReport || !currentReport.annotations) return 'pending';
    const annotation = currentReport.annotations.find(a => a.topic.id === topicId);
    return annotation ? annotation.status : 'pending';
  };

  const getStatusAttributes = (status) => {
    switch (status) {
      case 'answered':
        return { label: 'Answered', color: 'success', icon: <CheckCircleIcon /> };
      case 'unanswered':
        return { label: 'Unanswered', color: 'error', icon: <CancelIcon /> };
      default: // 'pending' or other
        return { label: 'Pending', color: 'default', icon: <HelpOutlineIcon /> };
    }
  };

  return (
    <Box sx={{height: '100%', display: 'flex', flexDirection: 'column'}}>
      <Typography variant="h6" sx={{ 
        px: 2, 
        pt: 2, 
        pb: 1,
        fontSize: { xs: '1rem', sm: '1.25rem' }
      }}>
        ESG Topics
        {topics.length > 0 && ` (${topics.length})`}
        {currentReport?.final_score !== undefined && ` | Score: ${currentReport.final_score}`}
      </Typography>
      <List dense sx={{ overflowY: 'auto', flexGrow: 1, pb: 2 }}>
        {topics.map(topic => {
          const isCurrentlyLoading = loadingSuggestionsForTopicId === topic.id;
          const status = getAnnotationStatus(topic.id);
          const statusAttrs = getStatusAttributes(status);
          const isActive = activeTopicId === topic.id;

          return (
            <ListItemButton
              key={topic.id}
              selected={isActive}
              onClick={() => onSelectTopic(topic)}
              sx={{
                mb: 0.5,
                borderRadius: 1,
                px: { xs: 1, sm: 2 },
                py: { xs: 0.5, sm: 1 },
                ...(isActive && { backgroundColor: 'action.selected', '&:hover': { backgroundColor: 'action.selected' }})
              }}
            >
              <ListItemText
                primary={`${topic.topic_number}. ${topic.name}`}
                primaryTypographyProps={{ 
                  fontWeight: isActive ? 'bold' : 'normal', 
                  noWrap: true,  
                  fontSize: { xs: '0.8rem', sm: '0.9rem' }
                }}
              />
              {isCurrentlyLoading ? (
                <CircularProgress size={16} sx={{ ml: 1 }} />
              ) : (
                <Tooltip title={statusAttrs.label} placement="top">
                    <Chip
                        icon={statusAttrs.icon}
                        label={statusAttrs.label}
                        color={statusAttrs.color}
                        size="small"
                        variant="outlined"
                        sx={{ 
                          ml: 1, 
                          minWidth: { xs: '70px', sm: '90px' },
                          fontSize: { xs: '0.7rem', sm: '0.75rem' }
                        }}
                    />
                </Tooltip>
              )}
            </ListItemButton>
          );
        })}
      </List>
    </Box>
  );
};

export default TopicList;