import React, { useEffect, useState } from 'react';
import { Worker, Viewer } from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';
import { Box, Typography, Paper } from '@mui/material';

import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';

const PDFViewer = ({
  pdfUrl,
  onDocumentLoadSuccess,
}) => {
  const [numPages, setNumPages] = useState(null);
  const [viewerKey, setViewerKey] = useState(0);

  // Create plugins once
  const defaultLayoutPluginInstance = defaultLayoutPlugin({
    sidebarTabs: (defaultTabs) => [
      defaultTabs[0], // Thumbnails
    ],
    toolbarPlugin: {
      moreActionsPopover: {
        direction: 'rtl',
      },
    },
  });

  // Handle document load
  const handleDocumentLoad = (e) => {
    const pageCount = e.doc.numPages;
    setNumPages(pageCount);
    if (onDocumentLoadSuccess) {
      onDocumentLoadSuccess(pageCount);
    }
  };

  // Only re-render when PDF URL changes
  useEffect(() => {
    setViewerKey(prev => prev + 1);
  }, [pdfUrl]);

  if (!pdfUrl) {
    return (
      <Paper
        sx={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          p: 2,
        }}
      >
        <Typography color="text.secondary">
          PDF will be displayed here once a report is loaded.
        </Typography>
      </Paper>
    );
  }

  return (
    <Box
      sx={{
        height: "100%",
        width: "100%",
        '& .rpv-core__viewer': {
          height: '100%',
        },
        '& .rpv-default-layout__container': {
          height: '100%',
        },
        '& .rpv-default-layout__toolbar': {
          flexShrink: 0,
          overflow: 'hidden',
          padding: { xs: '4px', sm: '8px' },
          '& button': {
            minWidth: { xs: '32px', sm: '40px' },
            padding: { xs: '4px', sm: '8px' },
          }
        },
        '& .rpv-default-layout__sidebar': {
          width: { xs: '200px !important', sm: '250px !important' },
        },
        '& .rpv-default-layout__main': {
          overflow: 'hidden',
        },
        // Mobile-specific adjustments
        '@media (max-width: 768px)': {
          '& .rpv-toolbar__item': {
            margin: '0 2px',
          },
          '& .rpv-toolbar__button': {
            fontSize: '12px',
            padding: '4px',
          }
        }
      }}
    >
      <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
        <Viewer
          key={viewerKey}
          fileUrl={pdfUrl}
          plugins={[defaultLayoutPluginInstance]}
          onDocumentLoad={handleDocumentLoad}
          theme="light"
        />
      </Worker>
    </Box>
  );
};

export default PDFViewer;