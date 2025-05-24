import React, { useEffect, useState } from 'react';
import { Worker, Viewer } from '@react-pdf-viewer/core';
import { defaultLayoutPlugin } from '@react-pdf-viewer/default-layout';
import { Box, Typography, Paper } from '@mui/material';

// Import CSS
import '@react-pdf-viewer/core/lib/styles/index.css';
import '@react-pdf-viewer/default-layout/lib/styles/index.css';

const PDFViewer = ({
  pdfUrl,
  pageNumberToDisplay,
  onDocumentLoadSuccess,
  onPageChange,
  activeHighlight,
}) => {
  const [numPages, setNumPages] = useState(null);
  const [viewerKey, setViewerKey] = useState(0);

  // Create plugins once
  const defaultLayoutPluginInstance = defaultLayoutPlugin({
    sidebarTabs: (defaultTabs) => [
      defaultTabs[0], // Thumbnails
      defaultTabs[1], // Bookmarks
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
    console.log('📚 PDF loaded with pages:', pageCount);
    if (onDocumentLoadSuccess) {
      onDocumentLoadSuccess(pageCount);
    }
  };

  // Handle page change
  const handlePageChange = (e) => {
    const currentPage = e.currentPage + 1; // Convert from 0-based to 1-based
    console.log('📄 PDF page changed to:', currentPage);
    if (onPageChange) {
      onPageChange(currentPage);
    }
  };

  // Force re-render when page changes
  useEffect(() => {
    console.log('📍 PDFViewer received pageNumberToDisplay:', pageNumberToDisplay);
    setViewerKey(prev => prev + 1);
  }, [pageNumberToDisplay]);

  // Debug active highlight
  useEffect(() => {
    console.log('🎯 PDFViewer received activeHighlight:', activeHighlight);
  }, [activeHighlight]);

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
        },
        '& .rpv-default-layout__main': {
          overflow: 'hidden',
        },
      }}
    >
      <Worker workerUrl="https://unpkg.com/pdfjs-dist@3.11.174/build/pdf.worker.min.js">
        <Viewer
          key={viewerKey}
          fileUrl={pdfUrl}
          plugins={[defaultLayoutPluginInstance]}
          initialPage={pageNumberToDisplay ? pageNumberToDisplay - 1 : 0}
          onDocumentLoad={handleDocumentLoad}
          onPageChange={handlePageChange}
          theme="light"
        />
      </Worker>
    </Box>
  );
};

export default PDFViewer;