import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css'; // Recommended for text selection, etc.
import 'react-pdf/dist/esm/Page/TextLayer.css';     // Recommended for text selection
import pdfWorkerSrc from 'pdfjs-dist/build/pdf.worker.min.mjs?url';

pdfjs.GlobalWorkerOptions.workerSrc = pdfWorkerSrc;

// interface HighlightRect { pageNumber: number; coordinates: { x0: number; y0: number; x1: number; y1: number; }; id: string|number; }
// interface PDFViewerProps {
//   pdfUrl: string;
//   pageNumberToDisplay: number;
//   onDocumentLoadSuccess?: (numPages: number) => void;
//   onPageChange?: (newPageNumber: number) => void;
//   activeHighlight?: HighlightRect | null;
// }

const PDF_RENDER_WIDTH = 600; // Define a fixed width for rendering the PDF page for consistent scaling

const PDFViewer = ({ pdfUrl, pageNumberToDisplay, onDocumentLoadSuccess, onPageChange, activeHighlight }) => {
  const [numPages, setNumPages] = useState(null);
  const [currentPageNumber, setCurrentPageNumber] = useState(pageNumberToDisplay || 1);
  const [pdfPageDetails, setPdfPageDetails] = useState(null); // To store original page dimensions

  const canvasRef = useRef(null); // Ref for the highlight canvas

  useEffect(() => {
    setCurrentPageNumber(pageNumberToDisplay || 1);
  }, [pageNumberToDisplay]);

  const onDocumentLoadSuccessInternal = useCallback(async ({ numPages: nextNumPages }) => {
    setNumPages(nextNumPages);
    if (onDocumentLoadSuccess) {
      onDocumentLoadSuccess(nextNumPages);
    }
    // Set the initial page or reset if current is out of bounds
    const newPage = Math.min(Math.max(1, currentPageNumber), nextNumPages);
    if (currentPageNumber !== newPage) {
        setCurrentPageNumber(newPage);
        if (onPageChange) onPageChange(newPage);
    }
  }, [onDocumentLoadSuccess, onPageChange, currentPageNumber]);


  const onPageLoadSuccessInternal = useCallback((page) => {
    // Store original page dimensions (unscaled)
    // PDF.js page object has getViewport method
    const originalViewport = page.getViewport({ scale: 1 });
    setPdfPageDetails({
        originalWidth: originalViewport.width,
        originalHeight: originalViewport.height,
        pdfJsPage: page, // Store the pdf.js page object if needed for more advanced ops
    });
  }, []);


  // Effect to draw highlight when activeHighlight or currentPageNumber changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !activeHighlight || !pdfPageDetails || activeHighlight.pageNumber !== currentPageNumber) {
      if (canvas) { // Clear canvas if no highlight or wrong page
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    // Clear previous highlight
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate scale
    // The Page component is rendered with a specific width (PDF_RENDER_WIDTH)
    // The scale is renderedWidth / originalWidth
    const scale = PDF_RENDER_WIDTH / pdfPageDetails.originalWidth;

    // Backend coordinates (x0, y0, x1, y1) from top-left of PDF page in points
    const { x0, y0, x1, y1 } = activeHighlight.coordinates;

    // Transform coordinates to canvas space
    const canvasX = x0 * scale;
    const canvasY = y0 * scale;
    const canvasWidth = (x1 - x0) * scale;
    const canvasHeight = (y1 - y0) * scale;

    // Draw the highlight
    ctx.fillStyle = 'rgba(255, 255, 0, 0.4)'; // Semi-transparent yellow
    ctx.fillRect(canvasX, canvasY, canvasWidth, canvasHeight);

    console.log(`Highlight drawn on page ${currentPageNumber}:`, {canvasX, canvasY, canvasWidth, canvasHeight, scale});

  }, [activeHighlight, currentPageNumber, pdfPageDetails]);


  const changePage = (offset) => {
    const newPage = Math.max(1, Math.min(currentPageNumber + offset, numPages || 1));
    setCurrentPageNumber(newPage);
    if (onPageChange) {
      onPageChange(newPage);
    }
    // setActiveHighlight(null); // Clear highlight on manual page change
  };

  const goToPrevPage = () => changePage(-1);
  const goToNextPage = () => changePage(1);


  if (!pdfUrl) {
    return <div className="card" style={{height: '100%', display: 'flex', alignItems:'center', justifyContent:'center'}}><p>PDF will be displayed here.</p></div>;
  }
  
  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <div style={{ paddingBottom: '10px', textAlign: 'center', flexShrink: 0 }}>
        <button onClick={goToPrevPage} disabled={currentPageNumber <= 1}>Prev</button>
        <span style={{ margin: '0 10px' }}>Page {currentPageNumber} of {numPages || '--'}</span>
        <button onClick={goToNextPage} disabled={!numPages || currentPageNumber >= numPages}>Next</button>
      </div>
      <div style={{ flexGrow: 1, border: '1px solid #ccc', overflow: 'auto', position: 'relative' }}> {/* Changed overflowY to auto for scrollbars if content overflows */}
        <Document
          file={pdfUrl}
          onLoadSuccess={onDocumentLoadSuccessInternal}
          onLoadError={(error) => console.error('PDF Load Error:', error.message)}
          loading={<div style={{padding: '20px', textAlign: 'center'}}>Loading PDF...</div>}
          error={<div style={{padding: '20px', textAlign: 'center', color: 'red'}}>Error loading PDF.</div>}
        >
          <div style={{ position: 'relative', width: `${PDF_RENDER_WIDTH}px`, margin: 'auto' }}> {/* Wrapper for positioning canvas */}
            <Page
              pageNumber={currentPageNumber}
              width={PDF_RENDER_WIDTH}
              onLoadSuccess={onPageLoadSuccessInternal} // Get page details after it's loaded
              onRenderError={(error) => console.error('Page Render Error:', error)}
            />
            {/* Canvas for highlighting, positioned over the Page */}
            {pdfPageDetails && ( // Only render canvas if we have page details (for width/height)
                 <canvas
                    ref={canvasRef}
                    width={PDF_RENDER_WIDTH} // Match the rendered page width
                    height={(pdfPageDetails.originalHeight * (PDF_RENDER_WIDTH / pdfPageDetails.originalWidth))} // Calculate scaled height
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        pointerEvents: 'none', // Make canvas non-interactive
                    }}
                />
            )}
          </div>
        </Document>
      </div>
    </div>
  );
};

export default PDFViewer;