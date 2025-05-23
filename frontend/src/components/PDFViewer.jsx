import React, { useState, useEffect } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import 'react-pdf/dist/esm/Page/AnnotationLayer.css';
import 'react-pdf/dist/esm/Page/TextLayer.css';

// Setup PDF.js worker
// You can download the worker file from a CDN or from the pdfjs-dist package
// For Vite, you might need to copy `pdf.worker.min.js` to your `public` folder
// and then set the workerSrc like this:
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.js',
  import.meta.url,
).toString();


const PDFViewer = ({ pdfUrl, onDocumentLoadSuccessProp, pageNumberProp }) => {
  const [numPages, setNumPages] = useState(null);
  const [pageNumber, setPageNumber] = useState(pageNumberProp || 1); // Use prop or default to 1

  useEffect(() => {
    setPageNumber(pageNumberProp || 1)
  }, [pageNumberProp])

  function onDocumentLoadSuccess({ numPages: nextNumPages }) {
    setNumPages(nextNumPages);
    if (onDocumentLoadSuccessProp) {
        onDocumentLoadSuccessProp(nextNumPages);
    }
    // Reset to page 1 if current pageNumber is out of bounds for new doc
    if (pageNumber > nextNumPages) setPageNumber(1);
  }

  function goToPrevPage() {
    setPageNumber(prevPageNumber => Math.max(prevPageNumber - 1, 1));
  }

  function goToNextPage() {
    setPageNumber(prevPageNumber => Math.min(prevPageNumber + 1, numPages));
  }


  if (!pdfUrl) {
    return <div className="card"><p>No PDF to display. Upload and select a report.</p></div>;
  }

  return (
    <div className="card">
      <h4>PDF Viewer</h4>
       <div style={{ marginBottom: '10px' }}>
        <button onClick={goToPrevPage} disabled={pageNumber <= 1}>Prev</button>
        <span style={{ margin: '0 10px' }}>Page {pageNumber} of {numPages || '--'}</span>
        <button onClick={goToNextPage} disabled={pageNumber >= numPages}>Next</button>
      </div>
      <div style={{ border: '1px solid #ccc', maxHeight: '70vh', overflowY: 'auto' }}>
        <Document
          file={pdfUrl}
          onLoadSuccess={onDocumentLoadSuccess}
          onLoadError={(error) => console.error('Failed to load PDF:', error.message)}
          loading="Loading PDF..."
        >
          <Page pageNumber={pageNumber} />
        </Document>
      </div>
    </div>
  );
};

export default PDFViewer;