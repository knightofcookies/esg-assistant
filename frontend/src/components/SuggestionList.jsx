import React from 'react';

const SuggestionList = ({ suggestions, isLoading, onSuggestionClick }) => {
  if (isLoading) return <p>Loading suggestions...</p>;
  if (!suggestions || suggestions.length === 0) return <p>No suggestions found for the current topic and threshold.</p>;

  return (
    <div className="card">
      <h4>Potential Disclosures (Suggestions)</h4>
      <ul style={{ listStyle: 'none', padding: 0, maxHeight: '40vh', overflowY: 'auto' }}>
        {suggestions.map((suggestion, index) => (
          <li
            key={suggestion.chunk_id || index} // Use chunk_id if available
            style={{
              padding: '10px',
              border: '1px solid #eee',
              marginBottom: '5px',
              borderRadius: '4px',
              cursor: onSuggestionClick ? 'pointer' : 'default'
            }}
            onClick={() => onSuggestionClick && onSuggestionClick(suggestion)}
          >
            <p><strong>Page: {suggestion.page_number} (Score: {(suggestion.entailment_score * 100).toFixed(1)}%)</strong></p>
            <p style={{ fontSize: '0.9em', color: '#555' }}>{suggestion.chunk_text}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default SuggestionList;