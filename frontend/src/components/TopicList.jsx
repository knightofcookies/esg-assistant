import React from 'react';

const TopicList = ({ topics, currentReport, onSelectTopic, activeTopicId, loadingSuggestionsForTopicId }) => {
  if (!topics || topics.length === 0) return <p>No ESG topics loaded.</p>;

  const getAnnotationStatus = (topicId) => {
    if (!currentReport || !currentReport.annotations) return 'pending';
    const annotation = currentReport.annotations.find(a => a.topic.id === topicId);
    return annotation ? annotation.status : 'pending';
  };

  const getStatusColor = (status) => {
    if (status === 'answered') return 'green';
    if (status === 'unanswered') return 'red';
    return 'grey';
  }

  return (
    <div className="card">
      <h4>ESG Topics ({currentReport?.final_score || 0}/{topics.length})</h4>
      <ul style={{ listStyle: 'none', padding: 0, maxHeight: 'calc(100vh - 180px)', overflowY: 'auto' }}> {/* Adjusted maxHeight */}
        {topics.map(topic => {
          const isCurrentlyLoading = loadingSuggestionsForTopicId === topic.id;
          const status = getAnnotationStatus(topic.id);
          return (
            <li
              key={topic.id}
              onClick={() => onSelectTopic(topic)}
              style={{
                padding: '10px',
                border: `1px solid ${activeTopicId === topic.id ? '#007bff' : '#eee'}`,
                marginBottom: '5px',
                cursor: 'pointer',
                borderRadius: '4px',
                backgroundColor: activeTopicId === topic.id ? '#e7f3ff' : 'white',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <span style={{ marginRight: '10px' }}>{topic.topic_number}. {topic.name}</span>
              {isCurrentlyLoading ? (
                <span style={{ fontSize: '0.8em', color: '#007bff' }}>Loading...</span>
              ) : (
                <span style={{
                    padding: '3px 8px',
                    borderRadius: '10px',
                    backgroundColor: getStatusColor(status),
                    color: 'white',
                    fontSize: '0.8em',
                    minWidth: '60px', // Give status some width
                    textAlign: 'center'
                }}>
                    {status}
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
};

export default TopicList;