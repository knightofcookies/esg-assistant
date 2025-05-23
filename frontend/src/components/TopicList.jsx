import React from 'react';

const TopicList = ({ topics, currentReport, onSelectTopic, activeTopicId }) => {
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
      <ul style={{ listStyle: 'none', padding: 0, maxHeight: '60vh', overflowY: 'auto' }}>
        {topics.map(topic => (
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
            <span>{topic.topic_number}. {topic.name}</span>
            <span style={{
                padding: '3px 8px',
                borderRadius: '10px',
                backgroundColor: getStatusColor(getAnnotationStatus(topic.id)),
                color: 'white',
                fontSize: '0.8em'
            }}>
                {getAnnotationStatus(topic.id)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TopicList;