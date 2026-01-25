import React, { useState, useEffect } from 'react';

/**
 * SearchInterface Component
 * 
 * A simple search interface for querying a RAG system.
 * Displays search results with highlighting and metadata.
 */
const SearchInterface = ({ apiEndpoint }) => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSearch = async (e) => {
    e.preventDefault();
    
    if (!query.trim()) {
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${apiEndpoint}/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          query: query,
          top_k: 5 
        }),
      });

      if (!response.ok) {
        throw new Error('Search request failed');
      }

      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          className="search-input"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}

      {results.length > 0 && (
        <div className="results-container">
          <h3>Results ({results.length})</h3>
          {results.map((result, index) => (
            <div key={index} className="result-card">
              <div className="result-header">
                <span className="result-score">
                  Score: {result.score.toFixed(3)}
                </span>
                <span className="result-source">
                  {result.metadata.file_name}
                </span>
              </div>
              <div className="result-content">
                {result.content}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default SearchInterface;
