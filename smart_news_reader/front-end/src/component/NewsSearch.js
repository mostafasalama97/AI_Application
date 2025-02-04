import React, { useState } from "react";
import axios from "axios";
import { Container, Form, Button, Row, Col, Card, Alert, Spinner } from "react-bootstrap";

const NewsSearch = () => {
  // State for search inputs
  const [query, setQuery] = useState("");
  const [numArticles, setNumArticles] = useState(3);
  const [articles, setArticles] = useState([]);
  const [summarizedArticles, setSummarizedArticles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Function to handle search
  const handleSearch = async () => {
    if (!query.trim()) {
      setError("⚠️ Please enter a search query.");
      return;
    }

    setError(null);
    setLoading(true);
    setArticles([]);
    setSummarizedArticles([]);

    try {
      const response = await axios.post("http://localhost:8009/search", {
        query,
        num_articles_tldr: numArticles
      });

      setArticles(response.data.original_articles || []);
      setSummarizedArticles(response.data.summarized_articles || []);

    } catch (error) {
      setError("❌ Error fetching news. Please try again.");
      console.error("Error fetching news:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container className="mt-5">
      <h2 className="text-center">🔍 AI News Search</h2>

      {error && <Alert variant="danger">{error}</Alert>}

      <Row className="mb-3">
        <Col md={6}>
          <Form.Control
            type="text"
            placeholder="Enter search query..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </Col>
        <Col md={3}>
          <Form.Select value={numArticles} onChange={(e) => setNumArticles(parseInt(e.target.value, 10))}>
            <option value="3">Top 3 Articles</option>
            <option value="5">Top 5 Articles</option>
            <option value="10">Top 10 Articles</option>
          </Form.Select>
        </Col>
        <Col md={3}>
          <Button variant="primary" onClick={handleSearch} disabled={loading}>
            {loading ? <Spinner as="span" animation="border" size="sm" role="status" aria-hidden="true" /> : "Search"}
          </Button>
        </Col>
      </Row>

      {/* Display results */}
      {articles.length > 0 && (
        <div>
          <h4>📰 Original Articles</h4>
          <Row>
            {articles.map((article, index) => (
              <Col md={6} lg={4} key={index} className="mb-3">
                <Card>
                  <Card.Body>
                    <Card.Title>{article.title}</Card.Title>
                    <Card.Text>
                      {article.description || "No description available."}
                    </Card.Text>
                    <Card.Link href={article.url} target="_blank">Read More</Card.Link>
                  </Card.Body>
                </Card>
              </Col>
            ))}
          </Row>
        </div>
      )}

      {summarizedArticles.length > 0 && (
        <div className="mt-4">
          <h4>📝 AI Summarized Articles</h4>
          <Card className="p-3">
            {summarizedArticles.map((summary, index) => (
              <p key={index}>{summary}</p>
            ))}
          </Card>
        </div>
      )}
    </Container>
  );
};

export default NewsSearch;
