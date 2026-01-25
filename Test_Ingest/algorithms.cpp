/**
 * Common algorithms for text similarity and ranking
 * Used in information retrieval and RAG systems
 */

#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <unordered_map>

/**
 * Calculate Levenshtein distance between two strings
 * Measures the minimum number of single-character edits needed
 */
int levenshteinDistance(const std::string& s1, const std::string& s2) {
    const size_t m = s1.size();
    const size_t n = s2.size();
    
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1));
    
    // Initialize base cases
    for (size_t i = 0; i <= m; i++) {
        dp[i][0] = i;
    }
    for (size_t j = 0; j <= n; j++) {
        dp[0][j] = j;
    }
    
    // Fill the DP table
    for (size_t i = 1; i <= m; i++) {
        for (size_t j = 1; j <= n; j++) {
            if (s1[i-1] == s2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            } else {
                dp[i][j] = 1 + std::min({
                    dp[i-1][j],     // deletion
                    dp[i][j-1],     // insertion
                    dp[i-1][j-1]    // substitution
                });
            }
        }
    }
    
    return dp[m][n];
}

/**
 * Calculate BM25 score for document ranking
 * k1 and b are tuning parameters
 */
double bm25Score(
    const std::vector<std::string>& query_terms,
    const std::unordered_map<std::string, int>& doc_term_freq,
    int doc_length,
    double avg_doc_length,
    const std::unordered_map<std::string, double>& idf_scores,
    double k1 = 1.5,
    double b = 0.75
) {
    double score = 0.0;
    
    for (const auto& term : query_terms) {
        auto it = doc_term_freq.find(term);
        if (it == doc_term_freq.end()) {
            continue;
        }
        
        int tf = it->second;
        double idf = idf_scores.at(term);
        
        // BM25 formula
        double numerator = tf * (k1 + 1);
        double denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length));
        
        score += idf * (numerator / denominator);
    }
    
    return score;
}

/**
 * Reciprocal Rank Fusion (RRF)
 * Combines rankings from multiple sources
 */
std::vector<std::pair<int, double>> reciprocalRankFusion(
    const std::vector<std::vector<int>>& rankings,
    int k = 60
) {
    std::unordered_map<int, double> scores;
    
    // Accumulate RRF scores
    for (const auto& ranking : rankings) {
        for (size_t rank = 0; rank < ranking.size(); rank++) {
            int doc_id = ranking[rank];
            scores[doc_id] += 1.0 / (k + rank + 1);
        }
    }
    
    // Convert to vector and sort by score
    std::vector<std::pair<int, double>> result(scores.begin(), scores.end());
    std::sort(result.begin(), result.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        });
    
    return result;
}

/**
 * Maximal Marginal Relevance (MMR)
 * Balances relevance and diversity in results
 */
std::vector<int> maximalMarginalRelevance(
    const std::vector<double>& relevance_scores,
    const std::vector<std::vector<double>>& similarity_matrix,
    int k,
    double lambda = 0.5
) {
    std::vector<int> selected;
    std::vector<bool> used(relevance_scores.size(), false);
    
    // Select the most relevant item first
    int first = std::distance(
        relevance_scores.begin(),
        std::max_element(relevance_scores.begin(), relevance_scores.end())
    );
    selected.push_back(first);
    used[first] = true;
    
    // Iteratively select k-1 more items
    while (selected.size() < k) {
        double best_score = -1e9;
        int best_idx = -1;
        
        for (size_t i = 0; i < relevance_scores.size(); i++) {
            if (used[i]) continue;
            
            // Calculate max similarity to already selected items
            double max_sim = 0.0;
            for (int selected_idx : selected) {
                max_sim = std::max(max_sim, similarity_matrix[i][selected_idx]);
            }
            
            // MMR formula: λ * relevance - (1-λ) * max_similarity
            double mmr_score = lambda * relevance_scores[i] - (1 - lambda) * max_sim;
            
            if (mmr_score > best_score) {
                best_score = mmr_score;
                best_idx = i;
            }
        }
        
        if (best_idx == -1) break;
        
        selected.push_back(best_idx);
        used[best_idx] = true;
    }
    
    return selected;
}
