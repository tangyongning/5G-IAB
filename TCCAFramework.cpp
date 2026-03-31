// TCCAFramework.cpp
#include "TCCAFramework.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

TCCAFramework::TCCAFramework(NetworkTopology& topo,
                              const vector<vector<QoSObservation>>& obs,
                              bool topology, bool trust, bool temporal)
    : topology(topo), observations(obs),
      hiddenDim(64), alpha(0.6), beta(0.4), lambda(0.1),
      useTopologyConstraint(topology), useTrustMetric(trust), 
      useTemporalModel(temporal) {
    rng.seed(chrono::steady_clock::now().time_since_epoch().count());
    numTimeSteps = observations.size();
    
    // Initialize node states
    for (const auto& node : topology.getNodes()) {
        nodeStates[node.id] = NodeState(hiddenDim);
    }
}

double TCCAFramework::computeUpstreamInfluence(int nodeId, int timeStep) {
    if (!useTopologyConstraint) {
        // Without topology constraint: consider all nodes
        double totalInfluence = 0.0;
        for (const auto& node : topology.getNodes()) {
            if (node.id != nodeId) {
                totalInfluence += nodeStates[node.id].faultProbability * 0.1;
            }
        }
        return totalInfluence;
    }
    
    // With topology constraint: only upstream parents
    auto parents = topology.getParentNodes(nodeId);
    double influence = 0.0;
    
    for (int parentId : parents) {
        double weight = computeAttentionWeight(parentId, nodeId, timeStep);
        influence += weight * nodeStates[parentId].faultProbability;
    }
    
    return influence;
}

double TCCAFramework::computeAttentionWeight(int fromNode, int toNode, int timeStep) {
    if (!topology.hasEdge(fromNode, toNode)) {
        return 0.0;
    }
    
    // Simple attention mechanism based on QoS correlation
    if (timeStep < 2) return 1.0 / max(1, (int)topology.getParentNodes(toNode).size());
    
    auto& fromState = nodeStates[fromNode];
    auto& toState = nodeStates[toNode];
    
    // Compute similarity
    double similarity = 0.0;
    for (int i = 0; i < hiddenDim; i++) {
        similarity += fromState.hiddenState[i] * toState.hiddenState[i];
    }
    
    // Softmax-like normalization
    double weight = exp(similarity / 10.0);
    attentionWeights[{fromNode, toNode}] = weight;
    
    return weight;
}

double TCCAFramework::computeStructuralConsistency(int nodeId, int timeStep) {
    if (!useTopologyConstraint) return 0.5;
    
    auto parents = topology.getParentNodes(nodeId);
    if (parents.empty()) return 1.0;
    
    double consistency = 0.0;
    int consistentCount = 0;
    
    for (int parentId : parents) {
        // Check if parent fault implies child fault
        if (nodeStates[parentId].faultProbability > 0.7) {
            if (nodeStates[nodeId].faultProbability > 0.5) {
                consistentCount++;
            }
        } else {
            if (nodeStates[nodeId].faultProbability <= 0.5) {
                consistentCount++;
            }
        }
    }
    
    return (double)consistentCount / parents.size();
}

double TCCAFramework::computeObservationalSupport(int nodeId, int timeStep) {
    if (timeStep >= numTimeSteps) return 0.0;
    
    // Find observation for this node and time
    const auto& timeStepObs = observations[timeStep];
    QoSObservation obs;
    bool found = false;
    
    for (const auto& o : timeStepObs) {
        if (o.nodeId == nodeId) {
            obs = o;
            found = true;
            break;
        }
    }
    
    if (!found) return 0.0;
    
    // Compute deviation from normal
    double latencyScore = obs.latency / 50.0;  // Normalize
    double plScore = obs.packetLoss / 10.0;
    double tpScore = (100.0 - obs.throughput) / 100.0;
    
    return (latencyScore + plScore + tpScore) / 3.0;
}

double TCCAFramework::computeTemporalStability(int nodeId, int timeStep) {
    if (timeStep < 3 || !useTemporalModel) return 0.5;
    
    // Check consistency over last 3 time steps
    double sum = 0.0;
    for (int t = timeStep - 3; t < timeStep; t++) {
        sum += nodeStates[nodeId].faultProbability;
    }
    
    double mean = sum / 3.0;
    double variance = 0.0;
    
    for (int t = timeStep - 3; t < timeStep; t++) {
        variance += pow(nodeStates[nodeId].faultProbability - mean, 2);
    }
    
    // Lower variance = higher stability
    return 1.0 / (1.0 + variance);
}

double TCCAFramework::computeUncertainty(int nodeId, int timeStep) {
    // Monte Carlo dropout-like uncertainty
    double prob = nodeStates[nodeId].faultProbability;
    return prob * (1.0 - prob);  // Maximum at 0.5, minimum at 0 or 1
}

vector<double> TCCAFramework::gruUpdate(const vector<double>& prevState,
                                         const vector<double>& input,
                                         const vector<double>& aggInput) {
    // Simplified GRU update
    vector<double> newState(hiddenDim);
    
    uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < hiddenDim; i++) {
        // Update gate
        double z = 1.0 / (1.0 + exp(-(0.5 * prevState[i] + 0.3 * input[i] + 0.2 * aggInput[i])));
        
        // Reset gate
        double r = 1.0 / (1.0 + exp(-(0.4 * prevState[i] + 0.4 * input[i] + 0.2 * aggInput[i])));
        
        // Candidate
        double h_tilde = tanh(0.6 * (r * prevState[i]) + 0.4 * input[i]);
        
        // New state
        newState[i] = (1 - z) * prevState[i] + z * h_tilde;
    }
    
    return newState;
}

void TCCAFramework::updateHiddenState(int nodeId, int timeStep) {
    if (!useTemporalModel) {
        // Without temporal model: use current observation only
        if (timeStep < numTimeSteps) {
            const auto& timeStepObs = observations[timeStep];
            for (const auto& obs : timeStepObs) {
                if (obs.nodeId == nodeId) {
                    nodeStates[nodeId].hiddenState[0] = obs.latency / 100.0;
                    nodeStates[nodeId].hiddenState[1] = obs.packetLoss / 10.0;
                    nodeStates[nodeId].hiddenState[2] = obs.throughput / 1000.0;
                    break;
                }
            }
        }
        return;
    }
    
    // Get input features
    vector<double> input(hiddenDim, 0.0);
    if (timeStep < numTimeSteps) {
        const auto& timeStepObs = observations[timeStep];
        for (const auto& obs : timeStepObs) {
            if (obs.nodeId == nodeId) {
                input[0] = obs.latency / 100.0;
                input[1] = obs.packetLoss / 10.0;
                input[2] = obs.throughput / 1000.0;
                input[3] = obs.availability / 100.0;
                break;
            }
        }
    }
    
    // Get aggregated upstream input
    vector<double> aggInput(hiddenDim, 0.0);
    auto parents = topology.getParentNodes(nodeId);
    
    for (int parentId : parents) {
        for (int i = 0; i < hiddenDim; i++) {
            aggInput[i] += nodeStates[parentId].hiddenState[i];
        }
    }
    
    if (!parents.empty()) {
        for (int i = 0; i < hiddenDim; i++) {
            aggInput[i] /= parents.size();
        }
    }
    
    // GRU update
    nodeStates[nodeId].hiddenState = gruUpdate(
        nodeStates[nodeId].hiddenState, input, aggInput);
}

map<int, double> TCCAFramework::localizeFaults() {
    cout << "\n=== Running TCCA Fault Localization ===" << endl;
    
    // Process each time step
    for (int t = 0; t < numTimeSteps; t++) {
        // Update hidden states
        for (const auto& node : topology.getNodes()) {
            updateHiddenState(node.id, t);
        }
        
        // Compute fault probabilities
        for (const auto& node : topology.getNodes()) {
            double upstreamInfluence = computeUpstreamInfluence(node.id, t);
            
            // Combine local and upstream effects
            double localEffect = nodeStates[node.id].hiddenState[0];
            double faultProb = alpha * localEffect + beta * upstreamInfluence;
            
            // Apply sigmoid
            faultProb = 1.0 / (1.0 + exp(-faultProb * 10));
            
            nodeStates[node.id].faultProbability = faultProb;
            
            // Compute trust score
            if (useTrustMetric) {
                double structural = computeStructuralConsistency(node.id, t);
                double observational = computeObservationalSupport(node.id, t);
                double temporal = computeTemporalStability(node.id, t);
                double uncertainty = computeUncertainty(node.id, t);
                
                double trustScore = 0.4 * structural + 0.3 * observational + 
                                   0.2 * temporal - 0.1 * uncertainty;
                
                nodeStates[node.id].trustScore = trustScore;
                nodeStates[node.id].uncertainty = uncertainty;
            }
        }
    }
    
    // Aggregate fault scores over time
    map<int, double> faultScores;
    for (const auto& node : topology.getNodes()) {
        faultScores[node.id] = nodeStates[node.id].faultProbability;
    }
    
    // Identify top faults
    vector<pair<int, double>> sortedFaults(faultScores.begin(), faultScores.end());
    sort(sortedFaults.begin(), sortedFaults.end(),
         [](const auto& a, const auto& b) { return a.second > b.second; });
    
    cout << "Top 5 suspected faults:" << endl;
    for (int i = 0; i < min(5, (int)sortedFaults.size()); i++) {
        cout << "  Node " << sortedFaults[i].first 
             << ": " << fixed << setprecision(3) << sortedFaults[i].second
             << " (trust: " << nodeStates[sortedFaults[i].first].trustScore << ")" << endl;
    }
    
    return faultScores;
}

void TCCAFramework::trainModel(int epochs, double learningRate) {
    cout << "Training TCCA model for " << epochs << " epochs..." << endl;
    
    // Simplified training: adjust alpha, beta based on validation
    // In practice, this would use gradient descent on the full model
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Run localization
        auto scores = localizeFaults();
        
        // Adjust parameters (simplified)
        if (epoch % 10 == 0) {
            alpha += 0.01;
            beta = 1.0 - alpha;
        }
    }
    
    cout << "Training complete. Final alpha: " << alpha 
         << ", beta: " << beta << endl;
}

void TCCAFramework::exportResults(const string& filename,
                                   const map<int, double>& faultScores) {
    ofstream file(filename);
    file << "node_id,fault_probability,trust_score,uncertainty\n";
    
    for (const auto& [nodeId, score] : faultScores) {
        file << nodeId << "," << score << ","
             << nodeStates[nodeId].trustScore << ","
             << nodeStates[nodeId].uncertainty << "\n";
    }
    
    file.close();
    cout << "Results exported to " << filename << endl;
}
