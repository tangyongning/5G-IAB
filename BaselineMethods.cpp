// BaselineMethods.cpp
#include "BaselineMethods.h"
#include <iostream>
#include <cmath>
#include <algorithm>

BaselineMethods::BaselineMethods(NetworkTopology& topo,
                                  const vector<vector<QoSObservation>>& obs)
    : topology(topo), observations(obs) {}

map<int, double> BaselineMethods::computeCorrelationScores() {
    map<int, double> scores;
    
    // Compute correlation between each node's QoS and overall degradation
    for (const auto& node : topology.getNodes()) {
        vector<double> latencies;
        vector<double> globalDegradation;
        
        for (const auto& timeStep : observations) {
            double nodeLatency = 0;
            double totalLatency = 0;
            int count = 0;
            
            for (const auto& obs : timeStep) {
                totalLatency += obs.latency;
                count++;
                if (obs.nodeId == node.id) {
                    nodeLatency = obs.latency;
                }
            }
            
            latencies.push_back(nodeLatency);
            globalDegradation.push_back(totalLatency / count);
        }
        
        // Compute Pearson correlation
        double meanX = accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
        double meanY = accumulate(globalDegradation.begin(), globalDegradation.end(), 0.0) / globalDegradation.size();
        
        double num = 0, denX = 0, denY = 0;
        for (size_t i = 0; i < latencies.size(); i++) {
            double dx = latencies[i] - meanX;
            double dy = globalDegradation[i] - meanY;
            num += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
        }
        
        double corr = (denX > 0 && denY > 0) ? num / sqrt(denX * denY) : 0;
        scores[node.id] = (corr + 1) / 2;  // Normalize to [0, 1]
    }
    
    return scores;
}

map<int, double> BaselineMethods::computeMLScores() {
    map<int, double> scores;
    
    // Simple neural network simulation
    for (const auto& node : topology.getNodes()) {
        double avgLatency = 0, avgPL = 0, avgTP = 0;
        int count = 0;
        
        for (const auto& timeStep : observations) {
            for (const auto& obs : timeStep) {
                if (obs.nodeId == node.id) {
                    avgLatency += obs.latency;
                    avgPL += obs.packetLoss;
                    avgTP += obs.throughput;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            avgLatency /= count;
            avgPL /= count;
            avgTP /= count;
        }
        
        // Simple scoring function (simulating NN output)
        double score = 0.4 * (avgLatency / 100.0) + 
                      0.3 * (avgPL / 10.0) + 
                      0.3 * (1.0 - avgTP / 1000.0);
        
        scores[node.id] = min(1.0, score);
    }
    
    return scores;
}

map<int, double> BaselineMethods::computeTopologyScores() {
    map<int, double> scores;
    
    // Topology-aware: consider node degree and position
    for (const auto& node : topology.getNodes()) {
        auto parents = topology.getParentNodes(node.id);
        auto children = topology.getChildNodes(node.id);
        
        double degreeCentrality = (parents.size() + children.size()) / 
                                  (double)topology.getNodes().size();
        
        // Compute average QoS degradation
        double avgDegradation = 0;
        int count = 0;
        
        for (const auto& timeStep : observations) {
            for (const auto& obs : timeStep) {
                if (obs.nodeId == node.id) {
                    avgDegradation += obs.latency / 100.0 + obs.packetLoss / 10.0;
                    count++;
                }
            }
        }
        
        if (count > 0) {
            avgDegradation /= count;
        }
        
        // Combine topology and QoS
        scores[node.id] = 0.5 * degreeCentrality + 0.5 * avgDegradation;
    }
    
    return scores;
}

map<int, double> BaselineMethods::correlationBased() {
    cout << "Running correlation-based fault localization..." << endl;
    return computeCorrelationScores();
}

map<int, double> BaselineMethods::mlBased() {
    cout << "Running ML-based fault localization..." << endl;
    return computeMLScores();
}

map<int, double> BaselineMethods::topologyAware() {
    cout << "Running topology-aware fault localization..." << endl;
    return computeTopologyScores();
}
