// BaselineMethods.h
#ifndef BASELINE_METHODS_H
#define BASELINE_METHODS_H

#include "NetworkTopology.h"
#include "QoSMetrics.h"
#include <vector>
#include <map>

class BaselineMethods {
private:
    NetworkTopology& topology;
    const vector<vector<QoSObservation>>& observations;
    
    // Correlation-based method
    map<int, double> computeCorrelationScores();
    
    // ML-based method (simple neural network)
    map<int, double> computeMLScores();
    
    // Topology-aware heuristic
    map<int, double> computeTopologyScores();
    
public:
    BaselineMethods(NetworkTopology& topo,
                    const vector<vector<QoSObservation>>& obs);
    
    map<int, double> correlationBased();
    map<int, double> mlBased();
    map<int, double> topologyAware();
};

#endif
