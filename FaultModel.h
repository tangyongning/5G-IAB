// FaultModel.h
#ifndef FAULT_MODEL_H
#define FAULT_MODEL_H

#include "NetworkTopology.h"
#include <vector>
#include <map>
#include <random>

struct Fault {
    int nodeId;
    int faultType;  // 0: node_failure, 1: congestion, 2: link_degradation, 3: beam_failure
    double severity;  // 0.0 to 1.0
    int startTime;
    int duration;
    bool active;
    
    Fault(int id, int type, double sev, int start, int dur)
        : nodeId(id), faultType(type), severity(sev), 
          startTime(start), duration(dur), active(true) {}
};

struct QoSDegradation {
    double latencyIncrease;
    double packetLossIncrease;
    double throughputDecrease;
    double availabilityDecrease;
    
    QoSDegradation(double lat = 0, double pl = 0, double tp = 0, double av = 0)
        : latencyIncrease(lat), packetLossIncrease(pl), 
          throughputDecrease(tp), availabilityDecrease(av) {}
};

class FaultModel {
private:
    NetworkTopology& topology;
    vector<Fault> faults;
    map<int, bool> groundTruth;  // nodeId -> isFaulty
    mt19937 rng;
    
    QoSDegradation calculateFaultImpact(const Fault& fault);
    void propagateFault(const Fault& fault, map<int, QoSDegradation>& degradation);
    
public:
    FaultModel(NetworkTopology& topo, int numFaults = 5);
    
    void injectRandomFaults();
    void injectCascadingFaults();
    void injectIABSpecificFaults();
    
    void updateFaults(int currentTime);
    
    const map<int, bool>& getGroundTruth() const { return groundTruth; }
    const vector<Fault>& getFaults() const { return faults; }
    
    map<int, QoSDegradation> getCurrentDegradation(int currentTime);
    
    void printFaults() const;
};

#endif
