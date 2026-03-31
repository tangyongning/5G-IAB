// NetworkTopology.h
#ifndef NETWORK_TOPOLOGY_H
#define NETWORK_TOPOLOGY_H

#include <vector>
#include <map>
#include <random>
#include <string>

using namespace std;

struct Edge {
    int from;
    int to;
    double capacity;
    double latency;
    double failureProb;
    
    Edge(int f, int t, double cap = 100.0, double lat = 1.0, double fp = 0.01)
        : from(f), to(t), capacity(cap), latency(lat), failureProb(fp) {}
};

struct Node {
    int id;
    string type;  // "core", "edge", "base_station", "iab_donor", "iab_node", "ue"
    double processingCapacity;
    vector<int> parentNodes;
    vector<int> childNodes;
    
    Node(int i, string t = "edge", double cap = 1000.0)
        : id(i), type(t), processingCapacity(cap) {}
};

class NetworkTopology {
private:
    int numNodes;
    vector<Node> nodes;
    vector<Edge> edges;
    map<pair<int,int>, double> adjacencyMatrix;
    mt19937 rng;
    
public:
    NetworkTopology(int n);
    
    void generateHierarchicalTopology();
    void generateIABTopology();
    void generateRandomTopology(double connectionProb = 0.1);
    
    const vector<Node>& getNodes() const { return nodes; }
    const vector<Edge>& getEdges() const { return edges; }
    const map<pair<int,int>, double>& getAdjacencyMatrix() const { 
        return adjacencyMatrix; 
    }
    
    vector<int> getParentNodes(int nodeId) const;
    vector<int> getChildNodes(int nodeId) const;
    bool hasEdge(int from, int to) const;
    double getEdgeWeight(int from, int to) const;
    
    void exportToGraphML(const string& filename);
    void printTopology() const;
};

#endif
