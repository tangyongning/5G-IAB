// main_simulation.cpp
#include <iostream>
#include <vector>
#include <map>
#include <random>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include "NetworkTopology.h"
#include "FaultModel.h"
#include "QoSMetrics.h"
#include "TCCAFramework.h"
#include "BaselineMethods.h"
#include "EvaluationMetrics.h"

using namespace std;

class SimulationRunner {
private:
    int numNodes;
    int numFaults;
    double noiseLevel;
    int numTimeSteps;
    mt19937 rng;
    
    vector<vector<double>> accuracyResults;
    vector<vector<double>> farResults;
    vector<double> scalabilityAccuracy;
    vector<double> scalabilityRuntime;
    vector<vector<double>> robustnessResults;
    
public:
    SimulationRunner(int nodes = 100, int faults = 5, 
                     double noise = 0.05, int timeSteps = 100)
        : numNodes(nodes), numFaults(faults), 
          noiseLevel(noise), numTimeSteps(timeSteps) {
        rng.seed(chrono::steady_clock::now().time_since_epoch().count());
    }
    
    void runFaultLocalizationExperiment() {
        cout << "=== Fault Localization Performance Experiment ===" << endl;
        
        vector<string> methods = {"Correlation", "ML", "Topology", "TCCA"};
        vector<vector<double>> accuracies(4, vector<double>(10, 0));
        vector<vector<double>> fars(4, vector<double>(10, 0));
        vector<vector<double>> mttds(4, vector<double>(10, 0));
        
        for (int run = 0; run < 10; run++) {
            cout << "Running experiment " << run + 1 << "/10..." << endl;
            
            // Generate network topology
            NetworkTopology topology(numNodes);
            topology.generateHierarchicalTopology();
            
            // Inject faults
            FaultModel faultModel(topology, numFaults);
            faultModel.injectRandomFaults();
            
            // Generate QoS observations
            QoSMetrics qosMetrics(topology, faultModel, noiseLevel);
            auto observations = qosMetrics.generateTimeSeries(numTimeSteps);
            
            // Run baseline methods
            BaselineMethods baselines(topology, observations);
            
            // Correlation-based
            auto corrResults = baselines.correlationBased();
            auto corrMetrics = EvaluationMetrics::calculateMetrics(
                corrResults, faultModel.getGroundTruth());
            accuracies[0][run] = corrMetrics.accuracy;
            fars[0][run] = corrMetrics.far;
            mttds[0][run] = corrMetrics.mttd;
            
            // ML-based
            auto mlResults = baselines.mlBased();
            auto mlMetrics = EvaluationMetrics::calculateMetrics(
                mlResults, faultModel.getGroundTruth());
            accuracies[1][run] = mlMetrics.accuracy;
            fars[1][run] = mlMetrics.far;
            mttds[1][run] = mlMetrics.mttd;
            
            // Topology-aware
            auto topoResults = baselines.topologyAware();
            auto topoMetrics = EvaluationMetrics::calculateMetrics(
                topoResults, faultModel.getGroundTruth());
            accuracies[2][run] = topoMetrics.accuracy;
            fars[2][run] = topoMetrics.far;
            mttds[2][run] = topoMetrics.mttd;
            
            // TCCA (Proposed)
            TCCAFramework tcca(topology, observations);
            auto tccaResults = tcca.localizeFaults();
            auto tccaMetrics = EvaluationMetrics::calculateMetrics(
                tccaResults, faultModel.getGroundTruth());
            accuracies[3][run] = tccaMetrics.accuracy;
            fars[3][run] = tccaMetrics.far;
            mttds[3][run] = tccaMetrics.mttd;
        }
        
        // Calculate averages and export
        exportFaultLocalizationResults(methods, accuracies, fars, mttds);
    }
    
    void runScalabilityExperiment() {
        cout << "\n=== Scalability Experiment ===" << endl;
        
        vector<int> networkSizes = {50, 200, 400, 600, 800, 1000};
        vector<double> tccaAccuracy, tccaRuntime;
        vector<double> mlAccuracy, mlRuntime;
        
        for (int size : networkSizes) {
            cout << "Testing network size: " << size << " nodes..." << endl;
            
            NetworkTopology topology(size);
            topology.generateHierarchicalTopology();
            
            FaultModel faultModel(topology, size / 20);
            faultModel.injectRandomFaults();
            
            QoSMetrics qosMetrics(topology, faultModel, noiseLevel);
            auto observations = qosMetrics.generateTimeSeries(numTimeSteps);
            
            // TCCA
            auto start = chrono::high_resolution_clock::now();
            TCCAFramework tcca(topology, observations);
            auto tccaResults = tcca.localizeFaults();
            auto end = chrono::high_resolution_clock::now();
            
            double runtime = chrono::duration<double>(end - start).count();
            auto metrics = EvaluationMetrics::calculateMetrics(
                tccaResults, faultModel.getGroundTruth());
            
            tccaAccuracy.push_back(metrics.accuracy);
            tccaRuntime.push_back(runtime);
            
            // ML baseline
            start = chrono::high_resolution_clock::now();
            BaselineMethods baselines(topology, observations);
            auto mlResults = baselines.mlBased();
            end = chrono::high_resolution_clock::now();
            
            runtime = chrono::duration<double>(end - start).count();
            metrics = EvaluationMetrics::calculateMetrics(
                mlResults, faultModel.getGroundTruth());
            
            mlAccuracy.push_back(metrics.accuracy);
            mlRuntime.push_back(runtime);
        }
        
        exportScalabilityResults(networkSizes, tccaAccuracy, tccaRuntime,
                                 mlAccuracy, mlRuntime);
    }
    
    void runRobustnessExperiment() {
        cout << "\n=== Robustness to Noise Experiment ===" << endl;
        
        vector<double> noiseLevels = {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3};
        vector<vector<double>> results(4, vector<double>(noiseLevels.size()));
        
        for (size_t i = 0; i < noiseLevels.size(); i++) {
            double noise = noiseLevels[i];
            cout << "Testing noise level: " << noise * 100 << "%" << endl;
            
            vector<double> runAccuracies(4, 0);
            
            for (int run = 0; run < 5; run++) {
                NetworkTopology topology(numNodes);
                topology.generateHierarchicalTopology();
                
                FaultModel faultModel(topology, numFaults);
                faultModel.injectRandomFaults();
                
                QoSMetrics qosMetrics(topology, faultModel, noise);
                auto observations = qosMetrics.generateTimeSeries(numTimeSteps);
                
                // Correlation
                BaselineMethods baselines(topology, observations);
                auto corrResults = baselines.correlationBased();
                auto corrMetrics = EvaluationMetrics::calculateMetrics(
                    corrResults, faultModel.getGroundTruth());
                runAccuracies[0] += corrMetrics.accuracy / 5;
                
                // ML
                auto mlResults = baselines.mlBased();
                auto mlMetrics = EvaluationMetrics::calculateMetrics(
                    mlResults, faultModel.getGroundTruth());
                runAccuracies[1] += mlMetrics.accuracy / 5;
                
                // Topology
                auto topoResults = baselines.topologyAware();
                auto topoMetrics = EvaluationMetrics::calculateMetrics(
                    topoResults, faultModel.getGroundTruth());
                runAccuracies[2] += topoMetrics.accuracy / 5;
                
                // TCCA
                TCCAFramework tcca(topology, observations);
                auto tccaResults = tcca.localizeFaults();
                auto tccaMetrics = EvaluationMetrics::calculateMetrics(
                    tccaResults, faultModel.getGroundTruth());
                runAccuracies[3] += tccaMetrics.accuracy / 5;
            }
            
            for (int j = 0; j < 4; j++) {
                results[j][i] = runAccuracies[j];
            }
        }
        
        exportRobustnessResults(noiseLevels, results);
    }
    
    void runAblationStudy() {
        cout << "\n=== Ablation Study ===" << endl;
        
        NetworkTopology topology(numNodes);
        topology.generateHierarchicalTopology();
        
        FaultModel faultModel(topology, numFaults);
        faultModel.injectRandomFaults();
        
        QoSMetrics qosMetrics(topology, faultModel, noiseLevel);
        auto observations = qosMetrics.generateTimeSeries(numTimeSteps);
        
        vector<string> variants = {"Full TCCA", "w/o Topology", 
                                    "w/o Trust", "w/o Temporal"};
        vector<double> accuracies;
        
        // Full model
        TCCAFramework tccaFull(topology, observations, true, true, true);
        auto resultsFull = tccaFull.localizeFaults();
        auto metricsFull = EvaluationMetrics::calculateMetrics(
            resultsFull, faultModel.getGroundTruth());
        accuracies.push_back(metricsFull.accuracy);
        
        // Without topology constraint
        TCCAFramework tccaNoTopo(topology, observations, false, true, true);
        auto resultsNoTopo = tccaNoTopo.localizeFaults();
        auto metricsNoTopo = EvaluationMetrics::calculateMetrics(
            resultsNoTopo, faultModel.getGroundTruth());
        accuracies.push_back(metricsNoTopo.accuracy);
        
        // Without trust metric
        TCCAFramework tccaNoTrust(topology, observations, true, false, true);
        auto resultsNoTrust = tccaNoTrust.localizeFaults();
        auto metricsNoTrust = EvaluationMetrics::calculateMetrics(
            resultsNoTrust, faultModel.getGroundTruth());
        accuracies.push_back(metricsNoTrust.accuracy);
        
        // Without temporal model
        TCCAFramework tccaNoTemp(topology, observations, true, true, false);
        auto resultsNoTemp = tccaNoTemp.localizeFaults();
        auto metricsNoTemp = EvaluationMetrics::calculateMetrics(
            resultsNoTemp, faultModel.getGroundTruth());
        accuracies.push_back(metricsNoTemp.accuracy);
        
        exportAblationResults(variants, accuracies);
    }
    
    void runQoSImprovementExperiment() {
        cout << "\n=== QoS Improvement Experiment ===" << endl;
        
        NetworkTopology topology(numNodes);
        topology.generateHierarchicalTopology();
        
        FaultModel faultModel(topology, numFaults);
        faultModel.injectRandomFaults();
        
        QoSMetrics qosMetrics(topology, faultModel, noiseLevel);
        auto baselineQoS = qosMetrics.getBaselineQoSDegradation();
        
        vector<string> methods = {"Correlation", "ML", "Topology", "TCCA"};
        vector<vector<double>> qosImprovements(3, vector<double>(4, 0));
        
        for (int run = 0; run < 10; run++) {
            QoSMetrics qosMetricsRun(topology, faultModel, noiseLevel);
            auto observations = qosMetricsRun.generateTimeSeries(numTimeSteps);
            
            BaselineMethods baselines(topology, observations);
            
            // Correlation
            auto corrResults = baselines.correlationBased();
            auto corrQoS = qosMetricsRun.calculatePostMitigationQoS(corrResults);
            qosImprovements[0][0] += (baselineQoS.latency - corrQoS.latency) / 
                                     baselineQoS.latency * 100 / 10;
            qosImprovements[1][0] += (baselineQoS.packetLoss - corrQoS.packetLoss) / 
                                     baselineQoS.packetLoss * 100 / 10;
            qosImprovements[2][0] += (corrQoS.throughput - baselineQoS.throughput) / 
                                     baselineQoS.throughput * 100 / 10;
            
            // ML
            auto mlResults = baselines.mlBased();
            auto mlQoS = qosMetricsRun.calculatePostMitigationQoS(mlResults);
            qosImprovements[0][1] += (baselineQoS.latency - mlQoS.latency) / 
                                     baselineQoS.latency * 100 / 10;
            qosImprovements[1][1] += (baselineQoS.packetLoss - mlQoS.packetLoss) / 
                                     baselineQoS.packetLoss * 100 / 10;
            qosImprovements[2][1] += (mlQoS.throughput - baselineQoS.throughput) / 
                                     baselineQoS.throughput * 100 / 10;
            
            // Topology
            auto topoResults = baselines.topologyAware();
            auto topoQoS = qosMetricsRun.calculatePostMitigationQoS(topoResults);
            qosImprovements[0][2] += (baselineQoS.latency - topoQoS.latency) / 
                                     baselineQoS.latency * 100 / 10;
            qosImprovements[1][2] += (baselineQoS.packetLoss - topoQoS.packetLoss) / 
                                     baselineQoS.packetLoss * 100 / 10;
            qosImprovements[2][2] += (topoQoS.throughput - baselineQoS.throughput) / 
                                     baselineQoS.throughput * 100 / 10;
            
            // TCCA
            TCCAFramework tcca(topology, observations);
            auto tccaResults = tcca.localizeFaults();
            auto tccaQoS = qosMetricsRun.calculatePostMitigationQoS(tccaResults);
            qosImprovements[0][3] += (baselineQoS.latency - tccaQoS.latency) / 
                                     baselineQoS.latency * 100 / 10;
            qosImprovements[1][3] += (baselineQoS.packetLoss - tccaQoS.packetLoss) / 
                                     baselineQoS.packetLoss * 100 / 10;
            qosImprovements[2][3] += (tccaQoS.throughput - baselineQoS.throughput) / 
                                     baselineQoS.throughput * 100 / 10;
        }
        
        exportQoSImprovementResults(methods, qosImprovements);
    }
    
private:
    void exportFaultLocalizationResults(vector<string>& methods,
                                        vector<vector<double>>& accuracies,
                                        vector<vector<double>>& fars,
                                        vector<vector<double>>& mttds) {
        ofstream file("results/fault_localization.csv");
        file << "Method,Avg_Accuracy(%),Avg_FAR(%),Avg_MTTD(s)\n";
        
        cout << "\n=== Fault Localization Results ===" << endl;
        cout << left << setw(15) << "Method" 
             << right << setw(15) << "Accuracy" 
             << setw(15) << "FAR" 
             << setw(15) << "MTTD" << endl;
        
        for (int i = 0; i < 4; i++) {
            double avgAcc = accumulate(accuracies[i].begin(), 
                                       accuracies[i].end(), 0.0) / 10;
            double avgFar = accumulate(fars[i].begin(), 
                                       fars[i].end(), 0.0) / 10;
            double avgMttd = accumulate(mttds[i].begin(), 
                                        mttds[i].end(), 0.0) / 10;
            
            file << methods[i] << "," << avgAcc << "," 
                 << avgFar << "," << avgMttd << "\n";
            
            cout << left << setw(15) << methods[i] 
                 << right << setw(15) << fixed << setprecision(1) << avgAcc
                 << setw(15) << avgFar
                 << setw(15) << avgMttd << endl;
        }
        file.close();
    }
    
    void exportScalabilityResults(vector<int>& sizes,
                                  vector<double>& tccaAcc,
                                  vector<double>& tccaRuntime,
                                  vector<double>& mlAcc,
                                  vector<double>& mlRuntime) {
        ofstream file("results/scalability.csv");
        file << "Network_Size,TCCA_Accuracy,TCCA_Runtime(s),ML_Accuracy,ML_Runtime(s)\n";
        
        cout << "\n=== Scalability Results ===" << endl;
        cout << left << setw(15) << "Size" 
             << right << setw(15) << "TCCA Acc" 
             << setw(15) << "TCCA Time"
             << setw(15) << "ML Acc"
             << setw(15) << "ML Time" << endl;
        
        for (size_t i = 0; i < sizes.size(); i++) {
            file << sizes[i] << "," << tccaAcc[i] << "," 
                 << tccaRuntime[i] << "," << mlAcc[i] << "," 
                 << mlRuntime[i] << "\n";
            
            cout << left << setw(15) << sizes[i]
                 << right << setw(15) << fixed << setprecision(1) << tccaAcc[i]
                 << setw(15) << setprecision(3) << tccaRuntime[i]
                 << setw(15) << setprecision(1) << mlAcc[i]
                 << setw(15) << setprecision(3) << mlRuntime[i] << endl;
        }
        file.close();
    }
    
    void exportRobustnessResults(vector<double>& noiseLevels,
                                 vector<vector<double>>& results) {
        ofstream file("results/robustness.csv");
        file << "Noise_Level,Correlation,ML,Topology,TCCA\n";
        
        cout << "\n=== Robustness Results ===" << endl;
        cout << left << setw(15) << "Noise %"
             << right << setw(15) << "Correlation"
             << setw(15) << "ML"
             << setw(15) << "Topology"
             << setw(15) << "TCCA" << endl;
        
        for (size_t i = 0; i < noiseLevels.size(); i++) {
            file << noiseLevels[i] << ",";
            cout << left << setw(15) << fixed << setprecision(0) 
                 << noiseLevels[i] * 100;
            
            for (int j = 0; j < 4; j++) {
                file << results[j][i];
                if (j < 3) file << ",";
                cout << right << setw(15) << setprecision(1) << results[j][i];
            }
            file << "\n";
            cout << endl;
        }
        file.close();
    }
    
    void exportAblationResults(vector<string>& variants,
                               vector<double>& accuracies) {
        ofstream file("results/ablation.csv");
        file << "Variant,Accuracy(%)\n";
        
        cout << "\n=== Ablation Study Results ===" << endl;
        cout << left << setw(20) << "Variant" 
             << right << setw(15) << "Accuracy" << endl;
        
        for (size_t i = 0; i < variants.size(); i++) {
            file << variants[i] << "," << accuracies[i] << "\n";
            cout << left << setw(20) << variants[i]
                 << right << setw(15) << fixed << setprecision(1) 
                 << accuracies[i] << endl;
        }
        file.close();
    }
    
    void exportQoSImprovementResults(vector<string>& methods,
                                     vector<vector<double>>& improvements) {
        ofstream file("results/qos_improvement.csv");
        file << "Method,Latency_Reduction(%),PacketLoss_Reduction(%),Throughput_Recovery(%)\n";
        
        cout << "\n=== QoS Improvement Results ===" << endl;
        cout << left << setw(15) << "Method"
             << right << setw(20) << "Latency Red"
             << setw(25) << "Packet Loss Red"
             << setw(25) << "Throughput Rec" << endl;
        
        for (int i = 0; i < 4; i++) {
            file << methods[i] << "," << improvements[0][i] << ","
                 << improvements[1][i] << "," << improvements[2][i] << "\n";
            
            cout << left << setw(15) << methods[i]
                 << right << setw(20) << fixed << setprecision(1) 
                 << improvements[0][i]
                 << setw(25) << improvements[1][i]
                 << setw(25) << improvements[2][i] << endl;
        }
        file.close();
    }
};

int main() {
    cout << "=== Globecom 2026: TCCA Framework Simulation ===" << endl;
    cout << "Topology-Constrained Fault Localization in 5G/6G Networks\n" << endl;
    
    // Create results directory
    system("mkdir -p results");
    
    SimulationRunner simulator(100, 5, 0.05, 100);
    
    // Run all experiments
    simulator.runFaultLocalizationExperiment();
    simulator.runScalabilityExperiment();
    simulator.runRobustnessExperiment();
    simulator.runAblationStudy();
    simulator.runQoSImprovementExperiment();
    
    cout << "\n=== All simulations completed ===" << endl;
    cout << "Results saved in results/ directory" << endl;
    
    return 0;
}
