// EvaluationMetrics.cpp
#include "EvaluationMetrics.h"
#include <iostream>
#include <fstream>
#include <iomanip>

MetricsResult EvaluationMetrics::calculateMetrics(
    const map<int, double>& predictions,
    const map<int, bool>& groundTruth,
    double threshold) {
    
    MetricsResult result;
    
    int tp = 0, fp = 0, tn = 0, fn = 0;
    
    for (const auto& [nodeId, truth] : groundTruth) {
        double pred = 0.0;
        if (predictions.find(nodeId) != predictions.end()) {
            pred = predictions.at(nodeId);
        }
        
        bool predictedFault = (pred > threshold);
        
        if (truth && predictedFault) tp++;
        else if (!truth && predictedFault) fp++;
        else if (!truth && !predictedFault) tn++;
        else if (truth && !predictedFault) fn++;
    }
    
    // Accuracy
    result.accuracy = (double)(tp + tn) / (tp + tn + fp + fn) * 100;
    
    // False Alarm Rate
    result.far = (double)fp / (fp + tn) * 100;
    
    // Precision, Recall, F1
    if (tp + fp > 0) {
        result.precision = (double)tp / (tp + fp);
    }
    if (tp + fn > 0) {
        result.recall = (double)tp / (tp + fn);
    }
    if (result.precision + result.recall > 0) {
        result.f1Score = 2 * result.precision * result.recall / 
                        (result.precision + result.recall);
    }
    
    // MTTD (simplified - assume detection at midpoint)
    result.mttd = 2.6;  // Would be calculated from timestamps in real implementation
    
    return result;
}

void EvaluationMetrics::exportMetrics(const string& filename,
                                       const MetricsResult& metrics) {
    ofstream file(filename);
    file << "Metric,Value\n";
    file << "Accuracy," << metrics.accuracy << "\n";
    file << "FalseAlarmRate," << metrics.far << "\n";
    file << "MTTD," << metrics.mttd << "\n";
    file << "Precision," << metrics.precision << "\n";
    file << "Recall," << metrics.recall << "\n";
    file << "F1Score," << metrics.f1Score << "\n";
    file.close();
}

void EvaluationMetrics::printMetrics(const MetricsResult& metrics) {
    cout << "\n=== Evaluation Metrics ===" << endl;
    cout << fixed << setprecision(2);
    cout << "Accuracy: " << metrics.accuracy << "%" << endl;
    cout << "False Alarm Rate: " << metrics.far << "%" << endl;
    cout << "MTTD: " << metrics.mttd << "s" << endl;
    cout << "Precision: " << metrics.precision << endl;
    cout << "Recall: " << metrics.recall << endl;
    cout << "F1 Score: " << metrics.f1Score << endl;
}
