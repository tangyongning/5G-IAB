// EvaluationMetrics.h
#ifndef EVALUATION_METRICS_H
#define EVALUATION_METRICS_H

#include <map>
#include <string>

struct MetricsResult {
    double accuracy;
    double far;  // False Alarm Rate
    double mttd; // Mean Time to Detect
    double precision;
    double recall;
    double f1Score;
    
    MetricsResult() : accuracy(0), far(0), mttd(0), 
                      precision(0), recall(0), f1Score(0) {}
};

class EvaluationMetrics {
public:
    static MetricsResult calculateMetrics(const map<int, double>& predictions,
                                          const map<int, bool>& groundTruth,
                                          double threshold = 0.7);
    
    static void exportMetrics(const string& filename,
                              const MetricsResult& metrics);
    
    static void printMetrics(const MetricsResult& metrics);
};

#endif
