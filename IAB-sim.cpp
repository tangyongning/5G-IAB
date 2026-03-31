// iab-fault-localization.cc
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"

using namespace ns3;
using namespace ns3::nr;

NS_LOG_COMPONENT_DEFINE("IabFaultLocalization");

class IabFaultLocalizationSimulator {
private:
    NodeContainer iabDonorNodes;
    NodeContainer iabNodes;
    NodeContainer ueNodes;
    
    // IAB topology parameters
    uint32_t m_numDonors;
    uint32_t m_numIabNodes;
    uint32_t m_numUePerIabNode;
    double m_iabNodeSpacing;
    
    // Fault injection parameters
    bool m_faultInjectionEnabled;
    Time m_faultStartTime;
    Time m_faultDuration;
    uint32_t m_faultNodeId;
    
    // Telemetry collection
    std::map<uint32_t, std::vector<double>> m_latencyTrace;
    std::map<uint32_t, std::vector<double>> m_throughputTrace;
    std::map<uint32_t, std::vector<double>> m_packetLossTrace;
    std::map<uint32_t, std::vector<double>> m_harqTrace;
    
public:
    IabFaultLocalizationSimulator(uint32_t donors, uint32_t iabNodes, 
                                   uint32_t uePerIab)
        : m_numDonors(donors),
          m_numIabNodes(iabNodes),
          m_numUePerIabNode(uePerIab),
          m_iabNodeSpacing(100.0),
          m_faultInjectionEnabled(false),
          m_faultStartTime(Seconds(1.0)),
          m_faultDuration(Seconds(5.0)),
          m_faultNodeId(0) {}
    
    void ConfigureIabTopology();
    void InjectFault(uint32_t nodeId, Time startTime, Time duration);
    void CollectTelemetryTraces();
    void ExportTracesToCsv(const std::string& filename);
};

void IabFaultLocalizationSimulator::ConfigureIabTopology()
{
    NS_LOG_FUNCTION(this);
    
    // Create IAB Donor (connected to core network via fiber)
    iabDonorNodes.Create(m_numDonors);
    
    // Create IAB Nodes (multi-hop wireless backhaul)
    iabNodes.Create(m_numIabNodes);
    
    // Create UEs attached to IAB nodes
    ueNodes.Create(m_numIabNodes * m_numUePerIabNode);
    
    // Configure mobility for IAB topology
    MobilityHelper mobility;
    mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                  "MinX", DoubleValue(0.0),
                                  "MinY", DoubleValue(0.0),
                                  "DeltaX", DoubleValue(m_iabNodeSpacing),
                                  "DeltaY", DoubleValue(m_iabNodeSpacing),
                                  "GridWidth", UintegerValue(m_numIabNodes),
                                  "LayoutType", StringValue("RowFirst"));
    
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(iabDonorNodes);
    mobility.Install(iabNodes);
    
    // UEs with random mobility near IAB nodes
    mobility.SetPositionAllocator("ns3::RandomRectanglePositionAllocator",
                                  "X", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"),
                                  "Y", StringValue("ns3::UniformRandomVariable[Min=0.0|Max=100.0]"));
    mobility.Install(ueNodes);
    
    // Configure 5G NR with IAB support
    Config::SetDefault("ns3::LteRlcUm::MaxTxBufferSize", UintegerValue(999999));
    Config::SetDefault("ns3::LteEnbRrc::SrsPeriodicity", UintegerValue(320));
    
    // IAB-specific: Configure backhaul links
    Config::SetDefault("ns3::NrGnbNetDevice::IabEnabled", BooleanValue(true));
    Config::SetDefault("ns3::NrGnbNetDevice::BackhaulLinkType", 
                       StringValue("wireless")); // or "wired"
    
    // Half-duplex constraint for IAB nodes
    Config::SetDefault("ns3::NrGnbPhy::HalfDuplex", BooleanValue(true));
    
    // mmWave configuration for backhaul (FR2)
    Config::SetDefault("ns3::NrHelper::CarrierFrequency", 
                       DoubleValue(28e9)); // 28 GHz
    Config::SetDefault("ns3::NrHelper::Bandwidth", 
                       DoubleValue(100e6)); // 100 MHz
    
    NS_LOG_INFO("IAB topology configured with " << m_numIabNodes << " IAB nodes");
}

void IabFaultLocalizationSimulator::InjectFault(uint32_t nodeId, 
                                                  Time startTime, 
                                                  Time duration)
{
    NS_LOG_FUNCTION(this << nodeId << startTime << duration);
    
    m_faultInjectionEnabled = true;
    m_faultNodeId = nodeId;
    m_faultStartTime = startTime;
    m_faultDuration = duration;
    
    // Schedule fault events
    Simulator::Schedule(startTime, &IabFaultLocalizationSimulator::StartFault, 
                       this, nodeId);
    Simulator::Schedule(startTime + duration, 
                       &IabFaultLocalizationSimulator::RecoverFromFault, 
                       this, nodeId);
}

void IabFaultLocalizationSimulator::StartFault(uint32_t nodeId)
{
    NS_LOG_FUNCTION(this << nodeId);
    
    // Type 1: Node failure (complete shutdown)
    // Config::Set("/NodeList/" + std::to_string(nodeId) + 
    //            "/*/DeviceList/*/TxQueue/MaxSize", 
    //            StringValue("0p")); // Drop all packets
    
    // Type 2: Wireless blockage (beam failure)
    // Increase path loss dramatically
    Config::Set("/NodeList/" + std::to_string(nodeId) + 
               "/*/DeviceList/*/PathlossModel/FadeMargin", 
               DoubleValue(100.0)); // Add 100dB loss
    
    // Type 3: Congestion (buffer overflow)
    Config::Set("/NodeList/" + std::to_string(nodeId) + 
               "/*/DeviceList/*/TxQueue/MaxSize", 
               StringValue("10p")); // Limit to 10 packets
    
    NS_LOG_WARN("Fault injected at node " << nodeId << " at time " 
                << Simulator::Now().GetSeconds());
}

void IabFaultLocalizationSimulator::RecoverFromFault(uint32_t nodeId)
{
    NS_LOG_FUNCTION(this << nodeId);
    
    // Restore normal operation
    Config::Set("/NodeList/" + std::to_string(nodeId) + 
               "/*/DeviceList/*/PathlossModel/FadeMargin", 
               DoubleValue(0.0));
    Config::Set("/NodeList/" + std::to_string(nodeId) + 
               "/*/DeviceList/*/TxQueue/MaxSize", 
               StringValue("999999p"));
    
    NS_LOG_INFO("Node " << nodeId << " recovered at time " 
                << Simulator::Now().GetSeconds());
}


void IabFaultLocalizationSimulator::CollectTelemetryTraces()
{
    NS_LOG_FUNCTION(this);
    
    // Enable tracing for all nodes
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/Rx",
        MakeCallback(&IabFaultLocalizationSimulator::TraceRx, this));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/Tx",
        MakeCallback(&IabFaultLocalizationSimulator::TraceTx, this));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/MacTx",
        MakeCallback(&IabFaultLocalizationSimulator::TraceMacTx, this));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/MacRx",
        MakeCallback(&IabFaultLocalizationSimulator::TraceMacRx, this));
    
    // HARQ statistics
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/HarqAckRx",
        MakeCallback(&IabFaultLocalizationSimulator::TraceHarqAck, this));
    
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/HarqNackRx",
        MakeCallback(&IabFaultLocalizationSimulator::TraceHarqNack, this));
    
    // RLC buffer monitoring
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/RlcTxQueueSize",
        MakeCallback(&IabFaultLocalizationSimulator::TraceRlcBuffer, this));
    
    // Schedule periodic telemetry collection (every 100ms for O-RAN compliance)
    Time collectionInterval = MilliSeconds(100);
    for (Time t = Seconds(0.1); t < Seconds(10.0); t += collectionInterval)
    {
        Simulator::Schedule(t, &IabFaultLocalizationSimulator::CollectPeriodicMetrics, 
                           this, t);
    }
}

void IabFaultLocalizationSimulator::TraceRx(Ptr<const Packet> packet)
{
    uint32_t nodeId = GetCurrentNodeId();
    double timestamp = Simulator::Now().GetSeconds();
    double latency = CalculateLatency(packet);
    
    m_latencyTrace[nodeId].push_back(latency);
    // Store in time-series format
}

void IabFaultLocalizationSimulator::TraceMacTx(Ptr<const Packet> packet)
{
    uint32_t nodeId = GetCurrentNodeId();
    double throughput = packet->GetSize() * 8.0 / 1e6; // Mbps
    
    m_throughputTrace[nodeId].push_back(throughput);
}

void IabFaultLocalizationSimulator::TraceHarqNack()
{
    uint32_t nodeId = GetCurrentNodeId();
    double timestamp = Simulator::Now().GetSeconds();
    
    m_harqTrace[nodeId].push_back(1.0); // NACK event
}

void IabFaultLocalizationSimulator::CollectPeriodicMetrics(Time timestamp)
{
    // Collect aggregated metrics every 100ms
    for (uint32_t nodeId = 0; nodeId < m_numIabNodes; nodeId++)
    {
        // Calculate current metrics
        double avgLatency = CalculateAverageLatency(nodeId);
        double packetLossRate = CalculatePacketLossRate(nodeId);
        double throughput = CalculateThroughput(nodeId);
        double harqNackRate = CalculateHarqNackRate(nodeId);
        double rlcBufferSize = GetRlcBufferSize(nodeId);
        
        // Store in CSV format
        std::ofstream traceFile("telemetry_trace.csv", std::ios::app);
        traceFile << timestamp.GetSeconds() << ","
                  << nodeId << ","
                  << avgLatency << ","
                  << packetLossRate << ","
                  << throughput << ","
                  << harqNackRate << ","
                  << rlcBufferSize << std::endl;
        traceFile.close();
    }
}


void CreateMultiHopIabScenario()
{
    // Scenario: Donor -> IAB-1 -> IAB-2 -> IAB-3 (3-hop backhaul)
    
    // Configure backhaul links with different qualities
    // Hop 1: Strong link (close proximity)
    Config::Set("/NodeList/0/*/DeviceList/*/PathlossModel/Shadowing",
               DoubleValue(3.0)); // 3dB shadowing
    
    // Hop 2: Medium link
    Config::Set("/NodeList/1/*/DeviceList/*/PathlossModel/Shadowing",
               DoubleValue(6.0)); // 6dB shadowing
    
    // Hop 3: Weak link (edge coverage)
    Config::Set("/NodeList/2/*/DeviceList/*/PathlossModel/Shadowing",
               DoubleValue(10.0)); // 10dB shadowing
    
    // Inject fault at IAB-2 (middle hop)
    simulator.InjectFault(1, Seconds(2.0), Seconds(5.0));
}


void CreateBeamFailureScenario()
{
    // Simulate mmWave beam blockage
    // Configure beam management
    Config::SetDefault("ns3::NrGnbPhy::BeamManagementEnabled", 
                       BooleanValue(true));
    Config::SetDefault("ns3::NrGnbPhy::BeamSweepingPeriod", 
                       TimeValue(MilliSeconds(20)));
    
    // Schedule beam failure event
    Simulator::Schedule(Seconds(3.0), []() {
        // Simulate blockage by setting high path loss
        Config::Set("/NodeList/1/*/DeviceList/*/BeamformingWeight",
                   DoubleValue(0.0)); // Null beam
    });
    
    // Schedule beam recovery
    Simulator::Schedule(Seconds(6.0), []() {
        // Restore beam
        Config::Set("/NodeList/1/*/DeviceList/*/BeamformingWeight",
                   DoubleValue(1.0));
    });
}



void IabFaultLocalizationSimulator::ExportTracesToCsv(const std::string& filename)
{
    std::ofstream outFile(filename);
    
    // CSV header
    outFile << "timestamp,node_id,latency_ms,packet_loss_percent,"
            << "throughput_mbps,harq_nack_rate,rlc_buffer_size,"
            << "fault_label" << std::endl;
    
    // Write time-series data
    for (const auto& [timestamp, metrics] : m_timeSeriesData)
    {
        outFile << timestamp << ","
                << metrics.nodeId << ","
                << metrics.latency << ","
                << metrics.packetLoss << ","
                << metrics.throughput << ","
                << metrics.harqNackRate << ","
                << metrics.rlcBufferSize << ","
                << (metrics.isFaulty ? "1" : "0") << std::endl;
    }
    
    outFile.close();
    NS_LOG_INFO("Traces exported to " << filename);
}

// Python script for post-processing (process_traces.py)
/*
import pandas as pd
import numpy as np

def process_iab_traces(csv_file):
    df = pd.read_csv(csv_file)
    
    # Create graph topology
    topology = {
        'nodes': df['node_id'].unique(),
        'edges': []  # Define parent-child relationships
    }
    
    # Generate fault labels
    df['fault_injected'] = df['fault_label'].shift(1).fillna(0)
    
    # Create sliding windows for temporal learning
    window_size = 10  # 1 second at 100ms intervals
    sequences = []
    labels = []
    
    for node_id in topology['nodes']:
        node_data = df[df['node_id'] == node_id].sort_values('timestamp')
        
        for i in range(len(node_data) - window_size):
            window = node_data.iloc[i:i+window_size][
                ['latency_ms', 'packet_loss_percent', 'throughput_mbps']
            ].values
            
            label = node_data.iloc[i+window_size]['fault_injected']
            
            sequences.append(window)
            labels.append(label)
    
    return np.array(sequences), np.array(labels), topology

# Export for PyTorch
sequences, labels, topology = process_iab_traces('telemetry_trace.csv')
np.save('sequences.npy', sequences)
np.save('labels.npy', labels)
import json
with open('topology.json', 'w') as f:
    json.dump(topology, f)
*/



// main-iab-simulation.cc
int main(int argc, char *argv[])
{
    LogComponentEnable("IabFaultLocalization", LOG_LEVEL_INFO);
    LogComponentEnable("NrHelper", LOG_LEVEL_WARN);
    
    // Simulation parameters
    uint32_t numDonors = 1;
    uint32_t numIabNodes = 20;
    uint32_t uePerIab = 2;
    double simDuration = 10.0; // seconds
    
    CommandLine cmd(__FILE__);
    cmd.AddValue("IabNodes", "Number of IAB nodes", numIabNodes);
    cmd.AddValue("Duration", "Simulation duration (s)", simDuration);
    cmd.Parse(argc, argv);
    
    // Initialize simulator
    IabFaultLocalizationSimulator simulator(numDonors, numIabNodes, uePerIab);
    
    // Configure topology
    simulator.ConfigureIabTopology();
    
    // Create traffic flows
    // (UDP/TCP flows from UEs to core network)
    
    // Enable telemetry collection
    simulator.CollectTelemetryTraces();
    
    // Inject faults for training data
    simulator.InjectFault(5, Seconds(2.0), Seconds(5.0));   // IAB node failure
    simulator.InjectFault(12, Seconds(6.0), Seconds(3.0));  // Wireless blockage
    
    // Run simulation
    Simulator::Stop(Seconds(simDuration));
    Simulator::Run();
    
    // Export results
    simulator.ExportTracesToCsv("iab_fault_traces.csv");
    
    Simulator::Destroy();
    return 0;
}




